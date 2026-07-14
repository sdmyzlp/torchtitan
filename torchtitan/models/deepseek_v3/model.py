# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    create_mask,
)

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
)
from torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.nn_modules import LayerNorm, Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module


@functools.cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    assert dim & (dim - 1) == 0, "Hadamard dim must be a power of two"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        index_n_heads: int
        index_head_dim: int
        rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: RoPE.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk

        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        idx_q = Indexer._hadamard_rotate(torch.cat([q_pe, q_nope], dim=-1))
        idx_k = Indexer._hadamard_rotate(torch.cat([k_pe.squeeze(2), k_nope], dim=-1))

        idx_w = self.weights_proj(x) * (self.n_heads**-0.5)
        idx_w = idx_w * (self.head_dim**-0.5)

        return idx_q, idx_w, idx_k

    @staticmethod
    def select(
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        block_mask: BlockMask,
        index_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Lq, _, _ = idx_q.shape
        Lkv = idx_k.shape[1]

        scores = torch.relu(
            torch.einsum("blhd,bsd->blhs", idx_q.float(), idx_k.float())
        )
        index_scores = (scores * idx_w.unsqueeze(-1).float()).sum(dim=2)

        valid = create_mask(
            block_mask.mask_mod, B, 1, Lq, Lkv, device=idx_q.device
        ).squeeze(1)
        index_scores = index_scores.masked_fill(~valid, float("-inf"))

        k = min(index_topk, Lkv)
        topk_scores, topk_indices = index_scores.topk(k, dim=-1)
        return topk_indices.where(topk_scores.isfinite(), -1), index_scores

    @staticmethod
    def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, device=x.device, dtype=x.dtype)
        return F.linear(x.reshape(-1, d), H).reshape(x.shape) * (d ** -0.5)


class DSAIndexerAuxLoss(LoggedAuxLoss):
    """Indexer alignment loss for the sparse training stage (DSA paper eq. 4).

    Follows the paper and Megatron-LM's reference implementation:
      1. Per-head attention scores: softmax over keys, restricted to S_t.
      2. Aggregate across heads: sum then L1-normalize → target p.
      3. Indexer distribution: softmax over index scores → q.
      4. KL(p || q) summed over query tokens; gradient flows only to
         the indexer (q/k are detached from the main model's graph).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        coeff: float = 1.0
        tag: str = "dsa_indexer_loss"
        reduce_mesh: str = "loss"

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float | None,
        selected: torch.Tensor,
        index_scores: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.einsum("blnh,bsnh->blns", q.float(), k.float())
        if scale is not None:
            logits = logits * scale

        logits = logits.masked_fill(~selected.unsqueeze(2), float("-inf"))
        p = F.softmax(logits, dim=-1).mean(dim=2)

        scores = index_scores.masked_fill(~selected, float("-inf"))
        q = F.softmax(scores, dim=-1)

        eps = 1e-10
        kl_loss = (p * ((p + eps).log() - (q + eps).log())).sum(dim=-1).mean()
        return self.inject(carrier, kl_loss)


class DeepSeekSparseAttention(FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        index_topk: int
        indexer_loss: DSAIndexerAuxLoss.Config = field(
            default_factory=DSAIndexerAuxLoss.Config
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = config.indexer_loss.build()

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        idx_q_BLNH: torch.Tensor,
        idx_k_BLH: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        *,
        attention_masks: BlockMask,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        topk_indices, index_scores = Indexer.select(
            idx_q_BLNH, idx_k_BLH, idx_w_BLN,
            attention_masks, self.index_topk,
        )

        def _build_selected(
            topk_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, BlockMask]:
            B, Lq = q_BLNH.shape[:2]
            Lkv = k_BLNH.shape[1]

            mask = torch.zeros(
                B, Lq, Lkv, dtype=torch.bool, device=topk_indices.device
            ).scatter_add_(-1, topk_indices.clamp(min=0), topk_indices != -1)

            with spmd.no_typecheck():
                block_mask = create_block_mask(
                    lambda b, h, q_idx, k_idx: mask[b, q_idx, k_idx],
                    B=B, H=None,
                    Q_LEN=Lq, KV_LEN=Lkv,
                    device=topk_indices.device,
                    BLOCK_SIZE=attention_masks.BLOCK_SIZE,
                    _compile=True,
                )
            return mask, block_mask

        selected, selected_bm = _build_selected(topk_indices)

        output = super().forward(
            q_BLNH, k_BLNH, v_BLNH,
            attention_masks=selected_bm,
            scale=scale,
        )
        if self.training:
            output = self.indexer_loss(
                q_BLNH.detach(), k_BLNH.detach(),
                scale, selected, index_scores, carrier=output,
            )
        return output


class Attention(BaseAttention):
    """
    Multi-head latent attention (MLA) module.

    This is DeepSeek V3-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        wq: Linear.Config | None = None
        wq_a: Linear.Config | None = None
        wq_b: Linear.Config | None = None
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope: RoPE.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)
        mscale: float = 1.0
        indexer: Indexer.Config | None = None

        def __post_init__(self):
            if self.q_lora_rank == 0:
                assert self.indexer is None, (
                    "indexer requires q_lora_rank > 0"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            assert config.wq is not None, "wq is required when q_lora_rank == 0"
            self.wq = config.wq.build()
        else:
            assert (
                config.wq_a is not None and config.wq_b is not None
            ), "wq_a and wq_b are required when q_lora_rank > 0"
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope.max_seq_len > config.rope.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()
        if config.indexer is not None:
            self.indexer = config.indexer.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            qr = self.q_norm(q)
            q = self.wq_b(qr)

        # TODO(pianpwk): same QKV:S(2) unflatten case handled by even sharding
        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types":
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():  # QKV even shard unflatten, but the expand is truly local SPMD
            kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, k_nope.size(2), -1)], dim=-1)
            if get_spmd_backend() == "spmd_types" and not torch.compiler.is_compiling():
                for t in [k, v]:
                    spmd.assert_type(
                        t,
                        {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                    )

        if self.indexer is not None:
            idx_q, idx_w, idx_k = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    idx_q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )
                spmd.assert_type(
                    idx_w,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )
                spmd.assert_type(
                    idx_k,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.R},
                )

            output = self.inner_attention(
                q, k, v,
                idx_q, idx_k, idx_w,
                attention_masks=attention_masks,
                scale=self.softmax_scale,
            )
        else:
            output = self.inner_attention(
                q, k, v,
                attention_masks=attention_masks,
                scale=self.softmax_scale,
            )
        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class DeepSeekV3TransformerBlock(TransformerBlock):
    """
    DeepSeek V3 Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class DeepSeekV3Model(Decoder):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.deepseek_v3.sharding import (
                set_deepseek_v3_sharding_config,
                set_deepseek_v3_indexer_sharding_config,
            )

            set_deepseek_v3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
            if self.use_sparse_attn:
                set_deepseek_v3_indexer_sharding_config(self)

        @property
        def use_sparse_attn(self) -> bool:
            return any(
                layer.attention.indexer is not None
                for layer in self.layers
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:

            assert isinstance(self.layers[0].attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                self.layers[0].attention.qk_nope_head_dim
                + self.layers[0].attention.qk_rope_head_dim
                + self.layers[0].attention.v_head_dim,
                seq_len,
            )
