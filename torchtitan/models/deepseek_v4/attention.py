# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import BaseAttention, FlexAttention
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


def _assert_spmd_attention_type(tensor, *, tp):
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


class CompressedSparseInnerAttention(FlexAttention):
    """DeepSeek sparse attention core for DeepSeek-V4.

    The core attends over the concatenated KV sequence ``[0, L + n_cmp + 1)``,
    where the first ``L`` positions are the uncompressed sliding-window KV
    (``swa_k``), the next ``n_cmp`` positions are the compressed KV
    (``cmp_k``), and the last position is a learned attention sink token:

    - sliding window: fixed pattern over ``swa_k``, expressed as a
      ``mask_mod`` predicate (no indices);
    - compressed blocks: for HCA (``compress_ratio=128``) all causal blocks
      are attendable, also a fixed ``mask_mod`` pattern; for CSA
      (``compress_ratio=4``) each query attends only its top-k selected
      compressed positions, which is the only dynamic (index-based) part;
    - attention sink: always attendable via ``score_mod``.

    The ``mask_mod`` is evaluated at token granularity inside flex_attention;
    the per-query-block KV block listing (``BlockMask.from_kv_blocks``) only
    restricts which blocks the kernel loads.

    Overrides can replace ``_build_block_mask`` (e.g. NPU varlen kernels) or
    the whole ``forward`` (e.g. fused SMLA/CSA kernels, which consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors). Under context
    parallelism, all-gathering ``idx_k`` and ``cmp_k`` at this module boundary
    enables global sparse selection.

    TODO: the indexer auxiliary loss is intentionally dropped for now; it will
    be re-added as a carrier-injected aux loss (see the NPU fork) once the
    general aux-loss mechanism lands.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.block_size = config.block_size

    def _build_block_mask(
        self,
        bsz: int,
        seqlen: int,
        n_cmp: int,
        topk_indices: torch.Tensor | None,
        device,
    ) -> BlockMask:
        """Build the DSA block mask for the concatenated KV sequence.

        Lists, per query block, the union of attendable KV blocks (sliding
        window, compressed top-k blocks, and the sink block) and returns a
        token-granular ``mask_mod`` that refines the listing.
        """
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        assert (
            seqlen % bq == 0
        ), f"seqlen ({seqlen}) must be divisible by Q block size ({bq})"
        kv_len = seqlen + n_cmp + 1
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq
        sink_idx = seqlen + n_cmp
        ratio = self.compress_ratio

        # Per-query-block KV block listing: 1 for every block attended by at
        # least one token in the query block, 0 otherwise.
        bm = torch.zeros(
            bsz, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device
        )
        q_block_ids = torch.arange(n_q_blocks, device=device).unsqueeze(1)
        q0 = q_block_ids * bq
        if topk_indices is not None:
            # compressed token c sits at KV position seqlen + c
            cmp_block_of = (seqlen + torch.arange(n_cmp, device=device)) // bk
            block_of_topk = cmp_block_of[topk_indices].reshape(
                bsz, n_q_blocks, bq * topk_indices.size(-1)
            )
            bm[:, 0].scatter_add_(
                -1,
                block_of_topk.clamp(0, n_kv_blocks - 1),
                torch.ones_like(block_of_topk, dtype=torch.int32),
            )
        elif ratio > 1:
            # HCA: every causal compressed block is attendable. The last
            # attendable compressed token of query block qb is
            # ((qb + 1) * bq) // ratio - 1; list its block (and earlier ones).
            last_cmp = ((q0 + bq) // ratio - 1).clamp_min(-1)
            first_cmp_block = seqlen // bk
            last_cmp_block = (seqlen + last_cmp).clamp_min(0) // bk
            kv_block_ids2 = torch.arange(n_kv_blocks, device=device).unsqueeze(0)
            cmp_blocks = (kv_block_ids2 >= first_cmp_block) & (
                kv_block_ids2 <= last_cmp_block
            )
            bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | cmp_blocks.to(torch.int32)
        first_window_block = (q0 - self.window_size + 1).clamp_min(0) // bk
        last_window_block = (q0 + bq - 1) // bk
        kv_block_ids = torch.arange(n_kv_blocks, device=device).unsqueeze(0)
        window_blocks = (kv_block_ids >= first_window_block) & (
            kv_block_ids <= last_window_block
        )
        bm[:, 0] = (bm[:, 0] > 0).to(torch.int32) | window_blocks.to(torch.int32)
        bm[:, 0, :, sink_idx // bk] = 1
        bm = (bm > 0).to(torch.int32)
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        # Per-token compressed selection lookup for the mask_mod gather.
        cmp_sel = torch.zeros(
            bsz, seqlen, max(n_cmp, 1), dtype=torch.bool, device=device
        )
        if topk_indices is not None:
            cmp_sel.scatter_(2, topk_indices.clamp(0, max(n_cmp, 1) - 1), True)

        def dsa_mask_mod(b, h, q_idx, kv_idx):
            swa = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < self.window_size)
            )
            is_sink = kv_idx == sink_idx
            if ratio > 1:
                cmp = (kv_idx >= seqlen) & (kv_idx < seqlen + n_cmp)
                causal = (kv_idx - seqlen) < (q_idx + 1) // ratio
                if topk_indices is not None:
                    topk_sel = cmp_sel[
                        b, q_idx, (kv_idx - seqlen).clamp(0, max(n_cmp, 1) - 1)
                    ]
                    return swa | (cmp & causal & topk_sel) | is_sink
                return swa | (cmp & causal) | is_sink
            return swa | is_sink

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=dsa_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attn_sink=None,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "CompressedSparseInnerAttention does not accept attention_masks; "
                "the DSA block mask is built internally."
            )
        if attn_sink is None:
            raise ValueError("CompressedSparseInnerAttention requires attn_sink")

        bsz, seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(1)
        sink_idx = seqlen + n_cmp

        topk_indices = None
        if self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError(
                    "CompressedSparseInnerAttention requires idx_q, idx_k, and idx_w "
                    "when compress_ratio=4"
                )
            topk_indices = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                seqlen=seqlen,
                ratio=self.compress_ratio,
                topk=self.index_topk,
            )

        kv = swa_k.unsqueeze(2)
        if cmp_k is not None:
            kv = torch.cat([kv, cmp_k.unsqueeze(2)], dim=1)
        sink_kv = kv.new_zeros((bsz, 1, 1, head_dim))
        kv = torch.cat([kv, sink_kv], dim=1)

        with spmd.no_typecheck():
            block_mask = self._build_block_mask(
                bsz, seqlen, n_cmp, topk_indices, q.device
            )

            def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
                return torch.where(kv_idx == sink_idx, attn_sink[h], score)

            return super().forward(
                q,
                kv,
                kv,
                attention_masks=block_mask,
                score_mod=v4_sink_score_mod,
                scale=self.softmax_scale,
                enable_gqa=True,
            )


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()

        # Build all sub-modules from their configs.
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build()

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()

    def forward(self, x, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        _assert_spmd_attention_type(qr, tp=spmd.R)
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(2))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        _assert_spmd_attention_type(kv, tp=spmd.R)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(2)], dim=-1)
        _assert_spmd_attention_type(q, tp=spmd.S(2))
        _assert_spmd_attention_type(kv, tp=spmd.R)

        cmp_k = idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        o = self.inner_attention(
            q,
            kv,
            cmp_k,
            idx_q,
            idx_k,
            idx_w,
            attn_sink=attn_sink_param,
            attention_masks=attention_masks,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)
        _assert_spmd_attention_type(o, tp=spmd.S(2))

        with spmd.local():
            n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
            o = o.view(bsz, seqlen, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(2))
            # wo_a is a Linear module; access its weight directly for the grouped
            # einsum (not a standard Linear forward).
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        with spmd.local():
            o = o.reshape(bsz, seqlen, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(2))
        return self.wo_b(o)
