# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM pooled indexing and NoPE MLA, with independent kernel boundaries.

T = tokens, D = hidden width, H = attention heads, L = KV latent width,
I = index heads, E = index width, P = pools, S = selected pools, W = pool size.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import register_dataclass

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseMetadata(VarlenAttentionMetadata):
    compressor_ratio: int
    cu_seq_pools: torch.Tensor
    pool_rows_PW: torch.Tensor  # noqa: N815
    pool_ids_P: torch.Tensor  # noqa: N815
    pool_offsets_W: torch.Tensor  # noqa: N815
    token_start_T: torch.Tensor  # noqa: N815
    pool_start_T: torch.Tensor  # noqa: N815
    pool_end_T: torch.Tensor  # noqa: N815
    tail_indices_TW: torch.Tensor  # noqa: N815
    positions_T: torch.Tensor  # noqa: N815

    def annotate_spmd_types(self) -> None:
        spmd.assert_type(
            (self.cu_seq_q, self.cu_seq_k, self.cu_seq_pools), self._OFFSETS_SPMD_TYPE
        )


register_dataclass(SparseMetadata)


class Compressor(Module):
    """Channel-wise weighted pooling of complete, document-local key groups."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        head_dim: int
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.ape = nn.Parameter(torch.empty(config.ratio, config.head_dim))

    def forward(self, k_TE, x_TD, *, attention_metadata: SparseMetadata):
        rows_PW = attention_metadata.pool_rows_PW
        k_PWE = k_TE[rows_PW]
        logits_PWE = self.gate(x_TD)[rows_PW].float() + self.ape.float()
        return (logits_PWE.softmax(1).to(k_TE.dtype) * k_PWE).sum(1)


class InnerIndexer(Module):
    """Select pool IDs and expose differentiable student logits; no token tails."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        topk: int
        ratio: int

    def __init__(self, config: Config):
        super().__init__()
        self.topk, self.ratio = config.topk, config.ratio

    def forward(self, q_TIE, k_PE, w_TI, *, attention_metadata: SparseMetadata):
        topk = min(self.topk, attention_metadata.max_k // self.ratio, k_PE.size(0))
        if topk == 0:
            return (
                torch.empty((q_TIE.size(0), 0), dtype=torch.int32, device=q_TIE.device),
                q_TIE[:, 0, :0],
            )
        with torch.no_grad(), torch.autocast(
            device_type=q_TIE.device.type, enabled=False
        ):
            logits_TIP = torch.einsum("tie,pe->tip", q_TIE.float(), k_PE.float())
            scores_TP = (logits_TIP.relu() * w_TI.float().unsqueeze(-1)).sum(1)
            valid_TP = (
                attention_metadata.pool_ids_P[None]
                >= attention_metadata.pool_start_T[:, None]
            ) & (
                attention_metadata.pool_ids_P[None]
                < attention_metadata.pool_end_T[:, None]
            )
            values_TS, rows_TS = scores_TP.masked_fill(~valid_TP, -torch.inf).topk(
                topk, dim=-1
            )
            indices_TS = (rows_TS - attention_metadata.pool_start_T[:, None]).int()
            indices_TS = indices_TS.masked_fill(~torch.isfinite(values_TS), -1)
        # FP32 scoring matches the fused indexer's accumulation contract.
        with torch.autocast(device_type=q_TIE.device.type, enabled=False):
            logits_TIS = torch.einsum(
                "tie,tse->tis", q_TIE.float(), k_PE[rows_TS].float()
            )
            scores_TS = (logits_TIS.relu() * w_TI.float().unsqueeze(-1)).sum(1)
        return indices_TS, scores_TS.masked_fill(indices_TS < 0, -torch.inf)


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_index_heads: int
        index_head_dim: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        compressor: Compressor.Config
        inner_indexer: InnerIndexer.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.compressor = config.compressor.build()
        self.inner_indexer = config.inner_indexer.build()

    def forward(self, q_TD, x_TD, *, attention_metadata: SparseMetadata):
        q_TIE = self.wq_b(q_TD).unflatten(
            -1, (self.num_index_heads, self.index_head_dim)
        )
        k_PE = self.compressor(
            self.k_norm(self.wk(x_TD)), x_TD, attention_metadata=attention_metadata
        )
        w_TI = (
            self.weights_proj(x_TD)
            * (self.index_head_dim * self.num_index_heads) ** -0.5
        )
        return self.inner_indexer(
            q_TIE, k_PE, w_TI, attention_metadata=attention_metadata
        )


class IndexerLoss(AuxLoss):
    """Mass-weighted KL on selected complete pools, with a detached teacher.

    p is the head-mean attention probability summed over each selected pool.
    The tail is excluded: Z=sum(p) can be less than one. The objective is
    Z * KL(p/Z || softmax(scores)); its score gradient is softmax(scores)*Z-p.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        chunk_size: int = 32

    def __init__(self, config: Config):
        super().__init__(config)
        self.chunk_size = config.chunk_size

    def kl(self, p_TS, scores_TS):
        valid_TS = torch.isfinite(scores_TS)
        p_TS = p_TS.masked_fill(~valid_TS, 0)
        t_TS = p_TS / p_TS.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        log_y_TS = scores_TS.float().masked_fill(~valid_TS, -1e30).log_softmax(-1)
        return (torch.special.xlogy(p_TS, t_TS) - p_TS * log_y_TS).sum()

    def forward(
        self,
        q_THL,
        kv_TL,
        selected_TSW,
        scores_TS,
        lse_TH,
        *,
        attention_metadata: SparseMetadata,
        scale,
        carrier,
        denominator,
        padding_mask
    ):
        if scores_TS.size(-1) == 0:
            return carrier
        if denominator is None:
            raise ValueError("IndexerLoss requires the step's valid-token denominator.")
        targets = []
        with torch.no_grad(), torch.autocast(
            device_type=q_THL.device.type, enabled=False
        ):
            for start in range(0, q_THL.size(0), self.chunk_size):
                stop = start + self.chunk_size
                selected_CSW = selected_TSW[start:stop]
                rows_CSW = (
                    selected_CSW
                    + attention_metadata.token_start_T[start:stop, None, None]
                ).clamp_min(0)
                logits_CHSW = torch.einsum(
                    "chl,cswl->chsw", q_THL[start:stop].float(), kv_TL[rows_CSW].float()
                )
                logits_CHSW = (
                    logits_CHSW * scale - lse_TH[start:stop, :, None, None].float()
                )
                p_CSW = (
                    logits_CHSW.masked_fill(selected_CSW[:, None] < 0, -torch.inf)
                    .exp()
                    .mean(1)
                )
                targets.append(p_CSW.sum(-1))
            p_TS = torch.cat(targets)
            if padding_mask is not None:
                p_TS = p_TS.masked_fill(padding_mask[:, None], 0)
        return self.inject(
            self.kl(p_TS, scores_TS), carrier=carrier, denominator=denominator
        )


class SparseAttention(InnerAttention):
    """Expand selected pools and append the incomplete tail before gather_attn."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        ratio: int
        aux_loss: IndexerLoss.Config | None

        @property
        def is_full_attention(self):
            return True

        def build_attention_metadata(
            self,
            positions,
            *,
            padding_mask=None,
            max_num_documents=None,
            max_context_length=None
        ):
            varlen = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            cu = varlen.cu_seq_q
            counts = (cu[1:] - cu[:-1]) // self.ratio
            cu_pools = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
            pools_P = torch.arange(
                int(cu_pools[-1]), dtype=torch.int32, device=positions.device
            )
            pool_docs_P = torch.searchsorted(cu_pools[1:], pools_P, right=True)
            tokens_T = torch.arange(positions.numel(), device=positions.device)
            docs_T = torch.searchsorted(cu[1:], tokens_T, right=True)
            offsets_W = torch.arange(
                self.ratio, dtype=torch.int32, device=positions.device
            )
            pool_starts_P = (
                cu[pool_docs_P] + (pools_P - cu_pools[pool_docs_P]) * self.ratio
            )
            visible_T = positions.to(torch.int32) + 1
            tail_count_T = visible_T % self.ratio
            tail_TW = (visible_T - tail_count_T)[:, None] + offsets_W[:-1]
            tail_TW = tail_TW.masked_fill(offsets_W[:-1] >= tail_count_T[:, None], -1)
            return SparseMetadata(
                cu_seq_q=cu,
                cu_seq_k=cu,
                max_q=varlen.max_q,
                max_k=varlen.max_k,
                compressor_ratio=self.ratio,
                cu_seq_pools=cu_pools,
                pool_rows_PW=pool_starts_P[:, None] + offsets_W,
                pool_ids_P=pools_P,
                pool_offsets_W=offsets_W,
                token_start_T=cu[docs_T],
                pool_start_T=cu_pools[docs_T],
                pool_end_T=cu_pools[docs_T] + visible_T // self.ratio,
                tail_indices_TW=tail_TW,
                positions_T=positions,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.ratio = config.ratio
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def selection(self, indices_TS, *, attention_metadata: SparseMetadata):
        selected_TSW = (
            indices_TS[..., None] * self.ratio + attention_metadata.pool_offsets_W
        )
        selected_TSW = selected_TSW.masked_fill(indices_TS[..., None] < 0, -1)
        return selected_TSW, attention_metadata.tail_indices_TW

    def forward(
        self,
        q_THL,
        kv_TL,
        indices_TS,
        scores_TS,
        *,
        attention_metadata: SparseMetadata,
        scale,
        aux_loss_denominator=None,
        padding_mask=None
    ):
        selected_TSW, tail_TW = self.selection(
            indices_TS, attention_metadata=attention_metadata
        )
        indices_TK = torch.cat((selected_TSW.flatten(-2), tail_TW), -1)
        aux_loss = self.aux_loss if self.training else None
        out_1HTL, aux = gather_attn(
            q_THL.transpose(0, 1)[None],
            None,
            kv_TL[None, None],
            indices_TK[None],
            sliding_window_size=0,
            cu_seqlens=attention_metadata.cu_seq_q,
            cu_seqlens_k=attention_metadata.cu_seq_k,
            scale=scale,
            impl="reference",
            return_aux=AuxRequest(lse=aux_loss is not None),
        )
        out_THL = out_1HTL[0].transpose(0, 1)
        if aux_loss is not None:
            assert aux.lse is not None
            out_THL = aux_loss(
                q_THL.detach(),
                kv_TL.detach(),
                selected_TSW,
                scores_TS,
                aux.lse[0].transpose(0, 1).detach(),
                attention_metadata=attention_metadata,
                scale=scale,
                carrier=out_THL,
                denominator=aux_loss_denominator,
                padding_mask=padding_mask,
            )
        return out_THL


class GlmDSA(Attention):
    """NoPE MLA projections, with detached inputs to the trainable indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        indexer: Indexer.Config
        inner_attention: SparseAttention.Config  # pyrefly: ignore[bad-override]

        def flops(self, positions):
            from torchtitan.models.common.flops import quadratic_attention_flops

            index = self.indexer
            ratio = self.inner_attention.ratio
            pools = (positions + 1) // ratio
            selected = pools.clamp(max=index.inner_indexer.topk) * ratio
            selected = selected + (positions + 1) % ratio
            return quadratic_attention_flops(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                num_keys=selected,
            ) + 2 * index.num_index_heads * index.index_head_dim * int(pools.sum())

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads, self.head_dim = config.n_heads, config.head_dim
        self.scale = config.head_dim**-0.5
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.indexer = config.indexer.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None
    ):
        q_TD = self.q_norm(self.wq_a(x_TD))
        q_THK = self.wq_b(q_TD).unflatten(-1, (self.n_heads, self.head_dim))
        kv_TL = self.kv_norm(self.wkv_a(x_TD))
        w_HKL = self.wkv_b.weight.unflatten(0, (self.n_heads, 2 * self.head_dim))
        w_uk_HKL, w_uv_HKL = w_HKL.chunk(2, dim=1)
        q_THL = torch.einsum("thk,hkl->thl", q_THK, w_uk_HKL)
        indices_TS, scores_TS = self.indexer(
            q_TD.detach(), x_TD.detach(), attention_metadata=attention_metadata
        )
        out_THL = self.inner_attention(
            q_THL,
            kv_TL,
            indices_TS,
            scores_TS,
            attention_metadata=attention_metadata,
            scale=self.scale,
            padding_mask=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        out_THK = torch.einsum("thl,hkl->thk", out_THL, w_uv_HKL)
        return self.wo(out_THK.flatten(-2))
