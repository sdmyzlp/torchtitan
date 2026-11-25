# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM sparse attention on Ascend 950: LI -> SMLA -> SMLAG -> SLIKG.

The scores edge transports a teacher, not the derivative of top-k selection.
T = tokens, H = heads, D = latent width, S = selected pools, W = pool size.
"""

from dataclasses import dataclass, field, fields, replace

import torch
import torch.nn.functional as F
from torch.export import register_dataclass

from torchtitan.config import derive, override
from torchtitan.models.glm53_flash.attention import (
    GlmDSA,
    InnerIndexer,
    SparseAttention,
    SparseMetadata,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AscendSparseMetadata(SparseMetadata):
    num_heads: int
    num_index_heads: int
    topk: int
    indexer_options: dict = field(default_factory=dict)
    sparse_options: dict = field(default_factory=dict)
    indexer_metadata: torch.Tensor | None = None
    indexer_grad_metadata: torch.Tensor | None = None
    sparse_metadata: torch.Tensor | None = None
    sparse_grad_metadata: torch.Tensor | None = None
    sinks: torch.Tensor | None = None


register_dataclass(AscendSparseMetadata)


def _global_metadata(config, positions, **kwargs):
    metadata = SparseAttention.Config.build_attention_metadata(
        config, positions, **kwargs
    )
    return AscendSparseMetadata(
        **{f.name: getattr(metadata, f.name) for f in fields(SparseMetadata)},
        num_heads=config.num_heads,
        num_index_heads=config.num_index_heads,
        topk=config.topk,
    )


def _build_kernel_metadata(metadata):
    ops = torch.ops.cann_ops_transformer
    ratio = metadata.compressor_ratio
    positions = metadata.positions_T
    topk = min(metadata.topk, metadata.max_k // ratio, metadata.pool_ids_P.numel())
    # This CANN pre-kernel must assign a nonempty output range to every core.
    cores = (
        torch.get_device_module(positions.device)
        .get_device_properties(positions.device)
        .vector_core_num
    )
    for count in (positions.numel() * (ratio - 1), positions.numel() * topk * ratio):
        if count:
            per_core = (count + cores - 1) // cores
            if (count + per_core - 1) // per_core < cores:
                raise ValueError(
                    "CANN sparse backward requires a larger query shard/top-k "
                    "to fill every vector core's initialization range."
                )
    end = metadata.token_start_T[0] + positions[0] + positions.numel()
    lengths = metadata.cu_seq_k.diff()
    visible = (end - metadata.cu_seq_k[:-1]).clamp_min(0).minimum(lengths)
    indexer_options = dict(
        cu_seqlens_q=metadata.cu_seq_q,
        cu_seqlens_k=metadata.cu_seq_pools,
        seqused_k=(visible // ratio).int(),
        cmp_residual_k=(visible % ratio).int(),
        layout_q="TND",
        layout_k="TND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    indexer_geometry = dict(
        num_heads_q=metadata.num_index_heads, num_heads_k=1, head_dim=128, topk=topk
    )
    has_cmp = topk > 0
    sparse_options = dict(
        cu_seqlens_q=metadata.cu_seq_q,
        cu_seqlens_ori_kv=metadata.cu_seq_k,
        cu_seqlens_cmp_kv=metadata.cu_seq_k if has_cmp else None,
        ori_topk_length=((positions + 1) % ratio).int()[:, None],
        cmp_topk_length=(((positions + 1) // ratio).clamp_max(topk) * ratio).int()[
            :, None
        ]
        if has_cmp
        else None,
        layout_q="TND",
        layout_kv="TND",
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
    )
    sparse_geometry = dict(
        num_heads_q=metadata.num_heads,
        num_heads_kv=1,
        head_dim=512,
        ori_topk=ratio - 1,
        cmp_topk=topk * ratio,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    return replace(
        metadata,
        topk=topk,
        indexer_options=indexer_options,
        sparse_options=sparse_options,
        indexer_metadata=ops.lightning_indexer_metadata(
            **indexer_geometry, **indexer_options
        )
        if has_cmp
        else None,
        indexer_grad_metadata=ops.sparse_lightning_indexer_kl_loss_grad_metadata(
            **indexer_geometry, **indexer_options
        )
        if has_cmp
        else None,
        sparse_metadata=ops.sparse_flash_mla_metadata(
            **sparse_geometry, **sparse_options
        ),
        sparse_grad_metadata=ops.sparse_flash_mla_grad_metadata(
            **sparse_geometry, **sparse_options
        ),
        # This CANN build requires sinks; -inf contributes exactly zero mass.
        sinks=torch.full(
            (metadata.num_heads,),
            -torch.inf,
            device=positions.device,
            dtype=torch.float32,
        ),
    )


class _LightningIndexer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, w, attention_metadata):  # pyrefly: ignore[bad-override]
        metadata = attention_metadata
        indices, scores = torch.ops.cann_ops_transformer.lightning_indexer(
            q,
            k,
            w,
            metadata.topk,
            return_value=1,
            metadata=metadata.indexer_metadata,
            **metadata.indexer_options,
        )
        ctx.attention_metadata = metadata
        ctx.save_for_backward(q, k, w, indices)
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return indices, scores

    @staticmethod
    def backward(ctx, grad_indices, teacher):  # pyrefly: ignore [bad-override]
        if teacher is None:
            return (None,) * 4
        q, k, w, indices = ctx.saved_tensors
        (
            dq,
            dk,
            dw,
            _,
        ) = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q,
            k,
            w,
            indices,
            teacher.contiguous(),
            metadata=ctx.attention_metadata.indexer_grad_metadata,
            **ctx.attention_metadata.indexer_options,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dw.to(w.dtype), None


class AscendInnerIndexer(InnerIndexer):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, q_TIE, k_PE, w_TI, *, attention_metadata):
        metadata = attention_metadata
        q, k, w = q_TIE, k_PE, w_TI
        if metadata.topk == 0:
            return super().forward(q, k, w, attention_metadata=metadata)
        # Zero-padding preserves the debug model's dot products and gradients.
        indices, scores = _LightningIndexer.apply(
            F.pad(q, (0, 128 - q.size(-1))),
            F.pad(k, (0, 128 - k.size(-1)))[:, None],
            w.float(),
            metadata,
        )
        return indices[:, 0], scores[:, 0]


class _SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        q,
        kv,
        selected,
        tail,
        scores,
        metadata,
        scale,
        aux_loss,
        denominator,
        padding_mask,
    ):
        ops = torch.ops.cann_ops_transformer
        has_cmp = selected.size(-1) > 0
        sinks = metadata.sinks
        out, lse = ops.sparse_flash_mla(
            q,
            ori_kv=kv,
            cmp_kv=kv if has_cmp else None,
            ori_sparse_indices=tail,
            cmp_sparse_indices=selected if has_cmp else None,
            sinks=sinks,
            softmax_scale=scale,
            return_softmax_lse=True,
            metadata=metadata.sparse_metadata,
            **metadata.sparse_options,
        )
        ctx.attention_metadata = metadata
        ctx.scale, ctx.ratio = scale, metadata.compressor_ratio
        ctx.aux_loss = aux_loss
        ctx.has_cmp = has_cmp
        ctx.save_for_backward(
            q, kv, selected, tail, scores, out, lse, sinks, denominator, padding_mask
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pyrefly: ignore [bad-override]
        (
            q,
            kv,
            selected,
            tail,
            scores,
            out,
            lse,
            sinks,
            denominator,
            padding_mask,
        ) = ctx.saved_tensors
        (
            dq,
            dori,
            dcmp,
            _,
            _,
            teacher,
        ) = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_out.contiguous(),
            out,
            lse,
            ori_kv=kv,
            cmp_kv=kv if ctx.has_cmp else None,
            ori_sparse_indices=tail,
            cmp_sparse_indices=selected if ctx.has_cmp else None,
            sinks=sinks,
            softmax_scale=ctx.scale,
            metadata=ctx.attention_metadata.sparse_grad_metadata,
            **ctx.attention_metadata.sparse_options,
        )
        teacher_TS = None
        if ctx.aux_loss is not None and scores.size(-1):
            # Mask kernel padding before summing token probabilities back into pools.
            p_TS = (
                teacher.masked_fill(selected < 0, 0)[:, 0]
                .unflatten(-1, (-1, ctx.ratio))
                .sum(-1)
            )
            if padding_mask is not None:
                p_TS = p_TS.masked_fill(padding_mask[:, None], 0)
            scale = denominator.clamp_min(1).reciprocal()
            ctx.aux_loss.instance_acc.add_(ctx.aux_loss.kl(p_TS, scores) * scale)
            teacher_TS = p_TS * (ctx.aux_loss.coeff * scale)
        # This is an auxiliary objective: grad_out scales trunk gradients, never the teacher.
        dkv = dori + dcmp if ctx.has_cmp else dori
        return (
            dq.to(q.dtype),
            dkv.to(kv.dtype),
            None,
            None,
            teacher_TS,
            None,
            None,
            None,
            None,
            None,
        )


class AscendSparseAttention(SparseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(SparseAttention.Config):
        num_heads: int
        num_index_heads: int
        topk: int

        def build_attention_metadata(self, positions, **kwargs):
            return _build_kernel_metadata(_global_metadata(self, positions, **kwargs))

    def forward(
        self,
        q_THL,
        kv_TL,
        indices_TS,
        scores_TS,
        *,
        attention_metadata,
        scale,
        aux_loss_denominator=None,
        padding_mask=None
    ):
        q, kv, scores = q_THL, kv_TL, scores_TS
        metadata = attention_metadata
        selected, tail = self.selection(indices_TS, attention_metadata=metadata)
        aux_loss = self.aux_loss if self.training else None
        if aux_loss is not None and aux_loss_denominator is None:
            raise ValueError("IndexerLoss requires the step's valid-token denominator.")
        width = kv.size(-1)
        return _SparseMLA.apply(
            F.pad(q, (0, 512 - width)),
            F.pad(kv, (0, 512 - width))[:, None],
            selected.flatten(-2)[:, None],
            tail[:, None],
            scores,
            metadata,
            scale,
            aux_loss,
            aux_loss_denominator,
            padding_mask,
        )[..., :width]


@override(
    target=GlmDSA.Config,
    description="GLM pooled sparse attention and indexer KL on Ascend 950",
)
def ascend_sparse_attention(cfg: GlmDSA.Config) -> GlmDSA.Config:
    import cann_ops_transformer  # noqa: F401 -- registers the four kernel pairs

    if cfg.wkv_a.out_features > 512 or cfg.indexer.index_head_dim > 128:
        raise ValueError(
            "Ascend sparse attention requires latent width <=512 and index width <=128."
        )
    return derive(
        cfg,
        GlmDSA.Config,
        inner_attention=derive(
            cfg.inner_attention,
            AscendSparseAttention.Config,
            num_heads=cfg.n_heads,
            num_index_heads=cfg.indexer.num_index_heads,
            topk=cfg.indexer.inner_indexer.topk,
        ),
        indexer=derive(
            cfg.indexer,
            type(cfg.indexer),
            inner_indexer=derive(cfg.indexer.inner_indexer, AscendInnerIndexer.Config),
        ),
    )
