# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSA2 attention for DeepSeek V4.1.

Shape legend for this file:
    T = packed tokens, D = model dimension,
    H = ``n_heads``, Dk = ``head_dim``, rd = ``rope_head_dim``,
    N = number of compressed KV entries, K = ``index_topk``.

Each query attends to two sources at once, combined into a single masked softmax by
Attention Gym's ``selected_attention``: its own sliding window over the layer's KV
``[T, Dk]``, and the ``K`` compressed entries selected by the indexer out of the shared
compressed KV ``[N, Dk]``. A learned per-head sink logit takes part in the softmax
denominator without contributing a value, so a row with no reachable entry still
produces zeros instead of NaN.

Packed documents are isolated by the metadata's ``doc_ids``, which is the only varlen
metadata: the operator applies it to the sliding-window branch, and the indexer uses it
to keep its top-``K`` inside the query's own document (the operator's contract leaves
the sparse branch to the caller). ``positions`` still drives RoPE; it is not segment
metadata.

The attention is also where the indexer's distillation loss is applied. The operator
returns the per-head log-sum-exp of the full softmax (window, selected compressed
entries and sink) and the student logits arrive as an input; the loss itself, including
the teacher it rebuilds from them, lives on ``IndexerDistillLoss``. The loss's inputs are
detached at that call site: the distillation must train the indexer and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from attn_gym.sparse.selected_attention import AuxRequest, selected_attention
from attn_gym.types import Impl
from torch import nn

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import BatchedLinear, Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor
from .indexer import HierarchicalIndexer

if TYPE_CHECKING:
    from .model import DeepSeekV41Metadata


class CompressedSparseInnerAttention2(Module):
    """CSA2's sparse core: sliding window plus the selected compressed entries.

    The whole core is Attention Gym's ``selected_attention`` in one call: ``q`` carries
    ``H`` heads while both KV sources carry a single shared head, and ``topk_indices``
    index the compressed KV with ``-1`` for unused slots.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        softmax_scale: float
        # Indexer distillation loss, attached only on layers that select compressed
        # entries (``compress_ratio > 0``).
        aux_loss: AuxLoss.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(
        self,
        q_THD: torch.Tensor,
        swa_k_TD: torch.Tensor,
        attn_sink_H: torch.Tensor,
        *,
        attention_masks: DeepSeekV41Metadata,
        cmp_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            q: Queries of shape ``[T, H, Dk]``.
            swa_k: Sliding-window KV of shape ``[T, Dk]``, shared across heads.
            attn_sink: Per-head sink logits of shape ``[H]``.
            attention_masks: The forward's varlen metadata; the operator reads its
                document ids for the sliding-window branch.
            cmp_k: Shared compressed KV of shape ``[N, Dk]``.
            topk_indices: Selected compressed entries ``[T, K]``, ``-1`` for unused.
            topk_scores: Student logits at those entries ``[T, K]``.

        Returns:
            Attention output of shape ``[T, H, Dk]``.
        """
        num_tokens, _, head_dim = q_THD.size()
        uses_cmp = cmp_k is not None and topk_indices is not None
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")

        q_BHTD = q_THD.transpose(0, 1).unsqueeze(0)
        local_kv_B1TD = swa_k_TD.view(1, 1, num_tokens, head_dim)
        if uses_cmp:
            sparse_kv_B1ND = cmp_k.view(1, 1, cmp_k.size(0), head_dim)
            kv_indices_BTK = topk_indices.unsqueeze(0)
        else:
            # Window-only layer: the sparse pool is empty and every query keeps the
            # window plus the sink.
            sparse_kv_B1ND = q_THD.new_zeros(1, 1, 0, head_dim)
            kv_indices_BTK = torch.empty(
                1, num_tokens, 0, dtype=torch.long, device=q_THD.device
            )

        wants_teacher = (
            self.training
            and self.aux_loss is not None
            and topk_scores is not None
            and uses_cmp
        )
        out = selected_attention(
            q_BHTD,
            local_kv_B1TD,
            sparse_kv_B1ND,
            kv_indices_BTK,
            attention_sink=attn_sink_H,
            doc_ids=attention_masks.doc_ids_T.unsqueeze(0),
            sliding_window_size=self.window_size,
            scale=self.softmax_scale,
            # TODO: switch to the fused implementation once it validates this path; the
            # reference one is what the CPU tests and the NPU ports run today.
            impl=Impl.REFERENCE,
            return_aux=AuxRequest(lse=True) if wants_teacher else None,
        )

        if not wants_teacher:
            return out.squeeze(0).transpose(0, 1)

        attn_BHTD, aux = out
        attn_THD = attn_BHTD.squeeze(0).transpose(0, 1)
        # The teacher's sources are constants: the distillation must train the indexer and
        # nothing else, and ``topk_scores`` is the one live input, carrying the gradient
        # that trains it. ``lse`` arrives as ``[B, H, T]``; the loss reads ``[T, H]``.
        return self.aux_loss(
            q_THD.detach(),
            cmp_k.detach(),
            topk_indices,
            aux.lse.squeeze(0).transpose(0, 1).detach(),
            topk_scores,
            carrier=attn_THD,
        )


class Attention(BaseAttention):
    """Latent attention with a grouped output projection, CSA2's per-layer wrapper.

    Projections, both RoPE phases, the compressor and the indexer live here; the sparse
    core is the ``inner_attention`` module. The shared cross-layer tensors (compressed
    KV, index keys, selected entries, student logits, candidate pool) are threaded
    through as ordinary inputs and returned updated, so a layer's role is visible at the
    call site rather than hidden in mutable state.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        head_dim: int = 512
        compress_ratio: int = 1
        inner_attention: CompressedSparseInnerAttention2.Config  # pyrefly: ignore [bad-override]
        rope: ComplexRoPE.Config
        compressor: Compressor.Config
        indexer: HierarchicalIndexer.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.compress_ratio = config.compress_ratio

        self.rope = config.rope.build()
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv = config.wkv.build()
        self.kv_norm = config.kv_norm.build()
        self.wo_a = config.wo_a.build()
        self.wo_b = config.wo_b.build()
        # One sink logit per head, fp32 as in the released checkpoint and the kernels.
        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = config.compressor.build()
        self.indexer = config.indexer.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        positions_T: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(o, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        The returned shared tensors are this layer's contribution to the chain: freshly
        computed where the layer is a source, otherwise the inputs unchanged.
        """
        num_tokens = x_TD.size(0)

        qr_TQ = self.q_norm(self.wq_a(x_TD))
        q_THD = self.wq_b(qr_TQ).unflatten(-1, (self.n_heads, self.head_dim))
        swa_k_TD = self.kv_norm(self.wkv(x_TD))

        cmp_k, latent_TD = self.compressor(x_TD, positions_T, cmp_k)
        # The indexer is trained by distillation alone, so it reads the trunk as
        # constants. The shared index keys are the exception: a Reindex Mode layer keeps
        # the ``idx_k`` it was handed live, so its consumers go on training the key's owner.
        idx_k, topk_indices, topk_scores, candidates = self.indexer(
            x_TD.detach(),
            qr_TQ.detach(),
            positions_T,
            attention_masks,
            latent_TDp=latent_TD.detach() if latent_TD is not None else None,
            idx_k_TpDi=idx_k,
            topk_indices_TK=topk_indices,
            topk_scores_TK=topk_scores,
            candidates_TN=candidates,
        )

        q_THD = self.rope(q_THD, positions=positions_T)
        # The shared KV latent is one rank-2 head; RoPE rotates rank-3 [T, N, H].
        swa_k_TD = self.rope(swa_k_TD.unsqueeze(1), positions=positions_T).squeeze(1)

        uses_cmp = self.compress_ratio > 0
        o_THD = self.inner_attention(
            q_THD,
            swa_k_TD,
            self.attn_sink,
            attention_masks=attention_masks,
            cmp_k=cmp_k if uses_cmp else None,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
        )
        o_THD = self.rope(o_THD, positions=positions_T, inverse=True)

        # The output projection is grouped: wo_a projects each query-head group on its
        # own, wo_b mixes the per-group results back to the model dimension. The group
        # count comes from the module so a group-wise sharding can narrow it later.
        o_TGR = self.wo_a(o_THD.view(num_tokens, self.wo_a.n_batches, -1))
        return (
            self.wo_b(o_TGR.flatten(-2)),
            cmp_k,
            idx_k,
            topk_indices,
            topk_scores,
            candidates,
        )
