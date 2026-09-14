# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning indexer for DeepSeek V4.1 (CSA2) and its distillation loss.

Shape legend for this file:
    T = packed tokens, D = model dimension,
    Hi = ``index_n_heads``, Di = ``index_head_dim``,
    N = number of compressed KV entries (``T // compress_ratio``),
    K = ``index_topk`` (entries selected per query).

Score of query ``t`` against compressed entry ``j``:

    S_{t,h,j} = <q^I_{t,h}, k^I_j>
    I_{t,j}   = sum_h w_{t,h} * relu(S_{t,h,j})

The top-``K`` entries of ``I_{t,.}`` are what the sparse attention reads. Selection is
discrete, hence carries no gradient: the indexer is trained *only* by
:class:`IndexerDistillLoss`, which distills each consumer layer's attention mass on those
entries -- a marginal weighted by that layer's own compressed share -- into
``softmax(I_{t,.})``.

The whole computation is single-pass: no query chunking, which is a kernel-side concern
and not something the reference implementation should carry. A kernel-backed variant
replaces the score-and-select half: the per-head weights and the relaxed score, the
visibility and candidate masking, and the top-k selection are what the NPU
``lightning_indexer`` forward and ``sparse_lightning_indexer_kl_loss_grad`` backward
implement.

Packed documents are handled exactly like ``selected_attention`` handles its window:
``doc_ids`` equality plus index arithmetic. Entry ``j`` covers tokens
``[j * compress_ratio, (j + 1) * compress_ratio)``, so its document is
``doc_ids[j * compress_ratio]`` and it is causally complete for query ``t`` iff
``j < (t + 1) // compress_ratio``. An entry is selectable by ``t`` iff both hold. This
relies on every document segment being a multiple of ``compress_ratio`` tokens (the
packing alignment), which is also what makes the compressor's reshape segment-exact.

Every layer owns a :class:`HierarchicalIndexer`, statically assigned one of CSA2's three
modes: Full Mode carries the parameters and produces both the index keys and the top-k,
Reindex Mode carries its own query and rescores the shared keys, and Reuse Mode carries
the shared top-k forward without computing anything. The mode is asserted in ``forward``,
so a misconfigured layer fails loudly instead of silently recomputing or silently reusing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from .model import DeepSeekV41Metadata


class IndexerMode(str, Enum):
    """CSA2's static per-layer modes (report section 2.3.1).

    ``FULL`` owns the main KV and projects its own index keys, ``REINDEX`` rescores the
    shared keys with its own query, and ``REUSE`` computes no index query at all.
    """

    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


# Bound at module level so the mode adapters in HierarchicalIndexer read as
# "if self.mode is REUSE".
FULL = IndexerMode.FULL
REINDEX = IndexerMode.REINDEX
REUSE = IndexerMode.REUSE


def _selection_mask(
    doc_ids_T: torch.Tensor, compress_ratio: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Document isolation plus causal completeness over the compressed-entry axis.

    Returns ``(visible_TN, newest_T1, newest_valid_T1)`` for one ``compress_ratio``.
    Entry ``j`` belongs to ``doc_ids_T[j * compress_ratio]`` and is complete for query
    ``t`` iff ``j < (t + 1) // compress_ratio``; the two conditions are exactly the
    ``selected_attention`` window rule one axis over.
    """
    num_tokens = doc_ids_T.size(0)
    num_cmp = num_tokens // compress_ratio
    entry_T = torch.arange(num_tokens, device=doc_ids_T.device).unsqueeze(-1)
    # Global number of complete groups up to and including each query.
    complete_T1 = (entry_T + 1) // compress_ratio
    entry_N = torch.arange(num_cmp, device=doc_ids_T.device).unsqueeze(0)
    cmp_doc_ids_N = doc_ids_T[::compress_ratio]
    visible_TN = (entry_N < complete_T1) & (
        cmp_doc_ids_N.unsqueeze(0) == doc_ids_T.unsqueeze(-1)
    )

    newest_T1 = complete_T1 - 1
    newest_valid_T1 = (newest_T1 >= 0) & (
        cmp_doc_ids_N[newest_T1.clamp_min(0)] == doc_ids_T.unsqueeze(-1)
    )
    return visible_TN, newest_T1, newest_valid_T1


def indexer_selection_masks(
    doc_ids_T: torch.Tensor, compress_ratios: tuple[int, ...]
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The indexer's selection masks for one forward, one per pooling ratio.

    The mask depends only on the forward's ``doc_ids`` and on a layer's
    ``compress_ratio``, so it is built once per forward and looked up by every indexer
    instead of being rebuilt per layer. Ratios of 0 are skipped: those layers reuse a
    selection rather than make one.
    """
    return {
        ratio: _selection_mask(doc_ids_T, ratio)
        for ratio in sorted({r for r in compress_ratios if r > 0})
    }


class HierarchicalIndexer(Module):
    """The Hierarchical Sparse Indexer (report section 2.3.2) across CSA2's layer modes.

    V4.1 is a Causal Encoder-Decoder (CED): the bottom ``L/2`` layers are the causal
    encoder and the top ``L/2`` the decoder, and this indexer is used in the decoder
    only. That is why the released config puts the candidate-pool source at the first
    decoder layer, ``candidate_source_layer_id = 20`` of 40.

    Each layer is statically assigned one mode (report section 2.3.1), and ``forward`` is
    the adapter over them:

    - ``FULL``: the layer owns the main KV, so it projects the index keys from the
      compressor latent and runs the indexer. With ``candidate_topk_blocks > 0`` it is
      also the group's candidate source and builds the shared pool.
    - ``REINDEX``: it reuses the main KV and index keys of a preceding Full Mode layer,
      computes its own index query and rescores them. With ``candidate_topk_blocks > 0``
      it searches only the shared candidate pool, otherwise every visible entry.
    - ``REUSE``: no index query and no scores; it carries the latest Top-K indices and
      the shared keys forward.

    The candidate pool is the hierarchy: the source scores every causally visible entry
    once, scores each block by its best entry and keeps ``candidate_topk_blocks`` blocks
    of ``candidate_block_size`` positions -- 2048 x 8 = 16384 candidates against
    ``index_topk`` 512 in V4.1-Flash -- and the Reindex Mode layers search only those.
    The pool boundary comes from a top-k over blocks, so it receives no gradient: it is a
    training/inference consistency and per-query cost device, not a learned component.

    The indexer is trained by distillation alone, so its graph starts at its own
    parameters: the caller detaches the trunk inputs (``Attention.forward``). The shared
    index keys are the one exception -- a Reindex Mode layer receives ``idx_k`` live, so
    that the consumers of a key keep training its owner.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mode: IndexerMode
        index_n_heads: int
        index_head_dim: int
        index_topk: int
        compress_ratio: int
        # Candidate pool: a Full Mode source builds it, Reindex Mode layers search it.
        # 0 leaves the mode's plain behaviour, scoring every visible entry.
        candidate_topk_blocks: int = 0
        candidate_block_size: int = 0
        # Present on Full and Reindex Mode layers:
        rope: ComplexRoPE.Config | None = None
        wq_b: Linear.Config | None = None
        weights_proj: Linear.Config | None = None
        # Present on Full Mode layers, which project their own keys:
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.mode = config.mode
        self.compress_ratio = config.compress_ratio
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        if self.mode is REUSE:
            return
        if config.rope is None or config.wq_b is None or config.weights_proj is None:
            raise ValueError(
                "A Full or Reindex Mode indexer requires rope, wq_b and weights_proj "
                "configs."
            )
        self.rope = config.rope.build()
        self.wq_b = config.wq_b.build()
        self.weights_proj = config.weights_proj.build()
        if self.mode is FULL:
            if config.wk is None or config.k_norm is None:
                raise ValueError(
                    "A Full Mode indexer requires wk and k_norm configs to project its "
                    "own index keys."
                )
            self.wk = config.wk.build()
            self.k_norm = config.k_norm.build()

    @staticmethod
    def select_candidate_blocks(
        scores_TN: torch.Tensor,
        newest_T1: torch.Tensor,
        newest_valid_T1: torch.Tensor,
        topk_blocks: int,
        block_size: int,
    ) -> torch.Tensor:
        """The candidate source's blockwise selection: keep the best-scoring blocks.

        Args:
            scores_TN: Index scores ``[T, N]``, already masked to ``-inf`` on entries the
                query cannot select (other documents and incomplete groups).
            newest_T1: Index of each query's newest selectable entry, ``[T, 1]``.
            newest_valid_T1: Whether that entry exists (the query's document has at least
                one complete group), ``[T, 1]``.
            topk_blocks: Maximum number of blocks to keep.
            block_size: Positions per block.

        Returns:
            Boolean mask ``[T, N]`` selecting every position of the kept blocks.
        """
        width = scores_TN.size(-1)
        if width % block_size != 0:
            scores_TN = F.pad(scores_TN, (0, -width % block_size), value=-torch.inf)
        # A block is scored by its best position, which is what makes the pool
        # recall-oriented rather than a second token-level selection.
        block_scores_TB = scores_TN.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = block_scores_TB.size(-1)

        # The block holding a query's newest selectable entry is only partly filled, and
        # must not be outscored by an older, full block. A query whose document has no
        # complete group yet owns no such block and pins nothing.
        last_T1 = newest_T1 // block_size
        pin_TB = (
            torch.arange(num_blocks, device=scores_TN.device).unsqueeze(0) == last_T1
        )
        block_scores_TB = block_scores_TB.masked_fill(
            pin_TB & newest_valid_T1, torch.inf
        )

        top = block_scores_TB.topk(min(topk_blocks, num_blocks), dim=-1)
        # Fewer reachable blocks than ``topk_blocks`` leaves -inf picks behind: drop them.
        keep_TB = torch.zeros_like(block_scores_TB, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > -torch.inf
        )
        return keep_TB.repeat_interleave(block_size, dim=-1)[..., :width]

    def _score_and_select(
        self,
        idx_q_THiDi: torch.Tensor,
        idx_k_NDi: torch.Tensor,
        weights_THi: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        candidates_TN: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The score-and-select half, and the half a fused kernel replaces.

        Everything between the projections and the returned tensors:

        1. the relaxed index score ``sum_h w * relu(q . k)`` over every entry the query
           can see. Selection is discrete and carries no gradient, so this runs outside
           the autograd graph;
        2. the hierarchy step: a Full Mode source builds the shared candidate pool, and a
           Reindex Mode layer searches the pool it was handed instead of every visible
           entry. Without ``candidate_topk_blocks`` both modes score everything visible;
        3. the top ``index_topk`` entries of the resulting scores;
        4. the same relaxed score recomputed at those selected entries, now *with*
           gradient: the student logits the distillation loss consumes. It is a second
           pass rather than a gather from step 1 for exactly the reason step 1 is outside
           the graph -- the full ``[T, Hi, N]`` tensor must not enter it.

        Returns:
            ``(topk_indices_TK, topk_scores_TK, candidates_TN)``. An unused slot of a
            padded row is ``-1`` in the indices and ``-inf`` in the student logits, so
            that the distillation's softmax drops it without looking at the indices.
        """
        visible_TN, newest_T1, newest_valid_T1 = attention_masks.selection_masks[
            self.compress_ratio
        ]

        with torch.no_grad():
            scores_THN = torch.einsum("thd,nd->thn", idx_q_THiDi, idx_k_NDi)
            scores_TN = (scores_THN.relu() * weights_THi.unsqueeze(-1)).sum(dim=1)
            scores_TN = scores_TN.masked_fill(~visible_TN, -torch.inf)

            if self.mode is FULL and self.candidate_topk_blocks > 0:
                candidates_TN = self.select_candidate_blocks(
                    scores_TN,
                    newest_T1,
                    newest_valid_T1,
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                )
            elif self.mode is REINDEX and self.candidate_topk_blocks > 0:
                assert candidates_TN is not None, (
                    "A Reindex Mode indexer with candidate_topk_blocks set searches the "
                    "pool the candidate source built, which no preceding layer produced."
                )
                scores_TN = scores_TN.masked_fill(~candidates_TN, -torch.inf)

            topk = min(self.index_topk, scores_TN.size(-1))
            selected_TK = (
                scores_TN.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            )
            # Entries the query cannot see yet, or that a pool excluded, come back as -1,
            # which the sparse attention and the loss both skip.
            topk_indices_TK = torch.where(
                visible_TN.gather(-1, selected_TK), selected_TK, -1
            )

        # The gradient this produces is what trains the indexer: it flows into ``wq_b``,
        # ``weights_proj``, and into the shared index keys' owner through ``idx_k``.
        selected_TKDi = idx_k_NDi[topk_indices_TK.clamp_min(0)]
        logits_THK = torch.einsum("thd,tkd->thk", idx_q_THiDi, selected_TKDi)
        logits_THK = logits_THK.relu() * weights_THi.unsqueeze(-1)
        # A ``-1`` slot scores against entry 0 and means nothing: it is marked with the
        # value that drops it from the distillation's student softmax.
        topk_scores_TK = logits_THK.sum(dim=1).masked_fill(
            topk_indices_TK < 0, -torch.inf
        )
        return topk_indices_TK, topk_scores_TK, candidates_TN

    def forward(
        self,
        x_TD: torch.Tensor,
        qr_TQ: torch.Tensor,
        positions_T: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        latent_TDp: torch.Tensor | None = None,
        idx_k_TpDi: torch.Tensor | None = None,
        topk_indices_TK: torch.Tensor | None = None,
        topk_scores_TK: torch.Tensor | None = None,
        candidates_TN: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """The mode adapter.

        A Reuse Mode layer does no work at all: it hands its arguments back untouched.
        The other two modes project the index query and the per-head weights, and only
        Full Mode also projects the keys, from its own compressor latent.

        Args:
            x: Hidden states of shape ``[T, D]``.
            qr: Query LoRA latent of shape ``[T, q_lora_rank]``.
            positions: Position ids of shape ``[T]``.
            attention_masks: The forward's varlen metadata; the indexer reads its
                document ids and its precomputed selection masks.
            latent: The compressor's pre-RoPE latent; a Full Mode layer projects its keys
                from it.
            idx_k: The shared index keys, rescored by a Reindex Mode layer.
            topk_indices: The shared top-k, carried by a Reuse Mode layer.
            topk_scores: The shared student logits at those entries. A Reuse Mode layer
                keeps them: they are what its own distillation loss compares its teacher
                against, and what the rest of its group inherits.
            candidates: The shared candidate pool, built by a Full Mode source and
                searched by the Reindex Mode layers after it.

        Returns:
            ``(idx_k, topk_indices, topk_scores, candidates)``. A Reuse Mode layer passes
            its inputs through: it has no parameters of its own to train, but it must not
            drop the student logits its group depends on.
        """
        # Reuse Mode carries the shared selection forward: nothing to compute.
        if self.mode is REUSE:
            return idx_k_TpDi, topk_indices_TK, topk_scores_TK, candidates_TN

        # Only Full Mode projects keys, from its own compressor latent; Reindex Mode
        # rescores the keys a preceding Full Mode layer produced.
        if self.mode is FULL:
            assert (
                latent_TDp is not None
            ), "A Full Mode indexer projects its keys from the compressor latent."
            idx_k_NDi = self.k_norm(self.wk(latent_TDp))
            # One rank-2 head; RoPE rotates rank-3 [T, N, H].
            idx_k_NDi = self.rope(
                idx_k_NDi.unsqueeze(1), positions=positions_T[:: self.compress_ratio]
            ).squeeze(1)
        else:
            assert idx_k_TpDi is not None, (
                "A Reindex Mode indexer rescores the shared index keys, which no "
                "preceding Full Mode layer produced."
            )
            idx_k_NDi = idx_k_TpDi

        # The index query is rotated at the token's own position.
        idx_q_THiDi = self.rope(
            self.wq_b(qr_TQ).unflatten(-1, (self.index_n_heads, self.index_head_dim)),
            positions=positions_T,
        )
        # Scaled by the index softmax scale and the head count, as in the reference: the
        # per-head scores are averaged rather than summed.
        weights_THi = self.weights_proj(x_TD) * (
            self.index_head_dim**-0.5 * self.index_n_heads**-0.5
        )

        topk_indices_TK, topk_scores_TK, candidates_TN = self._score_and_select(
            idx_q_THiDi,
            idx_k_NDi,
            weights_THi,
            attention_masks,
            candidates_TN=candidates_TN,
        )
        return idx_k_NDi, topk_indices_TK, topk_scores_TK, candidates_TN


class IndexerDistillLoss(AuxLoss):
    """Distill the attention's distribution over the top-k entries into the indexer.

    The teacher is the head-averaged attention mass on the selected entries, with the full
    softmax denominator (sliding window, selected compressed entries and sink alike).
    That mass is a *marginal* ``p`` whose row sum ``Z <= 1``: the window and the sink hold
    the rest of the probability. The objective scores the conditional teacher ``t = p / Z``
    against the student ``softmax(I)`` and weights the row by ``Z``::

        L = sum_t sum_j p_{t,j} (log t_{t,j} - log Y_{t,j}),   dI = Z * Y - p

    So it takes roughly the form of a KL from the student to the teacher, weighted by the
    row's compressed mass. ``p`` is deliberately left *unnormalised*: pre-normalising it to
    ``t`` would set ``Z = 1`` everywhere and quietly drop that weight.

    Like ``MicrobatchWiseLoadBalanceLoss``, the loss owns the whole computation: the
    attention hands over the tensors the teacher needs (queries, compressed keys, the
    selected entries and the operator's LSE) plus the student logits, and the forward
    builds the teacher, forms the weighted KL and injects the gradient on the carrier. The
    caller passes the teacher's sources as constants -- the distillation must train the
    indexer and nothing else -- which leaves ``topk_scores`` the only differentiable input.

    One instance is attached per layer that consumes the selection, and they all score the
    *same* student logits, because the indexer's ``topk_scores`` tensor is shared across
    the group. Each layer's backward therefore flows into that shared tensor, so the
    indexer accumulates the gradient of every consumer; ``dI`` being affine in the teacher
    is what makes this per-layer sum equal to the single pooled-teacher loss.

    The loss is summed over rows and normalized by the step's global valid-token count
    by the :class:`AuxLoss` framework, exactly like the MoE balance loss.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """The ``AuxLoss`` fields plus the teacher's temperature."""

        softmax_scale: float
        """Attention softmax scale; the teacher recomputes its logits with the same
        temperature as the sparse attention that produced the LSE."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.softmax_scale = config.softmax_scale

    def _teacher(
        self,
        q_THD: torch.Tensor,
        cmp_k_ND: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        lse_TH: torch.Tensor,
    ) -> torch.Tensor:
        """Raw head-averaged attention mass on the selected entries, ``[T, K]``.

        ``lse_TH`` is the operator's per-head log-sum-exp over window + selected compressed
        + sink, so ``exp(logit - lse)`` is each head's probability on that entry with the
        full denominator. Averaging over heads gives the *marginal* mass ``p``, whose row
        sum ``Z <= 1`` is the compressed slice's share of the full softmax: the window and
        the sink hold the rest. A head whose mass sits on the window or the sink therefore
        contributes ``m_h = exp(compressed_lse_h - lse_h) < 1``, instead of every head
        contributing unit compressed mass. The head mean (not the sum) is what the kernel
        reduces with, and it is what keeps ``Z`` a share rather than a multiple of it.

        ``p`` is deliberately returned unnormalised. The loss weights each row's KL by
        ``Z``, so pre-normalising to ``p / Z`` would set ``Z = 1`` and change the
        objective; the caller derives the conditional teacher.

        The per-head term is evaluated as ``m_h * softmax_h(logit)``: the conditional is a
        plain softmax over the selected support, stable without any shift, and only the
        mass needs the ``lse`` difference. A row whose mass underflows fp32 returns zeros,
        which is the correct limit: there is no compressed mass left to distil.
        """
        valid_TK = topk_indices_TK >= 0
        row_valid_T = valid_TK.any(dim=-1)
        selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)]
        logits_THK = (
            torch.einsum("thd,tkd->thk", q_THD, selected_TKD).float()
            * self.softmax_scale
        )
        logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -torch.inf)

        comp_lse_TH = torch.logsumexp(logits_THK, dim=-1)
        mass_TH = torch.exp(comp_lse_TH - lse_TH.float())
        # A row with no valid slot has an all -inf softmax row; its mass is zero anyway.
        conditional_THK = torch.softmax(
            logits_THK.masked_fill(~row_valid_T[:, None, None], 0.0), dim=-1
        )
        return (mass_TH.unsqueeze(-1) * conditional_THK).sum(dim=1) / q_THD.size(1)

    def forward(
        self,
        q_THD: torch.Tensor,
        cmp_k_ND: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        lse_TH: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Build the teacher, score the student against it, inject the gradient.

        The gradient is roughly a KL gradient: row ``t`` contributes
        ``sum_j p_j (log t_j - log Y_j)``, the conditional teacher ``t = p / Z`` scored
        against the student and weighted by the row's marginal mass ``Z`` through ``p``.
        That is ``Z * KL(t || Y)``, whose gradient w.r.t. the student logits is
        ``Z * Y - p``. ``p`` stays unnormalised on purpose: ``Z`` is the compressed
        slice's share of the full softmax, and normalising ``p`` first would set it to 1.

        Args:
            q_THD: Attention queries ``[T, H, Dk]``; a constant (detached by the caller).
            cmp_k_ND: Shared compressed KV ``[N, Dk]``; a constant.
            topk_indices_TK: Selected compressed entries ``[T, K]``; ``-1`` unused.
            lse_TH: Per-head log-sum-exp of the sparse softmax, ``[T, H]``; a constant.
            topk_scores_TK: Student logits at the selected entries, ``[T, K]``, with the
                indexer's ``-inf`` marking an unused slot. The one live input.
            carrier: Tensor whose backward path carries the injected gradient (the
                attention output).

        Returns:
            ``carrier`` unchanged.
        """
        p_TK = self._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        # The conditional teacher; a row whose mass underflowed keeps t = 0 and its p = 0
        # makes the contribution vanish.
        eps = torch.finfo(torch.float32).tiny
        t_TK = p_TK / p_TK.sum(dim=-1, keepdim=True).clamp_min(eps)

        logits_TK = topk_scores_TK.float()
        # A row with no reachable entry is all -inf, which log_softmax would turn into NaN.
        # It carries no teacher mass either, so it is zeroed and contributes nothing.
        row_valid_T = torch.isfinite(logits_TK).any(dim=-1)
        logits_TK = logits_TK.masked_fill(~row_valid_T.unsqueeze(-1), 0.0)
        log_student_TK = F.log_softmax(logits_TK, dim=-1)

        # xlogy keeps 0 * log(0) = 0: an entry with no teacher mass contributes nothing
        # even though its conditional is 0. The student term is a plain product, so the
        # slots the indexer marked unreachable would be 0 * -inf = NaN and are dropped.
        weighted_TK = torch.special.xlogy(p_TK, t_TK) - p_TK * log_student_TK
        weighted_TK = weighted_TK.masked_fill(~torch.isfinite(logits_TK), 0.0)
        return self.inject(weighted_TK.sum(), carrier=carrier)
