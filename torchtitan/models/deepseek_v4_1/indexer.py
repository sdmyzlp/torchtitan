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
:class:`IndexerKLLoss`, which distills the attention's own distribution over the selected
entries into ``softmax(I_{t,.})``.

The whole computation is single-pass: no query chunking, which is a kernel-side concern
and not something the reference implementation should carry. It is split in two so a
kernel-backed variant can replace the second half: :meth:`Indexer._project_qkw` runs the
matmuls that produce the index query, keys and per-head weights, and
:meth:`Indexer._select_topk` runs the relaxed score, the visibility and candidate
masking and the top-k selection. The latter is what the NPU ``lightning_indexer``
forward and ``sparse_lightning_indexer_kl_loss_grad`` backward implement.

Packed documents are handled exactly like ``selected_attention`` handles its window:
``doc_ids`` equality plus index arithmetic. Entry ``j`` covers tokens
``[j * compress_ratio, (j + 1) * compress_ratio)``, so its document is
``doc_ids[j * compress_ratio]`` and it is causally complete for query ``t`` iff
``j < (t + 1) // compress_ratio``. An entry is selectable by ``t`` iff both hold. This
relies on every document segment being a multiple of ``compress_ratio`` tokens (the
packing alignment), which is also what makes the compressor's reshape segment-exact.

Every layer owns an :class:`Indexer`, but only *source* layers carry parameters: an
index-source layer produces the top-k, a layer that owns the compressed KV also produces
the index keys, and a reuse layer returns what it was handed. ``is_source`` and
``owns_k`` encode that contract and are asserted in ``forward``, so a misconfigured layer
fails loudly instead of silently recomputing or silently reusing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

# log(float32 tiny): the point below which a compressed-key mass stops being
# representable in fp32, so the teacher row needs a shift to stay informative.
_LOG_FP32_TINY = math.log(torch.finfo(torch.float32).tiny)


def select_candidate_blocks(
    scores_TN: torch.Tensor,
    newest_T1: torch.Tensor,
    newest_valid_T1: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the hierarchical indexer: keep the best-scoring blocks.

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

    # The block holding a query's newest selectable entry is only partly filled, and must
    # not be outscored by an older, full block. A query whose document has no complete
    # group yet owns no such block and pins nothing.
    last_T1 = newest_T1 // block_size
    pin_TB = torch.arange(num_blocks, device=scores_TN.device).unsqueeze(0) == last_T1
    block_scores_TB = block_scores_TB.masked_fill(pin_TB & newest_valid_T1, torch.inf)

    top = block_scores_TB.topk(min(topk_blocks, num_blocks), dim=-1)
    # Fewer reachable blocks than ``topk_blocks`` leaves -inf picks behind: drop them.
    keep_TB = torch.zeros_like(block_scores_TB, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep_TB.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(Module):
    """Score compressed entries and keep the top ``index_topk`` per query.

    The indexer's inputs are detached: the distillation loss must train the indexer and
    nothing else, so its graph starts at the indexer's own parameters. One exception is
    ``idx_k`` on a layer that reuses the shared index keys, which is deliberately *not*
    detached so that the consumers of a shared key continue to train its owner.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        head_dim: int
        rope_head_dim: int
        index_n_heads: int
        index_head_dim: int
        index_topk: int
        compress_ratio: int
        is_source: bool
        # This layer projects the index keys from its own compressor latent.
        owns_k: bool
        # Hierarchical indexer: this layer builds the shared candidate pool.
        is_candidate_source: bool = False
        # Hierarchical indexer: this layer restricts its selection to the shared pool.
        uses_candidates: bool = False
        candidate_topk_blocks: int = 0
        candidate_block_size: int = 0
        # Whether a distillation loss consumes the student logits at the selected
        # entries. When no loss is attached (inference, or an LM-only run) the
        # gather-and-score recomputation is dead work and is skipped.
        needs_selection_scores: bool = False
        # Present on source layers:
        rope: ComplexRoPE.Config | None = None
        wq_b: Linear.Config | None = None
        weights_proj: Linear.Config | None = None
        # Present on key-owning layers:
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.rope_head_dim = config.rope_head_dim
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.is_source = config.is_source
        self.owns_k = config.owns_k
        self.is_candidate_source = config.is_candidate_source
        self.uses_candidates = config.uses_candidates
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.needs_selection_scores = config.needs_selection_scores
        if not self.is_source:
            return
        if config.rope is None or config.wq_b is None or config.weights_proj is None:
            raise ValueError(
                "An index-source layer requires rope, wq_b and weights_proj configs."
            )
        if self.owns_k and (config.wk is None or config.k_norm is None):
            raise ValueError("A key-owning indexer requires wk and k_norm configs.")
        self.rope = config.rope.build()
        self.wq_b = config.wq_b.build()
        self.weights_proj = config.weights_proj.build()
        if self.owns_k:
            self.wk = config.wk.build()
            self.k_norm = config.k_norm.build()

    def _selection_mask(
        self,
        doc_ids_T: torch.Tensor,
        num_cmp: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Document isolation plus causal completeness over the entry axis.

        Returns ``(visible_TN, newest_T1, newest_valid_T1)``. Entry ``j`` belongs to
        ``doc_ids_T[j * compress_ratio]`` and is complete for query ``t`` iff
        ``j < (t + 1) // compress_ratio``; the two conditions are exactly the
        ``selected_attention`` window rule one axis over.
        """
        num_tokens = doc_ids_T.size(0)
        ratio = self.compress_ratio
        entry_T = torch.arange(num_tokens, device=device).unsqueeze(-1)
        # Global number of complete groups up to and including each query.
        complete_T1 = (entry_T + 1) // ratio
        entry_N = torch.arange(num_cmp, device=device).unsqueeze(0)
        cmp_doc_ids_N = doc_ids_T[::ratio]
        visible_TN = (entry_N < complete_T1) & (
            cmp_doc_ids_N.unsqueeze(0) == doc_ids_T.unsqueeze(-1)
        )

        newest_T1 = complete_T1 - 1
        in_range = newest_T1 >= 0
        newest_valid_T1 = in_range & (
            cmp_doc_ids_N[newest_T1.clamp_min(0)] == doc_ids_T.unsqueeze(-1)
        )
        return visible_TN, newest_T1, newest_valid_T1

    def _selected_scores(
        self,
        idx_q_THiDi: torch.Tensor,
        idx_k_NDi: torch.Tensor,
        weights_THi: torch.Tensor,
        topk_indices_TK: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the index scores at the selected entries, with gradient.

        The gradient this produces is what trains the indexer: it flows into ``wq_b``,
        ``weights_proj``, and into the shared index keys' owner through ``idx_k``.
        """
        # Invalid (-1) slots gather entry 0; the loss masks them out.
        selected_TKDi = idx_k_NDi[topk_indices_TK.clamp_min(0)]
        logits_THK = torch.einsum("thd,tkd->thk", idx_q_THiDi, selected_TKDi)
        logits_THK = logits_THK.relu() * weights_THi.unsqueeze(-1)
        return logits_THK.sum(dim=1)

    def _project_qkw(
        self,
        x_TD: torch.Tensor,
        qr_TQ: torch.Tensor,
        positions_T: torch.Tensor,
        *,
        latent_TDp: torch.Tensor | None,
        idx_k_TpDi: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """First half of the indexer: the projections that produce q, k and w.

        A key-owning layer projects its keys from the compressor latent; a re-indexing
        layer scores the shared ``idx_k_TpDi`` instead. The caller has already checked
        that the required input is present.

        Returns:
            ``(idx_q_THiDi, idx_k_NDi, weights_THi)``.
        """
        idx_q_THiDi = self.rope(
            self.wq_b(qr_TQ).unflatten(-1, (self.index_n_heads, self.index_head_dim)),
            positions=positions_T,
        )
        if self.owns_k:
            idx_k_NDi = self.k_norm(self.wk(latent_TDp.detach()))
            # One rank-2 head; RoPE rotates rank-3 [T, N, H].
            idx_k_NDi = self.rope(
                idx_k_NDi.unsqueeze(1), positions=positions_T[:: self.compress_ratio]
            ).squeeze(1)
        else:
            # ``forward`` rejects a re-indexing layer that was not handed the shared
            # keys; this only narrows the type for the return.
            assert idx_k_TpDi is not None
            idx_k_NDi = idx_k_TpDi

        # ``weights_proj`` is scaled by the index softmax scale and the head count, as
        # in the reference: the per-head scores are averaged rather than summed.
        weights_THi = self.weights_proj(x_TD) * (
            self.index_head_dim**-0.5 * self.index_n_heads**-0.5
        )
        return idx_q_THiDi, idx_k_NDi, weights_THi

    def _select_topk(
        self,
        idx_q_THiDi: torch.Tensor,
        idx_k_NDi: torch.Tensor,
        weights_THi: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        candidates_TN: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Second half: score the visible entries and keep the top ``index_topk``.

        This is the half a fused kernel replaces: the relaxed score
        ``relu(q @ k^T) * w`` summed over heads, the visibility mask (document
        isolation plus causal completeness), the optional candidate-pool restriction,
        the top-k, and the differentiable student logits at the selected entries --
        the ``lightning_indexer`` forward and ``sparse_lightning_indexer_kl_loss_grad``
        backward of the NPU kernels.

        Returns:
            ``(topk_indices_TK, topk_scores_TK, candidates_TN)``. ``topk_scores_TK`` is
            ``None`` where no distillation loss consumes it.
        """
        num_tokens = idx_q_THiDi.size(0)
        num_cmp = idx_k_NDi.size(0)
        visible_TN, newest_T1, newest_valid_T1 = self._selection_mask(
            attention_masks.doc_ids_T, num_cmp, idx_q_THiDi.device
        )
        topk = min(self.index_topk, num_cmp)
        topk_indices_TK = torch.full(
            (num_tokens, topk), -1, dtype=torch.long, device=idx_q_THiDi.device
        )

        # Selection carries no gradient: the full ``[T, Hi, N]`` score tensor must stay
        # out of the autograd graph.
        with torch.no_grad():
            scores_THN = torch.einsum("thd,nd->thn", idx_q_THiDi, idx_k_NDi)
            scores_TN = (scores_THN.relu() * weights_THi.unsqueeze(-1)).sum(dim=1)
            scores_TN = scores_TN.masked_fill(~visible_TN, -torch.inf)

            if self.is_candidate_source:
                if candidates_TN is None:
                    candidates_TN = torch.zeros(
                        num_tokens, num_cmp, dtype=torch.bool, device=idx_q_THiDi.device
                    )
                candidates_TN = select_candidate_blocks(
                    scores_TN,
                    newest_T1,
                    newest_valid_T1,
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                )
            elif self.uses_candidates and candidates_TN is not None:
                scores_TN = scores_TN.masked_fill(~candidates_TN, -torch.inf)

            if topk > 0:
                selected_TK = (
                    scores_TN.topk(topk, dim=-1, sorted=False)
                    .indices.sort(dim=-1)
                    .values
                )
                # Entries the query cannot see yet (or that the pool excluded) come back
                # as -1, which the sparse attention and the loss both skip.
                topk_indices_TK = torch.where(
                    visible_TN.gather(-1, selected_TK), selected_TK, -1
                )

        # The student logits only exist to be distilled. Producing them when no loss
        # consumes them (inference, or a run with the coefficient set to ``None``)
        # would add a gather plus an einsum over the selected entries for nothing.
        topk_scores_TK = None
        if self.needs_selection_scores and self.training and torch.is_grad_enabled():
            topk_scores_TK = self._selected_scores(
                idx_q_THiDi, idx_k_NDi, weights_THi, topk_indices_TK
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
        """Args:
            x: Hidden states of shape ``[T, D]``.
            qr: Query LoRA latent of shape ``[T, q_lora_rank]``.
            positions: Position ids of shape ``[T]``.
            attention_masks: The forward's varlen metadata. Entry isolation reads its
                document ids; the caller does not hand them over separately.
            latent: The compressor's pre-RoPE latent; a key-owning layer projects its
                keys from it.
            idx_k: The shared index keys when this layer does not own them.
            topk_indices: The shared top-k when this layer does not produce it.
            topk_scores: The shared student logits at those entries. A reusing layer
                keeps them: they are what its own distillation loss compares its teacher
                against, and what the rest of its group inherits.
            candidates: The shared candidate pool mask.

        Returns:
            ``(idx_k, topk_indices, topk_scores, candidates)``. Reuse layers pass their
            inputs through: they have no parameters of their own to train, but they must
            not drop the student logits their group depends on.
        """
        # A source supersedes whatever was in flight, so these are consumption
        # contracts: an index source that does not own the keys must be handed the
        # shared ones, and a reusing layer must be handed a selection to reuse.
        if self.owns_k:
            assert (
                latent_TDp is not None
            ), "A key-owning indexer projects its keys from the compressor latent."
        elif self.is_source:
            assert idx_k_TpDi is not None, (
                "A re-indexing layer scores the shared index keys, which no preceding "
                "key-owning layer produced."
            )
        if self.compress_ratio > 0 and not self.is_source:
            assert topk_indices_TK is not None, (
                "A layer that reuses the top-k must receive it: no index source "
                f"precedes this one (compress_ratio={self.compress_ratio})."
            )
        if not self.is_source:
            return idx_k_TpDi, topk_indices_TK, topk_scores_TK, candidates_TN

        # The distillation loss must train the indexer and nothing else, so its graph
        # starts at the indexer's own parameters.
        idx_q_THiDi, idx_k_NDi, weights_THi = self._project_qkw(
            x_TD.detach(),
            qr_TQ.detach(),
            positions_T,
            latent_TDp=latent_TDp,
            idx_k_TpDi=idx_k_TpDi,
        )
        topk_indices_TK, topk_scores_TK, candidates_TN = self._select_topk(
            idx_q_THiDi,
            idx_k_NDi,
            weights_THi,
            attention_masks,
            candidates_TN=candidates_TN,
        )
        return idx_k_NDi, topk_indices_TK, topk_scores_TK, candidates_TN


class IndexerKLLoss(AuxLoss):
    """Distill the attention's distribution over the top-k entries into the indexer.

    The teacher ``p`` is the attention probability mass the sparse attention assigned to
    each selected entry: per head, with the full softmax denominator (sliding window,
    selected compressed entries and sink alike), then summed over heads and
    L1-normalized over the selected support. The student is ``softmax(I)`` over the same
    support. Both the teacher and the indexer's inputs are detached, so the gradient
    only ever reaches the indexer's parameters.

    Like ``MicrobatchWiseLoadBalanceLoss``, the loss owns the whole computation: the
    attention hands over the tensors the teacher needs (queries, compressed keys, the
    selected entries and the operator's LSE) plus the student logits, and the forward
    builds the teacher, forms the KL and injects the gradient on the carrier.

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
        lse_HT: torch.Tensor,
    ) -> torch.Tensor:
        """Attention mass on the selected compressed entries, ``[T, K]``.

        ``lse`` is the operator's per-head log-sum-exp over window + selected compressed
        + sink, so ``exp(logit - lse)`` is each head's probability on that entry with the
        full denominator. Heads are summed and the result is L1-normalized over the
        selected support, which is the DeepSeek-V4/Megatron teacher: a head whose mass
        sits on its window or the sink contributes little, instead of every head
        contributing unit compressed mass.

        The recomputation runs in fp32 and shifts each row when the window and sink push
        the compressed mass below the fp32 normal range; the shift is constant across
        heads and entries, so it cancels in the L1 normalization.
        """
        valid_TK = topk_indices_TK >= 0
        selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)]
        logits_THK = (
            torch.einsum("thd,tkd->thk", q_THD, selected_TKD).float()
            * self.softmax_scale
        )
        logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -torch.inf)

        lse_TH = lse_HT.transpose(0, 1).float()
        comp_lse_TH = torch.logsumexp(logits_THK, dim=-1)
        log_mass_TH = comp_lse_TH - lse_TH
        shift_T1 = log_mass_TH.amax(dim=1, keepdim=True)
        shift_T1 = torch.where(
            torch.isfinite(shift_T1) & (shift_T1 < _LOG_FP32_TINY),
            shift_T1,
            torch.zeros_like(shift_T1),
        )
        mass_THK = torch.exp(logits_THK - (lse_TH + shift_T1).unsqueeze(-1))
        teacher_TK = mass_THK.sum(dim=1)
        eps = torch.finfo(torch.float32).tiny
        return teacher_TK / teacher_TK.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _kl(
        teacher_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_indices_TK: torch.Tensor,
    ) -> torch.Tensor:
        """Unnormalized KL from the student to the teacher, summed over rows.

        Args:
            teacher_TK: Target distribution ``[T, K]``, detached, rows summing to 1.
            topk_scores_TK: Student logits at the selected entries, ``[T, K]``.
            topk_indices_TK: Selected entries ``[T, K]``; ``-1`` marks an unused slot.
        """
        valid_TK = topk_indices_TK >= 0
        row_valid_T = valid_TK.any(dim=-1)
        logits_TK = topk_scores_TK.float().masked_fill(~valid_TK, -torch.inf)
        # A row with no valid slot would produce NaN in log_softmax; it is zeroed below.
        logits_TK = logits_TK.masked_fill(~row_valid_T.unsqueeze(-1), 0.0)
        log_student_TK = F.log_softmax(logits_TK, dim=-1)

        eps = torch.finfo(torch.float32).tiny
        target_TK = teacher_TK.float().clamp_min(eps)
        # Unused slots carry no teacher mass but do carry -inf log-probabilities, so
        # they are dropped before summing rather than relying on 0 * -inf.
        kl_TK = target_TK * (target_TK.log() - log_student_TK)
        kl_TK = kl_TK.masked_fill(~valid_TK, 0.0)
        return kl_TK.sum(dim=-1).masked_fill(~row_valid_T, 0.0).sum()

    def forward(
        self,
        q_THD: torch.Tensor,
        cmp_k_ND: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        lse_HT: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Build the teacher, score the student against it, inject the gradient.

        Args:
            q_THD: Attention queries ``[T, H, Dk]``.
            cmp_k_ND: Shared compressed KV ``[N, Dk]``.
            topk_indices_TK: Selected compressed entries ``[T, K]``; ``-1`` unused.
            lse_HT: Per-head log-sum-exp of the sparse softmax, ``[H, T]``.
            topk_scores_TK: Student logits at the selected entries, ``[T, K]``.
            carrier: Tensor whose backward path carries the injected gradient (the
                attention output).

        Returns:
            ``carrier`` unchanged.
        """
        with torch.no_grad():
            teacher_TK = self._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_HT)
        return self.inject(
            self._kl(teacher_TK, topk_scores_TK, topk_indices_TK), carrier=carrier
        )
