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

For that reason the scores are computed twice: once without a graph over all candidates
to take the top-k, and once with a graph over the ``K`` selected entries only. Keeping
the full ``[T, Hi, N]`` score tensor in the autograd graph would pin gigabytes per layer
at long context, which is the standard failure mode of this loss.

Every layer owns an :class:`Indexer`, but only *source* layers carry parameters: an
index-source layer produces the top-k, a layer that owns the compressed KV also produces
the index keys, and a reuse layer returns what it was handed. ``is_source`` and
``owns_k`` encode that contract and are asserted in ``forward``, so a misconfigured layer
fails loudly instead of silently recomputing or silently reusing.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.module import Module


def select_candidate_blocks(
    scores_TN: torch.Tensor,
    visible_lens_T1: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the hierarchical indexer: keep the best-scoring blocks.

    Args:
        scores_TN: Index scores ``[T, N]``, already masked to ``-inf`` on entries the
            query cannot see.
        visible_lens_T1: Number of causally visible entries per query, ``[T, 1]``.
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

    # The block holding a query's newest visible entry is only partly filled, and must
    # not be outscored by an older, full block.
    last_T1 = (visible_lens_T1 - 1) // block_size
    block_scores_TB = block_scores_TB.masked_fill(
        torch.arange(num_blocks, device=scores_TN.device).unsqueeze(0) == last_T1,
        torch.inf,
    )

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
        # Queries scored per iteration of the chunked score computation.
        chunk_size: int = 512
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
        self.chunk_size = config.chunk_size
        self.is_source = config.is_source
        self.owns_k = config.owns_k
        self.is_candidate_source = config.is_candidate_source
        self.uses_candidates = config.uses_candidates
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
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
        num_tokens = idx_q_THiDi.size(0)
        chunks = []
        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            # Invalid (-1) slots gather entry 0; the loss masks them out.
            selected_CK = topk_indices_TK[start:end].clamp_min(0)
            selected_keys_CKDi = idx_k_NDi[selected_CK]
            logits_CHK = torch.einsum(
                "chd,ckd->chk", idx_q_THiDi[start:end], selected_keys_CKDi
            )
            logits_CHK = logits_CHK.relu() * weights_THi[start:end].unsqueeze(-1)
            chunks.append(logits_CHK.sum(dim=1))
        if not chunks:
            return idx_q_THiDi.new_zeros(num_tokens, 0)
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        x_TD: torch.Tensor,
        qr_TQ: torch.Tensor,
        positions_T: torch.Tensor,
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
            assert latent_TDp is not None, (
                "A key-owning indexer projects its keys from the compressor latent."
            )
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

        x_TD = x_TD.detach()
        qr_TQ = qr_TQ.detach()
        num_tokens = x_TD.size(0)
        hi, di = self.index_n_heads, self.index_head_dim

        idx_q_THiDi = self.rope(
            self.wq_b(qr_TQ).unflatten(-1, (hi, di)), positions=positions_T
        )
        if self.owns_k:
            idx_k_NDi = self.k_norm(self.wk(latent_TDp.detach()))
            # One rank-2 head; RoPE rotates rank-3 [T, N, H].
            idx_k_NDi = self.rope(
                idx_k_NDi.unsqueeze(1), positions=positions_T[:: self.compress_ratio]
            ).squeeze(1)
        else:
            idx_k_NDi = idx_k_TpDi
        num_cmp = idx_k_NDi.size(0)

        # ``weights_proj`` is scaled by the index softmax scale and the head count, as
        # in the reference: the per-head scores are averaged rather than summed.
        weights_THi = self.weights_proj(x_TD) * (
            self.index_head_dim**-0.5 * hi**-0.5
        )

        visible_lens_T1 = ((positions_T + 1) // self.compress_ratio).unsqueeze(-1)
        topk = min(self.index_topk, num_cmp)
        topk_indices_TK = torch.full(
            (num_tokens, topk), -1, dtype=torch.long, device=x_TD.device
        )

        # Selection carries no gradient, so it runs without a graph and in chunks: the
        # full ``[T, Hi, N]`` score tensor is the memory hazard this loss is famous for.
        with torch.no_grad():
            for start in range(0, num_tokens, self.chunk_size):
                end = min(start + self.chunk_size, num_tokens)
                scores_CHN = torch.einsum(
                    "chd,nd->chn", idx_q_THiDi[start:end], idx_k_NDi
                )
                scores_CHN = scores_CHN.relu() * weights_THi[start:end].unsqueeze(-1)
                scores_CN = scores_CHN.sum(dim=1)
                visible_CN = (
                    torch.arange(num_cmp, device=x_TD.device)
                    < visible_lens_T1[start:end]
                )
                scores_CN = scores_CN.masked_fill(~visible_CN, -torch.inf)

                if self.is_candidate_source:
                    if candidates_TN is None:
                        candidates_TN = torch.zeros(
                            num_tokens, num_cmp, dtype=torch.bool, device=x_TD.device
                        )
                    candidates_TN[start:end] = select_candidate_blocks(
                        scores_CN,
                        visible_lens_T1[start:end],
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                    )
                elif self.uses_candidates and candidates_TN is not None:
                    scores_CN = scores_CN.masked_fill(~candidates_TN[start:end], -torch.inf)

                if topk == 0:
                    continue
                selected_CK = (
                    scores_CN.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
                )
                # Entries the query cannot see yet (or that the pool excluded) come back
                # as -1, which the sparse attention and the loss both skip.
                topk_indices_TK[start:end] = torch.where(
                    visible_CN.gather(-1, selected_CK), selected_CK, -1
                )

        topk_scores_TK = self._selected_scores(
            idx_q_THiDi, idx_k_NDi, weights_THi, topk_indices_TK
        )
        return idx_k_NDi, topk_indices_TK, topk_scores_TK, candidates_TN


class IndexerKLLoss(AuxLoss):
    """Distill the attention's distribution over the top-k entries into the indexer.

    The teacher ``p`` is the head-averaged attention probability that the sparse
    attention assigned to each selected entry, computed with the full denominator
    (sliding window and sink included) and L1-normalized over the selected support. The
    student is ``softmax(I)`` over the same support. Both the teacher and the indexer's
    inputs are detached, so the gradient only ever reaches the indexer's parameters.

    The loss is summed over rows and normalized by the step's global valid-token count
    by the :class:`AuxLoss` framework, exactly like the MoE balance loss.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Same fields as ``AuxLoss.Config``; the loss adds no knobs.

        A distinct Config is required: ``Config.build()`` constructs the class that owns
        the config, so an ``AuxLoss.Config`` here would build a plain ``AuxLoss``.
        """

    def forward(
        self,
        teacher_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            teacher_TK: Target distribution ``[T, K]``, detached, rows summing to 1.
            topk_scores_TK: Student logits at the selected entries, ``[T, K]``.
            topk_indices_TK: Selected entries ``[T, K]``; ``-1`` marks an unused slot.
            carrier: Tensor whose backward path carries the injected gradient.

        Returns:
            ``carrier`` unchanged.
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
        raw_sum = kl_TK.sum(dim=-1).masked_fill(~row_valid_T, 0.0).sum()

        return self.inject(raw_sum, carrier=carrier)
