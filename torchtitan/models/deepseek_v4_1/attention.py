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

Each query attends to two sources at once, concatenated into a single masked softmax:
its own sliding window over the layer's KV ``[T, Dk]``, and the ``K`` compressed entries
selected by the indexer out of the shared compressed KV ``[N, Dk]``. A learned per-head
sink logit takes part in the softmax denominator without contributing a value, which is
why a row with no reachable entry still produces zeros instead of NaN.

The attention is also where the indexer's distillation loss is computed. It is the only
place that has the per-head probabilities over the selected entries, so the teacher
needs no second pass over the scores; the student logits arrive as an input, which keeps
the loss a function of tensors that are already in hand. The two halves of the indexer
contract are therefore: :class:`~.indexer.Indexer` produces the selection and the student
logits, and :class:`SparseAttention` compares the attention's own distribution against
them and injects the gradient on its output.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor
from .indexer import Indexer


class SparseAttention(Module):
    """Sparse attention over the sliding window plus the selected compressed entries.

    ``q`` carries ``H`` heads while the KV sources carry a single shared head, so the
    gathered KV is broadcast across heads by the einsum. ``topk_indices`` index the
    compressed KV and use ``-1`` for unused slots.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        softmax_scale: float
        # Queries processed per iteration: bounds the ``[chunk, K, Dk]`` gathered KV.
        chunk_size: int = 512
        # Indexer distillation loss, attached only on layers that select compressed
        # entries (``compress_ratio > 0``).
        aux_loss: AuxLoss.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.chunk_size = config.chunk_size
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def _window_indices(self, num_tokens: int, device) -> torch.Tensor:
        """Sliding-window KV slots for every query, ``[T, W]``, ``-1`` outside the window."""
        window = min(num_tokens, self.window_size)
        query_T1 = torch.arange(num_tokens, device=device).unsqueeze(1)
        idx_TW = (query_T1 - window + 1).clamp_min(0) + torch.arange(
            window, device=device
        )
        return torch.where(idx_TW <= query_T1, idx_TW, torch.full_like(idx_TW, -1))

    def forward(
        self,
        q_THD: torch.Tensor,
        swa_k_TD: torch.Tensor,
        attn_sink_H: torch.Tensor,
        *,
        cmp_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            q: Queries of shape ``[T, H, Dk]``.
            swa_k: Sliding-window KV of shape ``[T, Dk]``, shared across heads.
            attn_sink: Per-head sink logits of shape ``[H]``.
            cmp_k: Shared compressed KV of shape ``[N, Dk]``.
            topk_indices: Selected compressed entries ``[T, K]``, ``-1`` for unused.
            topk_scores: Student logits at those entries ``[T, K]``.

        Returns:
            Attention output of shape ``[T, H, Dk]``.
        """
        num_tokens, num_heads, _ = q_THD.size()
        uses_cmp = cmp_k is not None and topk_indices is not None
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")

        idx_TW = self._window_indices(num_tokens, q_THD.device)
        num_window = idx_TW.size(1)
        if uses_cmp:
            kv_ND = torch.cat([swa_k_TD, cmp_k], dim=0)
            # Offsets place the compressed entries after the window in the KV space.
            idx_TK = torch.cat(
                [
                    idx_TW,
                    torch.where(
                        topk_indices >= 0,
                        topk_indices + num_tokens,
                        torch.full_like(topk_indices, -1),
                    ),
                ],
                dim=-1,
            )
        else:
            kv_ND = swa_k_TD
            idx_TK = idx_TW

        wants_teacher = (
            self.training
            and self.aux_loss is not None
            and topk_scores is not None
            and uses_cmp
        )
        teacher_TK = None
        if wants_teacher:
            teacher_TK = torch.zeros(
                num_tokens,
                topk_indices.size(1),
                dtype=torch.float32,
                device=q_THD.device,
            )

        sink_CH = attn_sink_H.float().unsqueeze(0)
        outputs = []
        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            idx_CK = idx_TK[start:end]
            valid_CK = idx_CK >= 0
            gathered_CKD = kv_ND[idx_CK.clamp_min(0)]

            scores_CHK = (
                torch.einsum("chd,ckd->chk", q_THD[start:end], gathered_CKD)
                * self.softmax_scale
            )
            scores_CHK = scores_CHK.masked_fill(~valid_CK.unsqueeze(1), -torch.inf)
            # The sink joins the max as well, so an all-invalid row softmaxes to zeros
            # rather than to NaN.
            row_max_CH = torch.maximum(scores_CHK.amax(dim=-1), sink_CH)
            probs_CHK = torch.exp(scores_CHK - row_max_CH.unsqueeze(-1))
            denom_CH = probs_CHK.sum(dim=-1) + torch.exp(sink_CH - row_max_CH)
            out_CHD = torch.einsum(
                "chk,ckd->chd", probs_CHK.to(q_THD.dtype), gathered_CKD
            ) / denom_CH.unsqueeze(-1)
            outputs.append(out_CHD.to(q_THD.dtype))

            if teacher_TK is not None:
                # The teacher is the compressed half of the attention distribution:
                # per-head probabilities already carry the full denominator (window and
                # sink included). Detached: it must not train the attention.
                with torch.no_grad():
                    teacher_TK[start:end] = probs_CHK[:, :, num_window:].sum(dim=1)

        attn_out = torch.cat(outputs, dim=0)

        if teacher_TK is not None:
            # L1-normalize over the selected support, so the target is a distribution.
            eps = torch.finfo(torch.float32).tiny
            teacher_TK = teacher_TK / teacher_TK.sum(dim=-1, keepdim=True).clamp_min(eps)
            attn_out = self.aux_loss(
                teacher_TK, topk_scores, topk_indices, carrier=attn_out
            )
        return attn_out


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
        dim: int
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1536
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        inner_attention: SparseAttention.Config  # pyrefly: ignore [bad-override]
        rope: ComplexRoPE.Config
        compressor: Compressor.Config
        indexer: Indexer.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.n_groups
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.head_dim**-0.5

        self.rope = config.rope.build()
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv = config.wkv.build()
        self.kv_norm = config.kv_norm.build()
        self.wo_a = config.wo_a.build()
        self.wo_b = config.wo_b.build()
        # Holds one sink logit per head; the forward squeezes the trailing dim, matching
        # the parameter's meaning in the checkpoint.
        self.attn_sink = config.attn_sink.build()
        self.compressor = config.compressor.build()
        self.indexer = config.indexer.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        positions_T: torch.Tensor,
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
        idx_k, topk_indices, topk_scores, candidates = self.indexer(
            x_TD,
            qr_TQ,
            positions_T,
            latent_TDp=latent_TD,
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
            self.attn_sink.weight.squeeze(-1),
            cmp_k=cmp_k if uses_cmp else None,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
        )
        o_THD = self.rope(o_THD, positions=positions_T, inverse=True)

        # wo_a is block-diagonal over groups: each group projects only its own heads.
        n_local_heads = o_THD.size(1)
        n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
        o_TGD = o_THD.view(num_tokens, n_local_groups, -1)
        wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o_TGR = torch.einsum("tgd,grd->tgr", o_TGD, wo_a)
        return (
            self.wo_b(o_TGR.reshape(num_tokens, -1)),
            cmp_k,
            idx_k,
            topk_indices,
            topk_scores,
            candidates,
        )
