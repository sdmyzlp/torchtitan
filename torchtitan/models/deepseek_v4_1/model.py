# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 text backbone.

Shape legend for this file:
    T = packed tokens, D = model dimension, hc = ``hc_mult`` residual branches.

The block carries the residual stream as ``hc`` parallel branches and threads the
cross-layer attention state (compressed KV, index keys, selected entries, student
logits, candidate pool) explicitly: an attention source layer returns the tensors it
produced, every other layer returns what it was handed. That keeps the reuse structure
visible in the forward signature instead of in mutable module state, at the cost of a
longer tuple.

Single-Pass mHC means each sublayer collapses its input with the mixing coefficients
predicted by the *previous* sublayer, so the block returns the coefficients its
successor needs. The stack has no learned output head: the model collapses the branches
with the last block's coefficients and feeds the result straight to the output norm.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import ModuleDict

from .attention import Attention
from .mhc import HcPost, HcPre


class DeepSeekV41TransformerBlock(TransformerBlock):
    """Transformer block with HC mixing around attention and the MoE feed-forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Redeclared with the V4.1 types so config builders can reach the fields the
        # sharding and MTP helpers need.
        attention: Attention.Config  # pyrefly: ignore [bad-override]
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe = config.moe.build()
        self.moe_enabled = True
        self.hc_attn_pre = config.hc_attn_pre.build()
        self.hc_ffn_pre = config.hc_ffn_pre.build()
        self.hc_post = config.hc_post.build()

    def forward(
        self,
        x_THcD: torch.Tensor,
        positions_T: torch.Tensor,
        pre_mix_THc: torch.Tensor,
        *,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        ``pre_mix`` is the attention-input coefficient the next block must consume, and
        the shared attention tensors are this block's contribution to the chain.
        """
        residual_THcD = x_THcD
        y_TD, attn_pre_THc, attn_post_THc, attn_comb_THcHc = self.hc_attn_pre(
            x_THcD, pre_mix_THc
        )
        y_TD, cmp_k, idx_k, topk_indices, topk_scores, candidates = self.attention(
            self.attention_norm(y_TD),
            positions_T,
            cmp_k=cmp_k,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )
        x_THcD = self.hc_post(y_TD, residual_THcD, attn_post_THc, attn_comb_THcHc)

        residual_THcD = x_THcD
        y_TD, ffn_pre_THc, ffn_post_THc, ffn_comb_THcHc = self.hc_ffn_pre(
            x_THcD, attn_pre_THc
        )
        x_THcD = self.hc_post(
            self.moe(self.ffn_norm(y_TD)), residual_THcD, ffn_post_THc, ffn_comb_THcHc
        )
        return x_THcD, ffn_pre_THc, cmp_k, idx_k, topk_indices, topk_scores, candidates


class DeepSeekV41Model(Decoder):
    """DeepSeek V4.1 decoder with mHC residual branches and CSA2 sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        hc_mult: int = 4
        n_layers: int
        norm_eps: float = 1e-6
        compress_ratios: tuple[int, ...] = ()
        # Hierarchical sparse indexer switch: a layer in ``index_source_layers`` that
        # equals ``candidate_source_layer`` builds a shared pool of candidate entries,
        # and later index sources restrict their selection to it. Disabled by default;
        # the pool boundary receives no gradient, so it is a training/inference
        # consistency and cost device rather than a learned component.
        use_candidates: bool = False
        candidate_source_layer: int = -1
        candidate_topk_blocks: int = 0
        candidate_block_size: int = 0

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            # The block returns a tuple and the cross-layer state is threaded through
            # the stack, neither of which the pipeline scheduler handles.
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "DeepSeek V4.1 does not support pipeline parallelism: the sparse "
                    "attention threads compressed KV, index keys and top-k across layers."
                )
            # Compression and the sliding window both need the full sequence.
            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "DeepSeek V4.1 does not support context parallelism yet."
                )
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "DeepSeek V4.1 does not support tensor parallelism yet: the shared "
                    "cross-layer tensors and the single-head KV are not sharded."
                )

            if len(self.compress_ratios) < self.n_layers:
                raise ValueError(
                    f"compress_ratios must have at least n_layers ({self.n_layers}) "
                    f"entries, got {len(self.compress_ratios)}."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_flops = 0
            for layer_cfg in self.layers:
                attention = layer_cfg.attention
                # Sliding window, always attended.
                attention_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=seq_len,
                    sliding_window_size=attention.inner_attention.window_size,
                )
                if attention.compress_ratio > 0:
                    # The selected compressed entries, at most index_topk per query.
                    attention_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=min(
                            attention.indexer.index_topk,
                            seq_len // attention.compress_ratio,
                        ),
                    )
                    if attention.indexer.is_source:
                        # The indexer scores every causally visible compressed entry.
                        attention_flops += (
                            6
                            * attention.indexer.index_n_heads
                            * attention.indexer.index_head_dim
                            * (seq_len // attention.compress_ratio)
                        )
            return nparams, 6 * active_nparams + attention_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.hc_mult = config.hc_mult

    def get_attention_masks(
        self,
        positions,
        *,
        padding_mask=None,
        max_num_documents=None,
        max_context_length=None,
    ):
        """CSA2 builds its own sliding-window and top-k masks internally."""
        del positions, padding_mask, max_num_documents, max_context_length
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        del attention_masks
        if positions is None:
            positions = torch.arange(tokens.shape[0], device=tokens.device)

        h_TD = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h_THcD = h_TD.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix_THc = HcPre.identity_pre_mix(
            h_THcD.size(0), self.hc_mult, h_THcD.device
        )

        cmp_k: torch.Tensor | None = None
        idx_k: torch.Tensor | None = None
        topk_indices: torch.Tensor | None = None
        topk_scores: torch.Tensor | None = None
        candidates: torch.Tensor | None = None
        for block in cast(ModuleDict, self.layers).values():
            h_THcD, pre_mix_THc, cmp_k, idx_k, topk_indices, topk_scores, candidates = (
                block(
                    h_THcD,
                    positions,
                    pre_mix_THc,
                    cmp_k=cmp_k,
                    idx_k=idx_k,
                    topk_indices=topk_indices,
                    topk_scores=topk_scores,
                    candidates=candidates,
                )
            )

        # The stack has no learned output head: it collapses with the last block's
        # attention-input coefficients.
        h_TD = HcPre.collapse(h_THcD, pre_mix_THc)
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head or self.lm_head is None:
            return h_TD
        return self.lm_head(h_TD)
