# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compressed main KV for DeepSeek V4.1 (CSA2).

Shape legend for this file:
    T = packed tokens, D = model dimension, R = ``compress_ratio``,
    Dk = head dimension of a compressed KV entry (``head_dim``).

Every CSA2 layer owns a :class:`Compressor` instance, but only the *source* layers
carry parameters: a layer whose compressed KV is produced elsewhere holds no weights
and simply returns the tensor it was handed. ``is_source`` encodes that contract and
is asserted in ``forward``, so a misconfigured layer fails loudly instead of silently
compressing or silently passing through.

Entry ``j`` of the compressed KV stands for the ``R`` tokens of group ``j``, so it
takes the position of the group's first token for RoPE. The compressed KV is rotated
here rather than in the attention module, because the indexer needs the *un-rotated*
latent: CSA2 projects indexer keys from the main KV latent, so both consumers need a
different view of the same pooling result.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.module import Module


class Compressor(Module):
    """Pool ``compress_ratio`` tokens into one main-KV entry.

    ``R == 1`` is the uncompressed case: the entry is a plain projection of the token.
    ``R > 1`` pools each group of ``R`` tokens with a softmax gate over the group.
    CSA2 has neither the absolute positional embedding nor the overlapping groups of
    the earlier CSA compressor.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        head_dim: int
        rope_head_dim: int
        compress_ratio: int
        is_source: bool
        # Present only on source layers with ``compress_ratio > 0``:
        rope: ComplexRoPE.Config | None = None
        wkv: Linear.Config | None = None
        # Gate of the softmax pooling; only used when ``compress_ratio > 1``.
        wgate: Linear.Config | None = None
        norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.is_source = config.is_source
        if not self.is_source:
            return
        if config.rope is None or config.wkv is None or config.norm is None:
            raise ValueError(
                "A Compressor source layer requires rope, wkv and norm configs."
            )
        if config.compress_ratio > 1 and config.wgate is None:
            raise ValueError(
                "A Compressor with compress_ratio > 1 requires a wgate config."
            )
        self.wkv = config.wkv.build()
        if config.compress_ratio > 1:
            self.wgate = config.wgate.build()
        self.norm = config.norm.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        positions_T: torch.Tensor | None,
        cmp_k=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Args:
            x: Hidden states of shape ``[T, D]``.
            positions: Position ids of shape ``[T]`` (``None`` means ``0..T-1``).
            cmp_k: The shared compressed KV when this layer is not a source.

        Returns:
            ``(cmp_k, latent)``. A source layer returns its newly compressed, RoPE-rotated
            KV ``[T // R, head_dim]`` together with the pre-RoPE latent of the same shape
            that the indexer projects its keys from. A non-source layer returns the tensor
            it was handed and ``latent=None``.
        """
        # A source supersedes whatever compressed KV was in flight: the first source of
        # the stack sees ``None``, and a later group's source replaces the previous
        # group's tensor rather than consuming it. What must hold is that no *reusing*
        # layer is ever handed nothing.
        if self.compress_ratio > 0 and not self.is_source:
            assert cmp_k is not None, (
                "A layer that reuses the compressed KV must receive it: no source "
                f"layer precedes this one (compress_ratio={self.compress_ratio})."
            )
        if not self.is_source:
            return cmp_k, None

        ratio = self.compress_ratio
        seqlen = x_TD.size(0)
        if seqlen % ratio != 0:
            raise ValueError(
                f"token count ({seqlen}) must be divisible by compress_ratio ({ratio})"
            )

        if ratio == 1:
            latent_TD = self.norm(self.wkv(x_TD))
        else:
            # The softmax pooling runs in fp32; the projection itself stays in the model
            # dtype, as every other projection in the model does.
            kv_TKrD = self.wkv(x_TD).unflatten(0, (-1, ratio))
            gate_TKrD = self.wgate(x_TD).unflatten(0, (-1, ratio))
            pooled_TD = (
                kv_TKrD.float() * gate_TKrD.float().softmax(dim=1)
            ).sum(dim=1)
            latent_TD = self.norm(pooled_TD.to(x_TD.dtype))

        # Entry j stands for the group starting at token j * R, so it rotates at that
        # token's position. The latent is one rank-2 head; RoPE rotates rank-3 [T, N, H].
        rotated_TD = self.rope(
            latent_TD.unsqueeze(1), positions=positions_T[:: self.compress_ratio]
        ).squeeze(1)
        return rotated_TD, latent_TD
