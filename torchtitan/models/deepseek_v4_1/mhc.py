# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-Pass mHC (manifold-constrained Hyper-Connections) for DeepSeek V4.1.

Shape legend for this file:
    T = packed tokens, D = model dimension,
    hc = number of residual branches (``hc_mult``),
    M = (2 + hc) * hc = number of mixing coefficients predicted per token.

The residual stream is carried as ``hc`` parallel branches ``X_l``. Each sublayer
predicts its own coefficients ``(A, B, C) = H(X_l)`` from the stream it starts from,
but collapses its input with the ``A`` predicted by the *previous* sublayer. Shifting
input mixing by one sublayer is what makes the pass single: nothing has to wait for a
reduction over the whole hidden dimension before the stream can be mixed, so residual
update, coefficient prediction and input mixing can share one traversal.

The constraint is on the branch-mixing matrix ``C``: Sinkhorn normalization keeps it
doubly stochastic, which puts the mixing on the manifold of permutation-like mixings
instead of letting each branch rescale the stream arbitrarily.

There is no learned head at the top of the stack: the final collapse reuses the ``A``
predicted by the last block's FFN (see ``DeepSeekV41Model.forward``), which is why
``HcPre.collapse`` is a static method rather than a module of its own.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module


class HcPre(Module):
    """Collapse the residual branches into a sublayer input and emit its mixing
    coefficients.

    ``pre``/``post``/``comb`` are predicted from the stream this call starts from, while
    the collapse consumes the ``pre_mix`` produced by the previous sublayer. ``post`` and
    ``comb`` are handed to :class:`HcPost` for the residual update; ``pre`` is returned so
    the next sublayer can collapse with it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hc_mult: int = 4
        sinkhorn_iters: int = 20
        # Added to the normalized ``comb`` rows/columns while balancing, and to ``pre``.
        hc_eps: float = 1e-6
        # Epsilon of the RMS normalization folded into the coefficient projection.
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        # Kept in fp32: the coefficients steer the residual stream directly, and the
        # checkpoint stores them unquantized.
        self.hc_fn = nn.Parameter(
            torch.empty(mix_hc, config.hc_mult * config.dim, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    @staticmethod
    def identity_pre_mix(
        num_tokens: int, hc_mult: int, device: torch.device
    ) -> torch.Tensor:
        """Mixing coefficients that select branch 0.

        The first sublayer of the stack has no predecessor to inherit coefficients from.
        """
        pre_mix_THc = torch.zeros(
            num_tokens, hc_mult, dtype=torch.float32, device=device
        )
        pre_mix_THc[:, 0] = 1.0
        return pre_mix_THc

    @staticmethod
    def collapse(x_THcD: torch.Tensor, pre_mix_THc: torch.Tensor) -> torch.Tensor:
        """Collapse the branches into one sublayer input. ``[T, hc, D] x [T, hc] -> [T, D]``."""
        y_TD = torch.sum(pre_mix_THc.unsqueeze(-1) * x_THcD.float(), dim=-2)
        return y_TD.to(x_THcD.dtype)

    def _split_sinkhorn(
        self, mixes_TM: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the mixing logits and balance ``comb`` to be doubly stochastic.

        A row softmax followed by alternating row/column normalizations (Sinkhorn), so
        ``comb`` is a doubly stochastic mixing matrix over the branches.
        """
        hc, eps = self.hc_mult, self.hc_eps
        pre_TMc, post_TMc, comb_TMcc = mixes_TM.split([hc, hc, hc * hc], dim=-1)
        comb_THcHc = comb_TMcc.unflatten(-1, (hc, hc))

        pre_THc = torch.sigmoid(pre_TMc * self.hc_scale[0] + self.hc_base[:hc]) + eps
        post_THc = 2 * torch.sigmoid(
            post_TMc * self.hc_scale[1] + self.hc_base[hc : 2 * hc]
        )
        comb_THcHc = (
            comb_THcHc * self.hc_scale[2] + self.hc_base[2 * hc :].view(hc, hc)
        )

        comb_THcHc = torch.exp(comb_THcHc - comb_THcHc.amax(dim=-1, keepdim=True))
        comb_THcHc = comb_THcHc / comb_THcHc.sum(dim=-1, keepdim=True) + eps
        comb_THcHc = comb_THcHc / (comb_THcHc.sum(dim=-2, keepdim=True) + eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_THcHc = comb_THcHc / (comb_THcHc.sum(dim=-1, keepdim=True) + eps)
            comb_THcHc = comb_THcHc / (comb_THcHc.sum(dim=-2, keepdim=True) + eps)
        return pre_THc, post_THc, comb_THcHc

    def forward(
        self, x_THcD: torch.Tensor, pre_mix_THc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            x: Residual branches of shape ``[T, hc, D]``.
            pre_mix: Input-mixing coefficients of shape ``[T, hc]`` predicted by the
                previous sublayer.

        Returns:
            ``(y, pre, post, comb)``: the sublayer input ``[T, D]``, the input-mixing
            coefficients the next sublayer must consume ``[T, hc]``, and the ``post``
            ``[T, hc]`` / ``comb`` ``[T, hc, hc]`` consumed by :class:`HcPost`.
        """
        flat_TN = x_THcD.flatten(-2).float()
        # One RMS statistic per token over the whole flattened hc * D stream.
        rsqrt_T1 = torch.rsqrt(
            flat_TN.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        mixes_TM = F.linear(flat_TN, self.hc_fn) * rsqrt_T1
        pre_THc, post_THc, comb_THcHc = self._split_sinkhorn(mixes_TM)
        return self.collapse(x_THcD, pre_mix_THc), pre_THc, post_THc, comb_THcHc


class HcPost(Module):
    """Expand a sublayer output back to the residual branches and mix the residual in."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        y_TD: torch.Tensor,
        residual_THcD: torch.Tensor,
        post_THc: torch.Tensor,
        comb_THcHc: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            y: Sublayer output of shape ``[T, D]``.
            residual: Residual branches entering the sublayer, ``[T, hc, D]``.
            post: Output-mixing coefficients ``[T, hc]``, scaling ``y`` onto each branch.
            comb: Branch-mixing coefficients ``[T, hc, hc]`` applied to ``residual``.

        Returns:
            Updated residual branches of shape ``[T, hc, D]``.
        """
        out_THcD = post_THc.unsqueeze(-1) * y_TD.unsqueeze(-2) + torch.sum(
            comb_THcHc.unsqueeze(-1) * residual_THcD.unsqueeze(-2), dim=-2
        )
        return out_THcD.type_as(y_TD)
