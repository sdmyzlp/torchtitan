# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek V4.1 mHC residual update.

``HcPost`` contracts the residual-branch axis: entry ``(m, q)`` of ``comb``
weights residual branch ``m`` into output branch ``q``::

    out[..., q, d] = post[..., q] * y[..., d] + sum_m comb[..., m, q] * residual[..., m, d]

The product ``comb * residual`` ends in ``(..., m, q, d)``, so the contracted
axis is the third-from-last one. Contracting the output axis instead collapses
the update to ``residual * comb.sum(-1)``: a per-branch rescaling that is
approximately the identity, because Sinkhorn drives the row sums to one.
"""

import torch

from torchtitan.models.deepseek_v4_1.mhc import HcPost


def _post() -> HcPost:
    return HcPost.Config().build()


def _oracle(y, residual, post, comb):
    """Explicit per-branch sum in float64, one term at a time."""
    y64 = y.double()
    residual64 = residual.double()
    post64 = post.double()
    comb64 = comb.double()
    hc = residual.size(-2)
    out = post64.unsqueeze(-1) * y64.unsqueeze(-2)
    for m in range(hc):
        for q in range(hc):
            out[..., q, :] = out[..., q, :] + comb64[..., m, q].unsqueeze(-1) * residual64[..., m, :]
    return out


def test_matches_the_explicit_branch_sum():
    torch.manual_seed(0)
    num_tokens, hc, dim = 5, 4, 6
    y = torch.randn(num_tokens, dim)
    residual = torch.randn(num_tokens, hc, dim)
    post = torch.rand(num_tokens, hc)
    comb = torch.randn(num_tokens, hc, hc)

    got = _post()(y, residual, post, comb)

    torch.testing.assert_close(
        got, _oracle(y, residual, post, comb).float(), rtol=1e-5, atol=1e-6
    )


def test_permuted_comb_permutes_the_branches():
    """A permutation comb must permute the residual branches.

    A permutation has comb row sums of one, so a contraction over the wrong axis
    returns the residual unchanged instead of swapping the branches.
    """
    num_tokens, hc, dim = 3, 2, 4
    residual = torch.randn(num_tokens, hc, dim)
    y = torch.zeros(num_tokens, dim)
    post = torch.zeros(num_tokens, hc)
    comb = torch.zeros(num_tokens, hc, hc)
    comb[..., 0, 1] = 1.0  # residual branch 0 feeds output branch 1
    comb[..., 1, 0] = 1.0  # residual branch 1 feeds output branch 0

    got = _post()(y, residual, post, comb)

    torch.testing.assert_close(got, residual.flip(-2), rtol=0, atol=0)


def test_doubly_stochastic_comb_still_mixes():
    """Row sums of one do not make the update the identity."""
    num_tokens, hc, dim = 2, 2, 3
    residual = torch.randn(num_tokens, hc, dim)
    y = torch.zeros(num_tokens, dim)
    post = torch.zeros(num_tokens, hc)
    comb = torch.full((num_tokens, hc, hc), 0.5)

    got = _post()(y, residual, post, comb)

    expected = 0.5 * (residual[..., 0, :] + residual[..., 1, :])
    expected = expected.unsqueeze(-2).expand(num_tokens, hc, dim)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_leading_dimensions_do_not_move_the_axis():
    """Packed ``[T, hc, D]`` and batched ``[1, T, hc, D]`` must agree.

    A literal axis (``dim=2``) is only correct for one of the two, so this pins
    that the contraction follows the rank.
    """
    torch.manual_seed(3)
    num_tokens, hc, dim = 4, 3, 5
    y = torch.randn(num_tokens, dim)
    residual = torch.randn(num_tokens, hc, dim)
    post = torch.rand(num_tokens, hc)
    comb = torch.randn(num_tokens, hc, hc)

    packed = _post()(y, residual, post, comb)
    batched = _post()(
        y.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    )

    torch.testing.assert_close(batched.squeeze(0), packed, rtol=1e-5, atol=1e-6)
