# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Released attention geometry against an independent FP32 attention/KL oracle."""

from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_transformer")

from torchtitan.models.glm53_flash.attention import IndexerLoss
from torchtitan_recipes.overrides.glm53_flash_npu import (
    AscendAllGatherKVSparseAttention,
    AscendInnerIndexer,
    AscendSparseAttention,
)


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("trunk_scale", [0.0, 1.0])
@pytest.mark.parametrize("query_start", [0, 2048])
def test_teacher_bridge_matches_explicit_kl(trunk_scale, query_start):
    num_kv, heads, latent_dim, index_heads, index_dim, topk = (
        4096,
        64,
        512,
        32,
        128,
        512,
    )
    positions = torch.cat([torch.arange(n, device="npu") for n in (1024, 2048, 1024)])
    config = AscendSparseAttention.Config(
        ratio=4,
        aux_loss=IndexerLoss.Config(coeff=0.3),
        num_heads=64,
        num_index_heads=32,
        topk=512,
    )
    attention = config.build().to("npu")
    metadata = config.build_attention_metadata(
        positions, max_num_documents=4, max_context_length=num_kv
    )
    pools = metadata.pool_rows_PW.size(0)
    if query_start:
        # Exercise a CP query fragment against full keys without a second device.
        group = Mock()
        group.size.return_value = 2
        with patch(
            "torchtitan.models.glm53_flash.cp_dsa.spmd_mesh_group", return_value=group
        ), patch("torch.distributed.get_rank", return_value=1):
            metadata = AscendAllGatherKVSparseAttention.prepare_cp_metadata(
                metadata, permutation=None
            )
        positions = positions[query_start:]
    num_q = positions.numel()
    # All native kernels see the complete geometry. A sampled query objective
    # keeps the independent FP32 autograd oracle's gathered KV memory bounded.
    sample = torch.cat((torch.arange(8), torch.linspace(16, num_q - 1, 24).long())).to(
        "npu"
    )
    padding = torch.ones(num_q, device="npu", dtype=torch.bool)
    padding[sample] = False

    def tensor(shape, scale, dtype=torch.bfloat16):
        return (torch.randn(shape, dtype=dtype, device="npu") * scale).requires_grad_()

    q = tensor((num_q, heads, latent_dim), 0.3)
    kv = tensor((num_kv, latent_dim), 0.3)
    iq = tensor((num_q, index_heads, index_dim), 0.3)
    ik = tensor((pools, index_dim), 0.3)
    iw = tensor((num_q, index_heads), 0.1, torch.float32)
    indexer = AscendInnerIndexer.Config(topk=topk, ratio=4).build()
    indices, scores = indexer(iq, ik, iw, attention_metadata=metadata)
    scores.retain_grad()
    out = attention(
        q,
        kv,
        indices,
        scores,
        attention_metadata=metadata,
        scale=1 / 16,
        aux_loss_denominator=torch.tensor(4096.0, device="npu"),
        padding_mask=padding,
    )
    grad_out = torch.zeros_like(out)
    grad_out[sample] = torch.randn_like(out[sample]) * trunk_scale
    (out * grad_out).sum().backward()
    torch.accelerator.synchronize()

    rq, rkv, riq, rik, riw = [
        t.detach().float().cpu().requires_grad_()
        for t in (q[sample], kv, iq[sample], ik, iw[sample])
    ]
    idx = indices[sample].cpu().long()
    selected = idx[..., None] * 4 + torch.arange(4)
    selected = selected.masked_fill(idx[..., None] < 0, -1)
    visible = positions[sample].cpu() + 1
    remainder = visible % 4
    tail = visible[:, None] - remainder[:, None] + torch.arange(3)
    tail = tail.masked_fill(torch.arange(3) >= remainder[:, None], -1)
    slots = torch.cat((selected.flatten(-2), tail), -1)
    rows = (slots + metadata.token_start_T[sample].cpu()[:, None]).clamp_min(0)
    logits = torch.einsum("thd,tkd->thk", rq, rkv[rows]) / 16
    prob = logits.masked_fill(slots[:, None] < 0, -torch.inf).softmax(-1)
    expected_out = torch.einsum("thk,tkd->thd", prob, rkv[rows])
    teacher = (
        prob.detach().mean(1)[:, : topk * 4].reshape(sample.numel(), topk, 4).sum(-1)
    )
    pool_rows = (idx + metadata.pool_start_T[sample].cpu()[:, None]).clamp_min(0)
    student = (
        torch.einsum("tie,tse->tis", riq, rik[pool_rows]).relu() * riw[..., None]
    ).sum(1)
    log_y = student.masked_fill(idx < 0, -1e30).log_softmax(-1)
    # Cross entropy has the same student gradient as KL with a detached teacher.
    objective = (expected_out * grad_out[sample].float().cpu()).sum() - (
        teacher * log_y
    ).sum() * (0.3 / 4096)
    objective.backward()
    torch.testing.assert_close(
        out[sample].float().cpu(), expected_out, atol=0.003, rtol=0.02
    )
    assert scores.grad is not None
    torch.testing.assert_close(
        scores.grad[sample].cpu(), teacher * (0.3 / 4096), atol=3e-7, rtol=0.02
    )
    assert scores.grad[padding].count_nonzero() == 0
    for name, actual, ref, local in zip(
        ("q", "kv", "iq", "ik", "iw"),
        (q, kv, iq, ik, iw),
        (rq, rkv, riq, rik, riw),
        (True, False, True, False, True),
    ):
        assert actual.grad is not None and ref.grad is not None, name
        grad = actual.grad[sample] if local else actual.grad
        error = (grad.float().cpu() - ref.grad).norm()
        assert error <= 0.02 * ref.grad.norm() + 1e-6, (
            name,
            error.item(),
            ref.grad.norm().item(),
        )
        if local:
            assert actual.grad[padding].count_nonzero() == 0, name
    assert iq.grad is not None and iq.grad.abs().sum() > 0


def test_tail_without_complete_pools():
    # Single-token documents retain a large launch while exercising no cmp KV.
    num_tokens, heads, width = 4096, 64, 512
    config = AscendSparseAttention.Config(
        ratio=4,
        aux_loss=IndexerLoss.Config(coeff=0.3),
        num_heads=64,
        num_index_heads=32,
        topk=512,
    )
    metadata = config.build_attention_metadata(
        torch.zeros(num_tokens, dtype=torch.long, device="npu")
    )
    q = torch.randn(
        num_tokens, heads, width, device="npu", dtype=torch.bfloat16
    ).requires_grad_()
    kv = torch.randn(
        num_tokens, width, device="npu", dtype=torch.bfloat16
    ).requires_grad_()
    indices = torch.empty(num_tokens, 0, device="npu", dtype=torch.int32)
    scores = torch.empty(num_tokens, 0, device="npu", dtype=torch.float32)
    out = config.build().to("npu")(
        q,
        kv,
        indices,
        scores,
        attention_metadata=metadata,
        scale=1 / 16,
        aux_loss_denominator=torch.tensor(float(num_tokens), device="npu"),
    )
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()
    torch.testing.assert_close(out, kv[:, None].expand_as(out), atol=0, rtol=0)
    assert q.grad is not None
    # The two FP32 dot-product reductions in softmax backward need not cancel bitwise.
    assert q.grad.abs().max() <= width * torch.finfo(torch.float32).eps
    assert kv.grad is not None
    expected_grad = grad_out.float().sum(1)
    error = (kv.grad.float() - expected_grad).norm()
    assert error <= 0.02 * expected_grad.norm() + 1e-6


def test_rejects_partial_core_initialization_before_dispatch():
    config = AscendSparseAttention.Config(
        ratio=4, aux_loss=None, num_heads=64, num_index_heads=32, topk=512
    )
    with pytest.raises(ValueError, match="fill every vector core"):
        config.build_attention_metadata(torch.arange(1, device="npu"))
