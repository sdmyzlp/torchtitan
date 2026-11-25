# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.glm53_flash import build_model_config


def _check_kcp(rank, world_size, store):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("cp",))
        reference_config = build_model_config().layers[0].attention
        parallel_config = (
            build_model_config(
                parallelism=ParallelismConfig(
                    context_parallel_degree=world_size,
                    context_parallel_load_balancer=None,
                )
            )
            .layers[0]
            .attention
        )
        # A long document exercises composition across two earlier ranks;
        # resets exercise neutral summaries; length 6 needs a multi-rank halo.
        for lengths, padding in (
            ([24], ()),
            ([4, 16, 4], ()),
            ([8, 8, 8], ()),
            ([6], ()),
            # Per-document padding continues positions; row-tail padding is
            # one separate segment, as emitted by document-aware packing.
            ([4, 12, 4, 4], (3, 15, 19, 20, 21, 22, 23)),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(42)
                with torch.device("meta"):
                    reference = reference_config.build().to(dtype)
                    parallel = parallel_config.build().to(dtype)
                for module in (reference, parallel):
                    for _, parameter in module.named_preserve_dtype_params():
                        parameter.data = parameter.data.float()
                    module.to_empty(device="cpu")
                reference.init_states(buffer_device=torch.device("cpu"))
                parallel.load_state_dict(reference.state_dict())
                positions = torch.cat([torch.arange(length) for length in lengths])
                padding_mask = torch.zeros(len(positions), dtype=torch.bool)
                padding_mask[list(padding)] = True
                metadata = reference_config.inner_attention.build_attention_metadata(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=8,
                    max_context_length=24,
                )
                inputs = torch.randn(sum(lengths), 64, dtype=dtype, requires_grad=True)
                expected = reference(inputs, metadata)
                grad = torch.randn_like(expected)
                grad[padding_mask] = 0
                expected.backward(grad)
                shard = slice(
                    rank * len(inputs) // world_size,
                    (rank + 1) * len(inputs) // world_size,
                )
                local_inputs = inputs.detach()[shard].clone().requires_grad_()
                with set_current_spmd_mesh(mesh):
                    local_metadata = parallel.inner_attention.prepare_cp_metadata(
                        metadata, permutation=None
                    )
                    actual = parallel(local_inputs, local_metadata)
                    actual.backward(grad[shard])
                tolerance = (
                    dict(atol=2e-6, rtol=2e-4)
                    if dtype == torch.float32
                    else dict(atol=2e-4, rtol=0.02)
                )
                torch.testing.assert_close(actual, expected[shard], **tolerance)
                assert inputs.grad is not None
                torch.testing.assert_close(
                    local_inputs.grad, inputs.grad[shard], **tolerance
                )
                if padding:
                    assert local_inputs.grad is not None
                    assert (
                        torch.count_nonzero(local_inputs.grad[padding_mask[shard]]) == 0
                    )
                for (name, parameter), (_, ref) in zip(
                    parallel.named_parameters(),
                    reference.named_parameters(),
                    strict=True,
                ):
                    assert parameter.grad is not None, name
                    gradient = parameter.grad.float()
                    dist.all_reduce(gradient)
                    if dtype == torch.bfloat16:
                        # Weight GEMMs round their rank-local BF16 gradients
                        # before the FP32 reduction, unlike one global GEMM.
                        error = (gradient - ref.grad.float()).norm()
                        assert error <= 0.01 * ref.grad.float().norm() + 1e-7, (
                            lengths,
                            name,
                            error.item(),
                            ref.grad.float().norm().item(),
                        )
                    else:
                        torch.testing.assert_close(
                            gradient,
                            ref.grad,
                            **tolerance,
                            msg=lambda msg: f"{lengths}, {dtype}, {name}: {msg}",
                        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_kcp_packed_forward_and_all_gradients():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_check_kcp, args=(3, f"{directory}/store"), nprocs=3, join=True)
