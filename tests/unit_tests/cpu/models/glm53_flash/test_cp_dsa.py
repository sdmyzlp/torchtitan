# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CP collectives must preserve attention and indexer/compressor gradients."""

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.glm53_flash import build_model_config
from torchtitan.models.glm53_flash.attention import SparseMetadata
from torchtitan.models.glm53_flash.cp_dsa import AllGatherKVSparseAttention


class TestDSAContextParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_packed_padding_and_auxiliary_gradients(self):
        torch.set_num_threads(1)
        mesh = init_device_mesh(
            "cpu", (1, self.world_size, 1), mesh_dim_names=("dp", "cp", "tp")
        )
        attention_config = build_model_config().layers[3].attention
        cp_config = (
            build_model_config(
                parallelism=ParallelismConfig(context_parallel_degree=self.world_size)
            )
            .layers[3]
            .attention
        )
        lengths = (4, 12, 20, 4)
        real_lengths = (3, 11, 17, 0)
        positions = torch.cat([torch.arange(n) for n in lengths])
        padding_mask = torch.cat(
            [torch.arange(n) >= real for n, real in zip(lengths, real_lengths)]
        )
        denominator = (~padding_mask).sum()
        global_metadata = attention_config.inner_attention.build_attention_metadata(
            positions, padding_mask=padding_mask
        )
        assert isinstance(global_metadata, SparseMetadata)
        shard_len = positions.numel() // self.world_size
        start = self.rank * shard_len
        stop = start + shard_len
        with set_current_spmd_mesh(mesh):
            local_metadata = AllGatherKVSparseAttention.prepare_cp_metadata(
                global_metadata, permutation=None
            )
            local_metadata.annotate_spmd_types()
            local_metadata.annotate_spmd_types()
            spmd.assert_type(local_metadata.cu_seq_q, {"cp": spmd.V})
            spmd.assert_type(local_metadata.cu_seq_k, {"cp": spmd.R})
            for lm_scale in (1.0, 0.0):
                torch.manual_seed(42)
                reference = attention_config.build()
                reference.init_states(buffer_device=torch.device("cpu"))
                actual = cp_config.build()
                actual.load_state_dict(reference.state_dict())
                x = torch.randn(positions.numel(), 64, requires_grad=True)
                local_x = x.detach()[start:stop].clone().requires_grad_()
                grad = torch.randn_like(x) * lm_scale
                grad[padding_mask] = 0
                expected = reference(
                    x,
                    attention_metadata=global_metadata,
                    padding_mask=padding_mask,
                    aux_loss_denominator=denominator,
                )
                output = actual(
                    local_x,
                    attention_metadata=local_metadata,
                    padding_mask=padding_mask[start:stop],
                    aux_loss_denominator=denominator,
                )
                torch.testing.assert_close(
                    output, expected[start:stop], atol=3e-7, rtol=2e-5
                )
                output.backward(grad[start:stop])
                expected.backward(grad)
                assert x.grad is not None
                torch.testing.assert_close(
                    local_x.grad, x.grad[start:stop], atol=2e-7, rtol=2e-5
                )
                for (name, param), (_, ref) in zip(
                    actual.named_parameters(), reference.named_parameters()
                ):
                    assert param.grad is not None, name
                    assert ref.grad is not None, name
                    dist.all_reduce(param.grad)
                    torch.testing.assert_close(
                        param.grad,
                        ref.grad,
                        atol=2e-6,
                        rtol=3e-5,
                        msg=name,
                    )
                    if name.startswith("indexer."):
                        assert param.grad.norm() > 0, name
                    elif lm_scale == 0:
                        torch.testing.assert_close(param.grad, torch.zeros_like(param))
                metric = actual.inner_attention.aux_loss.instance_acc.clone()
                dist.all_reduce(metric)
                torch.testing.assert_close(
                    metric, reference.inner_attention.aux_loss.instance_acc
                )

    @with_comms
    def test_rejects_split_pools_and_token_permutations(self):
        mesh = init_device_mesh(
            "cpu", (1, self.world_size, 1), mesh_dim_names=("dp", "cp", "tp")
        )
        config = build_model_config().layers[3].attention.inner_attention
        with set_current_spmd_mesh(mesh):
            for lengths in ((5, 11), (12,)):
                positions = torch.cat([torch.arange(n) for n in lengths])
                metadata = config.build_attention_metadata(positions)
                assert isinstance(metadata, SparseMetadata)
                with pytest.raises(ValueError, match="whole pools|boundaries align"):
                    AllGatherKVSparseAttention.prepare_cp_metadata(
                        metadata, permutation=None
                    )
            metadata = config.build_attention_metadata(torch.arange(16))
            assert isinstance(metadata, SparseMetadata)
            with pytest.raises(ValueError, match="without a permutation"):
                AllGatherKVSparseAttention.prepare_cp_metadata(
                    metadata, permutation=torch.arange(16).reshape(1, -1)
                )
