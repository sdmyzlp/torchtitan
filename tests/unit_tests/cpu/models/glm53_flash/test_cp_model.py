# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The complete KDA/DSA/mHC/MoE backbone must agree with contiguous CP shards."""

from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.glm53_flash import build_model_config
from torchtitan.models.glm53_flash.cp_dsa import (
    AllGatherInnerIndexer,
    AllGatherKVSparseAttention,
)
from torchtitan.models.glm53_flash.cp_kda import ContextParallelKDA


class TestModelContextParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_packed_five_layer_model_and_all_parameter_gradients(self):
        torch.set_num_threads(1)
        parallelism = ParallelismConfig(
            context_parallel_degree=self.world_size,
            context_parallel_load_balancer=None,
        )
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=self.world_size,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context.build_mesh()
        torch.manual_seed(42)
        with torch.device("meta"):
            reference = build_model_config(seq_len=40).build()
            actual = build_model_config(seq_len=40, parallelism=parallelism).build()
        reference.to_empty(device="cpu")
        actual.to_empty(device="cpu")
        reference.init_states(buffer_device=torch.device("cpu"))
        actual.init_states(buffer_device=torch.device("cpu"))
        actual.load_state_dict(reference.state_dict())
        assert len(actual.layers) == 5
        for index, layer in enumerate(actual.layers.values()):
            if index == 3:
                assert (
                    type(layer.attention.indexer.inner_indexer) is AllGatherInnerIndexer
                )
                assert (
                    type(layer.attention.inner_attention) is AllGatherKVSparseAttention
                )
            else:
                assert type(layer.attention.inner_attention) is ContextParallelKDA

        # The packer continues positions through per-document padding. The last
        # segment is row-tail padding; rank 1 begins inside the third document.
        lengths = (4, 12, 20, 4)
        real_lengths = (3, 11, 17, 0)
        positions = torch.cat([torch.arange(n) for n in lengths])
        padding_mask = torch.cat(
            [torch.arange(n) >= real for n, real in zip(lengths, real_lengths)]
        )
        denominator = (~padding_mask).sum().reshape(1)
        tokens = torch.randint(0, reference.config.vocab_size, (positions.numel(),))
        expected = reference(
            tokens,
            positions=positions,
            padding_mask=padding_mask,
            aux_loss_denominators=denominator,
            attention_metadata=reference._get_attention_metadata(
                positions, padding_mask=padding_mask, max_num_documents=8
            ),
        )
        grad = torch.randn_like(expected) / denominator
        grad[padding_mask] = 0
        expected.backward(grad)

        with parallelism_context.activate_spmd():
            local_tokens, _, model_kwargs, _ = actual.preprocess_inputs(
                {
                    "input": tokens.clone(),
                    "labels": tokens.clone(),
                    "positions": positions.clone(),
                    "padding_mask": padding_mask.clone(),
                },
                parallelism_context=parallelism_context,
                parallelism=parallelism,
                max_num_documents=8,
                global_routing_token_counts=denominator,
            )
            start = self.rank * local_tokens.numel()
            end = start + local_tokens.numel()
            torch.testing.assert_close(
                model_kwargs["padding_mask"], padding_mask[start:end]
            )
            torch.testing.assert_close(
                model_kwargs["aux_loss_denominators"], denominator
            )
            output = actual(local_tokens, **model_kwargs)
            torch.testing.assert_close(
                output, expected[start:end], atol=3e-5, rtol=3e-4
            )
            output.backward(grad[start:end])
        for (name, parameter), (_, ref) in zip(
            actual.named_parameters(), reference.named_parameters(), strict=True
        ):
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, name
            assert ref.grad is not None, name
            dist.all_reduce(parameter.grad)
            assert torch.isfinite(parameter.grad).all(), name
            # Routed expert GEMMs accumulate BF16 operands; separate rank-local
            # weight-gradient GEMMs round before summing, unlike one global GEMM.
            error = (parameter.grad - ref.grad).norm()
            assert error <= 0.004 * ref.grad.norm() + 1e-7, (
                name,
                error.item(),
                ref.grad.norm().item(),
            )
            if ".attention.indexer." in name:
                assert parameter.grad.norm() > 0, name
        metric = actual.layers[
            "3"
        ].attention.inner_attention.aux_loss.instance_acc.clone()
        dist.all_reduce(metric)
        torch.testing.assert_close(
            metric,
            reference.layers["3"].attention.inner_attention.aux_loss.instance_acc,
            atol=1e-7,
            rtol=2e-4,
        )
