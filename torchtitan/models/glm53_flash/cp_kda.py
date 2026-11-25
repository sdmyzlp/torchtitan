# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA context parallelism using affine summaries and convolution halos.

Shape suffixes: T = tokens, C = channels, H = heads, K/V = head widths,
N = sequences, R = ranks, W = convolution width, F = V + K.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_single
from torch.export import register_dataclass

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.models.common.attention.cp_linear_attention import (
    prepare_cp_linear_attention_metadata,
)
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.attn_gym import (
    chunk_kda,
    context_parallel_conv_history,
    ContextParallelRouting,
)

from .model import KDAAttentionMetadata, ReferenceInnerKDA, ReferenceKDAKernel


@dataclass(frozen=True, slots=True)
class KCPContext(ContextParallelRouting):
    group: dist.ProcessGroup
    cu_seqlens_cpu: torch.Tensor
    first_continues: bool
    last_continues: bool
    sequence_ids: tuple[int, ...]
    summary_bounds: tuple[int, int]
    param_grad_reduce: str = "external"


@dataclass(frozen=True, slots=True)
class ContextParallelKDAAttentionMetadata(KDAAttentionMetadata):
    context: KCPContext

    _ROUTING_SPMD_TYPE = spmd.SpmdType(
        {MeshAxisName.DP: spmd.V, MeshAxisName.CP: spmd.V, MeshAxisName.TP: spmd.R}
    )

    def annotate_spmd_types(self):
        LinearAttentionMetadata.annotate_spmd_types(self)
        for value in vars(self.context).values():
            if isinstance(value, torch.Tensor):
                spmd.assert_type(value, self._ROUTING_SPMD_TYPE)


register_dataclass(KCPContext)
register_dataclass(ContextParallelKDAAttentionMetadata)


def reference_kcp(
    q_1THK,
    k_1THK,
    v_1THV,
    gate_1THK,
    beta_1TH,
    attention_metadata: ContextParallelKDAAttentionMetadata,
):
    """Exchange S_out = S_in @ A + B summaries with autograd collectives.

    Attention Gym uses V-first states and natural-log gates. Each rank sends
    only its final continuing document's [B; A], independent of token count.
    """
    context = attention_metadata.context
    routing = context
    routing.validate(q_1THK, context.group)
    with torch.autocast(device_type=q_1THK.device.type, enabled=False):
        num_heads, key_dim = k_1THK.shape[-2:]
        value_dim = v_1THV.shape[-1]
        identity_HKK = torch.eye(
            key_dim, device=q_1THK.device, dtype=torch.float32
        ).expand(num_heads, -1, -1)
        transition_HKK = identity_HKK
        # Even a terminal rank must participate in the collective's backward.
        bias_HVK = v_1THV[0, 0, :, :, None].expand(-1, -1, key_dim) * 0
        start, end = context.summary_bounds
        for token in range(start, end):
            key_HK = k_1THK[0, token]
            decay_HK = gate_1THK[0, token].exp()
            beta_H1 = beta_1TH[0, token, :, None]
            transition_HKK = transition_HKK * decay_HK[:, None]
            transition_HKK = transition_HKK - (
                (transition_HKK @ key_HK[:, :, None])
                * beta_H1[:, :, None]
                * key_HK[:, None]
            )
            bias_HVK = bias_HVK * decay_HK[:, None]
            delta_HV = (
                v_1THV[0, token] - (bias_HVK * key_HK[:, None]).sum(-1)
            ) * beta_H1
            bias_HVK = bias_HVK + delta_HV[:, :, None] * key_HK[:, None]

        summary_1HFK = torch.cat((bias_HVK, transition_HKK), dim=-2)[None]
        gathered_RHFK = all_gather_single(summary_1HFK, 0, context.group)
        neutral_1HFK = torch.cat((torch.zeros_like(bias_HVK), identity_HKK), dim=-2)[
            None
        ]
        predecessors_RHFK = torch.cat((gathered_RHFK, neutral_1HFK))[
            routing.predecessors[0]
        ]
        state_HVK = torch.zeros_like(bias_HVK)
        for summary_HFK in predecessors_RHFK.unbind():
            state_HVK = (
                state_HVK @ summary_HFK[:, value_dim:] + summary_HFK[:, :value_dim]
            )
        # Selecting a zero state still retains the all-gather autograd edge.
        initial_NHVK = torch.stack((state_HVK, state_HVK * 0))[
            routing.entry_sources[: len(context.sequence_ids)]
        ]
        return chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            gate_1THK,
            beta_1TH,
            cu_seqlens=context.cu_seqlens,
            initial_state=initial_NHVK,
            impl="reference",
        )[0]


class ContextParallelKDAKernel(ReferenceKDAKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(ReferenceKDAKernel.Config):
        pass

    def forward(
        self,
        q_1THK,
        k_1THK,
        v_1THV,
        raw_gate_1THK,
        raw_beta_1TH,
        A_log_H,
        dt_bias_HK,
        *,
        attention_metadata: KDAAttentionMetadata,
    ):
        assert isinstance(attention_metadata, ContextParallelKDAAttentionMetadata)
        q_1THK, k_1THK, gate_1THK, beta_1TH = self.prepare_inputs(
            q_1THK, k_1THK, raw_gate_1THK, raw_beta_1TH, A_log_H, dt_bias_HK
        )
        return reference_kcp(
            q_1THK,
            k_1THK,
            v_1THV.float(),
            gate_1THK,
            beta_1TH,
            attention_metadata,
        ).to(v_1THV.dtype)


class ContextParallelKDA(
    CPInnerAttention[KDAAttentionMetadata, ContextParallelKDAAttentionMetadata],
    ReferenceInnerKDA,
):
    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, ReferenceInnerKDA.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(attention_metadata: KDAAttentionMetadata, *, permutation):
        if permutation is not None:
            raise ValueError("GLM KCP requires contiguous token shards.")
        routing = prepare_cp_linear_attention_metadata(
            attention_metadata, permutation=None, family="GLM KDA"
        ).cp_routing
        group = spmd_mesh_group(MeshAxisName.CP)
        assert group is not None
        bounds = attention_metadata.offsets
        start = routing.cp_rank * routing.tokens
        end = start + routing.tokens
        sequence_ids = tuple(
            index
            for index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:]))
            if left < end and right > start
        )
        cu_seqlens = routing.cu_seqlens.unique_consecutive()
        cu_seqlens_cpu = cu_seqlens.cpu()
        context = KCPContext(
            **(vars(routing) | {"cu_seqlens": cu_seqlens}),
            group=group,
            cu_seqlens_cpu=cu_seqlens_cpu,
            first_continues=start > bounds[sequence_ids[0]],
            last_continues=end < bounds[sequence_ids[-1] + 1],
            sequence_ids=sequence_ids,
            summary_bounds=tuple(routing.forward_bounds[0].tolist()),
        )
        return ContextParallelKDAAttentionMetadata(
            varlen=attention_metadata.varlen,
            num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            offsets=tuple(cu_seqlens_cpu.tolist()),
            context=context,
        )

    def short_convolution(
        self, qkv_1TC, conv_weight_C1W, *, attention_metadata: KDAAttentionMetadata
    ):
        assert isinstance(attention_metadata, ContextParallelKDAAttentionMetadata)
        context = attention_metadata.context
        if context.tail_sources.shape[1] != conv_weight_C1W.shape[-1] - 1:
            raise ValueError("KCP convolution halo must match the kernel width.")
        history_NWC = context_parallel_conv_history(qkv_1TC, context, context.group)
        bounds = attention_metadata.offsets
        out_1TC = torch.zeros_like(qkv_1TC)
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            span_1TC = qkv_1TC[:, start:end]
            if history_NWC is not None:
                span_1TC = torch.cat((history_NWC[index : index + 1], span_1TC), 1)
            out_1TC[:, start:end] = F.silu(
                F.conv1d(
                    span_1TC.transpose(1, 2).to(conv_weight_C1W.dtype),
                    conv_weight_C1W,
                    groups=conv_weight_C1W.shape[0],
                ).transpose(1, 2)
            ).to(qkv_1TC.dtype)
        return out_1TC
