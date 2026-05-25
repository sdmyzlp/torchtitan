# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch_npu
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module

from .moe import DeepSeekV4MoE


class NPURMSNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.normalized_shape))
        self.eps = config.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.weight.to(dtype) if self.weight.dtype != dtype else self.weight
        resolved_eps = self.eps if self.eps is not None else torch.finfo(dtype).eps
        return torch_npu.npu_rms_norm(x, weight, resolved_eps)[0]


class NPURMSNormConverter(ModelConfigConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config) -> None:
        for fqn, cfg, parent, attr in model_config.traverse(RMSNorm.Config):
            new_config = NPURMSNorm.Config(
                **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)


class NPUDeepSeekV4MoE(DeepSeekV4MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MoE.Config):
        pass

    def forward(self, x, input_ids):
        from torch.distributed.tensor import Partial

        if isinstance(x, DTensor):
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        input_ids_flat = input_ids.flatten() if input_ids is not None else None

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat,
            input_ids_flat,
            self.expert_bias,
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        indices = selected_experts_indices.view(-1, self.router.top_k)
        routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x_flat, indices)

        routed_output = self.experts._experts_forward(
            routed_input, num_tokens_per_expert
        )

        if self.shared_experts is not None:
            out = self.shared_experts(x_flat)
        else:
            out = torch.zeros_like(x_flat)

        unpermuted = torch_npu.npu_moe_token_unpermute(
            routed_output,
            sorted_indices,
            top_scores.to(x_flat.dtype),
        )
        return (out + unpermuted).reshape(bs, slen, dim)


class NPUPermuteConverter(ModelConfigConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config) -> None:
        for fqn, cfg, parent, attr in model_config.traverse(DeepSeekV4MoE.Config):
            new_config = NPUDeepSeekV4MoE.Config(
                **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)


class GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list
        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


_npu_grouped_experts_cache: dict[type, type] = {}


def _get_npu_grouped_experts_cls(parent_cls: type) -> type:
    if parent_cls in _npu_grouped_experts_cache:
        return _npu_grouped_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class NPUGroupedExperts(parent_cls):
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            self.w13 = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim * 2, config.dim)
            )
            self.w1 = None
            self.w3 = None
            if self._param_init is not None and "w1" in self._param_init:
                self._param_init["w13"] = self._param_init["w1"]

        def _experts_forward(
            self,
            x: torch.Tensor,
            num_tokens_per_expert: torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(self.w13, DTensor):
                w13 = self.w13.to_local()
                w2 = self.w2.to_local()
            else:
                w13 = self.w13
                w2 = self.w2

            offsets = num_tokens_per_expert.to(torch.int64)

            h = GMMFunction.apply(
                x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets
            )
            h = torch_npu.npu_swiglu(h, dim=-1)
            out = GMMFunction.apply(
                h, w2.bfloat16().transpose(-2, -1), offsets
            ).type_as(x)
            return out

    NPUGroupedExperts.__name__ = f"NPU{parent_cls.__name__}"
    NPUGroupedExperts.__qualname__ = f"NPU{parent_cls.__name__}"
    _npu_grouped_experts_cache[parent_cls] = NPUGroupedExperts
    return NPUGroupedExperts


class NPUGMMConverter(ModelConfigConverter):
    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config) -> None:
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            base_module_cls = type(config)._owner
            npu_cls = _get_npu_grouped_experts_cls(base_module_cls)
            config_cls = npu_cls.Config
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
            )
            if new_config.param_init is not None:
                pi = dict(new_config.param_init)
                if "w1" in pi:
                    pi["w13"] = pi.pop("w1")
                pi.pop("w3", None)
                new_config.param_init = pi
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
