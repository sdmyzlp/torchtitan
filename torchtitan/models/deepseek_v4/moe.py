# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.protocols.module import Module


def _softplus_stable(x):
    return torch.log1p(torch.exp(-x.abs())) + torch.relu(x)


def _build_hash_routing_table(vocab_size, num_experts, top_k, device=None, chunk_size=8192):
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = (
            torch.rand((end - start, num_experts), device=device)
            .topk(top_k, dim=-1)
            .indices
        )
    return tid2eid


class DeepSeekV4Router(TokenChoiceTopKRouter):
    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        vocab_size: int
        n_hash_layers: int = 3
        layer_id: int = 0

    def __init__(self, config: Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_hash_layers = config.n_hash_layers
        self.layer_id = config.layer_id
        self.hash = config.layer_id < config.n_hash_layers
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(
                    self.vocab_size, self.num_experts, self.top_k
                ),
                persistent=True,
            )

    def _init_self_buffers(self, *, buffer_device=None):
        if self.hash:
            if buffer_device is not None:
                with torch.device(buffer_device):
                    self.tid2eid = _build_hash_routing_table(
                        self.vocab_size, self.num_experts, self.top_k,
                        device=buffer_device,
                    )
            else:
                self.tid2eid = _build_hash_routing_table(
                    self.vocab_size, self.num_experts, self.top_k,
                )

    def forward(self, x, input_ids, expert_bias=None):
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = _softplus_stable(scores.to(torch.float32)).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            selected_experts_indices = self.tid2eid[input_ids.flatten()]
        else:
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = self._debug_force_load_balance_routing(
                scores
            )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        return top_scores, selected_experts_indices, num_tokens_per_expert


class DeepSeekV4MoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def forward(self, x, input_ids):
        if isinstance(x, DTensor):
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        input_ids_flat = input_ids.flatten()

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat, input_ids_flat, self.expert_bias,
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        out = self.experts(
            x_flat, top_scores, selected_experts_indices,
            shared_experts=self.shared_experts,
        )
        return out.reshape(bs, slen, dim)
