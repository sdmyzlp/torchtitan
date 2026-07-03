# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module, ModuleDict


IGNORE_INDEX = -100


def roll_tensor(
    tensor: Tensor | None,
    shifts: int = -1,
    dims: int = -1,
    cp_group: ProcessGroup | None = None,
) -> Tensor | None:
    """Roll tensor left along seq dim with Context Parallelism support.

    When CP is enabled, adjacent CP ranks exchange boundary elements so
    the rolled boundaries stay contiguous across CP shards.  The element
    at the wrap-around position (``shifts`` in the rolled result) is
    zeroed to avoid leaking data across sequence boundaries.
    """
    if tensor is None:
        return None

    if cp_group is None or cp_group.size() == 1:
        rolled = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled.select(dims, shifts).fill_(0)
        return rolled

    cp_size = cp_group.size()
    local_rank = dist.get_rank(group=cp_group)
    global_ranks = dist.get_process_group_ranks(group=cp_group)

    tensor_list = tensor.chunk(2, dim=dims)
    rolled_list = [torch.roll(t, shifts=shifts, dims=dims) for t in tensor_list]

    send_list = [t.select(dims, shifts).contiguous() for t in rolled_list]
    recv_list = [torch.empty_like(s) for s in send_list]

    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    ops = []
    if local_rank != 0:
        ops.append(dist.isend(tensor=send_list[0], dst=prev_rank))
        ops.append(dist.irecv(tensor=recv_list[1], src=prev_rank))
    else:
        recv_list[1].zero_()
    if local_rank != cp_size - 1:
        ops.append(dist.irecv(tensor=recv_list[0], src=next_rank))
        ops.append(dist.isend(tensor=send_list[1], dst=next_rank))
    else:
        recv_list[0] = send_list[1]
    for op in ops:
        op.wait()

    idx = [slice(None)] * rolled_list[0].dim()
    idx[dims] = shifts
    for i in range(len(rolled_list)):
        rolled_list[i][tuple(idx)] = recv_list[i]

    return torch.cat(rolled_list, dim=dims)


@dataclass(kw_only=True, slots=True)
class MTPConfig:
    """Configuration for Multi-Token Prediction (DeepSeek-V3 style).

    Each MTP depth predicts one additional future token while keeping a
    complete causal chain::

        depth 0: predict token[i+1]    (standard next-token)
        depth 1: predict token[i+2]
        ...
        depth D: predict token[i+D+1]
    """

    num_layers: int = 0
    """Number of MTP depths (D = additional future tokens to predict)."""

    loss_scaling_factor: float = 0.1
    """Overall weight for the MTP auxiliary loss.  Divided evenly across depths."""

    use_repeated_layer: bool = False
    """Share a single MTP transformer layer across all depths (reduces params)."""

    detach_heads: bool = False
    """If True, detach MTP head input embeddings from the main model graph so
    MTP loss gradients do not flow back into the main model parameters."""

    inner_block_config: Any = None
    """Config for the inner transformer block in each MTP depth.  When set,
    this is used instead of the last decoder ``config.layers[-1]``.  Models
    with heterogeneous layer types (e.g. dense + MoE) must provide this
    explicitly since no single decoder layer is universally representative."""


class MTPLayer(Module):
    """One depth of Multi-Token Prediction.

    For each position ``i``:

        1. Roll ``input_ids`` left so position ``i`` sees token ``i+1``
        2. Embed the shifted ids via the shared ``tok_embeddings``
        3. Concatenate normalized embedding with normalized hidden states:
            ``[norm(decoder_input) || norm(hidden_states)]``  (shape ``[..., 2h]``)
        4. Project down to ``h`` via ``eh_proj``
        5. Run through a transformer block
        6. Apply final layer norm
    """

    def __init__(
        self,
        *,
        dim: int,
        inner_block_config,
        cp_group: ProcessGroup | None = None,
        detach_heads: bool = False,
        layer_number: int = 1,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.cp_group = cp_group
        self.detach_heads = detach_heads

        self.enorm = RMSNorm(RMSNorm.Config(normalized_shape=dim))
        self.hnorm = RMSNorm(RMSNorm.Config(normalized_shape=dim))
        self.eh_proj = Linear(Linear.Config(in_features=dim * 2, out_features=dim, bias=False))
        self.inner = inner_block_config.build()
        self.final_norm = RMSNorm(RMSNorm.Config(normalized_shape=dim))

    def forward(
        self,
        input_ids: Tensor,
        hidden_states: Tensor,
        positions: Tensor | None,
        attention_masks,
        embedding: Callable[[Tensor, Tensor | None], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        next_ids = roll_tensor(input_ids, cp_group=self.cp_group)
        next_pos = roll_tensor(positions, cp_group=self.cp_group)
        decoder_input = embedding(next_ids, next_pos)
        if self.detach_heads:
            decoder_input = decoder_input.detach()

        dec = self.enorm(decoder_input)
        hid = self.hnorm(hidden_states)
        fused = self.eh_proj(torch.cat([dec, hid], dim=-1))
        out = self.inner(fused, attention_masks, next_pos)
        return self.final_norm(out), next_ids, next_pos


class MTPBlock(Module):
    """Container for all MTP depths.

    Sequentially applies ``mtp_num_layers`` MTP layers and returns the
    concatenated output ``[B, (D + 1) * L, dim]`` where the first ``L``
    tokens are the original main-model hidden states and the remaining
    ``D * L`` are the per-depth MTP outputs.

    The loss function splits this tensor and computes per-depth losses.
    """

    def __init__(
        self,
        *,
        mtp_config: MTPConfig,
        dim: int,
        inner_block_config,
        cp_group: ProcessGroup | None = None,
    ):
        super().__init__()
        self.mtp_config = mtp_config
        self.num_depths = mtp_config.num_layers

        if self.num_depths == 0:
            self.layers = ModuleDict()
            return

        def _build_layer(k: int) -> MTPLayer:
            return MTPLayer(
                dim=dim,
                inner_block_config=inner_block_config,
                cp_group=cp_group,
                detach_heads=mtp_config.detach_heads,
                layer_number=k + 1,
            )

        if mtp_config.use_repeated_layer:
            self.layers = ModuleDict({"0": _build_layer(0)})
        else:
            self.layers = ModuleDict(
                {str(k): _build_layer(k) for k in range(self.num_depths)}
            )

    def forward(
        self,
        input_ids: Tensor,
        hidden_states: Tensor,
        positions: Tensor | None,
        attention_masks,
        embedding: Callable[[Tensor, Tensor | None], Tensor],
    ) -> Tensor:
        outputs = [hidden_states]
        h = hidden_states
        ids, pos = input_ids, positions

        for k in range(self.num_depths):
            layer_idx = "0" if self.mtp_config.use_repeated_layer else str(k)
            h, ids, pos = self.layers[layer_idx](
                ids, h, pos, attention_masks, embedding
            )
            outputs.append(h)

        return torch.cat(outputs, dim=1)


def mtp_auxiliary_loss(
    pred: Tensor,
    labels: Tensor,
    *,
    mtp_config: MTPConfig,
    lm_head: Callable[[Tensor], Tensor],
    loss_mask: Tensor | None = None,
    cp_group: ProcessGroup | None = None,
) -> Tensor:
    """Compute MTP auxiliary losses from the concatenated MTP block output.

    ``pred`` has shape ``[B, (D + 1) * L, dim]`` where ``D = mtp_config.num_layers``.

    For each depth ``k``:

        * Extract ``pred[:, (k+1)*L : (k+2)*L]``
        * Apply ``lm_head`` to get logits
        * Roll ``labels`` left by 1 (each depth sees one more future token)
        * Compute cross-entropy, mask with rolled ``loss_mask``

    Returns a scalar loss scaled by ``loss_scaling_factor / D``.
    """
    D = mtp_config.num_layers
    if D == 0:
        return pred.new_zeros(())

    chunk_len = pred.shape[1] // (D + 1)

    mtp_labels = labels.clone()
    mtp_mask = loss_mask.clone() if loss_mask is not None else None
    total = pred.new_zeros(())

    for k in range(D):
        h = pred[:, (k + 1) * chunk_len : (k + 2) * chunk_len]
        logits = lm_head(h)

        mtp_labels = roll_tensor(mtp_labels, cp_group=cp_group)
        if mtp_mask is not None:
            mtp_mask = roll_tensor(mtp_mask, cp_group=cp_group)

        ce = F.cross_entropy(
            logits.flatten(0, 1).float(),
            mtp_labels.flatten(0, 1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )

        if mtp_mask is not None:
            m = mtp_mask.flatten(0, 1).float()
            ce = ce * m
            num_valid = m.sum().clamp(min=1)
        else:
            num_valid = torch.tensor(ce.numel(), device=ce.device, dtype=ce.dtype)

        total = total + ce.sum() / num_valid

    return total * (mtp_config.loss_scaling_factor / D)
