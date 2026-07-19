# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configurable import Configurable
from torchtitan.models.common.attention import AttentionMasksType


_MASK_Q_SEQ_DIM = 2


class MaskHandler(Configurable):
    """Base class for attention mask processing.

    Two extension points:
    - ``post_process``: called every step after ``get_attention_masks``.
    - ``shard``: called only when CP is enabled, inside ``cp_shard``.
      Use for CP-specific mask sharding.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def post_process(
        self,
        masks: AttentionMasksType,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        return masks

    def shard(
        self,
        masks: AttentionMasksType,
        cp_mesh: DeviceMesh,
        load_balancer: Any | None,
    ) -> AttentionMasksType:
        return masks


class BlockMaskHandler(MaskHandler):
    """Default handler for BlockMask and dict[str, BlockMask].

    ``post_process`` is identity. ``shard`` mirrors the existing
    ``cp_shard`` BlockMask/dict logic.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MaskHandler.Config):
        pass

    def shard(
        self,
        masks: AttentionMasksType,
        cp_mesh: DeviceMesh,
        load_balancer: Any | None,
    ) -> AttentionMasksType:
        if isinstance(masks, dict):
            assert all(isinstance(v, BlockMask) for v in masks.values()), (
                "dict values must be BlockMask"
            )
            masks_list = list(masks.values())
        elif isinstance(masks, BlockMask):
            masks_list = [masks]
        else:
            raise TypeError(
                f"BlockMaskHandler expects BlockMask or dict[str, BlockMask], "
                f"got {type(masks)}"
            )

        masks_list = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks_list,
            seq_dims=(_MASK_Q_SEQ_DIM,) * len(masks_list),
            load_balancer=load_balancer,
        )

        if isinstance(masks, dict):
            return cast(
                AttentionMasksType,
                dict(zip(masks.keys(), masks_list)),
            )
        return cast(BlockMask, masks_list[0])
