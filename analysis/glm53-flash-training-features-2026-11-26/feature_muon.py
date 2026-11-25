# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.optim import AdamW, DistMuon, OptimizersContainer
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)


def configure_muon(config):
    with torch.device("meta"):
        model = config.model.build()
    layouts = {}
    for name, parameter in model.named_parameters():
        if not name.endswith(".weight") or parameter.ndim < 2:
            continue
        if "_conv." in name or name.startswith(("tok_embeddings.", "lm_head.")):
            continue
        placement = Owned()
        if ".routed_experts." in name:
            placement = Shard(0)
        elif any(
            name.endswith(suffix)
            for suffix in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "forget_b.weight",
                "output_gate.b.weight",
                "indexer.wq_b.weight",
            )
        ):
            placement = BlockShard(dim=0, block_sizes=(128,))
        elif name.endswith("attention.wq_b.weight"):
            placement = BlockShard(dim=0, block_sizes=(256,))
        elif name.endswith("attention.wkv_b.weight"):
            placement = BlockShard(dim=0, block_sizes=(256, 256))
        layouts[name] = ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": placement}, compute_dtype=torch.bfloat16
        )
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern="^(?:" + "|".join(re.escape(name) for name in layouts) + ")$",
                compute_sharding_by_fqn=layouts,
                bucket_configs=tuple(
                    BucketConfig(name=f"layer{i}", patterns=(f"layers.{i}.*",))
                    for i in range(len(config.model.layers))
                ),
                lr=8e-4,
                adjust_lr_fn="match_rms_adamw",
                ns_device="accelerator",
            ),
            AdamW.Config(pattern=r".*", lr=8e-4, fused=False, foreach=True),
        ]
    )
