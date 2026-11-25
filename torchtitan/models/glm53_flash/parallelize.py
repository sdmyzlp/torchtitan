# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the model's contiguous context-parallel boundaries."""

from torchtitan.config.transform import ContextParallelTransform, ModelConfigTransform
from torchtitan.config.transform.base import convert_config_type
from torchtitan.models.base.module import Module

from .attention import InnerIndexer, SparseAttention
from .cp_dsa import AllGatherInnerIndexer, AllGatherKVSparseAttention
from .cp_kda import ContextParallelKDA, ContextParallelKDAKernel
from .model import ReferenceInnerKDA, ReferenceKDAKernel


class GlmContextParallelTransform(ModelConfigTransform):
    def transform(self, model: Module.Config) -> Module.Config:
        ContextParallelTransform(
            inner_attention_map={
                SparseAttention: AllGatherKVSparseAttention,
                ReferenceInnerKDA: ContextParallelKDA,
            }
        ).transform(model)
        replacements: dict[type[Module], type[Module]] = {
            InnerIndexer: AllGatherInnerIndexer,
            ReferenceKDAKernel: ContextParallelKDAKernel,
        }
        for original, replacement in replacements.items():
            for _, config, parent, field_name in model.traverse(original.Config):
                assert parent is not None and isinstance(field_name, str)
                setattr(parent, field_name, convert_config_type(config, replacement))
        return model
