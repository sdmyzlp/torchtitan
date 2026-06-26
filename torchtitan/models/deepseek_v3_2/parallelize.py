# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig

from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3


def parallelize_deepseekv32(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """DeepSeek-V3.2 parallelize entry point.

    Delegates to the V3 parallelize logic but enforces the v32 constraint:
    Context Parallel requires ``spmd_backend="full_dtensor"`` (the legacy
    ``apply_cp_to_forward`` path cannot gather the indexer ``idx_k`` tensor).
    """
    if parallel_dims.cp_enabled and parallelism.spmd_backend != "full_dtensor":
        raise ValueError(
            "DeepSeek-V3.2 with Context Parallel requires "
            'parallelism.spmd_backend="full_dtensor". '
            f"Got spmd_backend={parallelism.spmd_backend!r}."
        )
    return parallelize_deepseekv3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
