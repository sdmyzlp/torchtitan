# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
from .converters import NPUGMMConverter, NPUPermuteConverter, NPURMSNormConverter


def deepseek_v4_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=256,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            enable_sequence_parallel=False,
            disable_loss_parallel=False,
            expert_parallel_degree=1,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        compile=CompileConfig(enable=False),
    )


def deepseek_v4_debugmodel_npu() -> Trainer.Config:
    config = deepseek_v4_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NPURMSNormConverter.Config(),
            NPUPermuteConverter.Config(),
            NPUGMMConverter.Config(),
        ],
    )
    return config
