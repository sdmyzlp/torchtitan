# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash debug training on the bundled C4 fixture."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm53_flash import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def glm53_flash_debugmodel(
    seq_len: int = 32,
    *,
    parallelism: ParallelismConfig | None = None,
    model_flavor: str = "debugmodel",
) -> Trainer.Config:
    parallelism = parallelism or ParallelismConfig(data_parallel_shard_degree=1)
    model = build_model_config(model_flavor, seq_len=seq_len, parallelism=parallelism)
    config = Trainer.Config(
        model=model,
        parallelism=parallelism,
        hf_assets_path="./tests/assets/tokenizer",
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size)
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=seq_len // 4
            if parallelism.context_parallel_degree > 1
            else None,
            per_sample_pad_multiple=4 if parallelism.context_parallel_degree > 1 else 1,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            )
        ),
        training=TrainingConfig(
            steps=10,
            disable_accelerator_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpointer=None,
        activation_remat=None,
    )
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def glm53_flash_debugmodel_cp2(seq_len: int = 32) -> Trainer.Config:
    return glm53_flash_debugmodel(
        seq_len,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1, context_parallel_degree=2
        ),
    )


def glm53_flash_debugmodel_npu(
    seq_len: int = 4096, context_parallel_degree: int = 1
) -> Trainer.Config:
    config = glm53_flash_debugmodel(
        seq_len,
        model_flavor="debugmodel_4layer",
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            context_parallel_degree=context_parallel_degree,
        ),
    )
    config.override = OverrideConfig(
        imports=[
            "torchtitan_recipes.overrides.glm53_flash_npu.ascend_sparse_attention",
            "torchtitan_recipes.overrides.glm53_flash_pypto.flash_kda",
        ]
    )
    if context_parallel_degree > 1:
        # PyPTO KCP requires continuing fragments to end on a 64-token boundary.
        if seq_len % (64 * context_parallel_degree):
            raise ValueError("PyPTO KCP requires seq_len divisible by 64 * CP degree.")
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        config.dataloader.per_sample_pad_multiple = 64
        config.dataloader.max_num_documents = seq_len // 64
    return config


def glm53_flash_debugmodel_npu_cp2(seq_len: int = 4096) -> Trainer.Config:
    return glm53_flash_debugmodel_npu(seq_len, context_parallel_degree=2)
