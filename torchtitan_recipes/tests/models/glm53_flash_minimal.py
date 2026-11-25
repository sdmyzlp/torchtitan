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
from torchtitan.config import TrainingConfig
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
