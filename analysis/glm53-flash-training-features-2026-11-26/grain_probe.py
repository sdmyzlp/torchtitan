# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import grain.python as grain
import numpy as np
import torch
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetMixConfig,
    GrainDataLoader,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from torchtitan.models.glm53_flash import build_model_config


@dataclass(frozen=True)
class Rows:
    rows: tuple

    def build(self):
        return self.rows


class Tokenizer:
    bos_id, eos_id = 1, 2


def child(length, token):
    return SingleDatasetConfig(
        source=Rows(
            tuple(
                TextSequence(
                    input_ids=np.full(length, token, dtype=np.int64),
                    labels=np.full(length, token, dtype=np.int64),
                )
                for _ in range(512)
            )
        )
    )


def mixed(children):
    return DatasetMixConfig(
        datasets=tuple(WeightedDataset(dataset=ds, weight=0.5) for ds in children)
    )


def loader(dataset, *, pad=1, dp=1):
    cfg = GrainDataLoader.Config(
        dataset=dataset,
        shuffle=False,
        repeat=True,
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=4),
        num_prefetch_microbatches=2,
        max_num_documents=(4096 // pad if pad > 1 else None),
        per_sample_pad_multiple=pad,
    )
    return cfg.build(
        dp_world_size=dp,
        dp_rank=0,
        tokenizer=Tokenizer(),
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
    )


def digest(batch):
    h = hashlib.sha256()
    for name, value in batch.as_input_dict().items():
        if isinstance(value, torch.Tensor):
            h.update(name.encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    children = (child(32, 42), child(256, 84))
    cases = {
        "mix_then_pack": ConcatThenSplitPackingConfig(dataset=mixed(children)),
        "pack_then_mix": mixed(
            tuple(ConcatThenSplitPackingConfig(dataset=ds) for ds in children)
        ),
    }
    results = {}
    for name, dataset in cases.items():
        pipe = loader(dataset)
        iterator = iter(pipe)
        counts = {"short": 0, "long": 0, "padding": 0}
        for _ in range(128):
            batch = next(iterator)
            counts["short"] += int((batch.input == 42).sum())
            counts["long"] += int((batch.input == 84).sum())
            counts["padding"] += int(batch.padding_mask.sum())
        state = copy.deepcopy(pipe.state_dict())
        expected = [digest(next(iterator)) for _ in range(8)]
        restored = loader(dataset)
        restored.load_state_dict(state)
        actual = [digest(batch) for _, batch in zip(range(8), iter(restored))]
        assert actual == expected
        counts["short_token_fraction"] = counts["short"] / (
            counts["short"] + counts["long"]
        )
        counts["resumed_8_microbatches_exact"] = True
        different_dp = loader(dataset, dp=2)
        try:
            different_dp.load_state_dict(state)
        except ValueError as error:
            counts["different_dp_error"] = str(error)
        else:
            raise AssertionError("DP change unexpectedly accepted")
        results[name] = counts
        for item in (pipe, restored, different_dp):
            item.close()
    # Exercise the actual document padding contract used by GLM DSA/KCP.
    dataset = ConcatThenSplitPackingConfig(
        dataset=mixed((child(31, 42), child(255, 84)))
    )
    attn = (
        build_model_config("debugmodel_4layer", seq_len=4096)
        .layers[3]
        .attention.inner_attention
    )
    for pad in (4, 64):
        pipe = loader(dataset, pad=pad)
        batch = next(iter(pipe))
        metadata = attn.build_attention_metadata(
            batch.positions,
            padding_mask=batch.padding_mask,
            max_num_documents=4096 // pad,
            max_context_length=4096,
        )
        assert not (metadata.cu_seq_q % pad).any()
        assert batch.input.numel() == 4096
        results[f"padding_{pad}"] = {
            "valid_tokens": int((~batch.padding_mask).sum()),
            "padding_tokens": int(batch.padding_mask.sum()),
            "document_offsets": metadata.cu_seq_q.unique_consecutive().tolist(),
            "complete_pools": metadata.pool_rows_PW.shape[0],
        }
        pipe.close()
    Path("grain.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
