# DeepSeek V4.1

This folder contains the TorchTitan implementation of the DeepSeek V4.1-Flash text
backbone. The model entry point is `torchtitan.models.deepseek_v4_1.model_registry`,
and the training configs are exposed from
`torchtitan.models.deepseek_v4_1.config_registry`. The currently registered configs are:

- `deepseek_v4_1_debugmodel`
- `deepseek_v4_1_debugmodel_candidates` (hierarchical sparse indexer enabled)

## Components

- `model.py`: decoder model and transformer block definitions.
- `attention.py`: CSA2 attention (sliding window plus selected compressed entries),
  the grouped output projection, and the indexer distillation loss site.
- `compressor.py`: the compressed main KV, produced by `kv_source_layers`.
- `indexer.py`: the lightning indexer, the hierarchical candidate pool, and
  `IndexerKLLoss`.
- `mhc.py`: Single-Pass mHC (manifold-constrained Hyper-Connections) branch
  mixing (`HcPre` / `HcPost`).
- `config_registry.py`: `Trainer.Config` entry points.

## Stack layout

`compress_ratios` places the layers into groups that share one compressed KV and one
top-k selection. The debug model exercises every variant at a small scale: layers 0-1
are sliding-window only, layers 2-4 compress 2:1 with layer 2 as their source, and
layers 5-7 use ratio 1 with layer 5 as their source. Layer 7 re-indexes: it borrows the
shared index keys but selects its own top-k.

## Cross-layer state is threaded, not stored

Three roles exist per group: a **Full** layer produces the compressed KV, the index keys
and the top-k; a **Reindex** layer borrows the keys but produces its own top-k; a
**Reuse** layer borrows everything. Those tensors are passed between layers as ordinary
inputs and returned updated, which keeps the reuse structure visible in the forward
signature:

```
(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)
```

`IndexerKLLoss` is the only training signal the indexer has: the top-k selection is
discrete and the teacher is detached, so without it none of the indexer's parameters
would update. It is attached to every layer with a compressed KV, not only to the layers
that own an indexer, because a reuse layer's own attention distribution is part of what
the shared selection must serve. Each layer's coefficient is divided by the size of the
group it belongs to, so the indexer's effective learning rate does not depend on the
sharing pattern.

The teacher is the head-averaged attention probability on the selected entries, computed
with the full softmax denominator (sliding window and attention sink included) and
L1-normalized over the selected support. The scores are computed twice per index source:
once without a graph over all candidates to take the top-k, and once with a graph over
the selected entries only, which is what keeps the full `[T, heads, entries]` score
tensor out of the autograd graph.

## Hierarchical sparse indexer

`use_candidates=False` by default, matching the report's statement that the candidate
restriction is introduced in post-training. When enabled, the layer named by
`candidate_source_layer` additionally builds a shared pool of blocks scored by their best
entry, and later index sources restrict their selection to it. The pool boundary is
derived by a top-k over blocks, so it receives no gradient from anywhere; it is a
training/inference consistency and per-query cost device, not a learned component.
Pass `--model.use_candidates` through `model_registry`, or use the
`deepseek_v4_1_debugmodel_candidates` config.

## Smoke test

Requires 8 GPUs for the full trainer path. The model itself also runs on CPU:

```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 MODULE=deepseek_v4_1 CONFIG=deepseek_v4_1_debugmodel ./run_train.sh \
  --training.steps 1 \
  --metrics.log_freq 1
```

## Status

Implemented: CSA2 attention (sliding window, compressed main KV, indexer top-k, attention
sink), cross-layer KV / index / top-k / student reuse with multi-layer distillation,
Single-Pass mHC (manifold-constrained Hyper-Connections), MoE with the common
sqrtsoftplus noaux_tc router, and the hierarchical
candidate pool behind a switch.

Not implemented yet:

- Engram conditional memory, the vision encoder, and the DSpark draft head.
- `state_dict_adapter.py` (HuggingFace conversion) and `sharding.py` (TP / EP
  placements); only data-parallel sharding works today. `parallelize_fn` is currently
  DeepSeek V3's, which applies the generic pieces and no-ops for tensor and expert
  parallelism.
- Context and pipeline parallelism are rejected in `update_from_config`. Tensor
  parallelism is rejected too: the shared cross-layer tensors and the single-head KV are
  not sharded.
- Indexer quantization (FP8 / FP4 QAT) and the fused-kernel override path.
- The indexer loss is always evaluated in the forward; a deployment build should skip
  the selected-entry score recomputation when it trains nothing.
