# DeepSeek V4.1

This folder contains the TorchTitan implementation of the DeepSeek V4.1-Flash text
backbone. The model entry point is `torchtitan.models.deepseek_v4_1.model_registry`,
and the training configs are exposed from
`torchtitan.models.deepseek_v4_1.config_registry`. The currently registered configs are:

- `deepseek_v4_1_debugmodel`
- `deepseek_v4_1_debugmodel_candidates` (hierarchical sparse indexer enabled)

## Components

- `model.py`: decoder model and transformer block definitions, plus
  `DeepSeekV41Metadata`, the per-forward varlen metadata.
- `attention.py`: CSA2 attention (sliding window plus selected compressed entries) on
  Attention Gym's `selected_attention`, the grouped output projection, and the site
  where the indexer's distillation loss is applied.
- `compressor.py`: the compressed main KV, produced by `kv_source_layers`.
- `indexer.py`: the lightning indexer (a projection half plus a swappable
  score-and-select half), the hierarchical candidate pool, and `IndexerKLLoss`, which
  owns the distillation teacher.
- `mhc.py`: Single-Pass mHC (manifold-constrained Hyper-Connections) branch
  mixing (`HcPre` / `HcPost`).
- `config_registry.py`: `Trainer.Config` entry points.

## Packed documents: `doc_ids` is the only metadata

The sparse core is Attention Gym's `selected_attention`, called in one shot with
`q`, the sliding-window KV, the compressed KV, the indexer's top-k, the sink and
`doc_ids`. The metadata is built once per forward in `get_attention_masks` (carried
through `preprocess_inputs` as `attention_masks`) and wrapped in
`DeepSeekV41Metadata`. Its single field is `doc_ids`: a non-decreasing document index
per token (`cumsum(positions == 0) - 1`), and it is the *only* varlen metadata:

- `selected_attention` applies `doc_ids[q] == doc_ids[k]` to the sliding-window branch.
- The indexer applies the same equality one axis over: entry `j` covers tokens
  `[j*ratio, (j+1)*ratio)`, so it belongs to `doc_ids[j*ratio]` and is selectable by
  query `t` iff `doc_ids[t] == doc_ids[j*ratio]` and `j < (t+1)//ratio`. The operator
  leaves that branch to the caller.
- The distillation loss needs no metadata: the LSE it consumes is already
  document-masked, and its support is the already-filtered top-k.

Both rules rely on every packed segment holding a multiple of `compress_ratio` tokens.
The training config sets `ConcatThenSplitPackingConfig.pad_segments_to_multiple` to the
largest ratio the model pools with, which pads each segment before the packer runs; the
packer's own cuts are multiples too, so the alignment survives concatenation and
splitting. Without it a pooling group could straddle a document boundary.


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

The teacher is the attention's own probability mass on the selected entries. The
operator returns the per-head log-sum-exp of the full softmax (sliding window, selected
compressed entries and sink alike), so each head contributes
`exp(logit - lse)`, the heads are summed, and the result is L1-normalized over the
selected support. Weighting each head by its own denominator is what keeps a head that
puts almost all of its mass on the window or the sink from outvoting a head that actually
uses the compressed entries; summing raw scores across heads would give every head unit
compressed mass. The recomputation runs in fp32 under `no_grad` and shifts each row when
the window and sink push the compressed mass below the fp32 normal range, so the relative
head weights survive. `IndexerKLLoss` owns this computation: its `forward` takes the
queries, compressed keys, selected entries, LSE and student logits and builds the teacher
itself, the way the MoE balance loss owns its formula.

The scores are computed twice per index source: once without a graph over all candidates
to take the top-k, and once with a graph over the selected entries only, which is what
keeps the full `[T, heads, entries]` score tensor out of the autograd graph. Both are
single-pass; there is no query chunking. The second pass only runs when a distillation
loss consumes it, so eval and LM-only runs skip it. `Indexer.forward` is split along the
same seam: `_project_qkw` runs the matmuls that produce the index query, keys and
per-head weights, and `_select_topk` runs the score, masking and top-k, so a
kernel-backed `lightning_indexer_v2` can override the second half alone.

## Hierarchical sparse indexer

`use_candidates=False` by default, matching the report's statement that the candidate
restriction is introduced in post-training. When enabled, the layer named by
`candidate_source_layer` additionally builds a shared pool of blocks scored by their best
entry, and later index sources restrict their selection to it. The pool boundary is
derived by a top-k over blocks, so it receives no gradient from anywhere; it is a
training/inference consistency and per-query cost device, not a learned component.
The switch is reached through `model_registry(..., use_candidates=True)` (the model config
is not a command-line surface), which is what the `deepseek_v4_1_debugmodel_candidates`
config does.

## Smoke test

Requires 8 GPUs for the full trainer path. The model itself also runs on CPU:

```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 MODULE=deepseek_v4_1 CONFIG=deepseek_v4_1_debugmodel ./run_train.sh \
  --training.steps 1 \
  --metrics.log_freq 1
```

## Status

Implemented: CSA2 attention (sliding window, compressed main KV, indexer top-k, attention
sink) on Attention Gym's `selected_attention` with `doc_ids` document isolation,
cross-layer KV / index / top-k / student reuse with multi-layer distillation, packing
alignment, Single-Pass mHC (manifold-constrained Hyper-Connections), MoE with the common
sqrtsoftplus noaux_tc router and the SwiGLU clamp on both the routed and the shared
experts, and the hierarchical candidate pool behind a switch.

Attention Gym is pinned to the reviewed commit; its `selected_attention` provides
`doc_ids`, `scale` and the `return_aux` LSE this model needs. Only the `eager` backend is
used today, so the CPU tests and the NPU ports run the same reference path; the fused
triton/cute kernels are a later switch.

Not implemented yet:

- Engram conditional memory, the vision encoder, and the DSpark draft head.
- `state_dict_adapter.py` (HuggingFace conversion) and `sharding.py` (TP / EP
  placements); only data-parallel sharding works today. `parallelize_fn` is currently
  DeepSeek V3's, which applies the generic pieces and no-ops for tensor and expert
  parallelism.
- Context and pipeline parallelism are rejected in `update_from_config`. Tensor
  parallelism is rejected too: the shared cross-layer tensors and the single-head KV are
  not sharded. CP will take the NPU-kernel path with its own varlen inputs rather than
  `doc_ids`; `selected_attention` is the non-CP core.
- Indexer quantization (FP8 / FP4 QAT) and the fused-kernel override path.

## Tests

`tests/unit_tests/cpu/test_deepseek_v4_1_indexer.py` pins the distillation teacher
against an explicit float64 oracle (per-head weighting, uniform-shift invariance, tiny
mass below the fp32 normal range), the packed-document isolation of `doc_ids`
(perturbing one segment leaves the other bit-identical, and the indexer never selects
across documents), and that the student-logit recomputation is skipped when nothing
consumes it. The segment-alignment helper is covered by
`tests/unit_tests/cpu/test_text_dataset_packing.py`.
