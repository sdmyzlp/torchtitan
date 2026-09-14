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
- `indexer.py`: `HierarchicalIndexer`, whose `forward` adapts across CSA2's three layer
  modes (Full / Reindex / Reuse) with the shared candidate pool as an option, plus
  `IndexerDistillLoss`, which owns the distillation teacher.
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

`IndexerDistillLoss` is the only training signal the indexer has: the top-k selection is
discrete, and both the indexer's own inputs and the teacher's sources are detached, so
without it none of the indexer's parameters would update. It is attached to every layer
with a compressed KV, not only to the layers that own an indexer, because a reuse layer's
own attention distribution is part of what the shared selection must serve. The layers of
a group all score the *same* student logits: the indexer's `topk_scores` tensor is computed
once at the index source and threaded through its consumers, so every consumer's backward
flows into that shared tensor and the indexer accumulates the sum of their gradients; the
`-inf` that tensor carries is how a padded row's unused slots are marked, so the loss drops
them without reading the indices. The loss gradient is affine in the teacher, so that
per-layer sum is exactly the single pooled-teacher loss the NPU kernel computes from
accumulated raw mass; no group-size divisor is applied.

The teacher is the attention's own mass on the selected entries. The operator returns the
per-head log-sum-exp of the full softmax (sliding window, selected compressed entries and
sink alike), so each head contributes `exp(logit - lse)` and the heads are *averaged*.
The result is the raw marginal `p`, whose row sum `Z <= 1` is the compressed slice's share
of the full softmax. The loss pairs the conditional `t = p / Z` with the student and
weights the row by `Z`:

```
L = sum_t sum_j p_{t,j} (log t_{t,j} - log Y_{t,j}),   dI = Z * Y - p
```

Weighting each head by its own denominator is what keeps a head that puts almost all of
its mass on the window or the sink from outvoting a head that actually uses the compressed
entries. Keeping `p` unnormalised is what keeps `Z` in the objective: normalising to `t`
first would set `Z = 1` and silently drop the weight. A row whose mass underflows fp32
gets no weight, which matches the kernel reading `Z` off the tensor it is handed.
`IndexerDistillLoss` owns this computation: its `forward` takes the queries, compressed keys,
selected entries, the operator's LSE (`[T, H]`) and the student logits, and builds the
teacher itself, the way the MoE balance loss owns its formula.

The scores are computed twice per index source: once without a graph over all candidates
to take the top-k, and once with a graph over the selected entries only, which is what
keeps the full `[T, heads, entries]` score tensor out of the autograd graph. Both are
single-pass; there is no query chunking, and the second pass is not a gather from the
first for exactly that reason. The score-and-select half is what a kernel-backed
`lightning_indexer_v2` replaces: the per-head weights, the relaxed score, the visibility
and candidate masking and the top-k.

## Hierarchical sparse indexer

Each layer is statically assigned one of CSA2's three modes (report 2.3.1), and
`HierarchicalIndexer.forward` is the adapter over them: **Full** owns the main KV and
projects its own index keys; **Reindex** reuses the main KV and index keys of a preceding
Full layer and rescores them with its own query; **Reuse** computes no index query at all
and carries the latest top-k forward.

The candidate pool (report 2.3.2) is the hierarchy on top of that, and it is optional in
both computing modes. With `candidate_topk_blocks > 0`, the Full layer named by
`candidate_source_layer` additionally builds a shared pool of blocks scored by their best
entry, and the Reindex layers after it search only that pool; with `0` both modes score
every visible entry. The pool boundary is derived by a top-k over blocks, so it receives
no gradient from anywhere; it is a training/inference consistency and per-query cost
device, not a learned component. `use_candidates=False` by default, matching the report's
statement that the restriction is introduced in post-training; the switch is reached
through `model_registry(..., use_candidates=True)` (the model config is not a
command-line surface), which is what the `deepseek_v4_1_debugmodel_candidates` config
does.

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

Attention Gym is pinned to the released `0.0.9`; its `selected_attention` provides
`doc_ids`, `scale` and the `return_aux` LSE this model needs. Only the reference
implementation (`impl=Impl.REFERENCE`) is selected today, so the CPU tests and the NPU
ports run the same path; the fused implementation is a later switch.

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
against an explicit float64 oracle (per-head weighting, uniform-lse scaling, the marginal
`Z` and its conditional), the objective's value and gradient `dI = Z * Y - p` against the
unweighted KL it replaces (both read back through `forward`: the value off the metric
accumulator, the gradient off the student), that a mass below the fp32 normal range carries
no weight, the `-inf` marking that stands in for a padded row's unused slots, the
packed-document isolation of `doc_ids` (perturbing one segment leaves the other
bit-identical, and the indexer never selects across documents), and the mode assignment and
candidate pool of `HierarchicalIndexer`. The segment-alignment helper is covered by
`tests/unit_tests/cpu/test_text_dataset_packing.py`.
