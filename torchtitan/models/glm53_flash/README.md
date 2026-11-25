# GLM-5.3-Flash text backbone

This model defines its own Compressor, Indexer, InnerIndexer, sparse attention,
and mHC modules. It uses common TorchTitan decoder, KDA projection, FFN and MoE
building blocks, with no imports from the DeepSeek or Kimi model packages.

The full flavor has 45 layers: 34 KDA and 11 NoPE sparse MLA layers, four mHC
streams, three dense FFNs followed by 42 MoEs, 288 routed experts with top-8
routing and one shared expert. The text model has 313,326,799,870 parameters,
including the LM head. The five-layer CPU debug flavor retains both attention
families, mHC, dense/MoE FFNs and a 2,048-token vocabulary. The
`debugmodel_4layer` flavor keeps the released hidden/attention widths with
three KDA layers and one DSA layer, a 2,048-token vocabulary and four routed
experts (top-2). Its 1,136,901,656 parameters fit the single-device NPU probe.

Architecture sources are the [official configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json),
[Transformers GLM5Next](https://github.com/huggingface/transformers/tree/v5.22.1/src/transformers/models/glm5_next),
and [vLLM GLM5Next](https://github.com/vllm-project/vllm/tree/08c497efa9a5213f2903670655f040977ba03fbd/vllm/models/glm5next/common).

## Kernel boundaries and indexer loss

```text
Indexer
  detached x/q -> index projections -> Compressor (complete groups of 4)
                                     -> InnerIndexer [AllGather idx_k under CP]
                                          | topk_indices, topk_scores
                                          v
InnerAttention [AllGather latent KV under CP]
  expand each selected pool into 4 tokens + append 0..3 incomplete tail tokens
  -> sparse attention -> output
```

`InnerIndexer` belongs to `Indexer`. Only its compressed-key input is gathered;
queries and index weights remain local. Its pool indices and scores go directly
to `InnerAttention`. The compressor never adds a tail. GLM has no overlapping
compression or sliding-window attention, so the two streams contain disjoint
tokens. The full model selects up to 512 pools (2,048 tokens) plus the tail.

All modules pass `attention_metadata` through their kernel boundaries. The
Decoder builds it once per backend and batch. Pool rows, visible pool limits,
tail indices and KDA document offsets are prepared there; forward only expands
the indexer's data-dependent selections. The NPU override also prepares all four
CANN metadata tensors in `build_attention_metadata`, or in `prepare_cp_metadata`
after query sharding. No CANN metadata kernel runs in a layer's forward.

The reference path uses Attention Gym `gather_attn(..., impl="reference")` and
`bound_gate`/`chunk_kda` reference implementations. These work on CPU as well
as accelerators. The four-tap convolution uses PyTorch because Attention Gym's
convolution has no reference backend. Packed document boundaries reset all
recurrences, convolution history and pooling.

`IndexerLoss` trains only indexer/compressor parameters; its inputs from the
trunk and its teacher are detached. For each query, let `p` be head-mean
attention probabilities summed within each selected four-token pool, and
`Z = sum(p)`. The objective is `Z * KL(p/Z || softmax(scores))`. The tail is
excluded from `p`, so `Z` can be less than one. Padding queries contribute zero;
the denominator is the step's global valid-token count.

The Ascend override uses four CANN kernels with a deliberate autograd protocol:

```text
forward:  lightning_indexer -> (indices, scores) -> sparse_flash_mla
backward: sparse_flash_mla_grad
            -> mask invalid token slots, sum each pool's teacher probabilities
            -> scores.grad = teacher * loss_coeff / global_valid_tokens
            -> sparse_lightning_indexer_kl_loss_grad
            -> d(index_q), d(index_k), d(index_weight)
```

Here `scores.grad` carries `attn_softmax_l1_norm`, not an ordinary score
Jacobian. The indexer backward computes `softmax(scores) * Z - p` itself.
The teacher is independent of the upstream LM gradient, including when that
gradient is zero. Do not insert a score transform between these two boundaries.
A fixed left window of two would include three tokens even when the tail is
empty (position 3), so the tail uses explicit indices rather than band mode.
SMLA treats expanded pools and tail as two full-resolution sparse streams
(`cmp_ratio=1`); its two KV gradients are added. The tested CANN build requires
an explicit sink tensor, so the adapter supplies `-inf` sinks with zero mass.

The tested CANN sparse backward initializes short probability tensors without
skipping vector cores beyond the final partition. The override checks the
rounded partition sizes against the device's actual vector-core count and
rejects affected shapes before dispatch. The NPU recipe uses 4,096 tokens and
the released attention geometry: 64 DSA heads of width 256, KV latent width
512, 32 index heads of width 128, and 64 KDA heads of width 128. Its CP2 query
shards of 2,048 tokens also satisfy this check. No native CANN package is changed.

## Context parallelism

`GlmContextParallelTransform` uses the upstream `ContextParallelTransform` to
replace the two inner attention types, and installs the CP indexer and KDA kernel
configs. The default model paths have no CP branch. DSA keeps
`Indexer.inner_indexer` as its compressed-key AllGather boundary.
Both AllGather adjoints reduce-scatter summed gradients, with FP32 reduction.
Queries are contiguous shards, with global key offsets and local query offsets.
Documents are padded internally to a multiple of four and shard boundaries
must also align to four. Padding positions continue the document's positions.
There is no HeadTail permutation or selective KV gathering.

KCP exchanges fixed-size affine summaries `S_out = S_in @ A + B`, reconstructs
initial states from preceding fragments of the same document, and exchanges
three-token convolution halos. Both exchanges participate in backward. This
implements [Kimi K3 section 5.1.2](https://arxiv.org/html/2607.24653v2#S5.SS1.SSS2)
with Attention Gym's value-first state layout. It does not gather the full QKV.

The PyPTO override delegates fused normalization, gate, recurrence and their
backward to PyPTO-Gym's public `chunk_kda`. Its KCP requires continuing fragments
to end on 64-token boundaries. The NPU CP recipe therefore uses per-document
padding of 64 and checks `seq_len % (64 * cp_degree) == 0`; this also satisfies
DSA's four-token alignment. See [PyPTO environment and compatibility patch](../../../torchtitan_recipes/overrides/GLM53_FLASH_PYPTO.md)
for the tested CANN 9.3.0/PyPTO 0.3.0 setup and the required isolated source patch.

## Run

```bash
source ~/pytorch/venv/3.12/bin/activate
OMP_NUM_THREADS=2 python -m pytest -q tests/unit_tests/cpu/models/glm53_flash
```

Runtime dependencies are the repository's pinned requirements, including
Attention Gym revision `755691f3`. Transformers 5.22.1 is optional and used only
by the reference parity tests. No vendor library is imported by the core model.

```python
import torch
from torchtitan.models.glm53_flash import build_model_config

config = build_model_config("debugmodel", seq_len=32)
model = config.build()
model.init_states()
positions = torch.arange(32)
logits = model(
    torch.randint(config.vocab_size, (32,)),
    positions=positions,
    attention_metadata=model._get_attention_metadata(positions),
    aux_loss_denominators=torch.tensor([32]),
)
logits.float().square().mean().backward()

with torch.device("meta"):
    full_model = build_model_config("glm53_flash").build()
```

The recipes use the bundled C4 fixture, seed 42, PyTorch deterministic settings
and 10 AdamW steps. After activating the CANN environment and patched PyPTO-Gym:

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.glm53_flash_minimal \
CONFIG=glm53_flash_debugmodel_npu ./run_train.sh

# Requires two devices; configured here, not validated on the one-device host.
NGPU=2 MODULE=torchtitan_recipes.tests.models.glm53_flash_minimal \
CONFIG=glm53_flash_debugmodel_npu_cp2 ./run_train.sh

python -m pytest -q tests/unit_tests/gpu/models/glm53_flash
```

`glm53_flash_debugmodel` and `glm53_flash_debugmodel_cp2` select the reference
kernels. The NPU recipes enable `glm53_flash_npu.ascend_sparse_attention` and
`glm53_flash_pypto.flash_kda` through the usual override mechanism.

## Validation and scope

- CPU: KDA, DSA, mHC and full BF16 backbone against Transformers; packed versus
  separate documents; causality; indexer-only auxiliary gradients; full meta
  construction and 10-step synthetic optimization.
- Real Gloo CP: three-rank KCP in FP32/BF16 (ten cases), two-rank DSA including
  zero LM gradient, and the complete five-layer model. Checks cover forward,
  input and every trainable parameter gradient, padding and KL metric sums.
- Ascend 950PR: DSA at 4,096 tokens and released head/latent widths against an
  independent FP32 attention/KL oracle, including CP query fragments and zero
  LM gradient. The oracle supervises 32 query rows to bound its gathered-KV
  memory, compares all resulting gradients, and verifies zero gradients on
  unsupervised queries. Gradient relative L2 tolerance is 2%. Additional tests
  cover all-tail documents and rejection of unsafe short launches.
- PyPTO on the same device: dense and packed `[1,4096,64,128]` output plus all
  seven input gradients against Attention Gym FP32, comparing every head/token.
  Worst gradient relative L2 error is 0.004918, below the 1% bound.
- Canonical Trainer, 950PR: both overrides active, four layers, 4,096 tokens and
  two runs of ten C4 steps. All 105 trainable parameter tensors have finite
  gradients on every step; indexer, convolution, gate parameters and LM head all update.
  The first run goes from loss 8.365148 to 5.877334; peak reserved memory is
  37.07 GiB with no allocation retries or OOM. This is a debug training check, not a full-model
  convergence or throughput claim.

The repeated Trainer run stays finite but is not bitwise identical: losses
first differ at step 6, and the second run ends at 6.284105. PyPTO uses FP32
atomic additions for gate-parameter gradients and does not honor PyTorch
deterministic settings. This is a known source of nondeterminism; it has not
been isolated as the sole cause of the observed training difference.

The single-device CP check uses PyTorch `LocalTensorMode`: collectives really
concatenate and sum the two logical ranks' tensors. A test-only adapter exposes
PyPTO's four pointer-based stages to that simulator and preserves the original
public `chunk_kda` autograd/summary exchange. It also adapts the installed
PyTorch/SPMD simulator API mismatch; production code contains neither adapter.
The test fixes every parameter, input and label, uses BF16 compute parameters
with FP32 preserved parameters, and compares CE loss and the global gradient
norm after summing replicated parameter gradients once.

Bitwise equality across CP degrees is not supported by this NPU stack. Changing
the number of rows changes rounding in ordinary token-local kernels, including
mHC and projections. BF16 rounding and discrete MoE routing amplify those
initial differences. Whole-model tests therefore report logits and every
parameter's gradient error alongside loss/norm; they do not claim per-parameter
1% parity. Separately, replaying identical real first-layer inputs through the
fused KDA CP paths checks the output and all seven gradients within 1%.

Multi-rank HCCL remains unverified because the host has one NPU. The reference
kernels use eager document/token loops and are intended for small numerical
checks. TP/EP/PP/FSDP combinations, compilation, router auxiliary objectives,
vision, MTP, inference caches and HF checkpoint conversion are outside this
port. Router correction biases remain frozen. Top-k ties retain backend-specific
tie breaking; numerical tests compare at the same selected indices.
