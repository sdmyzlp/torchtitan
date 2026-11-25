# GLM KDA on Ascend 950

`glm53_flash_pypto.flash_kda` replaces the configured KDA kernel, including its
CP variant. The GLM projections, four-tap convolution, and output gate stay in
the model. PyPTO-Gym's
public `chunk_kda` owns its two forward stages, saved activations, two backward
stages, and derivatives of the fused gate, sigmoid, and Q/K normalization:

```text
GLM projections -> convolution -> chunk_kda -> GLM output gate/projection
                                  |
                                  +-- dq, dk, dv, dgate, dbeta, dA_log, ddt_bias
```

The tested source is [PyPTO-Gym](https://gitcode.com/cann/pypto-gym) revision
`3d4361e0a90f4cd6326ac3be3dd2b0c5d6f1a4d8`, available in this workspace's
`~/cann/pypto-gym` checkout. Its `flash_kda` component is under
`ops/pypto_pro/kimi_k3`; this is an external kernel dependency, not a reused
TorchTitan Kimi model component. It requires BF16 Q/K/V, raw gate and beta,
FP32 `A_log`/`dt_bias`, and key/value width 128.

## Environment and activation

The tested 950PR host has PyPTO 0.3.0 bundled with CANN 9.3.0. Source
`/workspace/lrwei/ascend/cann/set_env.sh`; the host's default CANN 9.1 environment
does not select that installation. No PyPTO installation into the shared venv
is necessary. Use a separate PyPTO-Gym checkout at the revision above and apply
the explicit compatibility patch before putting its root on `PYTHONPATH`:

```bash
source /workspace/lrwei/ascend/cann/set_env.sh
export PYPTO_GYM=/path/to/isolated/pypto-gym
git -C "$PYPTO_GYM" apply --check \
  /path/to/torchtitan/torchtitan_recipes/overrides/patches/pypto_gym_flash_kda.patch
git -C "$PYPTO_GYM" apply \
  /path/to/torchtitan/torchtitan_recipes/overrides/patches/pypto_gym_flash_kda.patch
export PYTHONPATH="$PYPTO_GYM:$PYTHONPATH"
```

Enable the component with the usual recipe argument:

```bash
--override torchtitan_recipes.overrides.glm53_flash_pypto.flash_kda
```

It composes with `glm53_flash_npu` DSA overrides. Imports are lazy, so loading a
CPU model configuration does not import PyPTO. Missing dependencies or unsupported
inputs raise; there is no automatic fallback or runtime dependency patching.

## Compatibility patch

The pinned kernel fails to compile with the tested CANN build in three places.
The patch touches only `flash_kda_bwd.py`:

- 36 BF16 loads use `US_B16`, which this PyPTO IR incorrectly infers as a
  predicate register. `UNPK_B16` preserves the even lanes consumed by 35
  BF16-to-FP32 casts and one explicit even-index gather. The differing odd lanes
  are never consumed.
- Two loop assignments change the `single_token` parameter from INT64 to bool.
  Explicit INT64 casts preserve the predicate and satisfy the SSA type checker.
- A `[32, 128]` tile restores its valid shape to `[128, 128]`. Restoring
  `[32, 128]` respects its allocation and the neighboring uses of that tile.

The original file SHA256 is
`d5c248cdfba41e76c84a03e42c5e02a373de7d2a5878b55d249014c5a162fdce`;
the patched file SHA256 is
`88abd6ac95d88180735cf86483db2b350d3bdb6360edcc7a42cb10ed71b7b03c`.
Recheck the patch against the compiler and kernel revisions when upgrading.

## Numerical checks and CP

Run `tests/unit_tests/gpu/models/glm53_flash/test_pypto_kda.py` with `pytest -s`
in the CANN environment. On the single 950PR, `[1, 4096, 64, 128]` inputs
pass against Attention Gym's independent FP32 reference, both dense and packed
with document lengths `[3, 65, 7, 4021]`. The test checks every token and head,
processing eight independent oracle heads at a time to bound CPU memory. It
compares the output and all seven input gradients, including fused gate
parameters, and checks repeated empty metadata slots. Output relative L2 errors
are 0.004578 and 0.004576; the largest gradient relative L2 errors are 0.004917
and 0.004617 (`dbeta`). The asserted limit remains 0.01. PyPTO-Gym's packed
autograd test also passes, including initial/final recurrent states.

`glm53_flash_debugmodel_npu` completed ten canonical Trainer steps on the bundled
C4 fixture with both NPU overrides and PyTorch deterministic settings enabled.
The four-layer model uses three KDA blocks, one DSA block, sequence length 4096, full attention
head dimensions, four experts, and 1,136,901,656 parameters. All 105 trainable
parameter tensors had finite gradients at every step; the indexer, short
convolution, `A_log`, `dt_bias`, and language-model head all updated. Loss moved
from 8.365148 to 5.877334 across ten different batches. Peak reserved memory was
37.07 GiB. This is a training integration check, not evidence of full-model
convergence or a throughput benchmark.

A second ten-step run also has finite gradients, but its losses diverge from
the first starting at step 6 and end at 6.284105. The pinned
`flash_kda_bwd.py` uses FP32 `AtomicAdd` for `dA_log` and `ddt_bias` across
chunks (lines 3460/3462 and 5558/5560), without a fixed cross-core reduction
order. It neither reads PyTorch deterministic settings nor exposes a
deterministic option. Bitwise repeatability is therefore not guaranteed;
these atomics have not been established as the sole cause of this run
difference.

With KCP, `prepare_cp_metadata` builds the local document offsets and context
once; the adapter passes them through the public `chunk_kda(cp_context=...)`
interface and uses `param_grad_reduce="external"` because TorchTitan reduces replicated parameter
gradients. PyPTO requires every fragment that continues onto the next rank to
end on a 64-token chunk boundary; padding to four tokens alone is insufficient.
Its collective preflight rejects unsupported cuts. Multi-rank HCCL execution
has not been validated on the single-device host.

## Single-device context-parallel validation

`tests/unit_tests/gpu/models/glm53_flash/test_npu_cp.py` compares CP=1 with CP=2
on one 950PR using `LocalTensorMode`. It uses the four-layer, 4096-token model
with both overrides, BF16 compute parameters and the model's preserved FP32
parameters. Every initial weight, input, label, position, and denominator is
checked, including parameter dtypes, before computing cross-entropy and all
105 parameter gradients. It covers one document and packed lengths
`(1024, 2048, 1024)`, including a document that continues across the rank boundary. Parameter gradients are summed once
across CP in FP32.

The test harness runs PyPTO's four communication-free stage wrappers on each
rank's real NPU tensors. Its original autograd and summary tensor collectives
remain active; `LocalTensor` simulates the actual all-gather/reduction math.
Host metadata uses PyPTO's original preflight with the complete rank records.
All patches are restored by context managers. The harness also adapts the
installed Torch/spmd_types private tuple protocol for LocalTensor rank mapping;
no shared package or production module is patched. All four CANN operator
argument lists, including schema defaults, are saved per rank in
`kernel_arguments.json`. Each case records all four calls for CP=1 and both
logical CP=2 ranks.

```bash
pytest -s tests/unit_tests/gpu/models/glm53_flash/test_npu_cp.py \
  tests/unit_tests/gpu/models/glm53_flash/test_pypto_kda.py \
  --basetemp /path/to/new-validation-results
```

Gradient-norm relative differences are 0.0204% and 0.0158%. All parameter
gradients are finite. Logit relative L2 differences are 4.1765% and 3.5023%;
individual parameter-gradient vector differences can also exceed one percent.

Whole-model checks require loss difference below 0.01 and gradient-norm relative
difference below 0.01, and finite gradients everywhere. Logit and per-parameter
vector errors are recorded separately, not asserted to be below one percent.
Changing the row count selects different NPU reduction/GEMM implementations:
small rounding differences can grow through BF16 activations and MoE routing.
This limitation persists even with deterministic algorithms enabled. The same
test separately replays the same real first-layer KDA inputs through CP=1
and CP=2 and requires output and every one of the seven gradients to have
relative L2 error below 0.01. Both cases produce bitwise-identical KDA outputs
and `dq`; their largest gradient relative L2 errors are 0.000397 and 0.000262.
Single-device simulation does not validate physical multi-rank HCCL execution.

On the tested 950PR/CANN stack, the two whole-model cases produced:

| Documents | CE CP1 | CE CP2 | Grad norm CP1 | Grad norm CP2 | Logits relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 8.447333336 | 8.447729111 | 9.124147866 | 9.126008837 | 0.0417648 |
| 1024, 2048, 1024 | 8.446028709 | 8.446623325 | 9.108232735 | 9.106791620 | 0.0350234 |

These are fixed-weight forward/backward comparisons on seeded synthetic tokens,
not an optimizer trajectory or a physical multi-device performance measurement.
