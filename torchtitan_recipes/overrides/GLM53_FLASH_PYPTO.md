# GLM KDA on Ascend 950

`glm53_flash_pypto.flash_kda` replaces the configured single-device KDA kernel. The GLM projections, four-tap convolution, and output gate stay in
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

## Numerical checks

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
