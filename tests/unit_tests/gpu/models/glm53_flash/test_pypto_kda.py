# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guard the fused gate/L2Norm backward and packed document state resets."""

import json

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("pypto_pro")
pytest.importorskip("ops.pypto_pro.kimi_k3.flash_kda.op_kernel")

from torchtitan.models.glm53_flash.model import ReferenceInnerKDA, ReferenceKDAKernel
from torchtitan_recipes.overrides.glm53_flash_pypto import flash_kda


@pytest.mark.parametrize("lengths", [None, [3, 65, 7, 4021]])
def test_forward_and_all_input_gradients(lengths, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(42)
    tokens = 4096 if lengths is None else sum(lengths)
    heads, width = 64, 128
    raw = [(torch.randn(1, tokens, heads, width) * 0.2).bfloat16() for _ in range(4)]
    raw += [(torch.randn(1, tokens, heads) * 0.2).bfloat16()]
    raw += [torch.randn(heads) * 0.1]
    # GLM's dt initialization produces negative biases and slow forgetting.
    raw += [torch.linspace(-6.0, -2.5, heads * width).view(heads, width)]
    device = [x.to("npu").requires_grad_() for x in raw]
    positions = torch.cat([torch.arange(n) for n in (lengths or [tokens])])
    config = ReferenceInnerKDA.Config(
        head_dim=128, conv_kernel_size=4, kernel=ReferenceKDAKernel.Config()
    )
    # Reserved, empty metadata slots must not become spurious recurrent states.
    metadata = config.build_attention_metadata(
        positions.to("npu"),
        max_num_documents=len(lengths or [tokens]) + 2,
        max_context_length=tokens,
    )
    oracle_metadata = config.build_attention_metadata(positions)
    kernel = flash_kda(ReferenceKDAKernel.Config()).build()
    actual = kernel(*device, attention_metadata=metadata)
    upstream = (torch.randn_like(raw[0]) * 0.2).bfloat16()
    actual.backward(upstream.to("npu"))
    torch.accelerator.synchronize()

    observed = [actual.detach().float().cpu()]
    for x in device:
        assert x.grad is not None
        observed.append(x.grad.detach().float().cpu())
    del actual, device
    torch.accelerator.empty_cache()
    names = ("output", "dq", "dk", "dv", "dgate", "dbeta", "dA_log", "ddt_bias")
    report = {
        name: {"error_squared": 0.0, "reference_squared": 0.0, "max_abs": 0.0}
        for name in names
    }
    # Heads are independent; bound CPU oracle memory while checking all 64 heads.
    for start in range(0, heads, 8):
        span = slice(start, start + 8)
        oracle = [x[:, :, span].float().requires_grad_() for x in raw[:5]]
        oracle += [x[span].float().requires_grad_() for x in raw[5:]]
        expected = ReferenceKDAKernel.Config().build()(
            *oracle, attention_metadata=oracle_metadata
        )
        expected.backward(upstream[:, :, span].float())
        for index, (name, got, ref) in enumerate(
            zip(names, observed, (expected, *(x.grad for x in oracle)))
        ):
            assert ref is not None, name
            got = got[:, :, span] if index < 6 else got[span]
            ref = ref.detach().float()
            assert torch.isfinite(got).all() and torch.isfinite(ref).all(), name
            difference = got - ref
            report[name]["error_squared"] += difference.double().square().sum().item()
            report[name]["reference_squared"] += ref.double().square().sum().item()
            report[name]["max_abs"] = max(
                report[name]["max_abs"], difference.abs().max().item()
            )
        print(f"Checked KDA heads {start}:{start + 8}", flush=True)
    for name, values in report.items():
        values["relative_l2"] = (
            values.pop("error_squared") / max(values.pop("reference_squared"), 1e-24)
        ) ** 0.5
        assert values["relative_l2"] < 0.01, (name, values)
    print(json.dumps(report, sort_keys=True), flush=True)
