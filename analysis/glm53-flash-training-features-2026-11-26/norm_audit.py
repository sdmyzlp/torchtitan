# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare captured pre-clip GLM gradients and norm accumulation precision."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    torch.set_num_threads(2)
    reference = torch.load(
        root / "baseline_audit/raw_gradients.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    offloaded = torch.load(
        root / "cpu_offload_audit/raw_gradients.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    rows = []
    for name, gradient in reference.items():
        expected = gradient.double()
        actual = offloaded[name].double()
        delta = (actual - expected).norm().item()
        norm = expected.norm().item()
        rows.append(
            {
                "name": name,
                "shape": list(gradient.shape),
                "reference_norm": norm,
                "offloaded_norm": actual.norm().item(),
                "delta_norm": delta,
                "relative_l2": delta / max(norm, 1e-50),
                "max_abs": (actual - expected).abs().max().item(),
                "cpu_fp32": gradient.norm().item(),
                "cpu_fp64": norm,
                "npu_fp32": gradient.to("npu").norm().item(),
            }
        )
    totals = {
        key: sum(row[key] ** 2 for row in rows) ** 0.5
        for key in (
            "reference_norm",
            "offloaded_norm",
            "delta_norm",
            "cpu_fp32",
            "cpu_fp64",
            "npu_fp32",
        )
    }
    totals["relative_l2"] = totals["delta_norm"] / totals["reference_norm"]
    result = {"totals": totals, "parameters": rows}
    (root / "norm_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
