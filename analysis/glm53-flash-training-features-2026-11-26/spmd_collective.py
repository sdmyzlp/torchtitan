# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import spmd_types as spmd
import torch
import torch.distributed as dist
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.distributed.spmd_types import set_current_spmd_mesh

dist.init_process_group("gloo")
rank = dist.get_rank()
mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("cp",))
results = {}
key = (
    torch.arange(rank * 8, (rank + 1) * 8, dtype=torch.float32)
    .reshape(2, 4)
    .requires_grad_()
)
with set_current_spmd_mesh(mesh), typecheck(local=False):
    spmd.assert_type(key, {"cp": spmd.S(0)})
    (gathered,) = cp_all_gather_tensors((key,), reduce_dtype=torch.float32)
    spmd.assert_type(gathered, {"cp": spmd.R})
with spmd.no_typecheck():
    (gathered.sum() * (rank + 1)).backward()
assert torch.equal(key.grad, torch.full_like(key, 3))
results["gather_adjoint"] = {
    "shape": list(gathered.shape),
    "local_grad": key.grad.tolist(),
    "expected_grad_sum": 3,
}
wrong = torch.randn(2, 4)
try:
    with set_current_spmd_mesh(mesh), typecheck(local=False):
        spmd.assert_type(wrong, {"cp": spmd.R})
        cp_all_gather_tensors((wrong,), reduce_dtype=torch.float32)
except Exception as e:
    results["wrong_input_rejected"] = {"type": type(e).__name__, "error": str(e)}
else:
    raise AssertionError("Bad replication annotation accepted")
if rank == 0:
    Path("spmd_collective.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
dist.destroy_process_group()
