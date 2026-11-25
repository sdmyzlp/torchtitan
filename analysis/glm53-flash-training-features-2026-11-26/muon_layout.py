# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    DistMuon,
    Owned,
)
from torchtitan.models.glm53_flash import build_model_config

torch.set_num_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
model = build_model_config().build()
model.init_states(buffer_device=torch.device("cpu"))
tokens = torch.randint(0, 256, (32,))
positions = torch.arange(32)
logits = model(
    tokens,
    positions=positions,
    attention_metadata=model._get_attention_metadata(positions),
    aux_loss_denominators=torch.tensor([32]),
)
loss = F.cross_entropy(logits, tokens.roll(-1))
loss.backward()
source = model.layers["0"].attention.q_proj.weight
initial = source.detach().clone()
gradient = source.grad.detach().clone()
results = {
    "parameter": "layers.0.attention.q_proj.weight",
    "shape": list(initial.shape),
    "gradient_source_loss": loss.item(),
}
for name, placement, block in [
    ("Owned", Owned(), 256),
    ("BlockShard128", BlockShard(dim=0, block_sizes=(128,)), 128),
]:
    param = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
    opt = DistMuon(
        [{"params": [param], "param_names": ["projection"]}],
        compute_sharding_by_fqn={
            "projection": ComputeLayout(
                shardings_by_mesh_axis={"dp_shard": placement},
                compute_dtype=torch.bfloat16,
            )
        },
        bucket_configs=(BucketConfig(name="projection", patterns=("projection",)),),
        lr=1e-3,
        weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )
    refs = [torch.nn.Parameter(x.clone()) for x in initial.split(block)]
    oracle = torch.optim.Muon(
        refs,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    for step in range(10):
        grad = gradient.roll(step, 1)
        param.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        for ref, g in zip(refs, grad.split(block)):
            ref.grad = g.clone()
        opt.step()
        oracle.step()
    full = param.full_tensor().detach()
    expected = torch.cat([p.detach() for p in refs])
    delta = full - initial
    ref_delta = expected - initial
    error = float((delta - ref_delta).norm() / ref_delta.norm())
    assert error < 0.02, (name, error)
    results[name] = {
        "storage_local_shape": list(param.to_local().shape),
        "parameter_max_abs_error": float((full - expected).abs().max()),
        "update_relative_l2": error,
        "steps": 10,
    }
if dist.get_rank() == 0:
    Path("muon_layout.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
dist.destroy_process_group()
