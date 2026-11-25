# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import faulthandler
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torch_remat as remat
from torch.distributed.tensor import DTensor

from torchtitan.distributed.activation_offload import ActivationOffloader
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    BlockRematOverride,
    FullActivationRematConfig,
)
from torchtitan.observability.logging import init_logger
from torchtitan_recipes.tests.models.glm53_flash_minimal import (
    glm53_flash_debugmodel_npu,
)


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def main():
    parser = argparse.ArgumentParser()
    faulthandler.enable()
    parser.add_argument(
        "case",
        choices=(
            "baseline",
            "baseline_repeat",
            "remat_full",
            "remat_selective",
            "remat_recompute_attention",
            "remat_save_dsa",
            "activation_offload",
            "remat_save_all",
            "activation_offload_regions",
            "cpu_offload",
            "spmd",
            "distmuon",
            "muon_cpu_offload",
            "baseline_audit",
            "cpu_offload_audit",
            "cpu_offload_fp64_norm",
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--native-npu-placeholders", action="store_true")
    parser.add_argument("--backport-fsdp-copy-in-stream", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / (args.run_name or args.case)
    folder.mkdir(parents=True, exist_ok=True)
    report = {"case": args.case, "metrics": [], "steps": []}
    if args.native_npu_placeholders:
        from npu_remat import install_npu_placeholders

        install_npu_placeholders()
        report["native_npu_placeholders"] = True
    if args.backport_fsdp_copy_in_stream:
        from npu_remat import install_fsdp_copy_in_stream

        install_fsdp_copy_in_stream()
        report["fsdp_copy_in_stream_backport"] = True
    config = glm53_flash_debugmodel_npu()
    config.dump_folder = str(folder)
    config.debug.distinct_seed_mesh_axes = ["pp", "dp_shard"]
    # Compare the same Adam implementation on accelerator and host storage.
    adam = config.optim.optimizer.optimizers[0]
    adam.fused = False
    adam.foreach = True
    if args.case == "remat_full":
        config.activation_remat = FullActivationRematConfig()
    elif args.case == "remat_selective":
        config.activation_remat = ActivationRematConfig(
            save_regions=["*"], recompute_regions=["*routed_experts.w13.*"]
        )
    elif args.case == "remat_recompute_attention":
        config.activation_remat = ActivationRematConfig(
            save_regions=["*"],
            recompute_regions=[
                "*inner_attention",
                "*inner_indexer",
                "*routed_experts.w13.*",
            ],
        )
    elif args.case == "remat_save_dsa":
        config.activation_remat = ActivationRematConfig(
            save_regions=[],
            block_overrides=[
                BlockRematOverride(blocks=["layers.3"], save_regions=["*"])
            ],
        )
    elif args.case in ("remat_save_all", "activation_offload_regions"):
        offload = args.case == "activation_offload_regions"
        config.activation_remat = ActivationRematConfig(
            save_regions=["*"],
            offload_block_input=offload,
            offload_regions=["*"] if offload else [],
        )
    elif args.case == "activation_offload":
        config.activation_remat = FullActivationRematConfig(offload_block_input=True)
    elif args.case in (
        "cpu_offload",
        "cpu_offload_audit",
        "muon_cpu_offload",
        "cpu_offload_fp64_norm",
    ):
        if args.case == "muon_cpu_offload":
            from feature_muon import configure_muon

            configure_muon(config)
        config.parallelism.fsdp_enable_cpu_offloading = True
    elif args.case == "spmd":
        config.debug.spmd_typechecking = True
    elif args.case == "distmuon":
        from feature_muon import configure_muon

        configure_muon(config)
    trainer = None
    try:
        init_logger()
        if args.case.endswith("_audit"):
            config.training.steps = 1
        trainer = config.build()
        model = trainer.engine.model_parts[0]
        state = dict(model.named_parameters()) | dict(model.named_buffers())
        seed_path = args.root / "seed.pt"
        if not seed_path.exists():
            torch.save(
                {name: local(value).detach().cpu() for name, value in state.items()},
                seed_path,
            )
        seed_state = torch.load(seed_path, map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        with torch.no_grad():
            for name, value in state.items():
                source = seed_state[name]
                target = local(value)
                assert (
                    source.shape == target.shape and source.dtype == target.dtype
                ), name
                target.copy_(source)
                digest.update(name.encode())
                digest.update(
                    source.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                )
        report["initial_state_sha256"] = digest.hexdigest()
        del seed_state
        params = list(model.parameters())
        report["parameter_elements"] = sum(p.numel() for p in params)
        report["parameter_devices"] = dict(
            Counter(str(local(p).device) for p in params)
        )

        class AuditLogger:
            def log(self, metrics, step):
                row = {
                    "step": step,
                    **{key: float(value) for key, value in metrics.items()},
                }
                report["metrics"].append(row)
                print("FEATURE_METRIC " + json.dumps(row), flush=True)

            def close(self):
                pass

        trainer.metrics_processor.loggers.append(AuditLogger())
        if args.case == "cpu_offload_fp64_norm":
            assert torch.distributed.get_world_size() == 1

            def fp64_norm_clip(parameters, max_norm, **kwargs):
                grads = [local(p.grad) for p in parameters if p.grad is not None]
                assert all(g.device.type == "cpu" for g in grads)
                norm = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(g, dtype=torch.float64)
                            for g in grads
                        ]
                    )
                    .norm()
                    .float()
                )
                torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
                return norm

            trainer.engine.optim.gradient_clipper.clip_grad_norm_ = fp64_norm_clip
        if args.case.endswith("_audit"):
            original_clip = trainer.engine.optim.gradient_clipper.clip_grad_norm_

            def audit_clip(parameters, *clip_args, **clip_kwargs):
                gradients = {
                    name: local(p.grad).detach().cpu()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                torch.save(gradients, folder / "raw_gradients.pt")
                return original_clip(parameters, *clip_args, **clip_kwargs)

            trainer.engine.optim.gradient_clipper.clip_grad_norm_ = audit_clip
        # Record every native sparse call, including defaults, without tensor values.
        for op_name in (
            "lightning_indexer",
            "sparse_flash_mla",
            "sparse_flash_mla_grad",
            "sparse_lightning_indexer_kl_loss_grad",
        ):
            original_op = getattr(torch.ops.cann_ops_transformer, op_name)
            schema = next(iter(original_op._schemas.values()))

            def log_op(
                *op_args,
                original_op=original_op,
                schema=schema,
                op_name=op_name,
                **op_kwargs,
            ):
                bound = {a.name: a.default_value for a in schema.arguments}
                bound.update({a.name: v for a, v in zip(schema.arguments, op_args)})
                bound.update(op_kwargs)
                arguments = {
                    k: (
                        {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "device": str(v.device),
                        }
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in bound.items()
                }
                row = {
                    "step": len(report["steps"]) + 1,
                    "op": op_name,
                    "arguments": arguments,
                }
                with (folder / "kernel_arguments.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + chr(10))
                return original_op(*op_args, **op_kwargs)

            log_op.default = original_op.default
            if os.environ.get("GLM_STUDY_LOG_KERNEL_ARGS", "1") == "1":
                setattr(torch.ops.cann_ops_transformer, op_name, log_op)
        original_step = trainer.train_step

        def step(iterator):
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            original_step(iterator)
            torch.npu.synchronize()
            row = {
                "step": len(report["steps"]) + 1,
                "seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
            }
            report["steps"].append(row)
            (folder / "partial.json").write_text(json.dumps(report, indent=2))
            print("FEATURE_STEP " + json.dumps(row), flush=True)

        trainer.train_step = step
        # Audit all gradients once without adding host synchronization to measured steps.
        checked = False

        def check_grads(optimizer, unused_args, unused_kwargs):
            nonlocal checked
            if checked:
                return
            assert all(p.grad is not None for p in params)
            assert (
                torch.stack([torch.isfinite(local(p.grad)).all() for p in params])
                .all()
                .item()
            )
            report["finite_gradient_tensors"] = len(params)
            checked = True

        handle = trainer.engine.optim.optimizers.register_step_pre_hook(check_grads)
        with remat.collect_trace() as trace:
            trainer.train()
        handle.remove()
        report["activation_offload"] = [
            {
                key: value
                for key, value in vars(owner.forward_stats).items()
                if not key.startswith("_")
            }
            for hook in model._forward_pre_hooks.values()
            if isinstance(owner := getattr(hook, "__self__", None), ActivationOffloader)
        ]
        report["remat_trace"] = trace.format()
        report["status"] = "passed"
        report["completed_steps"] = trainer.engine.num_completed_steps
        report["optimizer_state"] = {}
        for index, opt in enumerate(trainer.engine.optim.optimizers.optimizers):
            counts = Counter()
            for state_values in opt.state.values():
                for name, value in state_values.items():
                    if isinstance(value, torch.Tensor):
                        tensor = local(value)
                        counts[f"{name}:{tensor.device}:{tensor.dtype}"] += (
                            tensor.numel() * tensor.element_size()
                        )
            report["optimizer_state"][str(index)] = dict(counts)
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2))
        print("FEATURE_RESULT " + str(folder / "result.json"), flush=True)
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
