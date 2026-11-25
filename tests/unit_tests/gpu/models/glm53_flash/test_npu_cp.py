# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-device CP validation with real fused compute and LocalTensor collectives."""

import gc
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode

pytest.importorskip("torch_npu")
pytest.importorskip("pypto_pro")
pytest.importorskip("ops.pypto_pro.kimi_k3.flash_kda.op_kernel")
pytest.importorskip("cann_ops_transformer")

from tests.unit_tests.gpu.models.glm53_flash._local_tensor import (
    capture_cann_arguments,
    local_tensor_metadata,
    local_tensor_pypto,
)
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.glm53_flash import build_model_config
from torchtitan.models.glm53_flash.cp_kda import (
    ContextParallelKDA,
    ContextParallelKDAKernel,
)
from torchtitan.models.glm53_flash.model import ReferenceKDAKernel
from torchtitan_recipes.overrides.glm53_flash_npu import (
    AscendAllGatherKVSparseAttention,
)
from torchtitan_recipes.overrides.glm53_flash_pypto import flash_kda


@pytest.mark.parametrize("lengths", [(4096,), (1024, 2048, 1024)])
def test_fused_model_cp_matches_unsharded(lengths, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with capture_cann_arguments(tmp_path / "kernel_arguments.json") as records:
        _check_model(lengths)
    # CP1 and both logical CP2 ranks must each execute all four fused DSA ops.
    assert len(records) == 12


def _check_model(lengths):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    tokens = 4096
    overrides = OverrideConfig(
        imports=[
            "torchtitan_recipes.overrides.glm53_flash_npu.ascend_sparse_attention",
            "torchtitan_recipes.overrides.glm53_flash_pypto.flash_kda",
        ]
    )

    def configuration(cp):
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=1,
            context_parallel_degree=cp,
            context_parallel_load_balancer=None,
        )
        config = build_model_config(
            "debugmodel_4layer", seq_len=tokens, parallelism=parallelism
        )
        apply_overrides(overrides, config)
        return config, parallelism

    def build(config):
        with torch.device("meta"):
            model = config.build().bfloat16()
        for _, param in model.named_preserve_dtype_params():
            param.data = param.data.float()
        model.to_empty(device="npu")
        model.init_states(buffer_device=torch.device("npu"))
        classes = [type(module).__name__ for module in model.modules()]
        counts = {
            "kda": classes.count("PyPTOKDAKernel")
            + classes.count("PyPTOContextParallelKDAKernel"),
            "indexer": classes.count("AscendInnerIndexer")
            + classes.count("AscendAllGatherInnerIndexer"),
            "attention": classes.count("AscendSparseAttention")
            + classes.count("AscendAllGatherKVSparseAttention"),
        }
        assert counts == {"kda": 3, "indexer": 1, "attention": 1}, counts
        print("FUSED_COMPONENTS " + json.dumps(counts), flush=True)
        return model

    reference_config, _ = configuration(1)
    reference = build(reference_config)
    weights = {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
    }
    input_ids = torch.randint(reference.config.vocab_size, (tokens,), device="cpu").to(
        "npu"
    )
    labels = torch.randint(reference.config.vocab_size, (tokens,), device="cpu").to(
        "npu"
    )
    positions = torch.cat([torch.arange(length, device="npu") for length in lengths])
    padding = torch.zeros(tokens, dtype=torch.bool, device="npu")
    denominator = (~padding).sum().reshape(1)
    metadata = reference._get_attention_metadata(
        positions, padding_mask=padding, max_num_documents=len(lengths) + 1
    )
    kernel_inputs = []

    def capture_inputs(module, args):
        kernel_inputs.extend(value.detach().cpu() for value in args)

    handle = reference.layers[
        "0"
    ].attention.inner_attention.kernel.register_forward_pre_hook(capture_inputs)
    expected = reference(
        input_ids,
        positions=positions,
        padding_mask=padding,
        aux_loss_denominators=denominator,
        attention_metadata=metadata,
    )
    handle.remove()
    kernel_metadata = next(
        value for value in metadata.values() if hasattr(value, "cu_seqlens_cpu")
    )
    loss = F.cross_entropy(expected.float(), labels, reduction="sum") / tokens
    loss.backward()
    ref_loss = float(loss.detach())
    ref_output = expected.detach().float().cpu()
    ref_grads = {
        name: p.grad.detach().float().cpu()
        for name, p in reference.named_parameters()
        if p.requires_grad
    }
    ref_kl = float(
        reference.layers["3"].attention.inner_attention.aux_loss.instance_acc
    )
    assert all(torch.isfinite(g).all() for g in ref_grads.values())
    print(
        "CP1_MODEL_DONE",
        json.dumps({"loss": ref_loss, "parameters": len(ref_grads), "kl": ref_kl}),
        flush=True,
    )
    del reference, expected, loss, metadata
    gc.collect()
    torch.accelerator.empty_cache()

    actual_config, parallelism = configuration(2)
    dist.init_process_group("fake", rank=0, world_size=2)
    try:
        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        pc.build_mesh()
        actual = build(actual_config)
        actual.load_state_dict(weights)
        for name, param in list(actual.named_parameters()):
            parent, _, field = name.rpartition(".")
            owner = actual.get_submodule(parent)
            setattr(
                owner,
                field,
                torch.nn.Parameter(
                    LocalTensor({rank: param.detach().clone() for rank in range(2)}),
                    requires_grad=param.requires_grad,
                ),
            )
        for name, buffer in list(actual.named_buffers()):
            parent, _, field = name.rpartition(".")
            setattr(
                actual.get_submodule(parent),
                field,
                LocalTensor({rank: buffer.detach().clone() for rank in range(2)}),
            )
        with (
            LocalTensorMode(2) as mode,
            local_tensor_pypto(),
            local_tensor_metadata(ContextParallelKDA, AscendAllGatherKVSparseAttention),
            pc.activate_spmd(),
        ):
            with mode.disable():
                for name, value in actual.state_dict().items():
                    for rank in range(2):
                        assert value._local_tensors[rank].dtype == weights[name].dtype
                        assert torch.equal(
                            value._local_tensors[rank].cpu(), weights[name]
                        ), ("weight mismatch", name, rank)
            print("CP2_WEIGHTS_IDENTICAL", len(weights), flush=True)
            local_input, local_labels, kwargs, _ = actual.preprocess_inputs(
                {
                    name: mode.rank_map(lambda rank: value.clone())
                    for name, value in {
                        "input": input_ids,
                        "labels": labels,
                        "positions": positions,
                        "padding_mask": padding,
                    }.items()
                },
                parallelism_context=pc,
                parallelism=parallelism,
                max_num_documents=len(lengths) + 1,
                global_routing_token_counts=denominator,
            )
            with mode.disable():
                for rank in range(2):
                    span = slice(rank * 2048, (rank + 1) * 2048)
                    assert torch.equal(
                        local_input._local_tensors[rank], input_ids[span]
                    )
                    assert torch.equal(local_labels._local_tensors[rank], labels[span])
                    assert torch.equal(
                        kwargs["positions"]._local_tensors[rank], positions[span]
                    )
                    assert torch.equal(
                        kwargs["padding_mask"]._local_tensors[rank], padding[span]
                    )
                    assert torch.equal(kwargs["aux_loss_denominators"], denominator)
            print("CP2_INPUTS_IDENTICAL", flush=True)
            output = actual(local_input, **kwargs)
            local_loss = (
                F.cross_entropy(output.float(), local_labels, reduction="sum") / tokens
            )
            local_loss.backward()
            print("CP2_MODEL_BACKWARD_DONE", flush=True)
            # CP-replicated parameter gradients are summed exactly once, in FP32.
            for name, param in actual.named_parameters():
                if not param.requires_grad:
                    continue
                assert param.grad is not None, name
                grad = param.grad.float()
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                with mode.disable():
                    value = grad.reconcile().cpu()
                    assert torch.isfinite(value).all(), name
                    ref = ref_grads[name]
                    error = (value - ref).double().norm().item()
                    norm = ref.double().norm().item()
                    ref_grads[name] = {
                        "reference_norm": norm,
                        "cp_norm": value.double().norm().item(),
                        "relative_l2": error / max(norm, 1e-20),
                        "max_abs": float((value - ref).abs().max()),
                    }
            assert isinstance(local_loss, LocalTensor)
            with mode.disable():
                cp_loss = sum(
                    float(local_loss._local_tensors[r].detach()) for r in range(2)
                )
                cp_output = torch.cat(
                    [output._local_tensors[r].detach().float().cpu() for r in range(2)]
                )
                kl = actual.layers["3"].attention.inner_attention.aux_loss.instance_acc
                cp_kl = sum(float(kl._local_tensors[r]) for r in range(2))
        report = {
            "loss_cp1": ref_loss,
            "loss_cp2": cp_loss,
            "indexer_kl_cp1": ref_kl,
            "indexer_kl_cp2": cp_kl,
            "grad_norm_cp1": sum(x["reference_norm"] ** 2 for x in ref_grads.values())
            ** 0.5,
            "grad_norm_cp2": sum(x["cp_norm"] ** 2 for x in ref_grads.values()) ** 0.5,
            "output_relative_l2": float(
                (cp_output - ref_output).double().norm() / ref_output.double().norm()
            ),
            "parameters": ref_grads,
        }
        Path("local_tensor_model_cp.json").write_text(json.dumps(report, indent=2))
        print(
            "LOCAL_TENSOR_MODEL_CP_RESULT "
            + json.dumps({k: v for k, v in report.items() if k != "parameters"}),
            flush=True,
        )
        worst = sorted(
            ref_grads.items(), key=lambda x: x[1]["relative_l2"], reverse=True
        )[:12]
        print("WORST_PARAMETERS " + json.dumps(worst), flush=True)
        # Different row counts select different NPU reduction/GEMM implementations.
        # Record vector errors; the full-model contract here is loss and grad norm.
        assert abs(cp_loss - ref_loss) < 0.01, report
        assert abs(report["grad_norm_cp2"] / report["grad_norm_cp1"] - 1) < 0.01, report
        report["kernel_cp"] = _check_kernel_inputs(kernel_inputs, kernel_metadata, pc)
        Path("local_tensor_model_cp.json").write_text(json.dumps(report, indent=2))
        print("LOCAL_TENSOR_MODEL_CP_PASS", flush=True)
    finally:
        dist.destroy_process_group()


def _check_kernel_inputs(raw, metadata, parallelism_context):
    """Replay real projected/convolved inputs with identical upstream gradients."""
    whole = [value.to("npu").requires_grad_() for value in raw]
    kernel = flash_kda(ReferenceKDAKernel.Config()).build()
    expected = kernel(*whole, attention_metadata=metadata)
    upstream = (torch.randn_like(raw[0]) * 0.2).bfloat16()
    expected.backward(upstream.to("npu"))
    observed = [expected.detach().float().cpu()]
    for value in whole:
        assert value.grad is not None
        observed.append(value.grad.detach().float().cpu())
    with (
        LocalTensorMode(2) as mode,
        local_tensor_pypto(),
        local_tensor_metadata(ContextParallelKDA),
        parallelism_context.activate_spmd(),
    ):
        inputs = [
            mode.rank_map(
                lambda rank: value[:, rank * 2048 : (rank + 1) * 2048].to("npu")
            ).requires_grad_()
            for value in raw[:5]
        ]
        inputs += [
            mode.rank_map(lambda rank: value.to("npu").clone()).requires_grad_()
            for value in raw[5:]
        ]
        local_metadata = ContextParallelKDA.prepare_cp_metadata(
            metadata, permutation=None
        )
        kernel = flash_kda(ContextParallelKDAKernel.Config()).build()
        actual = kernel(*inputs, attention_metadata=local_metadata)
        actual.backward(
            mode.rank_map(
                lambda rank: upstream[:, rank * 2048 : (rank + 1) * 2048].to("npu")
            )
        )
        with mode.disable():
            values = [
                torch.cat(
                    [actual._local_tensors[r].detach().float().cpu() for r in range(2)],
                    1,
                )
            ]
            for index, value in enumerate(inputs):
                grad = value.grad
                assert isinstance(grad, LocalTensor)
                shards = [grad._local_tensors[rank].float().cpu() for rank in range(2)]
                values.append(
                    torch.cat(shards, 1) if index < 5 else shards[0] + shards[1]
                )
    report = {}
    for name, value, reference in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta", "dA_log", "ddt_bias"),
        values,
        observed,
    ):
        assert torch.isfinite(value).all(), name
        relative_l2 = float(
            (value - reference).double().norm()
            / reference.double().norm().clamp_min(1e-20)
        )
        report[name] = {
            "relative_l2": relative_l2,
            "max_abs": float((value - reference).abs().max()),
            "bitwise_equal": torch.equal(value, reference),
        }
        assert relative_l2 < 0.01, (name, report[name])
    print("REAL_INPUT_KDA_CP " + json.dumps(report), flush=True)
    return report
