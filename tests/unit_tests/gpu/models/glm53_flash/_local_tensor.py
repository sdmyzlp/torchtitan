# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch opaque kernels locally while retaining their original CP autograd."""

import dataclasses
import importlib
import json
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    enabled_local_tensor_mode,
    LocalIntNode,
    LocalTensor,
)
from torch.utils import _pytree as pytree


def on_rank(value, rank):
    if isinstance(value, LocalTensor):
        return value._local_tensors[rank]
    if isinstance(value, torch.SymInt) and isinstance(value.node, LocalIntNode):
        return value.node._local_ints[rank]
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: pytree.tree_map(
                    lambda x: on_rank(x, rank), getattr(value, field.name)
                )
                for field in dataclasses.fields(value)
            },
        )
    return value


def combine(results):
    leaves = {rank: pytree.tree_flatten(value) for rank, value in results.items()}
    first = next(iter(leaves.values()))
    assert all(
        spec == first[1] for _, spec in leaves.values()
    ), "Rank result structures differ"
    output = []
    for index in range(len(first[0])):
        values = {rank: parts[0][index] for rank, parts in leaves.items()}
        value = next(iter(values.values()))
        if isinstance(value, torch.Tensor):
            value = LocalTensor(values)
        elif isinstance(value, int) and any(item != value for item in values.values()):
            value = torch.SymInt(LocalIntNode(values))
        else:
            assert all(item == value for item in values.values()), (index, values)
        output.append(value)
    return pytree.tree_unflatten(output, first[1])


def stage_dispatch(function):
    """Run only communication-free wrappers on the underlying rank tensors."""

    def call(*args, **kwargs):
        mode = enabled_local_tensor_mode()
        if mode is None:
            return function(*args, **kwargs)
        with mode.disable():
            return combine(
                {
                    rank: function(
                        *pytree.tree_map(lambda x: on_rank(x, rank), args),
                        **pytree.tree_map(lambda x: on_rank(x, rank), kwargs),
                    )
                    for rank in sorted(mode.ranks)
                }
            )

    return call


@contextmanager
def local_tensor_pypto():
    """Keep the original PyPTO autograd and its tensor summary collectives."""
    module = importlib.import_module(
        "ops.pypto_pro.kimi_k3.flash_kda.op_kernel.chunk_kda"
    )
    original_prepare = module.prepare_cp_metadata

    class CaptureRecordError(Exception):
        pass

    def prepare(*args, **kwargs):
        mode = enabled_local_tensor_mode()
        if mode is None:
            return original_prepare(*args, **kwargs)
        records, outputs = {}, {}
        ranks = sorted(mode.ranks)
        with mode.disable():
            # Reuse PyPTO's host preflight twice: collect its real rank records,
            # then validate the same complete record list on each logical rank.
            for replay in (False, True):
                for rank in ranks:

                    def gather(items, record, *, group):
                        if not replay:
                            records[rank] = record
                            raise CaptureRecordError
                        items[:] = [records[r] for r in ranks]

                    with (
                        patch.object(dist, "get_rank", lambda group=None: rank),
                        patch.object(
                            dist, "get_world_size", lambda group=None: len(ranks)
                        ),
                        patch.object(dist, "get_backend", lambda group=None: "gloo"),
                        patch.object(dist, "all_gather_object", gather),
                    ):
                        try:
                            outputs[rank] = original_prepare(
                                *pytree.tree_map(lambda x: on_rank(x, rank), args),
                                **pytree.tree_map(lambda x: on_rank(x, rank), kwargs),
                            )
                        except CaptureRecordError:
                            pass
            return combine(outputs)

    names = (
        "flash_kda_stage1_wrapper",
        "flash_kda_stage2_wrapper",
        "flash_kda_bwd_stage1_wrapper",
        "flash_kda_bwd_stage2_wrapper",
    )
    replacements = {name: stage_dispatch(getattr(module, name)) for name in names}
    replacements["prepare_cp_metadata"] = prepare
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(module, name, replacement))
        yield


@contextmanager
def local_tensor_metadata(*classes):
    originals = {cls: cls.prepare_cp_metadata for cls in classes}

    def wrap(original):
        def call(*args, **kwargs):
            mode = enabled_local_tensor_mode()
            if mode is None:
                return original(*args, **kwargs)
            outputs = {}
            with mode.disable():
                for rank in sorted(mode.ranks):
                    with patch.object(dist, "get_rank", lambda group=None: rank):
                        outputs[rank] = original(
                            *pytree.tree_map(lambda x: on_rank(x, rank), args),
                            **pytree.tree_map(lambda x: on_rank(x, rank), kwargs),
                        )
                return combine(outputs)

        return staticmethod(call)

    try:
        for cls, original in originals.items():
            cls.prepare_cp_metadata = wrap(original)
        import spmd_types._local as local_spmd

        original_layout = local_spmd._prepare_collective_groups

        def layout(group):
            result = original_layout(group)
            # The installed Torch returns two values; newer spmd_types unpacks
            # a third, unused offset in its LocalTensor-only rank mapping.
            return (*result, 0) if len(result) == 2 else result

        with patch.object(local_spmd, "_prepare_collective_groups", layout):
            yield
    finally:
        for cls, original in originals.items():
            cls.prepare_cp_metadata = staticmethod(original)


@contextmanager
def capture_cann_arguments(path):
    """Record each real rank's first call, including every schema default."""
    namespace = torch.ops.cann_ops_transformer
    names = (
        "lightning_indexer",
        "sparse_flash_mla",
        "sparse_flash_mla_grad",
        "sparse_lightning_indexer_kl_loss_grad",
    )
    originals = {name: getattr(namespace, name) for name in names}
    records, seen = [], set()

    def wrap(name, original):
        schema = next(iter(original._schemas.values()))

        def call(*args, **kwargs):
            mode = enabled_local_tensor_mode()
            ranks = sorted(mode.ranks) if mode else [0]
            bound = {
                argument.name: value for argument, value in zip(schema.arguments, args)
            }
            bound.update(kwargs)
            for argument in schema.arguments:
                bound.setdefault(argument.name, argument.default_value)
            for rank in ranks:
                key = (len(ranks), rank, name)
                if key in seen:
                    continue
                values = pytree.tree_map(lambda value: on_rank(value, rank), bound)
                values = pytree.tree_map(
                    lambda value: (
                        {"shape": list(value.shape)}
                        if isinstance(value, torch.Tensor)
                        else value
                    ),
                    values,
                )
                record = {
                    "cp": len(ranks),
                    "rank": rank,
                    "op": name,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "arguments": values,
                }
                records.append(record)
                seen.add(key)
                path.write_text(json.dumps(records, indent=2))
                print("CANN_ALL_ARGUMENTS " + json.dumps(record), flush=True)
            return original(*args, **kwargs)

        # LocalTensor resolves the native overload again during per-rank dispatch.
        call.default = original.default  # pyrefly: ignore[missing-attribute]
        return call

    try:
        for name, original in originals.items():
            setattr(namespace, name, wrap(name, original))
        yield records
    finally:
        for name, original in originals.items():
            setattr(namespace, name, original)
