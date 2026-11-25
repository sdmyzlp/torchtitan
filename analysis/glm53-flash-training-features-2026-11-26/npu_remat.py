# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit compatibility adapters for NPU rematerialization experiments."""

import torch_remat._placeholder as placeholder
from torch.utils.cpp_extension import load_inline


def install_fsdp_copy_in_stream():
    from torch.distributed.fsdp import FSDPModule

    # TODO: remove once the remote PyTorch has the public FSDP getter:
    # https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py
    assert not hasattr(FSDPModule, "get_all_gather_copy_in_stream")

    def get_all_gather_copy_in_stream(self):
        state = self._get_fsdp_state()
        assert state._is_root is not None, "FSDP lazy initialization has not run"
        return state._comm_ctx.all_gather_copy_in_stream

    FSDPModule.get_all_gather_copy_in_stream = get_all_gather_copy_in_stream


def install_npu_placeholders():
    # The generic pointer factory builds StorageImpl, but NPU set_ needs
    # NPUStorageImpl. The NPU factory rejects null pointers with nonzero size:
    # https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Module.cpp
    # Keep this experiment outside the offloader until remat supports it.
    extension = load_inline(
        name="glm53_npu_null_storage",
        cpp_sources="""
c10::Storage null_storage(size_t nbytes, c10::Device device) {
    auto tensor = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    auto storage = tensor.storage();
    TORCH_CHECK(storage.data_ptr().get() == nullptr);
    storage.set_nbytes(nbytes);
    return storage;
}
""",
        functions=["null_storage"],
        extra_cflags=["-O0"],
    )
    original = placeholder._null_storage

    def native_null_storage(nbytes, device):
        if device.type == "npu":
            return extension.null_storage(nbytes, device)
        return original(nbytes, device)

    placeholder._null_storage = native_null_storage


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    install_npu_placeholders()
    torch.npu.set_device(0)
    metadata = placeholder._TensorMetadata(
        (4096, 8192), (8192, 1), torch.bfloat16, torch.device("npu:0"), 67108864
    )
    before = torch.npu.memory_allocated()
    for _ in range(128):
        tensor = placeholder._make_placeholder_tensor(metadata, "inaccessible data")
        assert tensor.shape == metadata.shape and tensor.stride() == metadata.stride
        assert tensor.view(-1).numel() == 4096 * 8192
        assert tensor.untyped_storage().nbytes() == metadata.storage_nbytes
        try:
            tensor.data_ptr()
        except RuntimeError as error:
            assert "inaccessible data" in str(error)
        else:
            raise AssertionError("placeholder data is accessible")
    assert torch.npu.memory_allocated() == before
    print(
        "128 native placeholders: metadata/views valid, data poisoned, no NPU allocation"
    )
