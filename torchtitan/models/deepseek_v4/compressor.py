# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
def _rotate(x, freqs_cis, positions=None):
    """NPU-safe RoPE: no view_as_complex on the input tensor to avoid ACL format changes."""
    dtype = x.dtype
    x_f = x.float()
    x_f_pairs = x_f.unflatten(-1, (-1, 2))
    x1, x2 = x_f_pairs.unbind(-1)

    ndim = x1.ndim
    assert ndim > 1
    seqlen = x.shape[1]

    f_r = torch.view_as_real(freqs_cis)

    if positions is None:
        f_r = f_r[:seqlen]
    elif positions.size(0) == 1:
        f_r = f_r[positions.squeeze(0)]
    else:
        bsz = x.shape[0]
        assert positions.shape == (bsz, seqlen), (
            f"Expected positions shape ({bsz}, {seqlen}), got {positions.shape}"
        )
        f_r_expanded = f_r[None, :, None, :, :].expand(bsz, -1, -1, -1, -1)
        index = positions.view(bsz, seqlen, 1, 1).expand(
            bsz, seqlen, 1, f_r.shape[-2]
        )
        cos = torch.gather(f_r_expanded[..., 0], dim=1, index=index)
        sin = torch.gather(f_r_expanded[..., 1], dim=1, index=index)
        shape = [cos.shape[0], cos.shape[1]] + [1] * (ndim - 3) + [cos.shape[-1]]
        cos = cos.reshape(*shape)
        sin = sin.reshape(*shape)
        out_x1 = x1 * cos - x2 * sin
        out_x2 = x2 * cos + x1 * sin
        out = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)
        return out.to(dtype)

    cos = f_r[..., 0]
    sin = f_r[..., 1]

    shape = [1, cos.shape[0]] + [1] * (ndim - 3) + [cos.shape[1]]
    cos = cos.reshape(*shape)
    sin = sin.reshape(*shape)

    out_x1 = x1 * cos - x2 * sin
    out_x2 = x2 * cos + x1 * sin
    out = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)
    return out.to(dtype)


from torchtitan.protocols.module import Module


def _make_hadamard_mat(n: int, device: torch.device | str | None = None) -> torch.Tensor:
    n_pow2 = 2 ** math.ceil(math.log2(n))
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device)
    for _ in range(int(math.log2(n_pow2)) - 1):
        H = torch.kron(H, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device))
    return H


class Compressor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4
        rotate: bool = False
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rotate = cfg.rotate
        coff = 1 + self.overlap
        self.wkv = Linear.Config(
            in_features=cfg.dim, out_features=coff * cfg.head_dim, bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ).build()
        self.wgate = Linear.Config(
            in_features=cfg.dim, out_features=coff * cfg.head_dim, bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ).build()
        self.norm = RMSNorm.Config(
            normalized_shape=cfg.head_dim, eps=cfg.norm_eps,
            param_init={"weight": partial(nn.init.trunc_normal_, mean=1.0, std=0.02)},
        ).build()
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, coff * cfg.head_dim))
        if self._param_init is None:
            self._param_init = {}
        self._param_init["ape"] = partial(nn.init.trunc_normal_, std=0.02)

    def _overlap_transform(self, tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x, freqs_cis, positions=None):
        _, seqlen, _ = x.size()
        ratio = self.compress_ratio
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if seqlen % ratio != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
            )
        if positions is not None:
            comp_positions = positions[:, ::ratio]
        else:
            freqs_cis = freqs_cis[::ratio]
            comp_positions = None
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))
        kv_nope = kv[..., : -self.rope_head_dim]
        kv_rope = _rotate(
            kv[..., -self.rope_head_dim :], freqs_cis, positions=comp_positions
        )
        kv = torch.cat([kv_nope, kv_rope], dim=-1)
        return kv


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_index_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        compress_ratio: int = 4
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.dim = cfg.dim
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.index_topk = cfg.index_topk
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.compress_ratio = cfg.compress_ratio
        self.wq_b = Linear.Config(
            in_features=cfg.q_lora_rank,
            out_features=cfg.num_index_heads * cfg.index_head_dim,
            bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ).build()
        self.weights_proj = Linear.Config(
            in_features=cfg.dim, out_features=cfg.num_index_heads, bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ).build()
        self.compressor = Compressor.Config(
            dim=cfg.dim,
            head_dim=cfg.index_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            compress_ratio=cfg.compress_ratio,
            rotate=True,
            norm_eps=cfg.norm_eps,
        ).build()
        self.register_buffer("hadamard_mat", torch.empty(0), persistent=False)

    def _init_self_buffers(self, *, buffer_device=None):
        if buffer_device is not None:
            with torch.device(buffer_device):
                self.hadamard_mat = _make_hadamard_mat(self.head_dim, device=buffer_device)
        else:
            self.hadamard_mat = _make_hadamard_mat(self.head_dim)

    @staticmethod
    def _rotate_activation(x, hadamard_mat):
        x_shape = x.shape
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        log_dim = math.ceil(math.log2(dim))
        dim_padded = 2**log_dim
        if dim != dim_padded:
            x = F.pad(x, (0, dim_padded - dim))
        out = F.linear(x, hadamard_mat) * (dim**-0.5)
        return out[..., :dim].reshape(*x_shape)

    def forward(self, x, qr, freqs_cis, positions=None):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = _rotate(q_rope, freqs_cis, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = self._rotate_activation(q, self.hadamard_mat)
        k = self.compressor(x, freqs_cis, positions=positions)
        k = self._rotate_activation(k, self.hadamard_mat)
        weights = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return q, k, weights
