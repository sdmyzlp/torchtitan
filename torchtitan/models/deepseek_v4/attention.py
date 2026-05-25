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
def _rotate(x, freqs_cis, positions=None, inverse=False):
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
        if inverse:
            sin = -sin
        shape = [cos.shape[0], cos.shape[1]] + [1] * (ndim - 3) + [cos.shape[-1]]
        cos = cos.reshape(*shape)
        sin = sin.reshape(*shape)
        out_x1 = x1 * cos - x2 * sin
        out_x2 = x2 * cos + x1 * sin
        out = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)
        return out.to(dtype)

    cos = f_r[..., 0]
    sin = f_r[..., 1]
    if inverse:
        sin = -sin

    shape = [1, cos.shape[0]] + [1] * (ndim - 3) + [cos.shape[1]]
    cos = cos.reshape(*shape)
    sin = sin.reshape(*shape)

    out_x1 = x1 * cos - x2 * sin
    out_x2 = x2 * cos + x1 * sin
    out = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)
    return out.to(dtype)


from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


def _get_window_topk_idxs(window_size, bsz, seqlen, device):
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    window_topk = (base - window_size + 1).clamp(0) + torch.arange(
        min(seqlen, window_size), device=device
    )
    window_topk = torch.where(window_topk > base, -1, window_topk)
    return window_topk.unsqueeze(0).expand(bsz, -1, -1)


def _get_compress_topk_idxs_generic(bsz, seqlen, offset, compress_ratio, device):
    matrix = torch.arange(seqlen // compress_ratio, device=device).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // compress_ratio
    compress_topk = torch.where(mask, -1, matrix + offset)
    return compress_topk.unsqueeze(0).expand(bsz, -1, -1)


def _li_compute(q_indexer, k_indexer, weights, seqlen, offset, compress_ratio, index_topk):
    index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
    index_score = index_score.relu_() * weights.unsqueeze(-1)
    index_score = index_score.sum(dim=2)
    device = index_score.device
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    mask = (
        torch.arange(seqlen // compress_ratio, device=device).unsqueeze(0)
        >= (base + 1) // compress_ratio
    )
    index_score += torch.where(mask, torch.finfo(q_indexer.dtype).min, 0)
    index_score, topk_idxs = index_score.topk(
        min(index_topk, seqlen // compress_ratio), dim=-1
    )
    mask = topk_idxs >= (base + 1) // compress_ratio
    compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    return compress_topk_idxs, index_score


def _sparse_attention(query_states, kv_states, attn_sink, kv_compress, compress_topk_idxs,
                      window_size, compress_ratio, softmax_scale):
    bsz, seqlen, _, _ = query_states.size()

    topk_idxs = _get_window_topk_idxs(window_size, bsz, seqlen, query_states.device)
    if compress_ratio > 1:
        offset = kv_states.size(1)
        if compress_topk_idxs is None:
            compress_topk_idxs = _get_compress_topk_idxs_generic(
                bsz, seqlen, offset, compress_ratio, query_states.device
            )
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs.to(topk_idxs.device)], dim=-1)
    topk_idxs = topk_idxs.int()

    if compress_ratio > 1 and kv_compress is not None:
        kv_states = torch.cat([kv_states, kv_compress], dim=1)

    query_states = query_states.transpose(1, 2)
    kv_states = kv_states.unsqueeze(1)
    attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * softmax_scale

    fill_value = torch.finfo(torch.bfloat16).min
    index_mask = torch.full(
        (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
        fill_value, dtype=torch.bfloat16, device=query_states.device,
    ).scatter_(-1, topk_idxs.unsqueeze(1), 0)

    attn_weights = attn_weights + index_mask[..., :-1]
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(
        query_states.shape[0], -1, query_states.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits.float(), dim=-1).to(combined_logits.dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores, kv_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def _get_attn_dist(query, key, attention_masks, num_attn_heads, attn_scale):
    if num_attn_heads > 1:
        key = key.repeat_interleave(num_attn_heads, dim=1)
    attn = (query @ key.transpose(-1, -2)) * attn_scale
    if attention_masks is not None:
        attn.masked_fill_(attention_masks, float("-inf"))
    attn = F.softmax(attn.float(), dim=-1)
    attn = attn.sum(dim=1)
    return attn


def _li_loss(q, kv_compress, q_indexer, k_indexer, weights, compress_topk_idxs,
             index_score, attention_masks, offset, n_heads, softmax_scale,
             n_layers, layer_id, dsa_loss_tracker, eps=1e-10):
    cti_safe = torch.where(
        compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs - offset
    )
    main_dist = _get_attn_dist(
        q.transpose(1, 2).detach(), kv_compress.unsqueeze(1).detach(),
        attention_masks, n_heads, softmax_scale,
    )
    sel_dist = torch.gather(main_dist, dim=-1, index=cti_safe)
    attn = -(softmax_scale * sel_dist).log_softmax(dim=-1)
    idx_log = (softmax_scale * index_score).log_softmax(dim=-1)
    loss = (attn * (attn.exp() - idx_log.exp())).float()
    mask = compress_topk_idxs != -1
    loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
    if dsa_loss_tracker is not None:
        dsa_loss_tracker[layer_id] = loss.detach()
    return loss


class Attention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        window_size: int = 128
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        n_layers: int = 4
        layer_id: int = 0
        inner_attention: object | None = None
        mask_type: str = "causal"

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.window_size = cfg.window_size
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers

        rd = cfg.rope_head_dim
        hd = cfg.head_dim

        _l2 = partial(nn.init.trunc_normal_, std=0.02)
        _n2 = partial(nn.init.trunc_normal_, mean=1.0, std=0.02)
        _lp = {"weight": _l2}

        self.wq_a = Linear.Config(
            in_features=cfg.dim, out_features=cfg.q_lora_rank, bias=False,
            param_init=_lp,
        ).build()
        self.q_norm = RMSNorm.Config(
            normalized_shape=cfg.q_lora_rank, eps=cfg.norm_eps,
            param_init={"weight": _n2},
        ).build()
        self.wq_b = Linear.Config(
            in_features=cfg.q_lora_rank, out_features=cfg.n_heads * hd, bias=False,
            param_init=_lp,
        ).build()
        self.wkv = Linear.Config(
            in_features=cfg.dim, out_features=hd, bias=False,
            param_init=_lp,
        ).build()
        self.kv_norm = RMSNorm.Config(
            normalized_shape=hd, eps=cfg.norm_eps,
            param_init={"weight": _n2},
        ).build()

        per_group_in = (cfg.n_heads * hd) // cfg.n_groups
        per_group_out = cfg.n_groups * cfg.o_lora_rank
        self.wo_a = nn.Parameter(torch.empty(per_group_out, per_group_in))
        self.wo_b = Linear.Config(
            in_features=per_group_out, out_features=cfg.dim, bias=False,
            param_init=_lp,
        ).build()

        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))

        if self._param_init is None:
            self._param_init = {}
        self._param_init["wo_a"] = _l2
        self._param_init["attn_sink"] = _l2

        if cfg.compress_ratio == 4:
            self.compressor = Compressor.Config(
                dim=cfg.dim, head_dim=hd, rope_head_dim=rd,
                compress_ratio=cfg.compress_ratio, rotate=False,
                norm_eps=cfg.norm_eps,
            ).build()
            self.indexer = Indexer.Config(
                dim=cfg.dim, num_index_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim, index_topk=cfg.index_topk,
                rope_head_dim=rd, q_lora_rank=cfg.q_lora_rank,
                compress_ratio=cfg.compress_ratio, norm_eps=cfg.norm_eps,
            ).build()
        elif cfg.compress_ratio > 1:
            self.compressor_128 = Compressor.Config(
                dim=cfg.dim, head_dim=hd, rope_head_dim=rd,
                compress_ratio=cfg.compress_ratio, rotate=False,
                norm_eps=cfg.norm_eps,
            ).build()

        self._dsa_loss_tracker = None

    def set_dsa_loss_tracker(self, tracker):
        self._dsa_loss_tracker = tracker

    def _pre_phase(self, x, freqs_cis, positions):
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = _rotate(q_rope, freqs_cis, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = _rotate(kv_rope, freqs_cis, positions=positions)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        kv_compress = q_indexer = k_indexer = weights = None

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            q_indexer, k_indexer, weights = self.indexer(
                x.detach(), qr.detach(), freqs_cis, positions=positions,
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(x, freqs_cis, positions=positions)
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(x, freqs_cis, positions=positions)

        return q, kv, kv_compress, q_indexer, k_indexer, weights

    def _inner_phase(self, q, kv, kv_compress, q_indexer, k_indexer, weights,
                     seqlen, attention_masks):
        offset = kv.size(1)
        compress_topk_idxs = index_score = None
        has_li = (
            self.compress_ratio > 1
            and hasattr(self, "indexer")
            and q_indexer is not None
        )
        if has_li:
            compress_topk_idxs, index_score = _li_compute(
                q_indexer, k_indexer, weights, seqlen, offset,
                self.compress_ratio, self.indexer.index_topk,
            )
        o = _sparse_attention(
            q, kv, self.attn_sink, kv_compress, compress_topk_idxs,
            self.window_size, self.compress_ratio, self.softmax_scale,
        )
        if has_li:
            _li_loss(
                q, kv_compress, q_indexer, k_indexer, weights,
                compress_topk_idxs, index_score, attention_masks, offset,
                self.n_heads, self.softmax_scale, self.n_layers, self.layer_id,
                self._dsa_loss_tracker,
            )
        return o

    def _post_phase(self, o, freqs_cis, bsz, seqlen, positions):
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = _rotate(o_rope, freqs_cis, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a = self.wo_a.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.reshape(bsz, seqlen, -1))

    def forward(self, x, freqs_cis, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        freqs_cis = freqs_cis.to(x.device)

        q, kv, kv_compress, q_indexer, k_indexer, weights = self._pre_phase(
            x, freqs_cis, positions=positions,
        )

        o = self._inner_phase(
            q, kv, kv_compress, q_indexer, k_indexer, weights,
            seqlen, attention_masks,
        )

        return self._post_phase(o, freqs_cis, bsz, seqlen, positions=positions)
