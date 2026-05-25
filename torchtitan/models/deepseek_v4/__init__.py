# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    Embedding,
    FeedForward,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
)
from torchtitan.models.common.moe import GroupedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .attention import Attention
from .converters import NPUGMMConverter, NPUPermuteConverter, NPURMSNormConverter
from .model import DeepSeekV4Model, DeepSeekV4TransformerBlock
from .moe import DeepSeekV4MoE, DeepSeekV4Router
from .parallelize import parallelize_deepseek_v4
from .state_dict_adapter import DeepSeekV4StateDictAdapter

__all__ = [
    "parallelize_deepseek_v4",
    "DeepSeekV4Model",
    "deepseek_v4_configs",
    "model_registry",
    "NPUGMMConverter",
    "NPUPermuteConverter",
    "NPURMSNormConverter",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


_SINK_INIT = {"attn_sink": partial(nn.init.trunc_normal_, std=0.02)}
_WOA_INIT = {"wo_a": partial(nn.init.trunc_normal_, std=0.02)}


def _make_v4_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    n_layers: int,
) -> Attention.Config:
    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratio=compress_ratio,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        n_layers=n_layers,
        layer_id=layer_id,
        param_init={
            **_LINEAR_INIT,
            **_SINK_INIT,
            **_WOA_INIT,
        },
    )


def _make_v4_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    score_before_experts: bool = False,
) -> DeepSeekV4MoE.Config:
    router_cfg = DeepSeekV4Router.Config(
        num_experts=num_experts,
        gate=Linear.Config(
            in_features=dim,
            out_features=num_experts,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        top_k=top_k,
        score_func="sqrtsoftplus",
        route_norm=route_norm,
        route_scale=route_scale,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        layer_id=layer_id,
    )

    experts_cfg = make_experts_config(
        dim=dim,
        hidden_dim=moe_inter_dim,
        num_experts=num_experts,
        top_k=top_k,
        param_init={
            "w1": partial(nn.init.trunc_normal_, std=0.02),
            "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
            "w3": partial(nn.init.trunc_normal_, std=0.02),
        },
        score_before_experts=score_before_experts,
        comm_backend=moe_comm_backend,
    )

    shared_experts_cfg = (
        make_ffn_config(
            dim=dim,
            hidden_dim=moe_inter_dim * num_shared_experts,
            w1_param_init=_LINEAR_INIT,
            w2w3_param_init=_depth_init(layer_id),
        )
        if num_shared_experts > 0
        else None
    )

    return DeepSeekV4MoE.Config(
        num_experts=num_experts,
        router=router_cfg,
        experts=experts_cfg,
        shared_experts=shared_experts_cfg,
        load_balance_coeff=load_balance_coeff if layer_id >= n_hash_layers else None,
    )


def _make_v4_dense_config(
    *,
    layer_id: int,
    dim: int,
    hidden_dim: int,
) -> FeedForward.Config:
    return make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w1_param_init=_LINEAR_INIT,
        w2w3_param_init=_depth_init(layer_id),
    )


def _build_v4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratios: tuple[int, ...],
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    score_before_experts: bool = False,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    dense_hidden_dim: int | None = None,
    dense_layers: set[int] | None = None,
) -> list[DeepSeekV4TransformerBlock.Config]:
    if dense_layers is None:
        dense_layers = set()
    if dense_hidden_dim is None:
        dense_hidden_dim = moe_inter_dim * 4

    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 1

        attn_cfg = _make_v4_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            compress_ratio=cr,
            window_size=window_size,
            norm_eps=norm_eps,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            n_layers=n_layers,
        )

        if layer_id in dense_layers:
            ffn_cfg = _make_v4_dense_config(
                layer_id=layer_id,
                dim=dim,
                hidden_dim=dense_hidden_dim,
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = _make_v4_moe_config(
                layer_id=layer_id,
                dim=dim,
                moe_inter_dim=moe_inter_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                vocab_size=vocab_size,
                n_hash_layers=n_hash_layers,
                route_norm=route_norm,
                route_scale=route_scale,
                load_balance_coeff=load_balance_coeff,
                moe_comm_backend=moe_comm_backend,
                score_before_experts=score_before_experts,
            )

        layers.append(
            DeepSeekV4TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
                hc_mult=hc_mult,
                dim=dim,
                norm_eps=norm_eps,
                sinkhorn_iters=sinkhorn_iters,
                hc_eps=hc_eps,
            )
        )
    return layers


def _debugmodel(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> DeepSeekV4Model.Config:
    dim = 256
    n_layers = 4
    vocab_size = 2048
    n_heads = 16
    head_dim = 256
    rope_head_dim = 32
    q_lora_rank = 128
    o_lora_rank = 128
    n_groups = 2
    compress_ratios = (4, 1, 1, 4)
    window_size = 16
    norm_eps = 1e-6
    index_n_heads = 8
    index_head_dim = 64
    index_topk = 16
    moe_inter_dim = 256
    num_experts = 4
    num_shared_experts = 1
    top_k = 3
    n_hash_layers = 2
    route_norm = False
    route_scale = 1.5
    load_balance_coeff = 1e-3
    hc_mult = 4
    sinkhorn_iters = 20
    hc_eps = 1e-6
    dense_layers = set()
    max_seq_len = 2048
    seq_len = 2048
    compress_rope_theta = 40000.0
    original_seq_len = 65536

    _ = non_blocking_capacity_factor

    layers = _build_v4_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratios=compress_ratios,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        moe_comm_backend=moe_comm_backend,
        score_before_experts=False,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        dense_layers=dense_layers,
    )

    return DeepSeekV4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        norm_eps=norm_eps,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_head_dim,
            max_seq_len=seq_len,
            theta=10000.0,
            backend="complex",
            scaling="none",
        ),
        rope_compress=RoPE.Config(
            dim=rope_head_dim,
            max_seq_len=seq_len,
            theta=compress_rope_theta,
            backend="complex",
            scaling="yarn",
            rope_factor=4.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=original_seq_len,
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
    )


deepseek_v4_configs = {
    "debugmodel": _debugmodel,
}


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor not in deepseek_v4_configs:
        raise ValueError(
            f"Unknown deepseek_v4 flavor: {flavor}. "
            f"Available: {list(deepseek_v4_configs.keys())}"
        )
    config = deepseek_v4_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    if converters is not None:
        for converter_cfg in converters:
            converter_cfg.build().convert(config)
    return ModelSpec(
        name="deepseek_v4",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV4StateDictAdapter,
    )
