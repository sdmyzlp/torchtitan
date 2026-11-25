# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The released text backbone and a small, structurally identical debug flavor."""

import math
from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common import (
    Conv1d,
    GatedRMSNorm,
    LayerNorm,
    LimitedSwiGLU,
    Linear,
    RMSNorm,
    Sigmoid,
    VocabParallelEmbedding,
)
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher
from .attention import (
    Compressor,
    GlmDSA,
    Indexer,
    IndexerLoss,
    InnerIndexer,
    SparseAttention,
)

from .model import (
    Glm53FlashModel,
    GlmHcPost,
    GlmHcPre,
    GlmKDA,
    GlmTransformerBlock,
    LowRankProjection,
    ReferenceInnerKDA,
    ReferenceKDAKernel,
)


_INIT: dict[str, Callable] = {"weight": partial(nn.init.normal_, std=0.02)}
MODEL_FLAVORS = {"debugmodel_4layer": 4096, "debugmodel": 256, "glm53_flash": 1048576}


def _linear(in_features, out_features):
    return Linear.Config(
        in_features=in_features, out_features=out_features, param_init=_INIT
    )


def _norm(dim):
    return RMSNorm.Config(
        normalized_shape=dim, eps=1e-5, param_init={"weight": nn.init.ones_}
    )


def _dt_bias_init(tensor):
    with torch.no_grad():
        dt = (
            torch.empty_like(tensor)
            .uniform_(math.log(1e-3), math.log(1e-1))
            .exp()
            .clamp_min(1e-4)
        )
        tensor.copy_(dt + torch.log(-torch.expm1(-dt)))


def _kda(dim, heads):
    channels = heads * 128

    def conv():
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=4,
            groups=channels,
            bias=False,
            param_init=_INIT,
            preserve_dtype_params=("weight",),
        )

    return GlmKDA.Config(
        n_heads=heads,
        head_dim=128,
        conv_kernel_size=4,
        q_proj=_linear(dim, channels),
        k_proj=_linear(dim, channels),
        v_proj=_linear(dim, channels),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=_linear(dim, 128),
        forget_b=_linear(128, channels),
        beta=_linear(dim, heads),
        output_gate=LowRankProjection.Config(
            a=_linear(dim, 128), b=_linear(128, channels)
        ),
        output_norm=GatedRMSNorm.Config(
            dim=128,
            eps=1e-5,
            activation_fn=Sigmoid.Config(),
            param_init={"weight": nn.init.ones_},
        ),
        output_proj=_linear(channels, dim),
        inner_attention=ReferenceInnerKDA.Config(
            head_dim=128, conv_kernel_size=4, kernel=ReferenceKDAKernel.Config()
        ),
        param_init={"A_log": nn.init.zeros_, "dt_bias": _dt_bias_init},
    )


def _dsa(dim, heads, q_rank, kv_rank, head_dim, index_heads, index_dim, topk):
    return GlmDSA.Config(
        n_heads=heads,
        head_dim=head_dim,
        wq_a=_linear(dim, q_rank),
        q_norm=_norm(q_rank),
        wq_b=_linear(q_rank, heads * head_dim),
        wkv_a=_linear(dim, kv_rank),
        kv_norm=_norm(kv_rank),
        wkv_b=_linear(kv_rank, heads * head_dim * 2),
        wo=_linear(heads * head_dim, dim),
        indexer=Indexer.Config(
            num_index_heads=index_heads,
            index_head_dim=index_dim,
            wq_b=_linear(q_rank, index_heads * index_dim),
            wk=_linear(dim, index_dim),
            k_norm=LayerNorm.Config(
                normalized_shape=index_dim,
                eps=1e-6,
                param_init={"weight": nn.init.ones_, "bias": nn.init.zeros_},
            ),
            weights_proj=_linear(dim, index_heads),
            compressor=Compressor.Config(
                ratio=4,
                head_dim=index_dim,
                gate=Linear.Config(
                    in_features=dim,
                    out_features=index_dim,
                    param_init={"weight": nn.init.ones_},
                ),
                param_init={"ape": nn.init.zeros_},
            ),
            inner_indexer=InnerIndexer.Config(ratio=4, topk=topk // 4),
        ),
        inner_attention=SparseAttention.Config(
            ratio=4,
            aux_loss=IndexerLoss.Config(coeff=0.01, reduce_mesh="loss"),
        ),
    )


def _ffn(dim, hidden_dim):
    config = make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w13_param_init=_INIT,
        w2_param_init=_INIT,
        enable_sp=False,
    )
    config.activation_fn = LimitedSwiGLU.Config(limit=10.0)
    return config


def _moe(dim, hidden_dim, experts, topk):
    routed = make_routed_experts_config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=experts,
        top_k=topk,
        param_init={name: _INIT["weight"] for name in ("w1_EFD", "w2_EDF", "w3_EFD")},
        enable_ep=False,
    )
    routed.activation_fn = LimitedSwiGLU.Config(limit=10.0)
    routed.token_dispatcher = LocalTokenDispatcher.Config(
        num_experts=experts, top_k=topk
    )
    config = make_moe_config(
        num_experts=experts,
        routed_experts=routed,
        shared_experts=_ffn(dim, hidden_dim),
        router=make_router_config(
            dim=dim,
            num_experts=experts,
            gate_param_init=_INIT,
            score_func=Sigmoid.Config(),
            top_k=topk,
            route_norm=True,
            route_scale=2.5,
        ),
        load_balance_coeff=1e-3,
        enable_sp=False,
    )
    # Preserve checkpoint correction biases; this structural port adds no balancing objective.
    config.freeze_expert_bias = True
    return config


def build_model_config(
    flavor: str = "debugmodel",
    *,
    seq_len: int | None = None,
    parallelism: ParallelismConfig | None = None,
) -> Glm53FlashModel.Config:
    """Build the reference model with optional contiguous context parallelism."""
    if flavor not in MODEL_FLAVORS:
        raise ValueError(f"Unknown GLM-5.3-Flash flavor: {flavor}")
    p = parallelism or ParallelismConfig()
    if any(
        getattr(p, name) > 1
        for name in (
            "data_parallel_replicate_degree",
            "data_parallel_shard_degree",
            "tensor_parallel_degree",
            "pipeline_parallel_degree",
            "expert_parallel_degree",
        )
    ):
        raise NotImplementedError(
            "This minimal port supports single-device and contiguous context parallel execution."
        )
    seq_len = MODEL_FLAVORS[flavor] if seq_len is None else seq_len
    if not 0 < seq_len <= MODEL_FLAVORS[flavor]:
        raise ValueError(f"seq_len must be in [1, {MODEL_FLAVORS[flavor]}]")
    debug = flavor == "debugmodel"
    reduced = flavor == "debugmodel_4layer"
    dim, vocab, layers = (64, 2048, 5) if debug else (4096, 154880, 45)
    if reduced:
        vocab, layers = 2048, 4
    heads, q_rank, kv_rank, head_dim = (
        (2, 32, 32, 16) if debug else (64, 1536, 512, 256)
    )
    index_heads, index_dim, index_topk = (2, 16, 8) if debug else (32, 128, 2048)
    experts, expert_topk = (4, 2) if debug or reduced else (288, 8)

    def hc():
        return GlmHcPre.Config(
            dim=dim,
            hc_mult=4,
            norm_eps=1e-5,
            eps=1e-6,
            sinkhorn_iters=20,
            param_init={
                "hc_fn": _INIT["weight"],
                "hc_base": nn.init.zeros_,
                "hc_scale": nn.init.ones_,
            },
        )

    config = Glm53FlashModel.Config(
        dim=dim,
        vocab_size=vocab,
        max_context_length=seq_len,
        local_compile=LocalCompileConfig(regions=[]),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab, embedding_dim=dim, param_init=_INIT
        ),
        lm_head=_linear(dim, vocab),
        norm=_norm(dim),
        layers=[
            GlmTransformerBlock.Config(
                attention=_dsa(
                    dim,
                    heads,
                    q_rank,
                    kv_rank,
                    head_dim,
                    index_heads,
                    index_dim,
                    index_topk,
                )
                if i % 4 == 3
                else _kda(dim, heads),
                attention_norm=_norm(dim),
                ffn_norm=_norm(dim),
                feed_forward=_ffn(dim, dim * 3)
                if i < 3
                else _moe(dim, 32 if debug else 2048, experts, expert_topk),
                hc_attn_pre=hc(),
                hc_ffn_pre=hc(),
                hc_post=GlmHcPost.Config(),
            )
            for i in range(layers)
        ],
    )
    if p.context_parallel_degree > 1:
        from .parallelize import GlmContextParallelTransform

        if p.context_parallel_load_balancer is not None:
            raise ValueError(
                "GLM context parallel requires contiguous shards (no load balancer)."
            )
        GlmContextParallelTransform().transform(config)
    return config
