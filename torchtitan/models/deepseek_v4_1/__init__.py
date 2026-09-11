# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    RMSNorm,
)
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.deepseek_v3.parallelize import (
    # Interim: DeepSeek V3's parallelize applies the generic pieces (declarative
    # sharding, activation checkpointing, compile, FSDP) and is a no-op for tensor and
    # expert parallelism, which this model rejects. A V4.1-specific parallelize that
    # also declares the CSA2 placements belongs in a follow-up.
    parallelize_deepseekv3 as parallelize_deepseek_v4_1,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .attention import Attention, SparseAttention
from .compressor import Compressor
from .indexer import Indexer, IndexerKLLoss
from .mhc import HcPost, HcPre
from .model import DeepSeekV41Model, DeepSeekV41TransformerBlock

__all__ = [
    "parallelize_deepseek_v4_1",
    "DeepSeekV41Model",
    "deepseek_v4_1_configs",
    "model_registry",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_HC_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}
_EXPERTS_INIT = {
    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _served_group_sizes(
    *,
    n_layers: int,
    compress_ratios: tuple[int, ...],
    index_source_layers: tuple[int, ...],
) -> dict[int, int]:
    """Number of compressed layers each index source serves, itself included.

    The indexer distillation averages over the layers that consume its selection, so
    every layer in a group carries ``1 / group_size`` of the loss coefficient. Without
    it the indexer's effective learning rate would depend on the sharing pattern.
    """
    group_sizes: dict[int, int] = {}
    for source in index_source_layers:
        following = [s for s in index_source_layers if s > source]
        end = min(following) if following else n_layers
        group_sizes[source] = sum(
            1 for layer_id in range(source, end) if compress_ratios[layer_id] > 0
        )
    return group_sizes


def _make_attention_config(
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
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    rope: ComplexRoPE.Config,
    rope_compress: ComplexRoPE.Config,
    use_candidates: bool,
    candidate_source_layer: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    indexer_loss_coeff: float | None,
    served_group_sizes: dict[int, int],
) -> Attention.Config:
    owns_k = layer_id in kv_source_layers
    is_index_source = layer_id in index_source_layers
    is_candidate_source = use_candidates and layer_id == candidate_source_layer
    uses_candidates = (
        use_candidates
        and is_index_source
        and not is_candidate_source
        and 0 <= candidate_source_layer < layer_id
    )
    if is_candidate_source and not owns_k:
        raise ValueError(
            "The candidate-pool source must own the compressed KV it scores: "
            f"layer {layer_id} is not in kv_source_layers."
        )

    # Layers with a compressed main KV rotate at the compressed base, which is what
    # keeps the pooled entries' position spacing meaningful.
    layer_rope = rope_compress if compress_ratio > 0 else rope

    compressor = Compressor.Config(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        is_source=owns_k,
        rope=copy.deepcopy(layer_rope) if owns_k else None,
        wkv=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_k
            else None
        ),
        wgate=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_k and compress_ratio > 1
            else None
        ),
        norm=(
            RMSNorm.Config(
                normalized_shape=head_dim, eps=norm_eps, param_init=_NORM_INIT
            )
            if owns_k
            else None
        ),
    )

    indexer = Indexer.Config(
        dim=dim,
        q_lora_rank=q_lora_rank,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        is_source=is_index_source,
        owns_k=owns_k and is_index_source,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        rope=copy.deepcopy(layer_rope) if is_index_source else None,
        wq_b=(
            Linear.Config(
                in_features=q_lora_rank,
                out_features=index_n_heads * index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_index_source
            else None
        ),
        weights_proj=(
            Linear.Config(
                in_features=dim,
                out_features=index_n_heads,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_index_source
            else None
        ),
        wk=(
            Linear.Config(
                in_features=head_dim,
                out_features=index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_k and is_index_source
            else None
        ),
        k_norm=(
            RMSNorm.Config(
                normalized_shape=index_head_dim, eps=norm_eps, param_init=_NORM_INIT
            )
            if owns_k and is_index_source
            else None
        ),
    )

    # The distillation loss is attached on every layer that selects compressed entries,
    # including the reuse layers: their teachers are what make the shared indexer
    # predict a selection that is jointly useful for the whole group.
    aux_loss = None
    if indexer_loss_coeff is not None and compress_ratio > 0:
        source = max((s for s in index_source_layers if s <= layer_id), default=-1)
        if source >= 0:
            aux_loss = IndexerKLLoss.Config(
                coeff=indexer_loss_coeff / served_group_sizes[source],
                reduce_mesh="batch",
            )

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        inner_attention=SparseAttention.Config(
            window_size=window_size,
            softmax_scale=head_dim**-0.5,
            aux_loss=aux_loss,
        ),
        rope=copy.deepcopy(layer_rope),
        compressor=compressor,
        indexer=indexer,
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank, eps=norm_eps, param_init=_NORM_INIT
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=norm_eps, param_init=_NORM_INIT
        ),
        wo_a=Linear.Config(
            in_features=n_heads * head_dim // n_groups,
            out_features=n_groups * o_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=n_groups * o_lora_rank,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        attn_sink=Linear.Config(
            in_features=1,
            out_features=n_heads,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
    )


def _build_layers(
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
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    rope: ComplexRoPE.Config,
    rope_compress: ComplexRoPE.Config,
    use_candidates: bool,
    candidate_source_layer: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    indexer_loss_coeff: float | None,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    load_balance_coeff: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> list[DeepSeekV41TransformerBlock.Config]:
    served_group_sizes = _served_group_sizes(
        n_layers=n_layers,
        compress_ratios=compress_ratios,
        index_source_layers=index_source_layers,
    )
    layers = []
    for layer_id in range(n_layers):
        moe_cfg = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init=_LINEAR_INIT,
                top_k=top_k,
                score_func="sqrtsoftplus",
                route_norm=True,
                route_scale=route_scale,
            ),
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=moe_inter_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init=_EXPERTS_INIT,
                comm_backend="standard",
            ),
            shared_experts=(
                make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_inter_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
                )
                if num_shared_experts > 0
                else None
            ),
            load_balance_coeff=load_balance_coeff,
        )

        layers.append(
            DeepSeekV41TransformerBlock.Config(
                attention=_make_attention_config(
                    layer_id=layer_id,
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                    q_lora_rank=q_lora_rank,
                    o_lora_rank=o_lora_rank,
                    n_groups=n_groups,
                    compress_ratio=compress_ratios[layer_id],
                    window_size=window_size,
                    norm_eps=norm_eps,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    index_topk=index_topk,
                    kv_source_layers=kv_source_layers,
                    index_source_layers=index_source_layers,
                    rope=rope,
                    rope_compress=rope_compress,
                    use_candidates=use_candidates,
                    candidate_source_layer=candidate_source_layer,
                    candidate_topk_blocks=candidate_topk_blocks,
                    candidate_block_size=candidate_block_size,
                    indexer_loss_coeff=indexer_loss_coeff,
                    served_group_sizes=served_group_sizes,
                ),
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT
                ),
                moe=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    dim=dim,
                    hc_mult=hc_mult,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    dim=dim,
                    hc_mult=hc_mult,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )
    return layers


def _debugmodel(
    *,
    seq_len: int,
    use_candidates: bool = False,
    indexer_loss_coeff: float | None = 0.01,
) -> DeepSeekV41Model.Config:
    """Small model that exercises every structural variant.

    Layers 0-1 are sliding-window only; layers 2-4 compress 2:1 with layer 2 as the
    source; layers 5-7 have ratio 1 with layer 5 as the source. That gives every layer
    kind: a Full-mode source (2, 5), reuse layers that borrow both compressed KV and
    top-k (3, 4, 6), and a re-indexing layer that borrows the keys but selects its own
    top-k (7). With ``use_candidates`` the pool built by layer 5 restricts layer 7.
    """
    dim = 256
    n_layers = 8
    vocab_size = 2048
    n_heads = 8
    head_dim = 64
    rope_head_dim = 16
    q_lora_rank = 64
    o_lora_rank = 32
    n_groups = 2
    compress_ratios = (0, 0, 2, 2, 2, 1, 1, 1)
    window_size = 8
    norm_eps = 1e-6
    index_n_heads = 4
    index_head_dim = 32
    index_topk = 4
    kv_source_layers = (2, 5)
    index_source_layers = (2, 5, 7)
    candidate_source_layer = 5
    candidate_topk_blocks = 4
    candidate_block_size = 2
    moe_inter_dim = 128
    num_experts = 4
    num_shared_experts = 1
    top_k = 2
    route_scale = 1.5
    load_balance_coeff = 1e-3
    hc_mult = 4
    sinkhorn_iters = 3
    hc_eps = 1e-6
    compress_rope_theta = 160000.0
    original_seq_len = 65536

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=seq_len,
        theta=10000.0,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    layers = _build_layers(
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
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        rope=rope,
        rope_compress=rope_compress,
        use_candidates=use_candidates,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        indexer_loss_coeff=indexer_loss_coeff,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )

    return DeepSeekV41Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        norm_eps=norm_eps,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        n_layers=n_layers,
        compress_ratios=compress_ratios,
        use_candidates=use_candidates,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


deepseek_v4_1_configs = {
    "debugmodel": (_debugmodel, 4096),
}


def _post_optimizer_build_fn(optimizers, model_parts, parallel_dims):
    """Register the step pre-hooks for MoE load balancing and aux-loss accumulators."""
    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)
    register_aux_loss_zero_hook(optimizers, model_parts, parallel_dims)


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    use_candidates: bool = False,
    indexer_loss_coeff: float | None = 0.01,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor not in deepseek_v4_1_configs:
        raise ValueError(
            f"Unknown deepseek_v4_1 flavor: {flavor}. "
            f"Available: {list(deepseek_v4_1_configs.keys())}"
        )
    get_config, max_context_len = deepseek_v4_1_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        seq_len=context_len,
        use_candidates=use_candidates,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return ModelSpec(
        name="deepseek_v4_1",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_deepseek_v4_1,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_post_optimizer_build_fn,
        state_dict_adapter=None,
    )
