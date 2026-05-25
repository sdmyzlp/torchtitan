# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig
from torchtitan.protocols.types import MeshAxisName

DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
CP = MeshAxisName.CP
TP = MeshAxisName.TP

_dense_param_rep = dense_param_placement(tp=Replicate())
_act_shard0_tp_rep = dense_activation_placement(tp=Replicate())


def _replicate_param_config(param_names):
    return ShardingConfig(
        state_shardings={n: _dense_param_rep for n in param_names},
    )


def set_deepseek_v4_attention_sharding(attention_cfg, *, enable_sp):
    at = attention_cfg
    attn_x_placement = Shard(1) if enable_sp else Replicate()

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
        },
    )

    def _try_set(cfg, fn):
        if cfg is not None and hasattr(cfg, "sharding_config"):
            cfg.sharding_config = fn()

    _try_set(getattr(at, "wq_a", None), lambda: _replicate_param_config(["weight"]))
    _try_set(getattr(at, "q_norm", None), lambda: norm_config(enable_sp=False))
    _try_set(getattr(at, "wq_b", None), lambda: colwise_config())
    _try_set(getattr(at, "wkv", None), lambda: _replicate_param_config(["weight"]))
    _try_set(getattr(at, "kv_norm", None), lambda: norm_config(enable_sp=False))
    _try_set(getattr(at, "wo_b", None), lambda: rowwise_config(output_sp=enable_sp))

    if hasattr(at, "compressor") and at.compressor is not None:
        set_compressor_sharding(at.compressor)
    if hasattr(at, "compressor_128") and at.compressor_128 is not None:
        set_compressor_sharding(at.compressor_128)
    if hasattr(at, "indexer") and at.indexer is not None:
        set_indexer_sharding(at.indexer)


def set_compressor_sharding(compressor_cfg):
    compressor_cfg.sharding_config = _replicate_param_config(["weight"])
    compressor_cfg.wkv.sharding_config = _replicate_param_config(["weight"])
    compressor_cfg.wgate.sharding_config = _replicate_param_config(["weight"])
    compressor_cfg.norm.sharding_config = norm_config(enable_sp=False)
    compressor_cfg.ape_rep = _dense_param_rep


def set_indexer_sharding(indexer_cfg):
    indexer_cfg.wq_b.sharding_config = _replicate_param_config(["weight"])
    indexer_cfg.weights_proj.sharding_config = _replicate_param_config(["weight"])
    set_compressor_sharding(indexer_cfg.compressor)


def set_deepseek_v4_block_sharding(block_cfg, *, enable_sp):
    hc_rep = _replicate_param_config([
        "hc_attn_fn", "hc_ffn_fn",
        "hc_attn_base", "hc_ffn_base",
        "hc_attn_scale", "hc_ffn_scale",
    ])
    block_cfg.sharding_config = hc_rep

    block_cfg.attention_norm.sharding_config = norm_config(enable_sp=False)
    block_cfg.ffn_norm.sharding_config = norm_config(enable_sp=False)

    set_deepseek_v4_attention_sharding(block_cfg.attention, enable_sp=enable_sp)

    if block_cfg.moe is not None:
        block_cfg.moe.sharding_config = ShardingConfig(
            in_src_shardings={
                "x": dense_activation_placement(tp=Replicate()),
            },
            out_dst_shardings=dense_activation_placement(tp=Replicate()),
        )
        block_cfg.moe.router.gate.sharding_config = _replicate_param_config(["weight"])
    elif block_cfg.feed_forward is not None:
        block_cfg.feed_forward.sharding_config = ShardingConfig(
            in_src_shardings={
                "x": dense_activation_placement(tp=Replicate()),
            },
            in_dst_shardings={
                "x": dense_activation_placement(tp=Replicate()),
            },
        )
        block_cfg.feed_forward.w1.sharding_config = colwise_config()
        block_cfg.feed_forward.w3.sharding_config = colwise_config()
        block_cfg.feed_forward.w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_deepseek_v4_sharding_config(config, *, loss_parallel, enable_sp):
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )

    hc_rep = _replicate_param_config(["hc_head_fn", "hc_head_base", "hc_head_scale"])
    model_sharding = config.sharding_config or ShardingConfig()
    model_sharding.state_shardings.update(hc_rep.state_shardings)
    model_sharding.state_shardings["freqs_cis_compress"] = _dense_param_rep
    config.sharding_config = model_sharding

    for layer_cfg in config.layers:
        set_deepseek_v4_block_sharding(layer_cfg, enable_sp=enable_sp)
