# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import dataclasses as dc

import torch
from torch import nn

from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import ModuleDict

from .attention import Attention
from .mhc import HcHead, HcPost, HcPre
from .moe import DeepSeekV4MoE


class DeepSeekV4TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        hc_mult: int = 4
        dim: int
        norm_eps: float = 1e-6
        sinkhorn_iters: int = 20
        hc_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = (
            cfg.attention_norm.build() if cfg.attention_norm is not None else None
        )
        self.ffn_norm = (
            cfg.ffn_norm.build() if cfg.ffn_norm is not None else None
        )
        if cfg.moe is not None:
            self.moe = cfg.moe.build()
            self.feed_forward = None
            self.moe_enabled = True
        else:
            self.moe = None
            self.feed_forward = (
                cfg.feed_forward.build() if cfg.feed_forward is not None else None
            )
            self.moe_enabled = False

        self.hc_mult = cfg.hc_mult
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.dim

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

        self.hc_pre = HcPre.Config(
            hc_mult=cfg.hc_mult,
            dim=cfg.dim,
            sinkhorn_iters=cfg.sinkhorn_iters,
            eps=cfg.hc_eps,
            norm_eps=cfg.norm_eps,
        ).build()
        self.hc_post = HcPost.Config().build()

        if self._param_init is None:
            self._param_init = {}
        self._param_init.update({
            "hc_attn_fn": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_fn": partial(_init_trunc_normal, std=0.02),
            "hc_attn_base": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_base": partial(_init_trunc_normal, std=0.02),
            "hc_attn_scale": partial(_init_trunc_normal, std=0.02),
            "hc_ffn_scale": partial(_init_trunc_normal, std=0.02),
        })

    def _mhc_step(self, x, residual, hc_fn, hc_scale, hc_base, norm, fn, *a, **kw):
        x, post, comb = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        if norm is not None:
            x = norm(x)
        x = fn(x, *a, **kw)
        x = self.hc_post(x, residual, post, comb)
        return x

    def forward(self, x, input_ids, freqs_cis, attention_masks=None, positions=None):
        residual = x
        x = self._mhc_step(
            x, residual,
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.attention_norm,
            self.attention, freqs_cis, attention_masks, positions,
        )

        residual = x
        module = self.moe if self.moe is not None else self.feed_forward
        x = self._mhc_step(
            x, residual,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.ffn_norm,
            module, input_ids,
        )
        return x


class DeepSeekV4Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        hc_mult: int = 4
        compress_ratios: tuple[int, ...] = (1, 1, 4, 4)
        n_layers: int = 4
        rope_compress: RoPE.Config
        dim: int
        vocab_size: int
        norm_eps: float = 1e-6

        def update_from_config(self, *, trainer_config, **kwargs):
            parallelism = trainer_config.parallelism
            training = trainer_config.training
            debug = getattr(trainer_config, "debug", None)

            seq_len = training.seq_len
            self.rope = dc.replace(self.rope, max_seq_len=seq_len)
            self.rope_compress = dc.replace(self.rope_compress, max_seq_len=seq_len)

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(
                            f"n_heads ({n_heads}) must be divisible by tp ({tp})"
                        )
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(
                            f"n_groups ({n_groups}) must be divisible by tp ({tp})"
                        )

            if debug is not None and debug.moe_force_load_balance:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    if layer_cfg.moe is not None:
                        layer_cfg.moe.router._debug_force_load_balance = True

            from .sharding import set_deepseek_v4_sharding_config
            set_deepseek_v4_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(self, model, seq_len):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_embed_params = sum(
                p.numel()
                for n, p in model.named_parameters()
                if p.requires_grad and "tok_embeddings" not in n and "lm_head" not in n
            )
            n_layers = self.n_layers
            head_dim = self.layers[0].attention.head_dim
            n_heads = self.layers[0].attention.n_heads
            flops_per_token = 6 * non_embed_params + 12 * n_layers * n_heads * head_dim * seq_len
            return total_params, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.rope_compress = cfg.rope_compress.build()
        self.register_buffer(
            "freqs_cis_compress", self.rope_compress.cache, persistent=False
        )

        self.hc_mult = cfg.hc_mult
        self.compress_ratios = list(cfg.compress_ratios)[: cfg.n_layers]
        self.n_main_layers = cfg.n_layers

        hc_dim = cfg.hc_mult * cfg.dim
        self.hc_head_fn = nn.Parameter(torch.empty(cfg.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(cfg.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

        self.hc_head = HcHead.Config(
            hc_mult=cfg.hc_mult,
            dim=cfg.dim,
            norm_eps=cfg.norm_eps,
            eps=1e-6,
        ).build()

        if self._param_init is None:
            self._param_init = {}
        self._param_init.update({
            "hc_head_fn": partial(_init_trunc_normal, std=0.02),
            "hc_head_base": partial(_init_trunc_normal, std=0.02),
            "hc_head_scale": partial(_init_trunc_normal, std=0.02),
        })

        self._dsa_loss_tracker = {}

    def _init_self_buffers(self, *, buffer_device=None):
        assert buffer_device is None or buffer_device.type != "meta"
        if self.rope is not None:
            self.freqs_cis = self.rope.cache
        else:
            rope = self.config.rope.build()
            rope._init_self_buffers(buffer_device=buffer_device)
            self.freqs_cis = rope.cache
        if self.rope_compress is not None:
            self.freqs_cis_compress = self.rope_compress.cache

    def get_dsa_losses(self):
        losses = dict(self._dsa_loss_tracker)
        self._dsa_loss_tracker.clear()
        return losses

    def get_attention_masks(self, positions):
        return None

    def forward(self, tokens, attention_masks=None, positions=None):
        seq_len = tokens.shape[1]
        input_ids = tokens[:, :seq_len].detach().long()

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for i in range(self.n_main_layers):
            layer = self.layers[str(i)]
            cr = self.compress_ratios[i] if i < len(self.compress_ratios) else 1
            freqs = self.freqs_cis_compress if cr > 1 else self.freqs_cis
            h = layer(h, input_ids, freqs, attention_masks, positions)

        h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        h = self.norm(h.float()) if self.norm is not None else h

        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output


def _init_trunc_normal(x, std=0.02):
    nn.init.trunc_normal_(x, mean=0.0, std=std)
