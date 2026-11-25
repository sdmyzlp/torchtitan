# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash text backbone and reference KDA kernels.

Shape suffixes: T = packed tokens, D = hidden width, M = mHC streams,
H = heads, K = head width, C = channels, P = pools, W = convolution width.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn
from torch.export import register_dataclass

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attention.attention import (
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.attn_gym import bound_gate, chunk_kda
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE


class LowRankProjection(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        a: Linear.Config
        b: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.a = config.a.build()
        self.b = config.b.build()

    def forward(self, x_TD):
        return self.b(self.a(x_TD))


@dataclass(frozen=True, slots=True)
class KDAAttentionMetadata(LinearAttentionMetadata):
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    offsets: tuple[int, ...]

    def annotate_spmd_types(self):
        LinearAttentionMetadata.annotate_spmd_types(self)
        assert self.varlen is not None
        spmd.assert_type_like(self.cu_seqlens, self.varlen.cu_seq_q)


register_dataclass(KDAAttentionMetadata)


class ReferenceKDAKernel(Module):
    """Use Attention Gym's differentiable reference, including on CPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float = -5.0

        def __post_init__(self):
            if not -5.0 <= self.lower_bound < 0.0:
                raise ValueError("KDA lower_bound must be in [-5, 0).")

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound

    def prepare_inputs(
        self, q_1THK, k_1THK, raw_gate_1THK, raw_beta_1TH, A_log_H, dt_bias_HK
    ):
        gate_1THK = bound_gate(
            raw_gate_1THK,
            A_log_H.float(),
            dt_bias_HK.float(),
            lower_bound=self.lower_bound,
            impl="reference",
        )
        q_1THK, k_1THK = (
            x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            for x in (q_1THK, k_1THK)
        )
        return q_1THK, k_1THK, gate_1THK, raw_beta_1TH.sigmoid().float()

    def forward(
        self,
        q_1THK,
        k_1THK,
        v_1THV,
        raw_gate_1THK,
        raw_beta_1TH,
        A_log_H,
        dt_bias_HK,
        *,
        attention_metadata: KDAAttentionMetadata,
    ):
        q_1THK, k_1THK, gate_1THK, beta_1TH = self.prepare_inputs(
            q_1THK, k_1THK, raw_gate_1THK, raw_beta_1TH, A_log_H, dt_bias_HK
        )
        return chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV.float(),
            gate_1THK,
            beta_1TH,
            cu_seqlens=attention_metadata.cu_seqlens,
            impl="reference",
        )[0].to(v_1THV.dtype)


class ReferenceInnerKDA(InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        head_dim: int
        conv_kernel_size: int
        kernel: ReferenceKDAKernel.Config

        def __post_init__(self):
            if self.head_dim != 128:
                raise ValueError("GLM KDA requires head_dim=128.")

        def build_attention_metadata(
            self,
            positions,
            *,
            padding_mask=None,
            max_num_documents=None,
            max_context_length=None,
        ):
            varlen = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            cu_seqlens = varlen.cu_seq_q.unique_consecutive()
            cu_seqlens_cpu = cu_seqlens.cpu()
            return KDAAttentionMetadata(
                varlen=varlen,
                num_conv_history_tokens=self.conv_kernel_size - 1,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                offsets=tuple(cu_seqlens_cpu.tolist()),
            )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.kernel = config.kernel.build()

    def forward(
        self,
        query_TC,
        key_TC,
        value_TC,
        raw_gate_THK,
        raw_beta_TH,
        conv_q_weight_C1W,
        conv_k_weight_C1W,
        conv_v_weight_C1W,
        A_log_H,
        dt_bias_HK,
        *,
        attention_metadata: KDAAttentionMetadata,
    ):
        qkv_1TC = torch.cat((query_TC, key_TC, value_TC), dim=-1)[None]
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W)
        )
        qkv_1TC = self.short_convolution(
            qkv_1TC, conv_weight_C1W, attention_metadata=attention_metadata
        )
        q_1THK, k_1THK, v_1THV = (
            part.unflatten(-1, (-1, self.head_dim)) for part in qkv_1TC.chunk(3, -1)
        )
        return self.kernel(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_THK[None],
            raw_beta_TH[None],
            A_log_H,
            dt_bias_HK,
            attention_metadata=attention_metadata,
        )[0]

    def short_convolution(
        self, qkv_1TC, conv_weight_C1W, *, attention_metadata: KDAAttentionMetadata
    ):
        # Attention Gym's short convolution has no CPU/reference backend.
        bounds = attention_metadata.offsets
        out_1TC = torch.zeros_like(qkv_1TC)
        for start, end in zip(bounds[:-1], bounds[1:]):
            out_1TC[:, start:end] = F.silu(
                F.conv1d(
                    qkv_1TC[:, start:end].transpose(1, 2).to(conv_weight_C1W.dtype),
                    conv_weight_C1W,
                    padding=conv_weight_C1W.shape[-1] - 1,
                    groups=conv_weight_C1W.shape[0],
                )[:, :, : end - start].transpose(1, 2)
            ).to(qkv_1TC.dtype)
        return out_1TC


class GlmKDA(KDA):
    @dataclass(kw_only=True, slots=True)
    class Config(KDA.Config):
        output_gate: LowRankProjection.Config  # pyrefly: ignore [bad-override]

    def forward(
        self,
        x_TD,
        attention_metadata=None,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        return super().forward(x_TD, attention_metadata, positions)


class GlmHcPre(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hc_mult: int = 4
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-5
        preserve_dtype_params: tuple[str, ...] = ("hc_base", "hc_scale")

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps, self.norm_eps = config.eps, config.norm_eps
        self.sinkhorn_iters = config.sinkhorn_iters
        mixes = (2 + config.hc_mult) * config.hc_mult
        self.hc_fn = nn.Parameter(torch.empty(mixes, config.hc_mult * config.dim))
        self.hc_base = nn.Parameter(torch.empty(mixes))
        self.hc_scale = nn.Parameter(torch.empty(3))

    def forward(self, x_TMD):
        flat_TD = x_TMD.flatten(-2).float()
        flat_TD = flat_TD * torch.rsqrt(
            flat_TD.square().mean(-1, keepdim=True) + self.norm_eps
        )
        # Keep the mixer in FP32 even when the surrounding projections use autocast.
        with torch.autocast(device_type=x_TMD.device.type, enabled=False):
            mixes_TK = F.linear(flat_TD, self.hc_fn.float())
        m = self.hc_mult
        pre_TM, post_TM, comb_TK = mixes_TK.split((m, m, m * m), -1)
        pre_TM = (pre_TM * self.hc_scale[0] + self.hc_base[:m]).sigmoid() + self.eps
        post_TM = 2 * (post_TM * self.hc_scale[1] + self.hc_base[m : 2 * m]).sigmoid()
        comb_TMM = (comb_TK * self.hc_scale[2] + self.hc_base[2 * m :]).unflatten(
            -1, (m, m)
        )
        comb_TMM = comb_TMM.softmax(-1) + self.eps
        comb_TMM = comb_TMM / (comb_TMM.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_TMM = comb_TMM / (comb_TMM.sum(-1, keepdim=True) + self.eps)
            comb_TMM = comb_TMM / (comb_TMM.sum(-2, keepdim=True) + self.eps)
        return (pre_TM[..., None] * x_TMD).sum(-2).to(x_TMD.dtype), post_TM, comb_TMM


class GlmHcPost(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x_TD, residual_TMD, post_TM, comb_TMM):
        remat.recompute_needs_tensor(x_TD)
        return (
            post_TM.to(x_TD.dtype)[..., None] * x_TD[:, None]
            + comb_TMM.to(x_TD.dtype).transpose(-1, -2) @ residual_TMD
        )


class GlmTransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        hc_attn_pre: GlmHcPre.Config
        hc_ffn_pre: GlmHcPre.Config
        hc_post: GlmHcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.feed_forward = config.feed_forward.build()
        self.hc_attn_pre = config.hc_attn_pre.build()
        self.hc_ffn_pre = config.hc_ffn_pre.build()
        self.hc_post = config.hc_post.build()

    def forward(
        self,
        x_TMD,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        x_TD, post_TM, comb_TMM = self.hc_attn_pre(x_TMD)
        out_TD = self.attention(
            self.attention_norm(x_TD),
            attention_metadata,
            positions,
            padding_mask=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        x_TMD = self.hc_post(out_TD, x_TMD, post_TM, comb_TMM)
        x_TD, post_TM, comb_TMM = self.hc_ffn_pre(x_TMD)
        x_TD = self.ffn_norm(x_TD)
        if isinstance(self.feed_forward, MoE):
            out_TD = self.feed_forward(
                x_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            out_TD = self.feed_forward(x_TD)
        return self.hc_post(out_TD, x_TMD, post_TM, comb_TMM)


class Glm53FlashModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        hc_mult: int = 4

    def __init__(self, config: Config):
        super().__init__(config)
        self.hc_mult = config.hc_mult

    def forward_embeddings(self, h, *args, **kwargs):
        return super().forward_embeddings(
            h[:, None].expand(-1, self.hc_mult, -1), *args, **kwargs
        )

    def _pre_lm_head_norm_and_gather(self, h):
        return super()._pre_lm_head_norm_and_gather(h.mean(-2))

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook

        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
