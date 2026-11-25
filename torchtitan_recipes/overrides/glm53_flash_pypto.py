# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM KDA on Ascend 950 using PyPTO-Gym's fused forward/backward pair.

Put the PyPTO-Gym checkout on PYTHONPATH and source the CANN environment
before enabling ``glm53_flash_pypto.flash_kda``. The public ``chunk_kda``
entry owns both forward stages, saved activations, and both backward stages.
T = packed tokens, H = heads, K = key width, V = value width (K = V = 128).
"""

from dataclasses import dataclass

from torchtitan.config import derive, override
from torchtitan.models.glm53_flash.model import ReferenceKDAKernel


class PyPTOKDAKernel(ReferenceKDAKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(ReferenceKDAKernel.Config):
        pass

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
        attention_metadata,
    ):
        from ops.pypto_pro.kimi_k3.flash_kda.op_kernel import chunk_kda

        return chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK,
            lower_bound=self.lower_bound,
            cu_seqlens=attention_metadata.cu_seqlens,
            cu_seqlens_cpu=attention_metadata.cu_seqlens_cpu,
        )[0]


@override(target=ReferenceKDAKernel.Config)
def flash_kda(cfg: ReferenceKDAKernel.Config):
    return derive(cfg, PyPTOKDAKernel.Config)
