# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contiguous DSA query shards with all-gathered latent KV and compressed keys.

T = query tokens, H = attention heads, L = KV latent width,
I = index heads, E = index width, P = compressed pools.
"""

from dataclasses import dataclass, fields

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.export import register_dataclass

from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention.cp_attention import CPInnerAttention

from .attention import InnerIndexer, SparseAttention, SparseMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class CPSparseMetadata(SparseMetadata):
    def annotate_spmd_types(self) -> None:
        spmd.assert_type(
            self.cu_seq_q,
            {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
            partition_spec=spmd.PartitionSpec("dp"),
        )
        spmd.assert_type((self.cu_seq_k, self.cu_seq_pools), self._OFFSETS_SPMD_TYPE)


register_dataclass(CPSparseMetadata)


class AllGatherInnerIndexer(InnerIndexer):
    """Score local queries against all compressed keys; backward sums key grads."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, q_TIE, k_PE, w_TI, *, attention_metadata):
        (k_PE,) = cp_all_gather_tensors((k_PE,), reduce_dtype=torch.float32)
        return super().forward(q_TIE, k_PE, w_TI, attention_metadata=attention_metadata)


class AllGatherKVSparseAttention(
    CPInnerAttention[SparseMetadata, CPSparseMetadata], SparseAttention
):
    """Keep compression local by aligning document and shard boundaries to pools."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, SparseAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: SparseMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> CPSparseMetadata:
        if permutation is not None:
            raise ValueError(
                "GLM DSA CP requires contiguous shards without a permutation."
            )
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError("GLM DSA CP requires an active multi-rank CP mesh axis.")
        ratio = attention_metadata.compressor_ratio
        num_tokens = attention_metadata.positions_T.numel()
        if num_tokens % (cp_group.size() * ratio) != 0:
            raise ValueError(
                "GLM DSA CP requires each token shard to contain whole pools."
            )
        cu_seq = attention_metadata.cu_seq_k
        if (cu_seq % ratio != 0).any():
            raise ValueError(
                "GLM DSA CP requires per_sample_pad_multiple=compressor_ratio "
                "so document boundaries align to pools."
            )
        shard_len = num_tokens // cp_group.size()
        start = dist.get_rank(cp_group) * shard_len
        end = start + shard_len
        cu_seq_q = cu_seq.clamp(min=start, max=end) - start
        if spmd.is_type_checking():
            spmd.mutate_type(cu_seq_q, "cp", src=spmd.R, dst=spmd.V)
        # K offsets remain global; Q offsets describe fragments of the same docs.
        # Compression reads local groups, while all downstream gathers use global
        # offsets and original document positions after K/V and pool all-gathers.
        values = {
            field.name: getattr(attention_metadata, field.name)
            for field in fields(SparseMetadata)
        }
        values.update(
            cu_seq_q=cu_seq_q,
            max_q=min(attention_metadata.max_q, shard_len),
            pool_rows_PW=attention_metadata.pool_rows_PW[start // ratio : end // ratio]
            - start,
            token_start_T=attention_metadata.token_start_T[start:end],
            pool_start_T=attention_metadata.pool_start_T[start:end],
            pool_end_T=attention_metadata.pool_end_T[start:end],
            tail_indices_TW=attention_metadata.tail_indices_TW[start:end],
            positions_T=attention_metadata.positions_T[start:end],
        )
        return CPSparseMetadata(**values)

    def forward(
        self,
        q_THL,
        kv_TL,
        indices_TS,
        scores_TS,
        *,
        attention_metadata,
        **kwargs,
    ):
        # The adjoints sum contributions from every query shard with reduce-scatter.
        (kv_TL,) = cp_all_gather_tensors((kv_TL,), reduce_dtype=torch.float32)
        return super().forward(
            q_THL,
            kv_TL,
            indices_TS,
            scores_TS,
            attention_metadata=attention_metadata,
            **kwargs,
        )
