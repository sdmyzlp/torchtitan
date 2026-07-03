# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.utils import MoEStateDictAdapter
from .model import DeepSeekV3Model


# Abstract key constants shared by MTP and non-MTP maps.
_INNER_LAYER_KEYS: dict[str, str] = {
    # Attention Module
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
    "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    # MLP Module (dense)
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    # Transformer Layer
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    # MoE Module
    "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
    "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
    "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
    "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
}

# MTP-specific sub-modules (HF -> torchtitan).
_MTP_SPECIFIC_KEYS: dict[str, str] = {
    "model.layers.{}.enorm.weight": "mtp_block.layers.{}.enorm.weight",
    "model.layers.{}.hnorm.weight": "mtp_block.layers.{}.hnorm.weight",
    "model.layers.{}.eh_proj.weight": "mtp_block.layers.{}.eh_proj.weight",
    "model.layers.{}.shared_head.norm.weight": "mtp_block.layers.{}.final_norm.weight",
}

# MTP-specific reverse map (torchtitan -> HF).
_MTP_SPECIFIC_KEYS_TO_HF: dict[str, str] = {
    v: k for k, v in _MTP_SPECIFIC_KEYS.items()
}

# MTP sub-modules in the HF checkpoint that do NOT exist in torchtitan
# (torchtitan re-uses the main model's tok_embeddings and lm_head).
_MTP_SKIP_KEYS: frozenset = frozenset(
    [
        "model.layers.{}.embed_tokens.weight",
        "model.layers.{}.shared_head.head.weight",
    ]
)


def _mtp_inner_tt_suffix(tt_abstract_key: str) -> str | None:
    """If *tt_abstract_key* targets a non-MTP decoder layer, return the inner
    suffix used by MTP layers (e.g. ``"attention.wkv_a.weight"``), else None."""
    prefix = "layers.{}."
    if not tt_abstract_key.startswith(prefix):
        return None
    return tt_abstract_key[len(prefix):]


def _mtp_inner_hf_to_tt(hf_abstract_key: str, from_hf_map: dict[str, str]) -> str | None:
    """If *hf_abstract_key* targets an inner-decoder sub-module, return the
    torchtitan MTP form (with ``mtp_block.layers.{}.inner.`` prefix), else None."""
    if hf_abstract_key not in from_hf_map:
        return None
    tt_pattern = from_hf_map[hf_abstract_key]
    suffix = _mtp_inner_tt_suffix(tt_pattern)
    if suffix is None:
        return None
    return f"mtp_block.layers.{{}}.inner.{suffix}"


def _mtp_inner_tt_to_hf(tt_abstract_key: str, to_hf_map: dict[str, str]) -> str | None:
    """If *tt_abstract_key* is an MTP inner-layer key, return the HF form."""
    prefix = "mtp_block.layers.{}."
    if not tt_abstract_key.startswith(prefix):
        return None
    # Strip "mtp_block.layers.{depth}." -> then remove "inner." prefix.
    suffix = tt_abstract_key[len(prefix):]
    if not suffix.startswith("inner."):
        return None
    inner_suffix = suffix[len("inner."):]
    tt_plain = f"layers.{{}}.{inner_suffix}"
    return to_hf_map.get(tt_plain)


class DeepSeekV3StateDictAdapter(MoEStateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    def __init__(
        self,
        model_config: DeepSeekV3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            **_INNER_LAYER_KEYS,
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # Adjustments for from_hf_map based on model architecture
        if model_config.layers[0].attention.q_lora_rank != 0:
            self.from_hf_map.update(
                {
                    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
                    "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
                    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
                }
            )
        else:
            self.from_hf_map.update(
                {
                    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
                }
            )

        self._to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        # MTP support
        mtp_cfg = model_config.mtp
        self._mtp_num_layers = mtp_cfg.num_layers if mtp_cfg is not None else 0
        # HF stores MTP layers as extra indices after the main decoder layers.
        self._mtp_hf_offset = len(model_config.layers)

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            BLOCK_SIZE = 128
            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=BLOCK_SIZE,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    # ------------------------------------------------------------------
    # to_hf: torchtitan -> HuggingFace
    # ------------------------------------------------------------------

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = self._to_hf_map
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            # --- MTP layers ------------------------------------------------
            if key.startswith("mtp_block.layers."):
                self._to_hf_put_mtp(key, value, hf_state_dict)
                continue

            # --- Non-MTP layers (existing logic) ---------------------------
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    moe_layer = next(
                        l
                        for l in self.model_config.layers  # pyrefly: ignore [missing-attribute]
                        if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value,
                        moe_layer.moe.num_experts,
                    )
                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def _to_hf_put_mtp(
        self,
        key: str,
        value: torch.Tensor,
        hf_state_dict: dict[str, Any],
    ) -> None:
        """Convert a single ``mtp_block.layers.{depth}.{suffix}`` key and
        add the result (or split results for MoE experts) to
        *hf_state_dict*."""
        m = re.match(r"mtp_block\.layers\.(\d+)\.(.+)", key)
        if m is None:
            return
        mtp_depth = int(m.group(1))
        suffix = m.group(2)
        hf_layer = self._mtp_hf_offset + mtp_depth

        # 1) MTP-specific modules.
        tt_abstract = f"mtp_block.layers.{{}}.{suffix}"
        if tt_abstract in _MTP_SPECIFIC_KEYS_TO_HF:
            hf_key = _MTP_SPECIFIC_KEYS_TO_HF[tt_abstract].format(hf_layer)
            hf_state_dict[hf_key] = value
            return

        # 2) MTP inner-layer modules.
        inner_pattern = f"mtp_block.layers.{{}}.inner."
        if suffix.startswith("inner."):
            inner_suffix = suffix[len("inner."):]
            tt_plain = f"layers.{{}}.{inner_suffix}"
            hf_abstract = self._to_hf_map.get(tt_plain)
            if hf_abstract is None:
                return

            if "mlp.experts" in hf_abstract:
                # Split grouped MoE expert into individual expert weights.
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[tt_plain] = (
                        value.placements
                    )
                    self.grouped_expert_weight_shape[tt_plain] = value.shape
                    self.grouped_expert_weight_mesh[tt_plain] = value.device_mesh
                    local_expert_fqn = self._get_local_experts_weights(
                        hf_abstract,
                        tt_plain,
                        str(hf_layer),
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    moe_layer = _find_any_moe_layer(self.model_config)
                    split = self._split_experts_weights(
                        value, moe_layer.moe.num_experts
                    )
                    for en in range(moe_layer.moe.num_experts):
                        hf_state_dict[hf_abstract.format(hf_layer, en)] = (
                            split[en].squeeze()
                        )
                return

            hf_key = hf_abstract.format(hf_layer)
            hf_state_dict[hf_key] = value
            return

    # ------------------------------------------------------------------
    # from_hf: HuggingFace -> torchtitan
    # ------------------------------------------------------------------

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        self._validate_hf_rope_config(ComplexRoPE.Config)

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict = {}
        mtp_expert_groups: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            # Detect MTP layer (HF index >= num_hidden_layers).
            layer_match = re.match(r"model\.layers\.(\d+)\.", key)
            hf_layer = int(layer_match.group(1)) if layer_match else -1
            mtp_depth = hf_layer - self._mtp_hf_offset
            is_mtp = 0 <= mtp_depth < self._mtp_num_layers

            if is_mtp:
                self._from_hf_put_mtp(key, value, mtp_depth, state_dict, mtp_expert_groups)
                continue

            # --- Non-MTP keys (existing logic) -------------------------------
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                self._maybe_concat_experts(
                    expert_weights_by_layer, titan_abstract_key, layer_num, state_dict
                )

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        # Concatenate MTP expert weights after processing all keys.
        self._concat_mtp_experts(mtp_expert_groups, state_dict)

        return state_dict

    def _from_hf_put_mtp(
        self,
        hf_key: str,
        value: torch.Tensor,
        mtp_depth: int,
        state_dict: dict[str, Any],
        mtp_expert_groups: dict,
    ) -> None:
        """Convert one HF MTP key into torchtitan format and store it."""
        abstract_key = re.sub(
            r"model\.layers\.\d+", "model.layers.{}", hf_key, count=1
        )

        # 1) MTP-specific sub-modules.
        if abstract_key in _MTP_SPECIFIC_KEYS:
            state_dict[_MTP_SPECIFIC_KEYS[abstract_key].format(mtp_depth)] = value
            return

        # 2) Keys to skip.
        if abstract_key in _MTP_SKIP_KEYS:
            return

        # 3) MoE expert weights within MTP inner layer.
        if "mlp.experts" in hf_key:
            abstract_expert = re.sub(r"(\d+)", "{}", hf_key, count=2)
            _, expert_num = re.findall(r"\d+", hf_key)[:2]
            titan_abstract = self.from_hf_map.get(abstract_expert)
            if titan_abstract is None:
                return
            mtp_abstract = titan_abstract.replace(
                "layers.{}", "mtp_block.layers.{}.inner", 1
            )
            mtp_titan_key = mtp_abstract.format(mtp_depth)

            if mtp_titan_key not in mtp_expert_groups:
                mtp_expert_groups[mtp_titan_key] = {}
            if mtp_abstract not in mtp_expert_groups[mtp_titan_key]:
                mtp_expert_groups[mtp_titan_key][mtp_abstract] = {}
            mtp_expert_groups[mtp_titan_key][mtp_abstract][int(expert_num)] = value
            return

        # 4) Inner transformer layer (reuse standard layer mapping).
        mtp_pattern = _mtp_inner_hf_to_tt(abstract_key, self.from_hf_map)
        if mtp_pattern is not None:
            state_dict[mtp_pattern.format(mtp_depth)] = value

    def _maybe_concat_experts(
        self,
        expert_weights_by_layer: dict,
        titan_abstract_key: str,
        layer_num: str,
        state_dict: dict[str, Any],
    ) -> None:
        """Concatenate individual HF expert weights into a single grouped
        weight for *titan_abstract_key* (non-MTP layers)."""
        if titan_abstract_key in self.local_experts_indices:
            stacked = self._concatenate_expert_weights_dtensor(
                expert_weights_by_layer,
                titan_abstract_key,
                layer_num,
            )
        else:
            stacked = self._concatenate_expert_weights(
                expert_weights_by_layer,
                titan_abstract_key,
                layer_num,
                _find_any_moe_layer(self.model_config).moe.num_experts,
            )
        if stacked is not None:
            state_key = titan_abstract_key.format(layer_num)
            state_dict[state_key] = stacked

    def _concat_mtp_experts(
        self,
        mtp_expert_groups: dict,
        state_dict: dict[str, Any],
    ) -> None:
        """Concatenate individual HF expert weights inside MTP inner layers."""
        num_experts = _find_any_moe_layer(self.model_config).moe.num_experts
        for mtp_titan_key, groups in mtp_expert_groups.items():
            for mtp_abstract, expert_map in groups.items():
                if mtp_abstract in self.local_experts_indices:
                    stacked = self._concatenate_expert_weights_dtensor(
                        {mtp_titan_key: {mtp_abstract: expert_map}},
                        mtp_abstract,
                        mtp_titan_key,
                    )
                else:
                    stacked = self._concatenate_expert_weights(
                        {mtp_titan_key: {mtp_abstract: expert_map}},
                        mtp_abstract,
                        mtp_titan_key,
                        num_experts,
                    )
                if stacked is not None:
                    state_dict[mtp_titan_key] = stacked


def _find_any_moe_layer(model_config):
    return next(
        l for l in model_config.layers if l.moe is not None  # pyrefly: ignore
    )
