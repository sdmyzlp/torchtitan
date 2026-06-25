# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for StateDictAdapter's tied-lm_head handling.

A tied decoder shares one tensor between the LM head and the input embedding,
so HF stores only the embedding (model.safetensors.index.json omits, or should
omit, lm_head.weight). The base StateDictAdapter centralizes three pieces of
this convention:

- _drop_tied_lm_head_from_index_mapping: drop lm_head from the loaded shard
  index when tying is on, so HuggingFaceStorageWriter does not emit an empty,
  unreadable shard for the absent key.
- _drop_tied_lm_head: drop lm_head from a to_hf state dict when tying is on.
- _tie_lm_head: recreate lm_head from the embedding when an HF dict omits it.

These run on CPU.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

import torch

from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class _BareAdapter(StateDictAdapter):
    """Concrete adapter exercising only the base tied-lm_head helpers."""

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        return hf_state_dict


class _NestedEmbedAdapter(_BareAdapter):
    """Adapter whose embedding is nested (mirrors the qwen3_5 multimodal name)."""

    hf_embed_tokens_key = "model.language_model.embed_tokens.weight"


_WEIGHT_MAP = {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.norm.weight": "model-00001-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors",
}


def _build_index_adapter(adapter_cls, model_config: Any):
    index = {"weight_map": dict(_WEIGHT_MAP)}
    with tempfile.TemporaryDirectory() as assets_path:
        with open(
            os.path.join(assets_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(index, f)
        return adapter_cls(model_config, assets_path)


class TestIndexMapping(unittest.TestCase):
    def test_tied_lm_head_dropped(self) -> None:
        adapter = _build_index_adapter(
            _BareAdapter, SimpleNamespace(enable_weight_tying=True)
        )
        mapping = adapter.fqn_to_index_mapping
        assert mapping is not None
        self.assertNotIn("lm_head.weight", mapping)
        self.assertEqual(mapping["model.embed_tokens.weight"], 1)
        self.assertEqual(mapping["model.norm.weight"], 1)
        # The now-empty second shard is no longer referenced.
        self.assertNotIn(2, set(mapping.values()))

    def test_untied_lm_head_kept(self) -> None:
        adapter = _build_index_adapter(
            _BareAdapter, SimpleNamespace(enable_weight_tying=False)
        )
        mapping = adapter.fqn_to_index_mapping
        assert mapping is not None
        self.assertEqual(mapping["lm_head.weight"], 2)

    def test_missing_field_keeps_lm_head(self) -> None:
        adapter = _build_index_adapter(_BareAdapter, SimpleNamespace())
        mapping = adapter.fqn_to_index_mapping
        assert mapping is not None
        self.assertEqual(mapping["lm_head.weight"], 2)


class TestDropTiedLmHead(unittest.TestCase):
    def _adapter(self, **cfg: Any) -> _BareAdapter:
        return _BareAdapter(SimpleNamespace(**cfg), hf_assets_path=None)

    def test_tied_drops(self) -> None:
        adapter = self._adapter(enable_weight_tying=True)
        hf = {"model.embed_tokens.weight": 0, "lm_head.weight": 0}
        adapter._drop_tied_lm_head(hf)
        self.assertEqual(set(hf), {"model.embed_tokens.weight"})

    def test_untied_keeps(self) -> None:
        adapter = self._adapter(enable_weight_tying=False)
        hf = {"model.embed_tokens.weight": 0, "lm_head.weight": 0}
        adapter._drop_tied_lm_head(hf)
        self.assertIn("lm_head.weight", hf)

    def test_missing_field_keeps(self) -> None:
        adapter = self._adapter()
        hf = {"lm_head.weight": 0}
        adapter._drop_tied_lm_head(hf)
        self.assertIn("lm_head.weight", hf)


class TestTieLmHead(unittest.TestCase):
    def _adapter(self, cls=_BareAdapter) -> _BareAdapter:
        return cls(SimpleNamespace(enable_weight_tying=True), hf_assets_path=None)

    def test_present_is_noop(self) -> None:
        adapter = self._adapter()
        head = object()
        embed = object()
        hf = {"lm_head.weight": head, "model.embed_tokens.weight": embed}
        adapter._tie_lm_head(hf)
        self.assertIs(hf["lm_head.weight"], head)

    def test_absent_realiases_from_embed(self) -> None:
        adapter = self._adapter()
        embed = object()
        hf = {"model.embed_tokens.weight": embed}
        adapter._tie_lm_head(hf)
        self.assertIs(hf["lm_head.weight"], embed)

    def test_both_absent_raises(self) -> None:
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            adapter._tie_lm_head({"model.norm.weight": object()})

    def test_nested_embed_key(self) -> None:
        # A subclass overriding hf_embed_tokens_key resolves the nested name.
        adapter = self._adapter(_NestedEmbedAdapter)
        embed = object()
        hf = {"model.language_model.embed_tokens.weight": embed}
        adapter._tie_lm_head(hf)
        self.assertIs(hf["lm_head.weight"], embed)
        # The default (unnested) name is not what this adapter looks for.
        with self.assertRaises(ValueError):
            adapter._tie_lm_head({"model.embed_tokens.weight": object()})


class TestRealAdapterRoundTrip(unittest.TestCase):
    """Tied/untied to_hf + from_hf behavior through the real model adapters.

    The debug configs drive enable_weight_tying directly (it is the only switch
    the adapters consult), so a single config flavor exercises both paths.
    """

    def _adapter(self, model: str, tied: bool):
        if model == "llama3":
            cfg = llama3_configs["debugmodel"](attn_backend="flex")
            cfg.enable_weight_tying = tied
            return Llama3StateDictAdapter(cfg, hf_assets_path=None)
        cfg = qwen3_configs["debugmodel"](attn_backend="flex")
        cfg.enable_weight_tying = tied
        return Qwen3StateDictAdapter(cfg, hf_assets_path=None)

    def _check_tied(self, adapter) -> None:
        embed, head, norm = torch.randn(8, 4), torch.randn(8, 4), torch.randn(4)
        # to_hf omits lm_head (it aliases the embedding).
        hf = adapter.to_hf(
            {
                "tok_embeddings.weight": embed,
                "lm_head.weight": head,
                "norm.weight": norm,
            }
        )
        self.assertIn("model.embed_tokens.weight", hf)
        self.assertNotIn("lm_head.weight", hf)
        self.assertTrue(torch.equal(hf["model.embed_tokens.weight"], embed))
        # A tied HF checkpoint omits lm_head; from_hf rebuilds it from embed.
        tt = adapter.from_hf(
            {"model.embed_tokens.weight": embed, "model.norm.weight": norm}
        )
        self.assertTrue(torch.equal(tt["lm_head.weight"], embed))

    def _check_untied(self, adapter) -> None:
        embed, head, norm = torch.randn(8, 4), torch.randn(8, 4), torch.randn(4)
        # to_hf keeps the separate lm_head.
        hf = adapter.to_hf(
            {
                "tok_embeddings.weight": embed,
                "lm_head.weight": head,
                "norm.weight": norm,
            }
        )
        self.assertTrue(torch.equal(hf["lm_head.weight"], head))
        # An untied HF checkpoint includes lm_head; from_hf keeps it verbatim.
        tt = adapter.from_hf(
            {
                "model.embed_tokens.weight": embed,
                "lm_head.weight": head,
                "model.norm.weight": norm,
            }
        )
        self.assertTrue(torch.equal(tt["lm_head.weight"], head))

    def test_llama3_tied(self) -> None:
        self._check_tied(self._adapter("llama3", True))

    def test_llama3_untied(self) -> None:
        self._check_untied(self._adapter("llama3", False))

    def test_qwen3_tied(self) -> None:
        self._check_tied(self._adapter("qwen3", True))

    def test_qwen3_untied(self) -> None:
        self._check_untied(self._adapter("qwen3", False))


if __name__ == "__main__":
    unittest.main()
