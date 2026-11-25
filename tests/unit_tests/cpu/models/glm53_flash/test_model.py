# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.glm53_flash import build_model_config


def _model(dtype=torch.float32):
    torch.manual_seed(42)
    with torch.device("meta"):
        model = build_model_config().build().to(dtype)
        for _, param in model.named_preserve_dtype_params():
            param.data = param.data.float()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    return model


def _forward(model, tokens, lengths, padding=0, denominator=None):
    positions = torch.cat(
        [torch.arange(n) for n in lengths] + [torch.zeros(padding, dtype=torch.long)]
    )
    padding_mask = torch.arange(tokens.numel()) >= sum(lengths)
    return model(
        tokens,
        aux_loss_denominators=torch.tensor(
            [sum(lengths) if denominator is None else denominator]
        ),
        positions=positions,
        padding_mask=padding_mask,
        attention_metadata=model._get_attention_metadata(
            positions, padding_mask=padding_mask
        ),
    )


def test_packed_documents_and_padding_match_separate_forward_backward():
    model = _model()
    separate = copy.deepcopy(model)
    lengths = [3, 12, 17]  # short document, complete pools, and incomplete tail
    tokens = torch.randint(0, 256, (sum(lengths) + 2,))
    packed = _forward(model, tokens, lengths, padding=2)[: sum(lengths)]
    expected = torch.cat(
        [
            _forward(separate, doc, [len(doc)], denominator=sum(lengths))
            for doc in tokens[:-2].split(lengths)
        ]
    )
    torch.testing.assert_close(packed, expected, atol=5e-5, rtol=2e-4)
    packed.square().mean().backward()
    expected.square().mean().backward()
    for (name, param), (_, ref) in zip(
        model.named_parameters(), separate.named_parameters()
    ):
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            # RoutedExperts accumulates its GEMMs in BF16; splitting a batch
            # changes the rounding of its weight-gradient reductions.
            error = (param.grad - ref.grad).norm()
            assert error <= 0.004 * ref.grad.norm() + 1e-7, name


def test_causal_prefix_and_optimizer_steps():
    model = _model()
    tokens = torch.randint(0, 256, (22,))
    with torch.no_grad():
        full = _forward(model, tokens, [22])
        changed = tokens.clone()
        changed[13:] = torch.randint(0, 256, (9,))
        prefix = _forward(model, changed, [22])
    # Keep candidate capacity fixed: torch.topk tie-breaking is unspecified.
    torch.testing.assert_close(full[:13], prefix[:13], atol=2e-6, rtol=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = F.cross_entropy(_forward(model, tokens[:-1], [21]).float(), tokens[1:])
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses


def _hf_config():
    hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    config = hf.Glm5NextTextConfig(
        hidden_size=64,
        vocab_size=2048,
        num_hidden_layers=5,
        intermediate_size=192,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        index_kpool=4,
        linear_num_heads=2,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        pad_token_id=255,
    )
    config._attn_implementation = "eager"
    return hf, config


@torch.no_grad()
def _copy_attention(ours, ref, linear):
    if linear:
        for src, dst in [
            (ours.q_proj, ref.q_proj),
            (ours.k_proj, ref.k_proj),
            (ours.v_proj, ref.v_proj),
            (ours.forget_a, ref.forget_gate.f_a_proj),
            (ours.forget_b, ref.forget_gate.f_b_proj),
            (ours.beta, ref.b_proj),
            (ours.output_gate.a, ref.g_a_proj),
            (ours.output_gate.b, ref.g_b_proj),
            (ours.output_norm, ref.o_norm),
            (ours.output_proj, ref.o_proj),
        ]:
            dst.weight.copy_(src.weight)
        ref.conv1d.weight.copy_(
            torch.cat([ours.q_conv.weight, ours.k_conv.weight, ours.v_conv.weight])
        )
        ref.forget_gate.A_log.copy_(ours.A_log)
        ref.forget_gate.dt_bias.copy_(ours.dt_bias.flatten())
    else:
        for src, dst in [
            (ours.wq_a, ref.q_a_proj),
            (ours.q_norm, ref.q_a_layernorm),
            (ours.wq_b, ref.q_b_proj),
            (ours.wkv_a, ref.kv_a_proj_with_mqa),
            (ours.kv_norm, ref.kv_a_layernorm),
            (ours.wkv_b, ref.kv_b_proj),
            (ours.wo, ref.o_proj),
            (ours.indexer.wq_b, ref.indexer.wq_b),
            (ours.indexer.wk, ref.indexer.wk),
            (ours.indexer.k_norm, ref.indexer.k_norm),
            (ours.indexer.weights_proj, ref.indexer.weights_proj),
        ]:
            dst.weight.copy_(src.weight)
        ref.indexer.k_norm.bias.copy_(ours.indexer.k_norm.bias)
        ref.indexer.index_kpool_compress_gate.copy_(ours.indexer.compressor.gate.weight)
        ref.indexer.index_kpool_compress_ape.copy_(ours.indexer.compressor.ape)


@pytest.mark.parametrize("layer", [0, 3])
def test_attention_matches_transformers_forward_backward(layer):
    hf, config = _hf_config()
    ours = _model().layers[str(layer)].attention
    linear = layer == 0
    if not linear:
        with torch.no_grad():
            ours.indexer.compressor.gate.weight.normal_(std=0.1)
            ours.indexer.compressor.ape.normal_(std=0.2)
    ref = (hf.Glm5NextTextLinearAttention if linear else hf.Glm5NextTextAttention)(
        config, layer
    )
    _copy_attention(ours, ref, linear)
    x = torch.randn(23, 64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    metadata = (
        build_model_config()
        .layers[layer]
        .attention.inner_attention.build_attention_metadata(torch.arange(23))
    )
    actual = ours(x, metadata, aux_loss_denominator=torch.tensor(x.size(0)))
    expected = ref(x_ref[None], attention_mask=torch.ones(1, 23, dtype=torch.bool))
    expected = expected[0] if linear else expected[0][0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-6, rtol=2e-4)
    actual_proj = ours.output_proj if linear else ours.wo
    torch.testing.assert_close(
        actual_proj.weight.grad, ref.o_proj.weight.grad, atol=3e-5, rtol=2e-4
    )


def test_hyperconnection_matches_transformers_forward_backward():
    hf, config = _hf_config()
    ours = _model().layers["0"].hc_attn_pre
    ref = hf.Glm5NextTextHyperConnection(config)
    with torch.no_grad():
        ref.fn.copy_(ours.hc_fn)
        ref.base.copy_(ours.hc_base)
        ref.scale.copy_(ours.hc_scale)
    x = torch.randn(7, 4, 64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    actual = ours(x)
    post, comb, collapsed = ref(x_ref[None])
    expected = (collapsed[0], post[0], comb[0])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    torch.stack([t.square().sum() for t in actual]).sum().backward()
    torch.stack([t.square().sum() for t in expected]).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ours.hc_fn.grad, ref.fn.grad, atol=2e-5, rtol=2e-5)


def _model_pair():
    hf, config = _hf_config()
    ours = _model(torch.bfloat16)
    for layer in ours.layers.values():
        if hasattr(layer.feed_forward, "expert_bias_E"):
            layer.feed_forward.expert_bias_E.uniform_(-0.05, 0.05)
    ref = hf.Glm5NextTextModel(config).to(torch.bfloat16)
    for layer in ref.layers:
        if isinstance(layer.mlp, hf.Glm5NextTextMoE):
            layer.mlp.gate.e_score_correction_bias = (
                layer.mlp.gate.e_score_correction_bias.float()
            )
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.base.data = hc.base.data.float()
            hc.scale.data = hc.scale.data.float()
        if isinstance(layer.self_attn, hf.Glm5NextTextLinearAttention):
            layer.self_attn.conv1d.float()
            layer.self_attn.forget_gate.A_log.data = (
                layer.self_attn.forget_gate.A_log.data.float()
            )
            layer.self_attn.forget_gate.dt_bias.data = (
                layer.self_attn.forget_gate.dt_bias.data.float()
            )

    def copy_ffn(a, b):
        b.gate_proj.weight.copy_(a.w13.weight[0])
        b.up_proj.weight.copy_(a.w13.weight[1])
        b.down_proj.weight.copy_(a.w2.weight)

    with torch.no_grad():
        ref.embed_tokens.weight.copy_(ours.tok_embeddings.weight)
        ref.norm.weight.copy_(ours.norm.weight)
        for i, (a, b) in enumerate(zip(ours.layers.values(), ref.layers)):
            _copy_attention(a.attention, b.self_attn, i % 4 != 3)
            b.input_layernorm.weight.copy_(a.attention_norm.weight)
            b.post_attention_layernorm.weight.copy_(a.ffn_norm.weight)
            for hc_a, hc_b in [(a.hc_attn_pre, b.attn_hc), (a.hc_ffn_pre, b.ffn_hc)]:
                hc_b.fn.copy_(hc_a.hc_fn)
                hc_b.base.copy_(hc_a.hc_base)
                hc_b.scale.copy_(hc_a.hc_scale)
            if i < 3:
                copy_ffn(a.feed_forward, b.mlp)
            else:
                moe = a.feed_forward
                copy_ffn(moe.shared_experts, b.mlp.shared_experts)
                b.mlp.gate.weight.copy_(moe.router.gate.weight)
                b.mlp.gate.e_score_correction_bias.copy_(moe.expert_bias_E)
                b.mlp.experts.gate_up_proj.copy_(
                    moe.routed_experts.w13.weight.flatten(1, 2)
                )
                b.mlp.experts.down_proj.copy_(moe.routed_experts.w2.weight)
    return ours, ref


@torch.backends.mkldnn.flags(enabled=False)
def test_full_model_matches_transformers():
    ours, ref = _model_pair()
    tokens = torch.randint(0, 256, (23,))
    actual = _forward(ours, tokens, [23])
    expected = F.linear(
        ref(tokens[None], use_cache=False).last_hidden_state[0],
        ours.lm_head.weight.detach(),
    )
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.03)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    grad, ref_grad = (
        ours.tok_embeddings.weight.grad.float(),
        ref.embed_tokens.weight.grad.float(),
    )
    assert (grad - ref_grad).norm() <= 0.03 * ref_grad.norm()


def test_recipe_loads_and_full_flavor_builds_on_meta():
    from torchtitan.config import ConfigLoader
    from torchtitan.models.glm53_flash.model import Glm53FlashModel
    from torchtitan.trainer import Trainer

    recipe = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.glm53_flash_minimal",
            "--config",
            "glm53_flash_debugmodel",
        ]
    )
    assert isinstance(recipe, Trainer.Config)
    assert isinstance(recipe.model, Glm53FlashModel.Config)
    assert recipe.model.vocab_size >= 2020  # bundled tokenizer
    with torch.device("meta"):
        model = build_model_config("glm53_flash").build()
    assert len(model.layers) == 45
    assert all(p.is_meta for p in model.parameters())


def test_indexer_kl_preserves_selected_mass_and_empty_rows():
    from torchtitan.models.glm53_flash.attention import IndexerLoss

    loss = IndexerLoss.Config(coeff=0.01).build()
    scores = torch.tensor(
        [[0.2, -0.3, -torch.inf], [-torch.inf] * 3], requires_grad=True
    )
    teacher = torch.tensor([[0.1, 0.3, 0.0], [0.0] * 3])
    loss.kl(teacher, scores).backward()
    expected = torch.zeros_like(scores)
    expected[0, :2] = scores[0, :2].softmax(-1) * 0.4 - teacher[0, :2]
    torch.testing.assert_close(scores.grad, expected)


def test_indexer_loss_trains_compression_without_trunk_gradient():
    attention = _model().layers["3"].attention
    positions = torch.arange(23)
    metadata = (
        build_model_config()
        .layers[3]
        .attention.inner_attention.build_attention_metadata(positions)
    )
    x = torch.randn(23, 64, requires_grad=True)
    out = attention(x, metadata, aux_loss_denominator=torch.tensor(23))
    (out * 0).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    for name, param in attention.named_parameters():
        assert param.grad is not None, name
        if name.startswith("indexer."):
            assert param.grad.isfinite().all() and param.grad.abs().sum() > 0, name
        else:
            torch.testing.assert_close(param.grad, torch.zeros_like(param))
