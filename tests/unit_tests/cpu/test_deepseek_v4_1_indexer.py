# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek-V4.1 lightning-indexer distillation loss and packing.

The teacher is the sparse attention's own distribution over the selected compressed
entries, rebuilt by ``IndexerKLLoss`` from the queries, the compressed keys and the
operator's per-head log-sum-exp (window + selected compressed + sink), so it is
``sum_h exp(logit - lse)`` restricted to the selected support and L1-normalized. These
tests pin that formula against a float64 oracle, and pin the packed-document isolation
that the metadata's ``doc_ids`` is supposed to provide.
"""

import math
import unittest

import torch

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4_1 import model_registry
from torchtitan.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan.models.deepseek_v4_1.indexer import Indexer, IndexerKLLoss


def teacher_oracle(q_THD, cmp_k_ND, topk_indices_TK, lse_HT, softmax_scale):
    """Reference teacher from an explicit per-head ``lse``, in float64."""
    valid_TK = topk_indices_TK >= 0
    selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)].double()
    logits_THK = (
        torch.einsum("thd,tkd->thk", q_THD.double(), selected_TKD) * softmax_scale
    )
    logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -math.inf)
    mass_THK = torch.exp(logits_THK - lse_HT.double().transpose(0, 1).unsqueeze(-1))
    target_TK = mass_THK.sum(dim=1)
    return target_TK / target_TK.sum(dim=-1, keepdim=True)


class TestIndexerTeacher(unittest.TestCase):
    def _loss(self, head_dim: int) -> IndexerKLLoss:
        return IndexerKLLoss.Config(coeff=1.0, softmax_scale=head_dim**-0.5).build()

    def test_matches_oracle_with_masked_slots(self):
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 7, 3, 8, 5, 3
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.randint(0, num_cmp, (num_tokens, topk))
        topk_indices_TK[0, 1:] = -1  # a row with unused slots
        lse_HT = torch.randn(num_heads, num_tokens) + 2.0

        loss = self._loss(head_dim)
        got_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_HT)
        expected_TK = teacher_oracle(
            q_THD, cmp_k_ND, topk_indices_TK, lse_HT, loss.softmax_scale
        )
        self.assertLess((got_TK - expected_TK).abs().max().item(), 1e-6)
        self.assertTrue(
            torch.allclose(got_TK.sum(-1), torch.ones(num_tokens), atol=1e-6)
        )

    def test_heads_weighted_by_their_full_denominator(self):
        """A head whose lse is dominated by window/sink must not outvote the others."""
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 4, 2, 8, 4, 3
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.topk(
            torch.randn(num_tokens, num_cmp), topk, dim=-1
        ).indices
        # Head 0 concentrates on the compressed entries (small lse); head 1 spends its
        # mass on the window/sink (large lse).
        lse_HT = torch.stack(
            [
                torch.full((num_tokens,), -2.0),
                torch.full((num_tokens,), 6.0),
            ]
        )

        loss = self._loss(head_dim)
        got_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_HT)
        expected_TK = teacher_oracle(
            q_THD, cmp_k_ND, topk_indices_TK, lse_HT, loss.softmax_scale
        )
        self.assertLess((got_TK - expected_TK).abs().max().item(), 1e-6)

        # Summing raw scores across heads is the bug we fixed: build the "equal head
        # weight" target and confirm it differs.
        valid_TK = topk_indices_TK >= 0
        selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)].double()
        logits_THK = (
            torch.einsum("thd,tkd->thk", q_THD.double(), selected_TKD)
            * loss.softmax_scale
        ).masked_fill(~valid_TK.unsqueeze(1), -math.inf)
        equal_weight_TK = torch.exp(logits_THK - logits_THK.amax(-1, keepdim=True))
        equal_weight_TK = equal_weight_TK.sum(1)
        equal_weight_TK = equal_weight_TK / equal_weight_TK.sum(-1, keepdim=True)
        self.assertGreater((got_TK - equal_weight_TK).abs().max().item(), 1e-3)

    def test_uniform_lse_shift_cancels_but_per_head_does_not(self):
        """Only a *per-head* denominator difference reweights the teacher.

        A constant added to every head's ``lse`` scales by the same factor and cancels
        in the L1 normalization, which is exactly why the fix is about per-head mass
        (window/sink) rather than an absolute denominator.
        """
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 3, 2, 8, 4, 3
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.topk(
            torch.randn(num_tokens, num_cmp), topk, dim=-1
        ).indices
        base_HT = torch.randn(num_heads, num_tokens)

        loss = self._loss(head_dim)
        base_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, base_HT)
        uniform_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, base_HT + 40.0)
        per_head_HT = base_HT.clone()
        per_head_HT[0] += 40.0
        per_head_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, per_head_HT)

        self.assertLess((base_TK - uniform_TK).abs().max().item(), 1e-5)
        self.assertGreater((base_TK - per_head_TK).abs().max().item(), 1e-3)
        self.assertTrue(torch.isfinite(uniform_TK).all())
        self.assertTrue(torch.isfinite(per_head_TK).all())

    def test_tiny_compressed_mass_is_normalized_not_lost(self):
        """Mass below the fp32 normal range still yields the correct distribution."""
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 2, 1, 4, 5, 5
        q_THD = torch.zeros(num_tokens, num_heads, head_dim)
        q_THD[..., 0] = 1.0
        cmp_k_ND = torch.zeros(num_cmp, head_dim)
        cmp_k_ND[:, 0] = torch.tensor([0.0, math.log(2.0), 0.0, 0.0, 0.0])
        topk_indices_TK = torch.arange(num_cmp).expand(num_tokens, -1).contiguous()
        # A huge lse pushes every head's compressed mass below fp32 tiny.
        lse_HT = torch.full((num_heads, num_tokens), 300.0)

        loss = self._loss(head_dim)
        got_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_HT)
        self.assertTrue(torch.isfinite(got_TK).all())
        expected_TK = teacher_oracle(
            q_THD, cmp_k_ND, topk_indices_TK, lse_HT, loss.softmax_scale
        )
        self.assertLess((got_TK - expected_TK).abs().max().item(), 1e-6)
        # The logit gap is ``scale * log(2)``, so the target ratio is that exponent.
        expected_ratio = math.exp(loss.softmax_scale * math.log(2.0))
        self.assertAlmostEqual(
            got_TK[0, 1].item() / got_TK[0, 0].item(), expected_ratio, places=5
        )


class TestPackedDocuments(unittest.TestCase):
    """``doc_ids`` is the only varlen metadata; segments must not interact."""

    @staticmethod
    def _model():
        config = model_registry("debugmodel", seq_len=64, indexer_loss_coeff=None).model
        model = config.build()
        model.init_states()
        model.eval()
        return config, model

    def test_perturbing_one_document_leaves_the_other_untouched(self):
        torch.manual_seed(0)
        config, model = self._model()
        positions = torch.cat([torch.arange(32), torch.arange(32)])
        metadata = model.get_attention_masks(positions)
        tokens = torch.randint(0, config.vocab_size, (64,))
        with torch.no_grad():
            out_a = model(tokens, positions, metadata)
            perturbed = tokens.clone()
            perturbed[:32] = torch.randint(0, config.vocab_size, (32,))
            out_b = model(perturbed, positions, metadata)
        self.assertEqual(float((out_a[32:] - out_b[32:]).abs().max()), 0.0)
        self.assertGreater(float((out_a[:32] - out_b[:32]).abs().max()), 0.0)

    def test_indexer_selection_never_crosses_documents(self):
        """The operator's sparse branch is the caller's responsibility: pin it here."""
        torch.manual_seed(0)
        config, model = self._model()
        positions = torch.cat([torch.arange(32), torch.arange(32)])
        doc_ids = torch.cumsum((positions == 0).to(torch.int32), dim=0) - 1
        tokens = torch.randint(0, config.vocab_size, (64,))

        captured: list[tuple[int, torch.Tensor]] = []

        def hook(module, args, output):
            if module.is_source:
                captured.append((module.compress_ratio, output[1].clone()))

        for module in model.modules():
            if isinstance(module, Indexer):
                module.register_forward_hook(hook)
        with torch.no_grad():
            model(tokens, positions, model.get_attention_masks(positions))

        self.assertTrue(captured)
        for ratio, topk_indices_TK in captured:
            entry_doc = doc_ids[::ratio]
            valid_TK = topk_indices_TK >= 0
            same_doc_TK = entry_doc[topk_indices_TK.clamp_min(0)] == doc_ids.unsqueeze(
                -1
            )
            self.assertTrue(torch.all(same_doc_TK | ~valid_TK))


class TestSelectionScoreRecompute(unittest.TestCase):
    """The student logits are only computed where a distillation loss consumes them."""

    @staticmethod
    def _captured_topk_scores(model) -> list:
        captured = []

        def pre_hook(module, args, kwargs):
            # Window-only layers have no compressed selection to score.
            if kwargs.get("cmp_k") is not None:
                captured.append(kwargs.get("topk_scores"))

        for module in model.modules():
            if isinstance(module, CompressedSparseInnerAttention2):
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        return captured

    def _forward(self, indexer_loss_coeff, *, eval_mode):
        torch.manual_seed(0)
        model_config = model_registry(
            "debugmodel", seq_len=32, indexer_loss_coeff=indexer_loss_coeff
        ).model
        model = model_config.build()
        model.init_states()
        AuxLoss.set_step_denominator(torch.tensor(32.0))
        model.eval() if eval_mode else model.train()
        captured = self._captured_topk_scores(model)
        tokens = torch.randint(0, model_config.vocab_size, (32,))
        positions = torch.arange(32)
        metadata = model.get_attention_masks(positions)
        if eval_mode:
            with torch.no_grad():
                model(tokens, positions, metadata)
        else:
            model(tokens, positions, metadata)
        return captured

    def test_training_with_loss_produces_selection_scores(self):
        captured = self._forward(0.01, eval_mode=False)
        self.assertTrue(captured)
        self.assertTrue(all(scores is not None for scores in captured))

    def test_eval_skips_selection_scores(self):
        captured = self._forward(0.01, eval_mode=True)
        self.assertTrue(captured)
        self.assertTrue(all(scores is None for scores in captured))

    def test_training_without_loss_skips_selection_scores(self):
        captured = self._forward(None, eval_mode=False)
        self.assertTrue(captured)
        self.assertTrue(all(scores is None for scores in captured))


class TestIndexerKLLoss(unittest.TestCase):
    def test_kl_is_zero_for_matching_distributions(self):
        loss = IndexerKLLoss.Config(coeff=1.0, softmax_scale=1.0).build()
        teacher = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
        indices = torch.tensor([[0, 1], [0, 1]])
        # log-student equal to log-teacher.
        raw = loss._kl(teacher, teacher.log(), indices)
        self.assertAlmostEqual(float(raw), 0.0, places=6)

    def test_kl_positive_for_mismatched_distributions(self):
        loss = IndexerKLLoss.Config(coeff=1.0, softmax_scale=1.0).build()
        teacher = torch.tensor([[0.9, 0.1]])
        indices = torch.tensor([[0, 1]])
        student = torch.tensor([[0.0, 5.0]])
        self.assertGreater(float(loss._kl(teacher, student, indices)), 0.0)

    def test_forward_builds_the_teacher_and_injects_its_kl(self):
        """The loss owns the teacher: ``forward`` takes the raw tensors, not ``p``."""
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp = 3, 2, 4, 4
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.tensor([[0, 1], [1, 2], [2, 3]])
        lse_HT = torch.zeros(num_heads, num_tokens)

        loss = IndexerKLLoss.Config(coeff=1.0, softmax_scale=head_dim**-0.5).build()
        AuxLoss.set_step_denominator(torch.tensor(1.0))
        teacher_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_HT)
        carrier = torch.zeros(num_tokens, num_heads, head_dim)

        # A student that matches the teacher has zero KL, so the metric is zero.
        out = loss(
            q_THD,
            cmp_k_ND,
            topk_indices_TK,
            lse_HT,
            teacher_TK.log(),
            carrier=carrier,
        )
        self.assertTrue(torch.equal(out, carrier))
        self.assertAlmostEqual(float(loss.instance_acc), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
