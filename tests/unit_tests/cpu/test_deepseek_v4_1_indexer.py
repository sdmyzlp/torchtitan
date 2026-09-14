# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek-V4.1 lightning-indexer distillation loss and packing.

The teacher is the sparse attention's own mass on the selected compressed entries,
rebuilt by ``IndexerDistillLoss`` from the queries, the compressed keys and the operator's
per-head log-sum-exp (window + selected compressed + sink). It is the *raw marginal*
``p = mean_h exp(logit - lse)``, whose row sum ``Z <= 1`` is the compressed slice's share
of the full softmax; the loss weights the conditional's KL by ``Z``, so its gradient is
``dI = Z * Y - p``. These tests pin that formula against a float64 oracle and against that
gradient, always through ``IndexerDistillLoss.forward`` (the value is read back off its
metric accumulator), and pin the packed-document isolation that the metadata's ``doc_ids``
provides plus the ``-inf`` marking the loss reads the student's unused slots from.
"""

import math
import unittest

import torch

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4_1 import model_registry
from torchtitan.models.deepseek_v4_1.indexer import (
    FULL,
    HierarchicalIndexer,
    IndexerDistillLoss,
    REINDEX,
    REUSE,
)

_FP32_EPS = torch.finfo(torch.float32).tiny


def teacher_oracle(q_THD, cmp_k_ND, topk_indices_TK, lse_TH, softmax_scale):
    """Reference raw marginal teacher from an explicit per-head ``lse``, in float64.

    Returns ``p`` (row sum ``Z``), not the conditional ``p / Z``: the loss weights each
    row by ``Z``, so a normalised oracle would pin the wrong quantity.
    """
    valid_TK = topk_indices_TK >= 0
    row_valid_T = valid_TK.any(dim=-1)
    selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)].double()
    logits_THK = (
        torch.einsum("thd,tkd->thk", q_THD.double(), selected_TKD) * softmax_scale
    )
    logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -math.inf)
    mass_TH = torch.exp(torch.logsumexp(logits_THK, dim=-1) - lse_TH.double())
    conditional_THK = torch.softmax(
        logits_THK.masked_fill(~row_valid_T[:, None, None], 0.0), dim=-1
    )
    return (mass_TH.unsqueeze(-1) * conditional_THK).sum(dim=1) / q_THD.size(1)


def conditional(p_TK):
    """``t = p / Z``, the conditional teacher the student is scored against."""
    return p_TK / p_TK.sum(dim=-1, keepdim=True).clamp_min(_FP32_EPS)


def valid_lse(q_THD, cmp_k_ND, topk_indices_TK, scale):
    """A denominator holding the compressed slice plus a positive window/sink share.

    Returns ``lse`` in ``[T, H]`` with every head's ``Z`` strictly between 0 and 1, which
    is the shape the operator's LSE has when the window and sink take the rest.
    """
    valid_TK = topk_indices_TK >= 0
    selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)]
    logits_THK = (
        torch.einsum("thd,tkd->thk", q_THD, selected_TKD) * scale
    ).masked_fill(~valid_TK.unsqueeze(1), -torch.inf)
    extra_TH = torch.rand(q_THD.size(0), q_THD.size(1)) * 3.0 + 1.0
    return torch.logsumexp(logits_THK, dim=-1) + extra_TH.log()


def student_logits(topk_indices_TK, *, requires_grad):
    """Student logits with the indexer's ``-inf`` marking on its unused slots."""
    logits_TK = torch.randn(*topk_indices_TK.shape).masked_fill(
        topk_indices_TK < 0, -torch.inf
    )
    return logits_TK.requires_grad_(requires_grad)


def loss_value(
    loss, q_THD, cmp_k_ND, topk_indices_TK, lse_TH, topk_scores_TK
) -> torch.Tensor:
    """The loss's raw per-microbatch value, read back off its metric accumulator.

    ``forward`` returns the carrier rather than the value; with the step denominator at 1
    the accumulator holds exactly the sum the gradient is injected from. The teacher's
    sources are passed detached, as the loss's own call site does.
    """
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    loss.instance_acc.zero_()
    loss(
        q_THD.detach(),
        cmp_k_ND.detach(),
        topk_indices_TK,
        lse_TH.detach(),
        topk_scores_TK,
        carrier=torch.zeros_like(topk_scores_TK),
    )
    return loss.instance_acc.clone()


def loss_student_gradient(
    loss, q_THD, cmp_k_ND, topk_indices_TK, lse_TH, topk_scores_TK
) -> torch.Tensor:
    """The gradient ``forward`` injects into the student logits, through the carrier."""
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    carrier = loss(
        q_THD.detach(),
        cmp_k_ND.detach(),
        topk_indices_TK,
        lse_TH.detach(),
        topk_scores_TK,
        carrier=torch.zeros_like(topk_scores_TK),
    )
    carrier.backward(torch.ones_like(carrier))
    return topk_scores_TK.grad


class TestIndexerTeacher(unittest.TestCase):
    def _loss(self, head_dim: int) -> IndexerDistillLoss:
        return IndexerDistillLoss.Config(
            coeff=1.0, softmax_scale=head_dim**-0.5
        ).build()

    def test_matches_oracle_with_masked_slots(self):
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 7, 3, 8, 5, 3
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.randint(0, num_cmp, (num_tokens, topk))
        topk_indices_TK[0, 1:] = -1  # a row with unused slots

        loss = self._loss(head_dim)
        # A real denominator (compressed slice plus a window/sink share), so Z <= 1.
        lse_TH = valid_lse(q_THD, cmp_k_ND, topk_indices_TK, loss.softmax_scale)
        got_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        expected_TK = teacher_oracle(
            q_THD, cmp_k_ND, topk_indices_TK, lse_TH, loss.softmax_scale
        )
        torch.testing.assert_close(got_TK, expected_TK.float(), rtol=1e-5, atol=1e-6)
        # The marginal is not a distribution: its row mass is Z <= 1, and it is the
        # conditional p / Z that sums to one.
        Z_T = got_TK.sum(-1)
        self.assertTrue(bool((Z_T > 0).all()))
        self.assertTrue(bool((Z_T <= 1.0 + 1e-6).all()))
        self.assertTrue(
            torch.allclose(
                conditional(got_TK).sum(-1), torch.ones(num_tokens), atol=1e-6
            )
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
        lse_TH = torch.stack(
            [
                torch.full((num_tokens,), -2.0),
                torch.full((num_tokens,), 6.0),
            ],
            dim=-1,
        )

        loss = self._loss(head_dim)
        got_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        expected_TK = teacher_oracle(
            q_THD, cmp_k_ND, topk_indices_TK, lse_TH, loss.softmax_scale
        )
        torch.testing.assert_close(got_TK, expected_TK.float(), rtol=1e-5, atol=1e-6)

        # Giving every head unit compressed mass is the defect this weighting avoids:
        # that target is a plain mean over the per-head softmaxes.
        valid_TK = topk_indices_TK >= 0
        selected_TKD = cmp_k_ND[topk_indices_TK.clamp_min(0)].double()
        logits_THK = (
            torch.einsum("thd,tkd->thk", q_THD.double(), selected_TKD)
            * loss.softmax_scale
        ).masked_fill(~valid_TK.unsqueeze(1), -math.inf)
        equal_weight_TK = torch.softmax(logits_THK, dim=-1).mean(dim=1)
        self.assertGreater(
            (conditional(got_TK) - equal_weight_TK).abs().max().item(), 1e-3
        )

    def test_uniform_lse_shift_scales_the_mass_but_not_the_conditional(self):
        """Only a *per-head* denominator difference reweights the heads.

        A constant added to every head's ``lse`` is a per-row scale of the mass, so it
        cancels in the conditional and the weighted loss direction is unchanged. That is
        why the fix is about per-head mass (window/sink) rather than an absolute
        denominator.
        """
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 3, 2, 8, 4, 3
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.topk(
            torch.randn(num_tokens, num_cmp), topk, dim=-1
        ).indices
        base_TH = torch.randn(num_tokens, num_heads)

        loss = self._loss(head_dim)
        base_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, base_TH)
        uniform_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, base_TH + 40.0)
        per_head_TH = base_TH.clone()
        per_head_TH[:, 0] += 40.0
        per_head_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, per_head_TH)

        self.assertLess(
            (conditional(base_TK) - conditional(uniform_TK)).abs().max().item(), 1e-5
        )
        self.assertGreater(
            (conditional(base_TK) - conditional(per_head_TK)).abs().max().item(), 1e-3
        )
        # The uniform shift is exactly a per-row scale of the marginal.
        self.assertTrue(
            torch.allclose(
                uniform_TK.sum(-1) / base_TK.sum(-1),
                torch.full((num_tokens,), math.exp(-40.0)),
                rtol=1e-3,
            )
        )
        self.assertTrue(torch.isfinite(uniform_TK).all())
        self.assertTrue(torch.isfinite(per_head_TK).all())

    def test_tiny_compressed_mass_carries_no_weight(self):
        """Mass below the fp32 normal range is zero, exactly as it is in the kernel.

        The kernel reads ``Z`` off the teacher tensor's row sum, so a mass it cannot
        represent contributes nothing; rescuing it would up-weight a row whose compressed
        slice holds no probability.
        """
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 2, 1, 4, 5, 5
        q_THD = torch.zeros(num_tokens, num_heads, head_dim)
        q_THD[..., 0] = 1.0
        cmp_k_ND = torch.zeros(num_cmp, head_dim)
        cmp_k_ND[:, 0] = torch.tensor([0.0, math.log(2.0), 0.0, 0.0, 0.0])
        topk_indices_TK = torch.arange(num_cmp).expand(num_tokens, -1).contiguous()
        # A huge lse pushes every head's compressed mass below fp32 tiny.
        lse_TH = torch.full((num_tokens, num_heads), 300.0)

        loss = self._loss(head_dim)
        p_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        self.assertTrue(torch.equal(p_TK, torch.zeros_like(p_TK)))

        student = student_logits(topk_indices_TK, requires_grad=False)
        raw = loss_value(loss, q_THD, cmp_k_ND, topk_indices_TK, lse_TH, student)
        self.assertEqual(float(raw), 0.0)
        self.assertTrue(math.isfinite(float(raw)))

    def test_student_gradient_is_z_times_y_minus_p(self):
        """Pin the objective: ``dI = Z * Y - p``, not ``Y - t``."""
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim, num_cmp, topk = 5, 2, 8, 6, 4
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.randint(0, num_cmp, (num_tokens, topk))
        topk_indices_TK[0, 1:] = -1  # a row with unused slots

        loss = self._loss(head_dim)
        lse_TH = valid_lse(q_THD, cmp_k_ND, topk_indices_TK, loss.softmax_scale)
        p_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        Z_T = p_TK.sum(-1, keepdim=True)

        student = student_logits(topk_indices_TK, requires_grad=True)
        dI_TK = loss_student_gradient(
            loss, q_THD, cmp_k_ND, topk_indices_TK, lse_TH, student
        )

        # The student's own -inf slots are what the loss reads as unreachable.
        logits_TK = student.detach()
        logits_TK = logits_TK.masked_fill(
            ~torch.isfinite(logits_TK).any(-1).unsqueeze(-1), 0.0
        )
        Y_TK = torch.softmax(logits_TK, dim=-1)
        torch.testing.assert_close(dI_TK, Z_T * Y_TK - p_TK, atol=1e-6, rtol=1e-5)


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
            if module.mode is not REUSE:
                captured.append((module.compress_ratio, output[1].clone()))

        for module in model.modules():
            if isinstance(module, HierarchicalIndexer):
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


class TestIndexerDistillLoss(unittest.TestCase):
    """The loss's value and gradient, exercised through ``forward``.

    ``forward`` is the only entry point: it builds the teacher from the raw attention
    tensors, forms the weighted KL and injects the gradient on the carrier. These tests
    read the value back off the metric accumulator (with the step denominator at 1) and
    the gradient back off the student, so the formula and the injection are both pinned.
    """

    def _case(
        self, *, num_tokens=4, num_heads=2, head_dim=8, num_cmp=5, topk=3, seed=0
    ):
        """A teacher with a real denominator (0 < Z < 1) and a padded row."""
        torch.manual_seed(seed)
        q_THD = torch.randn(num_tokens, num_heads, head_dim)
        cmp_k_ND = torch.randn(num_cmp, head_dim)
        topk_indices_TK = torch.randint(0, num_cmp, (num_tokens, topk))
        topk_indices_TK[0, 1:] = -1  # a row with unused slots
        loss = IndexerDistillLoss.Config(
            coeff=1.0, softmax_scale=head_dim**-0.5
        ).build()
        lse_TH = valid_lse(q_THD, cmp_k_ND, topk_indices_TK, loss.softmax_scale)
        p_TK = loss._teacher(q_THD, cmp_k_ND, topk_indices_TK, lse_TH)
        return loss, q_THD, cmp_k_ND, topk_indices_TK, lse_TH, p_TK

    def test_weighted_kl_is_the_marginal_times_the_conditional_kl(self):
        loss, q_THD, cmp_k_ND, indices, lse_TH, p_TK = self._case()
        t_TK = conditional(p_TK)
        student = student_logits(indices, requires_grad=False)

        raw = loss_value(loss, q_THD, cmp_k_ND, indices, lse_TH, student)

        valid_TK = torch.isfinite(student)
        log_student_TK = torch.log_softmax(student, dim=-1)
        weighted_TK = torch.special.xlogy(t_TK, t_TK) - t_TK * log_student_TK
        per_row_kl = weighted_TK.masked_fill(~valid_TK, 0.0).sum(-1)
        self.assertAlmostEqual(
            float(raw), float((p_TK.sum(-1) * per_row_kl).sum()), places=5
        )
        # The unweighted KL would be larger by 1 / Z: this is the objective fix.
        self.assertGreater(
            float(per_row_kl.mean()), float((p_TK.sum(-1) * per_row_kl).mean())
        )

    def test_loss_is_zero_for_a_student_matching_the_teacher(self):
        loss, q_THD, cmp_k_ND, indices, lse_TH, p_TK = self._case(seed=1)
        # log(p / Z) is -inf exactly on the unused slots, which is the indexer's own
        # marking for them, so matching the teacher needs no extra masking.
        student = conditional(p_TK).log()
        self.assertAlmostEqual(
            float(loss_value(loss, q_THD, cmp_k_ND, indices, lse_TH, student)),
            0.0,
            places=5,
        )

    def test_loss_is_positive_for_a_mismatched_student(self):
        loss, q_THD, cmp_k_ND, indices, lse_TH, _ = self._case(seed=2)
        student = student_logits(indices, requires_grad=False)
        self.assertGreater(
            float(loss_value(loss, q_THD, cmp_k_ND, indices, lse_TH, student)), 0.0
        )

    def test_a_row_reaching_nothing_stays_finite(self):
        """Row 0 keeps a real denominator from the window, but selects nothing.

        Its conditional is an all ``-inf`` softmax row and its mass is exactly zero. This
        is what the two dead-row guards buy, and why they are not redundant with the
        cleanup in ``forward``: without them ``p`` is NaN there, a ``log_softmax`` over an
        all ``-inf`` row backpropagates NaN whatever the incoming gradient is, and the
        cleanup only zeroes values that reach the sum -- it cannot undo either.
        """
        loss, q_THD, cmp_k_ND, indices, lse_TH, _ = self._case(seed=3)
        indices = indices.clone()
        indices[0] = -1

        p_TK = loss._teacher(q_THD, cmp_k_ND, indices, lse_TH)
        self.assertTrue(bool(torch.isfinite(p_TK).all()))
        self.assertEqual(float(p_TK[0].abs().max()), 0.0)

        student = student_logits(indices, requires_grad=True)
        raw = loss_value(loss, q_THD, cmp_k_ND, indices, lse_TH, student)
        self.assertTrue(math.isfinite(float(raw)))

        dI_TK = loss_student_gradient(loss, q_THD, cmp_k_ND, indices, lse_TH, student)
        self.assertTrue(bool(torch.isfinite(dI_TK).all()))
        self.assertEqual(float(dI_TK[0].abs().max()), 0.0)

        # A zero mass means that row's student cannot move the loss at all.
        disturbed = student.detach().clone()
        disturbed[0] = torch.randn(disturbed.size(-1))
        self.assertAlmostEqual(
            float(loss_value(loss, q_THD, cmp_k_ND, indices, lse_TH, disturbed)),
            float(raw),
            places=7,
        )

    def test_forward_builds_the_teacher_itself_and_returns_the_carrier(self):
        """The loss owns the teacher: ``forward`` takes the raw tensors, not ``p``."""
        loss, q_THD, cmp_k_ND, indices, lse_TH, p_TK = self._case()
        AuxLoss.set_step_denominator(torch.tensor(1.0))
        carrier = torch.zeros(3, 3)
        out = loss(
            q_THD,
            cmp_k_ND,
            indices,
            lse_TH,
            conditional(p_TK).log(),
            carrier=carrier,
        )
        self.assertTrue(torch.equal(out, carrier))


class TestHierarchicalIndexer(unittest.TestCase):
    """The mode adapter: Full / Reindex (with or without the pool) / Reuse."""

    @staticmethod
    def _model(use_candidates: bool):
        torch.manual_seed(0)
        model_config = model_registry(
            "debugmodel",
            seq_len=32,
            use_candidates=use_candidates,
            indexer_loss_coeff=None,
        ).model
        model = model_config.build()
        model.init_states()
        model.eval()
        return model_config, model

    def test_modes_follow_the_layer_roles(self):
        """Full owns the index keys, Reindex rescores them, the rest carry the top-k."""
        _, model = self._model(use_candidates=True)
        indexers = [layer.attention.indexer for layer in model.layers.values()]
        # Debug model roles: kv sources (2, 5) are Full, index source 7 reuses the KV of
        # 5 and so is Reindex, and every non-source layer carries the shared top-k.
        self.assertIs(indexers[2].mode, FULL)
        self.assertIs(indexers[5].mode, FULL)
        self.assertIs(indexers[7].mode, REINDEX)
        self.assertIs(indexers[3].mode, REUSE)

    def test_only_the_pool_layers_carry_candidate_topk_blocks(self):
        """The Full source and the Reindex layers after it; layers before it are plain."""
        _, model = self._model(use_candidates=True)
        pooled = [
            i
            for i, layer in enumerate(model.layers.values())
            if layer.attention.indexer.candidate_topk_blocks > 0
        ]
        self.assertEqual(pooled, [5, 7])

    def test_without_candidates_every_layer_scores_everything(self):
        _, model = self._model(use_candidates=False)
        self.assertTrue(
            all(
                layer.attention.indexer.candidate_topk_blocks == 0
                for layer in model.layers.values()
            )
        )

    @classmethod
    def _captured(cls):
        model_config, model = cls._model(use_candidates=True)
        captured = []

        def hook(module, args, kwargs, output):
            # (mode, topk_indices, topk_scores, candidate pool)
            captured.append((module.mode, output[1], output[2], output[3]))

        for module in model.modules():
            if isinstance(module, HierarchicalIndexer):
                module.register_forward_hook(hook, with_kwargs=True)
        tokens = torch.randint(0, model_config.vocab_size, (32,))
        positions = torch.arange(32)
        with torch.no_grad():
            model(tokens, positions, model.get_attention_masks(positions))
        return captured

    def test_full_source_builds_the_pool_and_reindex_searches_it(self):
        captured = self._captured()
        pooled = [c for c in captured if c[0] is FULL and c[3] is not None]
        self.assertEqual(len(pooled), 1)
        _, source_topk, _, source_pool = pooled[0]
        # The pool holds whole blocks, so a late row's pool is wider than the top-k it
        # feeds; early rows have too few complete blocks to fill it.
        self.assertGreater(int(source_pool.sum(dim=-1).max()), source_topk.size(-1))

        consumers = [c for c in captured if c[0] is REINDEX]
        self.assertTrue(consumers)
        for _, consumer_topk, _, consumer_pool in consumers:
            self.assertTrue(torch.equal(consumer_pool, source_pool))
            valid_TK = consumer_topk >= 0
            in_pool_TK = consumer_pool.gather(-1, consumer_topk.clamp_min(0))
            self.assertTrue(bool(in_pool_TK[valid_TK].all()))

    def test_unused_slots_carry_no_logit(self):
        """The distillation reads ``-inf``, not the ``-1`` indices, as "unreachable".

        That marking is what keeps an unused slot out of the student's softmax, so it is
        pinned here on every layer that computes a selection rather than reuses one.
        """
        captured = self._captured()
        padded = False
        for mode, topk_indices_TK, topk_scores_TK, _ in captured:
            if mode is REUSE:
                continue
            valid_TK = topk_indices_TK >= 0
            padded |= bool((~valid_TK).any())
            self.assertTrue(bool(torch.isfinite(topk_scores_TK[valid_TK]).all()))
            self.assertTrue(bool(torch.isinf(topk_scores_TK[~valid_TK]).all()))
            self.assertTrue(bool((topk_scores_TK[~valid_TK] < 0).all()))
        self.assertTrue(padded, "no padded row to check the marking on")


if __name__ == "__main__":
    unittest.main()
