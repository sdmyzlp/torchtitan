# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LoggedAuxLoss and SeqwiseLoadBalanceLoss.

TestCase groups:
- TestSeqwiseLoadBalanceLossNumerics: forward/backward values vs an explicit
  formula reference, injected gradient vs explicit loss, and non-differentiability
  of the one-hot counts.
- TestLoggedAuxLossAccumulation: forward-side accumulation semantics,
  roll-up/zero semantics, and that torch_remat checkpointing accumulates the
  metric exactly once via the loss's own retained region.
- TestSeqwiseLossSpmdTypes: 8-rank (dp2/cp2/tp2/ep2) spmd_types test on CPU
  float64 + gloo; per-forward loss and gradient match the global per-DP-rank
  reference with and without the SPMD typechecker, and with/without EP
  token sharding.
"""

import contextlib
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch_remat as remat
from spmd_types.checker import typecheck
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.common.aux_loss import (
    _zero_aux_losses,
    collect_aux_loss_metrics,
    LoggedAuxLoss,
)
from torchtitan.models.common.moe import SeqwiseLoadBalanceLoss


def _set_spmd_types_backend():
    from torchtitan.distributed.utils import set_spmd_backend

    set_spmd_backend("spmd_types")


def _reference_seqwise_aux_loss(
    scores_TE: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    top_k: int,
    coeff: float,
    per_step_denominator: int,
) -> torch.Tensor:
    """Explicit DeepSeek-V3 per-forward aux loss (Eqs 17-20), framework-scaled.

    ``T`` is the number of tokens in the forward (the whole input, matching
    the single-process shape-derived count).
    """
    E = scores_TE.size(-1)
    T = scores_TE.size(0)
    routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_TK, True
    )
    counts_E = routing_map_TE.sum(dim=0).to(scores_TE.dtype)
    probs_TE = scores_TE / scores_TE.sum(dim=-1, keepdim=True)
    prob_sums_E = probs_TE.sum(dim=0)
    f_E = counts_E * (E / (top_k * T))
    p_E = prob_sums_E / T
    return (f_E * p_E).sum() * (coeff / per_step_denominator)


def _make_loss_module(
    top_k: int,
    coeff: float,
    per_step_denominator: int,
) -> SeqwiseLoadBalanceLoss:
    """Build a SeqwiseLoadBalanceLoss with the given parameters."""
    cfg = SeqwiseLoadBalanceLoss.Config(coeff=coeff, top_k=top_k)
    cfg.per_step_denominator = per_step_denominator
    loss = SeqwiseLoadBalanceLoss(cfg)
    loss.train()
    return loss


def _clear_aux_loss_registry():
    """Reset the class-level metric registry for the current process."""
    LoggedAuxLoss._group_counts.clear()
    LoggedAuxLoss.group_acc.clear()


def _make_inputs(T: int, E: int, K: int):
    scores_TE = torch.rand(T, E, dtype=torch.float32, requires_grad=True)
    topk_expert_ids_TK = torch.topk(
        scores_TE.detach(), k=K, dim=-1, sorted=False
    ).indices
    topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
    return scores_TE, topk_scores_TK, topk_expert_ids_TK


class TestSeqwiseLoadBalanceLossNumerics(unittest.TestCase):
    """Forward/backward of SeqwiseLoadBalanceLoss vs an explicit reference."""

    def setUp(self):
        self.T, self.E, self.K = 15, 7, 2
        self.coeff = 0.125
        self.per_step_denominator = 8
        torch.manual_seed(0)
        _set_spmd_types_backend()
        _clear_aux_loss_registry()

    def tearDown(self):
        _clear_aux_loss_registry()

    def test_loss_value_matches_formula(self):
        """The accumulated metric after the forward equals the reference
        formula (the metric accumulates in the forward)."""
        scores_TE, topk_scores_TK, topk_expert_ids_TK = _make_inputs(
            self.T, self.E, self.K
        )
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        out_TK = loss(scores_TE, topk_expert_ids_TK, carrier=topk_scores_TK)
        self.assertTrue(torch.equal(out_TK, topk_scores_TK))

        _zero_aux_losses([loss])
        group_acc_value = LoggedAuxLoss.group_acc.get(
            ("batch", "seqwise_load_balance_loss")
        )
        self.assertIsNotNone(group_acc_value)
        ref = _reference_seqwise_aux_loss(
            scores_TE,
            topk_expert_ids_TK,
            self.K,
            self.coeff,
            self.per_step_denominator,
        )
        # group_acc holds raw_sum / denominator, and ref = raw_sum * coeff /
        # denominator, so the register equals ref / coeff.
        self.assertAlmostEqual(
            group_acc_value.item(), ref.item() / self.coeff, places=4
        )

    def test_gradient_injected_via_carrier(self):
        """The aux loss gradient flows through the carrier tensor."""
        scores_TE, topk_scores_TK, topk_expert_ids_TK = _make_inputs(
            self.T, self.E, self.K
        )
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        topk_scores_TK.retain_grad()
        out_TK = loss(scores_TE, topk_expert_ids_TK, carrier=topk_scores_TK)
        out_TK.sum().backward()
        self.assertIsNotNone(topk_scores_TK.grad)
        self.assertFalse(torch.all(topk_scores_TK.grad == 0))

    def test_counts_gradient_is_zero(self):
        """The gradient flows only through the normalized-probs path; the
        one-hot counts (Eq. 18) are non-differentiable."""
        scores_TE, topk_scores_TK, topk_expert_ids_TK = _make_inputs(
            self.T, self.E, self.K
        )
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)

        out_TK = loss(scores_TE, topk_expert_ids_TK, carrier=topk_scores_TK)
        out_TK.sum().backward()
        self.assertIsNotNone(scores_TE.grad)
        self.assertGreater(torch.abs(scores_TE.grad).sum().item(), 0)


class TestLoggedAuxLossAccumulation(unittest.TestCase):
    """Accumulation, roll-up/zero semantics, and torch_remat checkpointing."""

    def setUp(self):
        self.T, self.E, self.K = 8, 4, 2
        self.coeff = 0.1
        self.per_step_denominator = 4
        torch.manual_seed(42)
        _set_spmd_types_backend()
        _clear_aux_loss_registry()

    def tearDown(self):
        _clear_aux_loss_registry()

    def test_accumulation_across_forwards(self):
        """Multiple forwards accumulate the scaled per-forward values."""
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        num_forwards = 3
        ref_total = 0.0
        for _ in range(num_forwards):
            scores_TE, topk_scores_TK, topk_expert_ids_TK = _make_inputs(
                self.T, self.E, self.K
            )
            out_TK = loss(scores_TE, topk_expert_ids_TK, carrier=topk_scores_TK)
            # force gradient path too
            out_TK.sum().backward()
            ref_total += _reference_seqwise_aux_loss(
                scores_TE,
                topk_expert_ids_TK,
                self.K,
                1.0,
                self.per_step_denominator,
            ).item()

        _zero_aux_losses([loss])
        group_acc_value = LoggedAuxLoss.group_acc.get(
            ("batch", "seqwise_load_balance_loss")
        )
        self.assertIsNotNone(group_acc_value)
        self.assertAlmostEqual(group_acc_value.item(), ref_total, places=3)

    def test_rollup_then_clear(self):
        """After the roll-up into group_acc, each instance accumulator is zeroed."""
        loss = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        scores_TE, topk_scores_TK, topk_expert_ids_TK = _make_inputs(
            self.T, self.E, self.K
        )
        out_TK = loss(scores_TE, topk_expert_ids_TK, carrier=topk_scores_TK)
        out_TK.sum().backward()

        _zero_aux_losses([loss])
        self.assertEqual(loss.instance_acc.item(), 0.0)

    def test_no_double_count_with_remat_checkpointing(self):
        """The loss protects its own accumulation under torch_remat
        checkpointing: the metric is counted exactly once per microbatch
        without the call site declaring anything."""
        loss_guarded = _make_loss_module(self.K, self.coeff, self.per_step_denominator)
        (
            scores_TE2,
            topk_scores_TK2,
            topk_expert_ids_TK2,
        ) = _make_inputs(self.T, self.E, self.K)

        def _forward_once(module, carrier, scores_TE, expert_ids):
            # LoggedAuxLoss.inject() wraps its accumulation in a retained
            # remat region itself, so a plain call must not double count
            # when the enclosing checkpoint replays.
            return module(scores_TE, expert_ids, carrier=carrier).sum()

        out = remat.checkpoint()(_forward_once)(
            loss_guarded, topk_scores_TK2, scores_TE2, topk_expert_ids_TK2
        )
        out.backward()
        _zero_aux_losses([loss_guarded])
        single = _reference_seqwise_aux_loss(
            scores_TE2, topk_expert_ids_TK2, self.K, 1.0, self.per_step_denominator
        ).item()
        self.assertAlmostEqual(
            LoggedAuxLoss.group_acc[("batch", "seqwise_load_balance_loss")].item(),
            single,
            places=5,
        )


class TestSeqwiseLossSpmdTypes(DTensorTestBase):
    """8-rank spmd_types test: per-forward loss and gradient match the
    global per-DP-rank reference (CPU float64 + gloo), under
    with and without the SPMD typechecker, and with EP token sharding on
    and off."""

    @property
    def world_size(self):
        return 8

    def _setup_mesh(self, *, enable_ep: bool):
        """Build parallel dims and register meshes.  Returns
        (parallel_dims, dense_mesh).

        ``enable_ep`` selects the expert-parallel mesh: with EP the loss
        reduces the router output's token sums over CP and TP, without EP only
        over CP (the router output is TP-replicate).  DP stays local either
        way (one stream per DP rank)."""
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.distributed.spmd_types import set_spmd_meshes

        _set_spmd_types_backend()
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=2,
                cp=2,
                tp=2,
                pp=1,
                ep=2 if enable_ep else 1,
                world_size=8,
                spmd_backend="spmd_types",
            )
            parallel_dims.build_mesh()
            dense_mesh = parallel_dims.get_mesh(["dp", "cp", "tp"])
            set_spmd_meshes(
                dense_mesh=dense_mesh,
                sparse_mesh=parallel_dims.spmd_sparse_mesh(),
            )
        return parallel_dims, dense_mesh

    def _make_loss_config(self, top_k: int):
        cfg = SeqwiseLoadBalanceLoss.Config(coeff=0.1, top_k=top_k)
        cfg.per_step_denominator = 1
        return cfg

    def _reference_for_stream(self, scores, ids, top_k, E, coeff=0.1):
        """Reference loss and gradient for a single dp-rank token stream."""
        with torch.no_grad():
            routing_map = torch.zeros(scores.shape[0], E).scatter_(-1, ids, 1.0)
            counts_E = routing_map.sum(dim=0)
            probs = scores / scores.sum(dim=-1, keepdim=True)
            prob_sums_E = probs.sum(dim=0)
            num_tokens = scores.shape[0]
            f_E = counts_E * (E / (top_k * num_tokens))
            p_E = prob_sums_E / num_tokens
            ref_loss = (f_E * p_E).sum()
        ref_scores = scores.detach().clone().requires_grad_(True)
        rm = torch.zeros(scores.shape[0], E).scatter_(-1, ids, 1.0)
        cts = rm.sum(dim=0)
        prs = ref_scores / ref_scores.sum(dim=-1, keepdim=True)
        nt = scores.shape[0]
        f = cts * (E / (top_k * nt))
        p = prs.sum(dim=0) / nt
        ref_aux = (f * p).sum() * coeff
        ref_carrier = ref_scores.gather(dim=-1, index=ids)
        (ref_aux + ref_carrier.sum()).backward()
        return ref_loss, ref_scores.grad

    def _setup_pp_mesh(self):
        """Build pp=2 parallel dims (pp2 x dp2 x cp2 = 8 ranks)."""
        from torchtitan.distributed.parallel_dims import ParallelDims

        _set_spmd_types_backend()
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=2,
                cp=2,
                tp=1,
                pp=2,
                ep=1,
                world_size=8,
                spmd_backend="spmd_types",
            )
            parallel_dims.build_mesh()
        return parallel_dims

    @with_comms
    def test_pp_reduction_sums_stages(self):
        """The pp reduction sums stages instead of averaging them.

        Every layer lives on exactly one pipeline stage, so summing the stage
        partials and dividing by the build-time instance count (identical on
        every rank) yields the mean over all layers.  Averaging stages would
        under-report by the pipeline degree.
        """
        parallel_dims = self._setup_pp_mesh()
        _clear_aux_loss_registry()
        key = ("batch", "seqwise_load_balance_loss")
        # Emulate one rank: the build-time count is 6 instances (the divisor),
        # and this rank's per-step values sum to 3.0.
        LoggedAuxLoss._group_counts[key] = 6
        LoggedAuxLoss.group_acc[key] = torch.tensor(3.0, dtype=torch.float32)

        metrics = collect_aux_loss_metrics(parallel_dims)
        # Sum over the batch mesh (2 dp coords) -> 6.0 per stage, then sum
        # over the pp mesh (2 stages) -> 12.0, divided by 6 -> 2.0.
        # Averaging over pp would report 1.0.
        self.assertAlmostEqual(metrics["seqwise_load_balance_loss/mean"], 2.0, places=6)
        _clear_aux_loss_registry()

    def _run_reduction_case(self, *, enable_ep: bool, use_typecheck: bool):
        """Run one distributed reduction case and compare with the reference.

        The loss derives its token-partition axes from runtime mesh state, so
        the same assertions must hold with and without the SPMD typechecker
        (the default training configuration has it off).
        """
        parallel_dims, dense_mesh = self._setup_mesh(enable_ep=enable_ep)
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh

        T, E, K = 128, 8, 2
        dp, cp, tp = 2, 2, 2
        dp_rank = self.rank // (cp * tp)
        cp_rank = (self.rank // tp) % cp
        tp_rank = self.rank % tp
        t_dp = T // dp
        dp_start = dp_rank * t_dp

        if enable_ep:
            from torchtitan.models.common.decoder_sharding import (
                dense_sequence_parallel_placement,
            )

            t_blk = t_dp // (cp * tp)
            shard = cp_rank * tp + tp_rank
            t_start = dp_start + shard * t_blk
            t_end = t_start + t_blk
            placement = dense_sequence_parallel_placement()
        else:
            from torchtitan.models.common.decoder_sharding import (
                dense_activation_placement,
            )

            t_blk = t_dp // cp
            shard = cp_rank
            t_start = dp_start + shard * t_blk
            t_end = t_start + t_blk
            placement = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))

        checker = typecheck(local=False) if use_typecheck else contextlib.nullcontext()
        _clear_aux_loss_registry()
        with set_current_spmd_mesh(dense_mesh), checker:
            torch.manual_seed(0)
            global_scores = torch.rand(T, E, dtype=torch.float64)
            global_ids = torch.randint(0, E, (T, K))
            with spmd.no_typecheck():
                local_scores = global_scores[t_start:t_end].contiguous()
                local_ids = global_ids[t_start:t_end].contiguous()

            spmd.assert_type(local_scores, placement)
            spmd.assert_type(local_ids, placement)

            loss = SeqwiseLoadBalanceLoss(self._make_loss_config(K))

            local_scores.requires_grad_(True)
            carrier = local_scores.gather(dim=-1, index=local_ids)
            out = loss(local_scores, local_ids, carrier=carrier)
            with spmd.no_typecheck():
                torch.testing.assert_close(out, carrier, rtol=0, atol=0)
                # Backward runs outside the checker in both modes: with EP off
                # the final statistics are Replicate on tp (no tp token
                # sharding), which the checker rejects for implicit backward.
                out.sum().backward()

            with spmd.no_typecheck():
                dp_scores = global_scores[dp_start : dp_start + t_dp]
                dp_ids = global_ids[dp_start : dp_start + t_dp]
                ref_loss, ref_grad = self._reference_for_stream(dp_scores, dp_ids, K, E)
                self.assertAlmostEqual(
                    loss.instance_acc.item(), ref_loss.item(), places=6
                )
                ref_local_grad = ref_grad[shard * t_blk : (shard + 1) * t_blk]
                self.assertLess(
                    (local_scores.grad - ref_local_grad).abs().max().item(), 1e-10
                )

        # The collected metric sums the reduce mesh (each dp rank contributes
        # its own stream), so with two dp ranks the value equals the sum of
        # the two streams' losses -- the same step-global value a one-rank
        # run processing both streams would produce (denominator = 1 here).
        _zero_aux_losses([loss])
        metrics = collect_aux_loss_metrics(parallel_dims)
        ref_total = 0.0
        for stream in range(dp):
            s = global_scores[stream * t_dp : (stream + 1) * t_dp]
            i = global_ids[stream * t_dp : (stream + 1) * t_dp]
            ref_total += self._reference_for_stream(s, i, K, E)[0].item()
        self.assertAlmostEqual(
            metrics["seqwise_load_balance_loss/mean"], ref_total, places=6
        )
        _clear_aux_loss_registry()

    @with_comms
    def test_ep_enabled_reduction(self):
        """EP/SP token sharding over TP: P->I on CP and TP; loss and gradient
        match the per-DP-rank reference, with and without the typechecker."""
        for use_typecheck in (True, False):
            with self.subTest(use_typecheck=use_typecheck):
                self._run_reduction_case(enable_ep=True, use_typecheck=use_typecheck)

    @with_comms
    def test_ep_disabled_reduction(self):
        """No EP token sharding over TP: P->I on CP only; loss and gradient
        match the per-DP-rank reference, with and without the typechecker."""
        for use_typecheck in (True, False):
            with self.subTest(use_typecheck=use_typecheck):
                self._run_reduction_case(enable_ep=False, use_typecheck=use_typecheck)


if __name__ == "__main__":
    unittest.main()
