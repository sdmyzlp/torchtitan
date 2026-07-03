# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.loss import CrossEntropyLoss, MTPAwareLoss
from torchtitan.models.common.mtp import (
    MTPBlock,
    MTPConfig,
    MTPLayer,
    mtp_auxiliary_loss,
    roll_tensor,
)


def _make_dummy_block_config(dim: int):
    """A minimal transformer block for MTP testing."""
    from dataclasses import dataclass

    from torchtitan.models.common.nn_modules import Linear, RMSNorm

    class DummyBlock(torch.nn.Module):
        @dataclass(kw_only=True, slots=True)
        class Config:
            dim: int

            def build(self):
                return DummyBlock(self)

        def __init__(self, config: Config):
            super().__init__()
            self.norm = RMSNorm(RMSNorm.Config(normalized_shape=config.dim))
            self.linear = Linear(Linear.Config(in_features=config.dim, out_features=config.dim, bias=False))

        def forward(self, x, attention_masks=None, positions=None):
            return x + self.linear(self.norm(x))

    return DummyBlock.Config(dim=dim)


class TestRollTensor(unittest.TestCase):
    def test_roll_left_zeroes_boundary(self):
        t = torch.arange(10).unsqueeze(0)
        rolled = roll_tensor(t, shifts=-1)
        expected = torch.cat([torch.arange(1, 10), torch.zeros(1)]).unsqueeze(0)
        self.assertTrue(rolled.equal(expected))

    def test_roll_none(self):
        self.assertIsNone(roll_tensor(None))

    def test_roll_preserves_shape(self):
        t = torch.randn(2, 8, 16)
        rolled = roll_tensor(t, shifts=-1, dims=1)
        self.assertEqual(rolled.shape, t.shape)


class TestMTPLayer(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.batch = 2
        self.seq_len = 8

        class DummyEmbedding(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(100, dim))

            def forward(self, tokens, positions=None):
                return self.weight[tokens]

        inner_cfg = _make_dummy_block_config(self.dim)
        self.layer = MTPLayer(
            dim=self.dim,
            inner_block_config=inner_cfg,
            layer_number=1,
        )
        self.embedding = DummyEmbedding(self.dim)

    def test_forward_shape(self):
        ids = torch.randint(0, 100, (self.batch, self.seq_len))
        h = torch.randn(self.batch, self.seq_len, self.dim)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch, -1)

        out, next_ids, next_pos = self.layer(ids, h, pos, None, self.embedding)

        self.assertEqual(out.shape, (self.batch, self.seq_len, self.dim))
        self.assertEqual(next_ids.shape, (self.batch, self.seq_len))
        self.assertEqual(next_pos.shape, (self.batch, self.seq_len))

    def test_next_ids_are_shifted(self):
        ids = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch, -1)
        h = torch.randn(self.batch, self.seq_len, self.dim)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch, -1)

        _, next_ids, _ = self.layer(ids, h, pos, None, self.embedding)

        expected = ids.roll(shifts=-1, dims=1)
        expected[:, -1] = 0
        self.assertTrue(next_ids.equal(expected))


class TestMTPBlock(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.batch = 2
        self.seq_len = 8
        self.num_layers = 2

        class DummyEmbedding(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(100, dim))

            def forward(self, tokens, positions=None):
                return self.weight[tokens]

        inner_cfg = _make_dummy_block_config(self.dim)
        self.block = MTPBlock(
            mtp_config=MTPConfig(num_layers=self.num_layers, loss_scaling_factor=0.3),
            dim=self.dim,
            inner_block_config=inner_cfg,
        )
        self.embedding = DummyEmbedding(self.dim)

    def test_output_shape(self):
        ids = torch.randint(0, 100, (self.batch, self.seq_len))
        h = torch.randn(self.batch, self.seq_len, self.dim)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch, -1)

        out = self.block(ids, h, pos, None, self.embedding)

        expected_seq = (self.num_layers + 1) * self.seq_len
        self.assertEqual(out.shape, (self.batch, expected_seq, self.dim))

    def test_first_chunk_is_original(self):
        ids = torch.randint(0, 100, (self.batch, self.seq_len))
        h = torch.randn(self.batch, self.seq_len, self.dim)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch, -1)

        out = self.block(ids, h, pos, None, self.embedding)
        first_chunk = out[:, :self.seq_len]
        self.assertTrue(torch.equal(first_chunk, h))

    def test_repeated_layer_flag(self):
        inner_cfg = _make_dummy_block_config(self.dim)
        block = MTPBlock(
            mtp_config=MTPConfig(
                num_layers=3, loss_scaling_factor=0.3, use_repeated_layer=True
            ),
            dim=self.dim,
            inner_block_config=inner_cfg,
        )
        self.assertEqual(len(block.layers), 1)

    def test_zero_layers(self):
        block = MTPBlock(
            mtp_config=MTPConfig(num_layers=0),
            dim=self.dim,
            inner_block_config=_make_dummy_block_config(self.dim),
        )
        self.assertEqual(block.num_depths, 0)
        self.assertEqual(len(block.layers), 0)


class TestMTPAuxiliaryLoss(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.batch = 2
        self.seq_len = 8
        self.vocab_size = 100
        self.num_layers = 2

        self.lm_head = torch.nn.Linear(self.dim, self.vocab_size, bias=False)
        self.mtp_config = MTPConfig(
            num_layers=self.num_layers, loss_scaling_factor=0.3
        )

        self.pred = torch.randn(
            self.batch,
            (self.num_layers + 1) * self.seq_len,
            self.dim,
        )
        self.labels = torch.randint(
            0, self.vocab_size, (self.batch, self.seq_len)
        )

    def test_mtp_auxiliary_loss_shape(self):
        loss = mtp_auxiliary_loss(
            self.pred,
            self.labels,
            mtp_config=self.mtp_config,
            lm_head=self.lm_head,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0)

    def test_mtp_auxiliary_loss_zero_when_D_is_zero(self):
        cfg = MTPConfig(num_layers=0)
        loss = mtp_auxiliary_loss(
            self.pred,
            self.labels,
            mtp_config=cfg,
            lm_head=self.lm_head,
        )
        self.assertEqual(loss.item(), 0.0)

    def test_mtp_auxiliary_loss_scaling(self):
        loss1 = mtp_auxiliary_loss(
            self.pred,
            self.labels,
            mtp_config=MTPConfig(
                num_layers=self.num_layers, loss_scaling_factor=0.5
            ),
            lm_head=self.lm_head,
        )
        loss2 = mtp_auxiliary_loss(
            self.pred,
            self.labels,
            mtp_config=MTPConfig(
                num_layers=self.num_layers, loss_scaling_factor=1.0
            ),
            lm_head=self.lm_head,
        )
        self.assertAlmostEqual(loss2.item() / loss1.item(), 2.0, places=4)

    def test_mtp_auxiliary_loss_with_mask(self):
        loss_mask = torch.ones(self.batch, self.seq_len, dtype=torch.long)
        loss_mask[:, -2:] = 0

        loss = mtp_auxiliary_loss(
            self.pred,
            self.labels,
            mtp_config=self.mtp_config,
            lm_head=self.lm_head,
            loss_mask=loss_mask,
        )
        self.assertGreater(loss.item(), 0)


class TestMTPAwareLoss(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.batch = 2
        self.seq_len = 8
        self.vocab_size = 100

        self.lm_head = torch.nn.Linear(self.dim, self.vocab_size, bias=False)
        self.mtp_config = MTPConfig(num_layers=2, loss_scaling_factor=0.3)

        self.loss_fn = MTPAwareLoss(
            MTPAwareLoss.Config(
                loss_fn=CrossEntropyLoss.Config(
                    global_vocab_size=self.vocab_size
                ),
                global_vocab_size=self.vocab_size,
            )
        )
        self.loss_fn.set_lm_head(self.lm_head)
        self.loss_fn.set_mtp(self.mtp_config)

    def test_loss_output_shape_and_nonzero(self):
        pred = torch.randn(
            self.batch,
            (self.mtp_config.num_layers + 1) * self.seq_len,
            self.dim,
        )
        labels = torch.randint(0, self.vocab_size, (self.batch, self.seq_len))

        loss, metrics = self.loss_fn(pred, labels)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0)

    def test_without_mtp_delegates_to_inner(self):
        loss_fn = MTPAwareLoss(
            MTPAwareLoss.Config(
                loss_fn=CrossEntropyLoss.Config(
                    global_vocab_size=self.vocab_size
                ),
                global_vocab_size=self.vocab_size,
            )
        )
        loss_fn.set_lm_head(self.lm_head)
        # No set_mtp — D=0 path

        pred = torch.randn(self.batch, self.seq_len, self.dim)
        labels = torch.randint(0, self.vocab_size, (self.batch, self.seq_len))
        loss, metrics = loss_fn(pred, labels)
        self.assertEqual(loss.ndim, 0)

    def test_aux_loss_adds_to_main_loss(self):
        pred = torch.randn(
            self.batch,
            (self.mtp_config.num_layers + 1) * self.seq_len,
            self.dim,
        )
        labels = torch.randint(0, self.vocab_size, (self.batch, self.seq_len))

        loss_with_mtp, _ = self.loss_fn(pred, labels)

        # Loss without MTP (D=0): use lm_head directly
        loss_fn_no_mtp = MTPAwareLoss(
            MTPAwareLoss.Config(
                loss_fn=CrossEntropyLoss.Config(
                    global_vocab_size=self.vocab_size
                ),
            )
        )
        loss_fn_no_mtp.set_lm_head(self.lm_head)
        main_only = pred[:, :self.seq_len]
        loss_no_mtp, _ = loss_fn_no_mtp(main_only, labels)

        self.assertGreater(loss_with_mtp.item(), loss_no_mtp.item())


if __name__ == "__main__":
    unittest.main()
