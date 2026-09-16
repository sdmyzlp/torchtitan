# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import _PAD_ID, HuggingFaceTextDataset

_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "tokenizer")


def _build_dataset(
    seq_len: int, *, pad_segments_to_multiple: int = 1
) -> HuggingFaceTextDataset:
    return HuggingFaceTextDataset(
        dataset_name="c4_test",
        dataset_path=None,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=True,
        pad_segments_to_multiple=pad_segments_to_multiple,
    )


class TestTextDatasetPacking(unittest.TestCase):
    """Greedy packing must emit exactly seq_len tokens with in-range positions.

    Inputs and labels are shifted per document at tokenization time, so a
    packed sample is seq_len long (not seq_len + 1). Emitting one extra token
    pushes the largest position to seq_len, which is one past the last entry of
    a RoPE cache sized at max_seq_len == seq_len and only surfaces as an async
    device-side assert deep inside the model.
    """

    def test_positions_are_contiguous_per_document_runs(self):
        it = iter(_build_dataset(256))
        for _ in range(100):
            positions = next(it)[0]["positions"]
            steps = positions[1:] - positions[:-1]
            # Each position either continues the current document (+1) or
            # restarts a new one (back to 0).
            self.assertTrue(bool(torch.all((steps == 1) | (positions[1:] == 0))))

    def test_no_cross_document_targets(self):
        """The last token of a document must never predict the next document."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        ds = _build_dataset(256)
        it = iter(ds)
        interior_doc_starts = 0
        for _ in range(100):
            input_dict, labels = next(it)
            input_ids = input_dict["input"]
            positions = input_dict["positions"]

            # EOS closes a document and is never fed back in; BOS opens one and
            # is never a target.
            self.assertFalse(bool(torch.any(input_ids == tokenizer.eos_id)))
            self.assertFalse(bool(torch.any(labels == tokenizer.bos_id)))

            starts = (positions == 0).nonzero().flatten()
            starts = starts[starts > 0]
            interior_doc_starts += len(starts)
            self.assertTrue(bool(torch.all(input_ids[starts] == tokenizer.bos_id)))
            # The token right before a document start predicts that document's
            # own EOS, not the next document's first token.
            self.assertTrue(bool(torch.all(labels[starts - 1] == tokenizer.eos_id)))

        # Guard against the assertions above passing vacuously.
        self.assertGreater(interior_doc_starts, 0)


class TestSegmentAlignment(unittest.TestCase):
    """Pad every document segment to a multiple for token-pooling models.

    A model that pools k consecutive tokens into one compressed entry pools two
    documents together whenever a document boundary is not a multiple of k, so
    the token stream pads each document to a multiple of k first.
    """

    _MULTIPLE = 4

    def test_every_document_segment_is_aligned(self):
        it = iter(_build_dataset(256, pad_segments_to_multiple=self._MULTIPLE))
        num_segments = 0
        for _ in range(50):
            positions = next(it)[0]["positions"]
            starts = (positions == 0).nonzero().flatten().tolist()
            for start, end in zip(starts, starts[1:] + [len(positions)]):
                self.assertEqual((end - start) % self._MULTIPLE, 0)
            num_segments += len(starts)

        # Guard against the assertions above passing vacuously.
        self.assertGreater(num_segments, 1)

    def test_pads_are_ignored_and_end_their_segment(self):
        it = iter(_build_dataset(256, pad_segments_to_multiple=self._MULTIPLE))
        num_pads = 0
        for _ in range(50):
            input_dict, labels = next(it)
            input_ids = input_dict["input"]
            positions = input_dict["positions"]

            # Inputs and labels are shifted per document, so every real token
            # predicts something: only an alignment pad carries IGNORE_INDEX.
            is_pad = labels == IGNORE_INDEX
            self.assertTrue(bool(torch.all(input_ids[is_pad] == _PAD_ID)))

            # A pad fills the tail of its own segment, and positions keep
            # running through it instead of restarting.
            starts = (positions == 0).nonzero().flatten().tolist()
            for start, end in zip(starts, starts[1:] + [len(positions)]):
                segment_pads = is_pad[start:end]
                first_pad = segment_pads.nonzero().flatten()
                if len(first_pad) == 0:
                    continue
                self.assertTrue(bool(torch.all(segment_pads[first_pad[0] :])))
                self.assertGreater(int(first_pad[0]), 0)
                pad_start = start + int(first_pad[0])
                self.assertEqual(
                    int(positions[pad_start]), int(positions[pad_start - 1]) + 1
                )
                num_pads += int(segment_pads.sum())

        # Guard against the assertions above passing vacuously.
        self.assertGreater(num_pads, 0)

    def test_rejects_unaligned_seq_len(self):
        with self.assertRaisesRegex(ValueError, "seq_len"):
            _build_dataset(255, pad_segments_to_multiple=2)
        with self.assertRaisesRegex(ValueError, "pad_segments_to_multiple"):
            _build_dataset(256, pad_segments_to_multiple=0)


class TestTextDatasetBufferCheckpointing(unittest.TestCase):
    def test_labels_buffer_round_trips(self):
        ds = _build_dataset(256)
        it = iter(ds)
        for _ in range(5):
            next(it)
        # Leave a partial sample in the buffers to checkpoint.
        self.assertGreater(len(ds._inputs_buffer), 0)

        state = ds.state_dict()
        self.assertIn("labels_buffer", state)

        resumed = _build_dataset(256)
        resumed.load_state_dict(state)
        self.assertEqual(resumed._inputs_buffer, ds._inputs_buffer)
        self.assertEqual(resumed._labels_buffer, ds._labels_buffer)
        self.assertEqual(resumed._positions_buffer, ds._positions_buffer)


if __name__ == "__main__":
    unittest.main()
