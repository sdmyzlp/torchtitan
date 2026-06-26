# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity test for the DeepSeek-V3.2 attention (MLA + Lightning
Indexer + DeepSeek Sparse Attention) against DeepSeek's official inference
implementation.

The golden reference is a CPU/fp32 port of ``DeepSeek-V3.2/inference/model.py``
(fp8/CUDA kernels replaced by their float idealizations; ``fast_hadamard_transform``
replaced by a pure-torch Sylvester Hadamard; see ``model_cpu.py``). Both sides run
in fp32 on CPU with the same random weights, so any difference is an *algorithmic*
discrepancy rather than bf16/fp8 deployment noise.

Set ``DSV32_REF_DIR`` to the directory containing the CPU-ported ``model_cpu.py``.
The test skips if the reference is unavailable.
"""

import os
import sys

import pytest
import torch

import torchtitan.tools.utils as _ttu

# The reference is CPU-only; force torchtitan onto CPU before anything reads
# the auto-detected device type.
_ttu.device_type = "cpu"
_ttu.device_module = torch.cpu

torch.set_default_dtype(torch.float32)

_REF_DIR = os.environ.get("DSV32_REF_DIR", "/home/lrwei/tmp/DeepSeek-V3.2/inference")


def _import_reference():
    if _REF_DIR and _REF_DIR not in sys.path:
        sys.path.insert(0, _REF_DIR)
    try:
        import model_cpu  # type: ignore

        return model_cpu
    except Exception:
        return None


M = _import_reference()

pytestmark = pytest.mark.skipif(
    M is None,
    reason=(
        "DeepSeek-V3.2 CPU reference not importable; set DSV32_REF_DIR to the "
        "directory containing the CPU-ported model_cpu.py."
    ),
)

# ---- matched (small) config; structurally faithful rope/eps ----
DIM = 256
N_HEADS = 16
Q_LORA = 64
KV_LORA = 128
QK_NOPE = 32
QK_ROPE = 16
V_HEAD = 32
MSCALE = 0.7
IDX_HEADS = 4
IDX_HEAD_DIM = 32
B, L = 2, 32
TOL = dict(atol=1e-4, rtol=1e-4)
# Large random weights deliberately amplify any latent floating-point
# divergence between the two implementations (a subtle algorithmic bug that
# is sub-ULP at small magnitudes becomes visible at large ones).
WEIGHT_STD = 0.5
NORM_STD = 0.2


def _apply_eager_flex_patches():
    """CPU analog of titan-demo's flex patches: run flex_attention and
    create_block_mask eagerly so the DSA path works on CPU without inductor."""
    from torch.nn.attention import flex_attention as _fa
    from torch.nn.attention.flex_attention import (
        create_block_mask as _eager_cbm,
        flex_attention as _eager_flex,
    )

    import torchtitan.models.common.attention as _ttattn
    import torchtitan.models.deepseek_v3_2.model as _dsmodel

    _ttattn.FlexAttention._compiled_flex_attn = staticmethod(
        lambda *a, **k: _eager_flex(*a, **k)
    )

    def _cbm_eager(*a, **k):
        k.pop("separate_full_blocks", None)
        k["_compile"] = False
        return _eager_cbm(*a, **k)

    _ttattn._compiled_create_block_mask = _cbm_eager
    _dsmodel.create_block_mask = _cbm_eager
    if hasattr(_fa, "_validate_device"):
        _fa._validate_device = lambda q, k, v: None


if M is not None:
    _apply_eager_flex_patches()


def _build_reference(index_topk):
    args = M.ModelArgs(
        max_batch_size=B, max_seq_len=16384, dim=DIM, n_heads=N_HEADS,
        q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE, v_head_dim=V_HEAD, original_seq_len=4096,
        rope_theta=10000.0, rope_factor=40, beta_fast=32, beta_slow=1, mscale=MSCALE,
        index_n_heads=IDX_HEADS, index_head_dim=IDX_HEAD_DIM, index_topk=index_topk,
    )
    mla = M.MLA(args)
    with torch.no_grad():
        for name, p in mla.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.normal_(1.0, NORM_STD)
            elif name.endswith("bias"):
                p.normal_(0.0, NORM_STD)
            else:
                p.normal_(0.0, WEIGHT_STD)
    freqs = M.precompute_freqs_cis(args)
    return mla, freqs


def _build_titan(index_topk):
    from torchtitan.models.deepseek_v3_2 import model_registry

    spec = model_registry("debugmodel")
    cfg = spec.model.layers[0].attention
    # NOTE: no eps / rope overrides -- this validates the shipped config.
    cfg.inner_attention.index_topk = index_topk
    cfg.indexer.index_topk = index_topk
    return cfg.build()


def _copy_weights(ref_mla, titan_attn):
    ref = dict(ref_mla.named_parameters())
    with torch.no_grad():
        for name, p in titan_attn.named_parameters():
            assert name in ref, f"missing in reference: {name}"
            assert ref[name].shape == p.shape, (
                f"shape mismatch {name}: {ref[name].shape} vs {p.shape}"
            )
            p.copy_(ref[name])


def _causal_mask_mod(b, h, q, kv):
    return q >= kv


def _build_case(index_topk, seed=0):
    torch.manual_seed(seed)
    ref, freqs = _build_reference(index_topk)
    titan = _build_titan(index_topk)
    _copy_weights(ref, titan)
    x = torch.randn(B, L, DIM)
    positions = torch.arange(L).unsqueeze(0).expand(B, L)
    fc = freqs[:L]
    causal2d = torch.full((L, L), float("-inf")).triu_(1)
    return ref, titan, x, positions, fc, causal2d


# Packed-document layout within the length-L sequence (positions reset per doc).
DOC_LENGTHS = (8, 12, 12)


def _build_packed_case(index_topk, doc_lengths=DOC_LENGTHS, seed=0):
    assert sum(doc_lengths) == L
    torch.manual_seed(seed)
    ref, freqs = _build_reference(index_topk)
    titan = _build_titan(index_topk)
    _copy_weights(ref, titan)
    x = torch.randn(B, L, DIM)

    # per-document positions reset to 0 at each document boundary
    posvec = torch.cat([torch.arange(dl) for dl in doc_lengths])  # (L,)
    positions = posvec.unsqueeze(0).expand(B, L)  # torchtitan: (B, L)

    # reference rope: gather freqs by per-document position
    fc_packed = freqs[posvec]  # (L, d/2)

    # reference mask: block-diagonal causal (same-document AND causal)
    doc_ids = torch.cumsum((posvec == 0).int(), dim=0) - 1
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    allowed = same_doc & causal  # (L, L)
    mask2d = torch.zeros(L, L).masked_fill(~allowed, float("-inf"))
    return ref, titan, x, positions, fc_packed, mask2d, allowed


def _titan_packed_block_mask(positions):
    """Build the same causal+packed-document flex BlockMask the decoder uses."""
    from torch.nn.attention.flex_attention import and_masks

    from torchtitan.models.common.attention import (
        create_attention_mask,
        get_causal_mask_mod,
        get_efficient_causal_mask_mod_for_packed_document,
    )

    mask_mods = [
        get_causal_mask_mod(),
        get_efficient_causal_mask_mod_for_packed_document(positions),
    ]
    return create_attention_mask(
        and_masks(*mask_mods), B, None, L, L, device="cpu", BLOCK_SIZE=128
    )


def test_hadamard_involution():
    """The pure-torch Hadamard must be a valid (involutive) normalized transform."""
    x = torch.randn(3, IDX_HEAD_DIM)
    y = M.rotate_activation(M.rotate_activation(x))
    assert torch.allclose(y, x, atol=1e-5)


def test_indexer_parity():
    ref, titan, x, positions, fc, causal2d = _build_case(index_topk=64)
    with torch.no_grad():
        qr_ref = ref.q_norm(ref.wq_a(x))
        ref.indexer(x, qr_ref, 0, fc, causal2d)
        dref = ref.indexer._dbg

        qr_t = titan.q_norm(titan.wq_a(x))
        idx_q_t, idx_w_t, idx_k_t = titan.indexer(x, qr_t, positions=positions)
        scores = torch.relu(
            torch.einsum("blhd,bsd->blhs", idx_q_t.float(), idx_k_t.float())
        )
        index_score_t = (scores * idx_w_t.unsqueeze(-1).float()).sum(dim=2)

    assert torch.allclose(qr_ref, qr_t, **TOL)
    assert torch.allclose(dref["idx_q"], idx_q_t, **TOL)
    assert torch.allclose(dref["idx_k"], idx_k_t, **TOL)
    assert torch.allclose(dref["weights"], idx_w_t, **TOL)
    assert torch.allclose(dref["index_score"], index_score_t, **TOL)


def test_dense_mla_parity():
    """index_topk >= seq_len -> all keys selected -> pure dense causal MLA."""
    from torch.nn.attention.flex_attention import create_block_mask

    ref, titan, x, positions, fc, causal2d = _build_case(index_topk=64)
    bm = create_block_mask(_causal_mask_mod, B, None, L, L, device="cpu", _compile=False)
    with torch.no_grad():
        ref_out = ref(x, 0, fc, causal2d)
        tt_out = titan(x, bm, positions)
    assert torch.allclose(ref_out, tt_out, **TOL)


def test_sparse_dsa_parity():
    """index_topk < seq_len -> genuine sparse selection; check selection + output."""
    from torch.nn.attention.flex_attention import create_block_mask

    from torchtitan.models.deepseek_v3_2.model import Indexer as TIndexer

    topk = 8
    ref, titan, x, positions, fc, causal2d = _build_case(index_topk=topk, seed=1)
    bm = create_block_mask(_causal_mask_mod, B, None, L, L, device="cpu", _compile=False)
    causal_valid = torch.tril(torch.ones(L, L, dtype=torch.bool))
    with torch.no_grad():
        ref_out = ref(x, 0, fc, causal2d)
        tt_out = titan(x, bm, positions)

        qr_t = titan.q_norm(titan.wq_a(x))
        iq, iw, ik = titan.indexer(x, qr_t, positions=positions)
        sel_t = TIndexer.select(iq, ik, iw, bm, topk) & causal_valid

        qr_ref = ref.q_norm(ref.wq_a(x))
        topk_ref = ref.indexer(x, qr_ref, 0, fc, causal2d)
        sel_ref = torch.zeros(B, L, L, dtype=torch.bool).scatter_(-1, topk_ref, True)
        sel_ref = sel_ref & causal_valid

    assert sel_ref.sum() > 0  # selection is actually sparse/non-trivial
    assert torch.equal(sel_t, sel_ref), "DSA top-k selection differs from reference"
    assert torch.allclose(ref_out, tt_out, **TOL)


def test_document_packing_dense():
    """Packed documents (positions reset per doc), index_topk >= L -> dense
    block-diagonal causal MLA. Output must match the reference fed an equivalent
    block-diagonal mask + per-document rope positions."""
    ref, titan, x, positions, fc, mask2d, _allowed = _build_packed_case(index_topk=64)
    bm = _titan_packed_block_mask(positions)
    with torch.no_grad():
        ref_out = ref(x, 0, fc, mask2d)
        tt_out = titan(x, bm, positions)
    assert torch.allclose(ref_out, tt_out, **TOL)


def test_document_packing_sparse():
    """Packed documents + sparse DSA: selection must stay within documents and
    match the reference, and the output must match."""
    from torchtitan.models.deepseek_v3_2.model import Indexer as TIndexer

    topk = 6
    ref, titan, x, positions, fc, mask2d, allowed = _build_packed_case(
        index_topk=topk, seed=2
    )
    bm = _titan_packed_block_mask(positions)
    with torch.no_grad():
        ref_out = ref(x, 0, fc, mask2d)
        tt_out = titan(x, bm, positions)

        qr_t = titan.q_norm(titan.wq_a(x))
        iq, iw, ik = titan.indexer(x, qr_t, positions=positions)
        # effective selection = top-k masked to same-document causal positions
        sel_t = TIndexer.select(iq, ik, iw, bm, topk) & allowed

        qr_ref = ref.q_norm(ref.wq_a(x))
        topk_ref = ref.indexer(x, qr_ref, 0, fc, mask2d)
        sel_ref = torch.zeros(B, L, L, dtype=torch.bool).scatter_(-1, topk_ref, True)
        sel_ref = sel_ref & allowed

    assert sel_ref.sum() > 0
    assert torch.equal(sel_t, sel_ref), "packed-doc DSA selection differs from reference"
    # output parity also guarantees no cross-document attention leaked through
    assert torch.allclose(ref_out, tt_out, **TOL)
