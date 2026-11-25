# GLM-5.3-Flash text backbone

This model defines its own Compressor, Indexer, InnerIndexer, sparse attention,
and mHC modules. It uses common TorchTitan decoder, KDA projection, FFN and MoE
building blocks, with no imports from the DeepSeek or Kimi model packages.

The full flavor has 45 layers: 34 KDA and 11 NoPE sparse MLA layers, four mHC
streams, three dense FFNs followed by 42 MoEs, 288 routed experts with top-8
routing and one shared expert. The text model has 313,326,799,870 parameters,
including the LM head. The five-layer CPU debug flavor retains both attention
families, mHC, dense/MoE FFNs and a 2,048-token vocabulary. The
`debugmodel_4layer` flavor keeps the released hidden/attention widths with
three KDA layers and one DSA layer, a 2,048-token vocabulary and four routed
experts (top-2). Its 1,136,901,656 parameters fit the single-device NPU probe.

Architecture sources are the [official configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json),
[Transformers GLM5Next](https://github.com/huggingface/transformers/tree/v5.22.1/src/transformers/models/glm5_next),
and [vLLM GLM5Next](https://github.com/vllm-project/vllm/tree/08c497efa9a5213f2903670655f040977ba03fbd/vllm/models/glm5next/common).

## Kernel boundaries and indexer loss

```text
Indexer
  detached x/q -> index projections -> Compressor (complete groups of 4)
                                     -> InnerIndexer
                                          | topk_indices, topk_scores
                                          v
InnerAttention
  expand each selected pool into 4 tokens + append 0..3 incomplete tail tokens
  -> sparse attention -> output
```

`InnerIndexer` belongs to `Indexer`. Its pool indices and scores go directly
to `InnerAttention`. The compressor never adds a tail. GLM has no overlapping
compression or sliding-window attention, so the two streams contain disjoint
tokens. The full model selects up to 512 pools (2,048 tokens) plus the tail.

All modules pass `attention_metadata` through their kernel boundaries. The
Decoder builds it once per backend and batch. Pool rows, visible pool limits,
tail indices and KDA document offsets are prepared there; forward only expands
the indexer's data-dependent selections.

The reference path uses Attention Gym `gather_attn(..., impl="reference")` and
`bound_gate`/`chunk_kda` reference implementations. These work on CPU as well
as accelerators. The four-tap convolution uses PyTorch because Attention Gym's
convolution has no reference backend. Packed document boundaries reset all
recurrences, convolution history and pooling.

`IndexerLoss` trains only indexer/compressor parameters; its inputs from the
trunk and its teacher are detached. For each query, let `p` be head-mean
attention probabilities summed within each selected four-token pool, and
`Z = sum(p)`. The objective is `Z * KL(p/Z || softmax(scores))`. The tail is
excluded from `p`, so `Z` can be less than one. Padding queries contribute zero;
the denominator is the step's global valid-token count.


## Run

```bash
source ~/pytorch/venv/3.12/bin/activate
OMP_NUM_THREADS=2 python -m pytest -q tests/unit_tests/cpu/models/glm53_flash
```

Runtime dependencies are the repository's pinned requirements, including
Attention Gym revision `755691f3`. Transformers 5.22.1 is optional and used only
by the reference parity tests. No vendor library is imported by the core model.

```python
import torch
from torchtitan.models.glm53_flash import build_model_config

config = build_model_config("debugmodel", seq_len=32)
model = config.build()
model.init_states()
positions = torch.arange(32)
logits = model(
    torch.randint(config.vocab_size, (32,)),
    positions=positions,
    attention_metadata=model._get_attention_metadata(positions),
    aux_loss_denominators=torch.tensor([32]),
)
logits.float().square().mean().backward()

with torch.device("meta"):
    full_model = build_model_config("glm53_flash").build()
```

The debug recipe uses the bundled C4 fixture, seed 42, deterministic settings
and 10 AdamW steps.

This commit supports the single-device reference model. Tensor, pipeline, expert,
data and context parallel degrees greater than one are rejected.
