# Non-parametric memory — KV-cached generation (validated, production form)

> **Historical record — 2026-07-14.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](../../docs/DOC_STATUS.md).

**Date:** 2026-06-27 · **Model:** Qwen2.5-7B-Instruct-4bit (MLX) · `python -m aura_bench.nonparametric_probe`

Full pipeline: ingestion engine + real MLX encoder + **KV-cached** `generate_with_memory`.
Fictional facts the model cannot know → any correct output is purely datastore-sourced.

## Result (KV-cached + anisotropy-corrected gate)

| test | bare | + memory |
|---|:--:|:--:|
| **end-to-end GENERATION** | 0/5 | **5/5** ✅ |
| next-token, exact context | 0/8 | **8/8** |
| next-token, paraphrase | 0/8 | 1/8 |
| control ("Paris") preserved | — | ✅ |

```
[Tessaly]  mem='Tessaly. She has a'
[Myrrhal]  mem='Myrrhal, a city'
[Brannoch] mem='Brannoch. Brann'
[Velreth]  mem='Velreth, the goddess'
[Sethryn]  mem='Sethryn. The Sethryn'
```

## How it got here (three real fixes, each caught by a real run)

1. **First run: generation 0/5.** Only first-token pairs were ingested → mid-answer positions had no
   correct neighbor. Fix: **full-sequence ingestion** (prefix-hidden → next-token at every position).
2. **λ floor injected garbage.** The additive λ never reached 0, so spurious far neighbors corrupted
   mid-generation ("Br Br Br"). Fix: **gated λ** (zero below the confidence gate) + **anti-stutter**
   (the entry that fired last step is excluded next step).
3. **The raw-cosine gate was invalid.** Hidden states are **anisotropic**: measured, UNRELATED prompts
   score raw cosine **0.81–0.93** (identical = 1.0), so a 0.55 raw gate separates nothing. Fix:
   **mean-centred similarity** (unrelated ≤0.36) with calibrated thresholds
   (`min_similarity()`: 0.60 centred / 0.98 raw fallback).
4. **Latency:** the first working loop recomputed the forward per token (O(n²)). Fix: **KV-cached**
   loop — prefill once, then **O(1) per token**. This is the production form.

## Honest reading

- **Capacity injection is proven in GENERATION (5/5)** at production latency. The model outputs
  knowledge it cannot have from weights, sourced token-by-token from the datastore.
- **Paraphrase next-token is 1/8** — the strict, correctly-calibrated gate trades recall for
  precision. It fires only on confident recall; that's the right default (a wrong injection is worse
  than none), but it means real-world paraphrased questions often won't benefit yet.
- **No corruption** of known facts (control preserved). **n small** — proof of mechanism, not a
  population number.
- **Live status:** ingestion is background-wired (flag+pressure gated). The KV-cached generation is
  validated standalone. Making it the live FOREGROUND path still requires a worker-side
  `generate_step` that yields hidden states while preserving the worker's streaming, samplers,
  logits-processors and stop-sequences — a distinct refactor of the live generation core, not done here.
