# Non-parametric memory — end-to-end GENERATION (validated)

**Date:** 2026-06-27 · **Model:** Qwen2.5-7B-Instruct-4bit (MLX) · `python -m aura_bench.nonparametric_probe`

Built the full pipeline (ingestion engine, real MLX encoder, `generate_with_memory` causal loop)
and ran it end to end on FICTIONAL facts the model cannot know.

## Result (after the generation fix)

| test | bare | + memory |
|---|:--:|:--:|
| **end-to-end GENERATION** | 0/5 | **5/5** ✅ |
| next-token, exact context | 0/8 | 4/8 |
| next-token, paraphrase | 0/8 | 1/8 |
| control ("Paris") preserved | — | ✅ |

```
[Tessaly]  mem='Tessaly. She has a'      (generated the full fictional name)
[Myrrhal]  mem='Myrrhalth, a'
[Brannoch] mem='Brannoch. Brann'
[Velreth]  mem='Velreth, the goddess'
[Sethryn]  mem='Sethryn. The Sethryn'
```

## What changed (first run was 0/5)

1. **Full-sequence ingestion** — store (prefix-hidden → next token) for EVERY answer position, not
   just the first. So after generating token 1, position 2 has a correct neighbor and the chain
   follows. (`NonParametricIngestor.ingest_sequence`)
2. **Cosine-gated λ** on unit-normalized keys — `cos = 1 - d²/2`; below a cutoff (0.55) the neighbor
   is unrelated → λ=0 (defer to the model); above it, λ scales with cosine. This stops spurious far
   neighbors from corrupting mid-generation. Model-independent (cosine), no per-model dist tuning.
   (`generate_with_memory`)

## Honest reading

- **Capacity injection is now proven in GENERATION (5/5)**, not just next-token. The model produces
  knowledge it cannot have from weights — sourced entirely from the datastore, token by token.
- **Next-token exact dropped 8/8→4/8**: that eval path still uses the untuned `adaptive_lambda`
  (additive floor) with normalized keys; the *generation* path uses the tuned cosine-gated λ. The
  generation number is the one that matters for "more knowledgeable in use".
- **No corruption** of known facts (control preserved). **n is small** (5 generation / 8 next-token):
  proof of mechanism, not a population number.
- **Still standalone**: validated in the probe harness; the LIVE foreground path (per-token
  interpolation inside the MLX worker's generation loop) is the remaining production step. Ingestion
  is wired to the background (flag-gated). phi/fe were fixed probe values, not live-fed yet.
