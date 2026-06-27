# Non-parametric memory probe — does it inject knowledge the weights lack?

**Date:** 2026-06-27 · **Model:** Qwen2.5-7B-Instruct-4bit (MLX) · **Datastore:** 8 fictional facts
**Method:** facts the base model CANNOT know (invented entities), so any correct recall is purely
from the datastore. Real hidden states (`model.model(ids)[0,-1]`, dim 3584) as keys; next-token
prediction compared BARE vs INTERPOLATED. Run: `python -m aura_bench.nonparametric_probe`.

## Result

| condition | bare top-1 | + memory top-1 | mean gold-token prob |
|---|:--:|:--:|:--:|
| exact context | **0/8** | **8/8** | 0.013 → 0.501 |
| paraphrased context | **0/8** | **3/8** | 0.007 → 0.252 |
| control (a fact it knows) | — | **preserved** | "Paris" unchanged |

## Honest reading

- **The mechanism is real and works.** Bare model 0/8 on knowledge it lacks → 8/8 exact recall
  purely from the datastore. Capacity was genuinely *added* (token-level), not coincidence.
- **Paraphrase recall (the realistic case) is partial: 3/8 (~38%).** Real questions are worded
  differently than what was stored, so the hidden-state neighborhood only sometimes matches. This
  is the honest real-world ceiling of this tiny datastore — semantic recall, but not reliable yet.
- **No corruption.** A fact the model already knew ("Paris") was preserved — distant fictional
  neighbors → low adaptive-λ → no false injection. The safety property holds.
- **n is tiny (8 facts).** This proves the mechanism; it is not a population-level capability number.

## What this does and does NOT establish

- DOES: token-level non-parametric memory injects recall the frozen weights don't have, on a real
  model, fail-open and non-corrupting. Capacity accumulates as facts are added (0→8).
- DOES NOT: it is still a STANDALONE probe — NOT wired into Aura's live generation loop. The Φ/FE
  inputs were fixed test values (phi=0.5, fe=0.9), not fed from the live consciousness substrate.
  So: mechanism validated; live/causal integration is the remaining step (background-first).

## Next to make it live (safe order)

1. Ingest trusted content (solved-cache answers, beliefs) → real keys, in the IDLE loop.
2. Apply `interpolate`/`apply_to_logits` in the BACKGROUND generation path first (no foreground
   latency, can't destabilize the live model), measure real-conversation lift.
3. Graduate to foreground with a latency budget + the Φ-gate feeding real phi/free-energy.
