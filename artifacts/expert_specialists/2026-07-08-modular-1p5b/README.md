# Domain-specialist proof bundle — 2026-07-08, `modular` on Qwen2.5-1.5B-4bit

The supply side of the expert-adapter architecture, proven end to end in one
unsupervised run (`proof_run.log` is the full transcript). No human touched
anything between harvest and verdict.

## The chain

1. **Self-play harvest** — the base model sampled 4 attempts per task at
   temperature 0.85 against 40 seeded `modular` arithmetic tasks (training
   seeds < 1000). Every attempt graded by the task's exact checker; 32
   verified win/loss contrasts became DPO pairs
   (`selfplay_preferences.jsonl`, rows carry domain provenance). Base
   correct-rate at temperature: 23.8% — real headroom, real contrast.
2. **DPO train** — 230 s on the contrast pairs (`run/data/` holds the exact
   rows).
3. **Two-sided sealed gate** (`run/modular-1783521611.json` is the receipt):
   - domain-concentrated battery (specialist seeds ≥ 2000, disjoint from all
     training seeds): **base 0.25 → candidate 0.50** — the specialist
     doubled the base model's domain accuracy;
   - general battery (seed 1500): 0.625 → 0.5625, inside the 10 pp collapse
     tolerance — the domain gain was not bought with general collapse.
   - Every raw model response for all four evals is in
     `run/*.responses.jsonl`, regradeable offline.
4. **Registration** — promoted into the ExpertLoRALibrary as
   `modular-specialist-20260708-074401` (`library_manifest.json`), the same
   registry the live router consults for background reasoning work.
5. **Hot-attach verification** — the adapter applied to the RESIDENT model
   via the worker's own attach helpers (112 layers wrapped, ~0.01 s):
   sealed-domain accuracy base **0.250 → attached 0.312 → detached 0.250**
   (byte-exact restore; the personality-adapter-safe bookkeeping held).

## Honest observations

- **Attached 0.312 vs gate 0.50**: the gate evaluates via
  `mlx_lm.load(model, adapter_path=…)` in a fresh process; the hot-attach
  measures the in-place wrap. Both beat base and detach restores exactly,
  but the gap between load-path and wrap-path effective weights is a real,
  open observation — tracked for investigation before specialists serve
  user-facing traffic (routing is background-only and env-gated off by
  default, `AURA_EXPERT_LORA_ROUTING`).
- The first proof attempt (same pipeline, `arithmetic_chain`) was REFUSED:
  the base already scores 1.000 on that domain gate, so no gain was
  claimable — the gate held instead of manufacturing a specialist. That
  refusal receipt is in the compounding work log; refusals print with the
  same prominence as promotions throughout.

Adapter weights (~40 MB) are not committed; every decision they fed is
reproducible from the receipts and data above.
