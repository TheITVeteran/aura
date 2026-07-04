# Ablation legibility — reviewer-runnable subsystem deltas

External reviewers ask one thing above all else:

> Run Aura baseline vs *without-memory / without-Will / without-substrate /
> without-reasoning-amplifier / without-verifier / without-System-2* and see
> **measurable deltas.**

This is that, in one command. Each condition drives a **real organ** (no mocks,
no clamps, no hardcoded statistics) intact vs lesioned and reports the measured
metric for both, the delta, and an honest load-bearing verdict. A no-delta
result is reported as *not* load-bearing — never hidden.

## One command

```bash
.venv/bin/python tools/ablation_runner.py
```

Writes `artifacts/ablation/ablation_scorecard.json` and prints:

```
condition             metric delta    baseline→ablated      load-bearing
--------------------------------------------------------------------------
without_substrate     0.0850          0.085 → 0.000         True
without_system2       1.0000          1.000 → 0.000         True
without_verifier      0.5000          1.000 → 0.500         True
```

`--list` shows every condition; `--conditions substrate,verifier` runs a subset.
Exit code is 0 iff every executed condition is load-bearing.

## What each condition actually does

| Condition | Real organ | Metric | Method |
|-----------|-----------|--------|--------|
| `without_substrate` | `core.being.policy_coupler.ClosedLoopPolicyCoupler` | policy divergence across 7 causal self-states (mean pairwise normalized L1) | intact coupler vs the same organ blinded to a neutral state |
| `without_system2` | `core.reasoning.proof_answer_solver` | strict-proof exact-answer rate | real symbolic solver vs no deterministic derivation path |
| `without_verifier` | `…proof_answer_solver.validate_strict_proof_answer` | balanced accuracy (accept correct + reject wrong) | real validator on (correct, wrong) pairs vs a null verifier that accepts everything |

The `without_substrate` metric is exactly the felt-state → action-policy claim:
uncertainty, trust debt, resource pressure, governance pressure, identity
tension and workspace coherence measurably change temperature, verification
threshold, tool-risk budget, planning depth and memory retrieval depth. Blind
the coupler to that state and every context collapses to one identical policy
(distinct policies 7 → 1, divergence → 0).

## Context conditions — honest deferral, not a null

`will`, `memory`, and `reasoning_amplifier` only exercise their organ with
runtime or model context. A bare offline harness starves the Will of its
stakes/permission context and starves memory/amplifier of the model, so a flat
offline "0" would be *misleading theater*, not evidence. Each therefore points
to the dedicated runner that sets the context up honestly:

- **will** → `tools/agi/run_will_governance_ablation.py` — real `UnifiedWill`
  decisions vs a neutral governance packet, on a survival/critical/initiative
  benchmark.
- **memory** → a retention battery against `/api/chat` on the running kernel
  (plant a fact, recall it after a gap; grounded recall vs no-memory).
- **reasoning_amplifier** → `ReasoningAmplifierV2(generate=…)` with the live
  model: verifier-filtered self-consistency vs single-sample.
- **causal agency** → `tools/agi/run_causal_agency_lesion.py` — the same
  substrate coupling proven with a permutation p-value and receipt coverage.

## Honesty guarantees

- Every number is measured from the real organ at run time.
- No clamps, floors, or hardcoded statistics (the causal-agency runner was
  rewritten in July 2026 to remove exactly those; see its module docstring).
- A condition that shows no delta is reported `load_bearing: false`. That is a
  real, publishable result — it means the subsystem is not load-bearing on that
  battery, and the battery/claim should be revised, not the number.
