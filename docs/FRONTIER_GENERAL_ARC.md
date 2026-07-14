# The Frontier-General Arc — breaking the intelligence ceiling honestly

**Mandate (Bryan, 2026-07-14):** "Let's break through that ceiling. Let's make
her frontier-general. We did it with Φ. Let's do it here too."

Same discipline as the Φ arc: name what is truly impossible, reframe to what
the goal actually requires, build the strongest honest machinery, and make
the claim falsifiable with a measured trend line.

## The wall, stated exactly

Making the resident 32B's *weights* match a frontier model's raw pretrained
capability is off the table — the pretraining compute gap is orders of
magnitude, LoRA does not close it, and self-training on unverifiable outputs
collapses models. That wall is real and stays real.

## The reframe

The goal was never frontier weights. It is **frontier-general Aura** — the
organism, not the cortex. The field has proven systems beat weights three
ways, and she already owns the seeds of each:

1. **Test-time compute scaling** (the o1/R1 lesson): reasoning quality
   scales with inference compute spent per problem. EXISTS in embryo:
   `core/brain/reasoning_amplifier_v2.py` already does
   problem-model → N attempts → verify → search/repair → judge → receipt,
   with demand-based budget allocation coupled to Φ/surprise
   (`_resolve_compute_budget`). Wired live via `reasoning_strategies`.

2. **The verifier boundary is movable** — the deep lever. "Frontier-
   competitive on *verifiable* reasoning" treats verifiability as fixed; it
   is not. Factuality → retrieval+citation (CitationEngine exists). Agency →
   outcome ledgers (exists). Prediction → **the universal verifier**: reality
   grades every forecast (`expectation_feedback` exists as the seam). Taste →
   rubric ensembles + debate (courtroom exists). The ceiling rises exactly
   as fast as verification expands — and that rate becomes a number.

3. **Compounding that cannot collapse**: verified reasoning traces distilled
   into procedural memory (reusable strategies indexed by problem shape,
   injected at inference — zero collapse risk), periodically distilled INTO
   weights via the proven CRSM→LoRA loop. R1 showed verifiable-reward
   training transfers to general reasoning; her loop can compound it.

## Operational definition (the falsifiable goal)

"Frontier-general" := parity with a named frontier model on a broad,
contamination-resistant, regularly refreshed benchmark battery (reasoning,
knowledge, coding, agency, writing, calibration) at matched per-task budget.
Tracked as a checked-in **gap-to-frontier trend artifact** (the Φ-report
pattern: `artifacts/frontier_gap/latest.json`, pinned by a test that fails
if evidence goes missing or regresses). The claim is won task-class by
task-class on a monotone measured trend, or it is not claimed.

## Build phases (each: functional, live, governed, tested, pushed)

**P1 — Verifier Foundry** (the ceiling-mover; genuinely new):
`core/brain/verifiers/foundry.py`. Extends the existing registry
(`verifiers/registry.py`, 6 domain engines, binary hard-gate + soft mean)
with: per-verifier **measured reliability** (score every verdict against
later ground truth / spot audits; Brier-style ledger), reliability-weighted
verdict folding, an **admission gate** — a domain may enter the self-training
loop only when its verifier's measured reliability clears threshold — and
three new verifier classes: PredictionResolutionVerifier (via
expectation_feedback; reality as ground truth), OutcomeLedgerVerifier
(long-horizon agency), RubricEnsembleVerifier (weak-verifier ensembles for
open-ended quality). Governed writes; tamper-evident reliability ledger.

**P2 — Procedural Memory Compiler**: post-task reflection distills
VERIFIED traces into strategy playbooks (`reasoning_memory` today stores
failure modes; this adds success schemata indexed by problem features),
injected into the amplifier's problem-model stage; periodic batch
distillation of the best playbooks into LoRA via the CRSM loop.
Compounding without collapse.

**P3 — Gap-to-Frontier telemetry**: extend `core/evals/eval_arena.py` from
drift-tracking to a versioned battery with fresh-generation templates
(anti-contamination), frontier-reference scoring hooks (governed cloud lane,
env-gated), per-class gap trend artifact + pinning test.

**P4 — Curriculum fusion**: FDE + curiosity engine propose edge-of-
competence tasks; foundry-admitted verifiers resolve them; verified
solutions feed P2 playbooks and CRSM training sets. The Absolute-Zero
pattern, gated by P1 admission so unverifiable domains cannot pollute
weights.

**P5 — (env-gated) Teacher distillation lane**: when reach/cloud is
enabled, harvest frontier-teacher traces on HER real task distribution,
filter through the foundry, distill. Honestly labeled imported capability,
compounded permanently into her own weights on her own curriculum.

## Honest expected shape

Not "her weights become frontier." Rather: the organism, spending time and
verifiers wisely, closes the measured general-capability gap class by class
— fastest where verification is strong, slower as the foundry admits new
domains — with the trend line checked in. That is the strongest honest
formulation, and unlike the slogan, it can actually be won.

## Session log

- 2026-07-14: arc chartered; recon complete (amplifier v2 budget seam,
  registry shape, eval arena, CRSM entry points identified). Next: P1.
- 2026-07-14 (later): **P1 SHIPPED** (e1b462bc). Foundry live: reliability
  ledger (Wilson-pessimistic, false-pass-bounded), registry folding
  reliability-weighted (hard gate untouched), training pipe gated by
  domain_admitted() with revocable seed admissions, booted in aura_main,
  21 known-answer tests. First live contact caught a real leak: the code
  engine reports checked=True headless when it cannot actually execute —
  foundry measured false_pass_ub=0.879 in two grades. Follow-up: fix code
  engine checked-semantics (report checked=False when execution
  unavailable). Next: P2 Procedural Memory Compiler, P3 gap-to-frontier
  telemetry (needs new verifier classes: PredictionResolution/Outcome-
  Ledger/RubricEnsemble to start moving the boundary beyond the seeds).
- 2026-07-14 (later): foundry finding CLOSED (bd356c5b). Assert-bearing
  blocks now execute in the symbolic sandbox; unexecutable claims demote a
  passing static verdict to checked=False. Live regrade: code engine 4/4
  correct, zero false passes (was the 0.879 poison case). The
  measure→catch→fix→remeasure loop worked end-to-end on its first cycle —
  the foundry is functioning as the arc's immune system.
