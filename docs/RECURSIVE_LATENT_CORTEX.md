# Recursive Latent Cortex (RLC)

Turn the frozen resident checkpoint from a fixed-depth 64-layer pipeline into
a **programmable, stateful, self-configuring reasoning machine** — without
changing a single stored weight. This is the productionization of the
"Anima Rationis" spec: latent workspace, controlled recurrence, layer-schedule
programs, virtual-width branches, hidden-state optimization, episode-scoped
fast weights, adaptive halting, and a falsification harness that keeps every
capability claim honest.

Package: `core/brain/llm/latent_cortex/` (worker-side, pure MLX, lazy imports)
Service: `ServiceNames.LATENT_CORTEX = "latent_cortex"` (orchestrator-side)
Worker action: `latent_reason` (runs on the RESIDENT model, no reload)

## The invariant (checked, not promised)

```
Checkpoint bytes:        unchanged  (SHA-256 cached per (path, mtime, size))
Permanent parameters:    unchanged  (sampled-tensor fingerprint pre/post episode)
Episode fast weights:    provably erased (post-erase probe-batch equality)
No hidden fine-tuning:   consolidation only via the governed LoRA queue
```

`governance.CheckpointInvariant` enforces all four and emits a receipt per
episode. A violated invariant is a CRITICAL degradation and the episode's
output is discarded.

## Architecture

```
prompt ──prefill (all 64 layers, standard)──▶ prompt KV (read-only memory)
seed M thought slots (mean prompt embedding + role anchors + jitter)
slots ──prelude [0..p)──▶ Z₀ at layer p          (slot KV persists for [0..p))
loop over schedule program π (windows within [p..c), repeats, α):
    Z̃   = Window(Zₜ)          # slots attend to prompt KV + own KV, RoPE-stable
    Uₜ   = RMSMatch((1-αₜ)Zₜ + αₜ·RMSMatch(Z̃, A), A)
    Zₜ₊₁ = Uₜ if CalibratedAccept(Evidence, Zₜ, Uₜ, A) else Zₜ
                                                        # A: fixed post-prelude anchor
    slot KV REWOUND every pass  (only clean final pass persists)
    halting: fixed-point residual, divergence guard, overthinking revert
branches: K independent workspaces, Exchange every E steps via comm slot
optional: latent optimization of Z (∇_Z on reconstruction+manifold proxy,
          verifier accept/reject); episode fast-weights ΔW=UVᵀ (identity-start)
final clean pass [p..c) persists slot KV; coda [c..64) persists slot KV
decode: answer tokens attend to [prompt; refined slots] at every layer
```

### Why slots ride the KV cache
The decoded answer must be **causally downstream of the latent computation**,
not decoration. Persisting the refined slots' K/V at every layer means every
generated token attends to them. Ablating a slot (Experiment 3) measurably
changes the answer — that is the causality contract.

### Controlled recurrence (not naive looping)
2026 frozen-loop studies show naive repetition is unstable. Controls:
- **RMSMatch**: per-position RMS rescaling toward the immutable post-prelude anchor,
  ratio-clamped — keeps Z on the activation manifold the next layers expect.
- **α-interpolation** with configurable schedule (constant / cosine decay).
- **Calibrated update admission**: a pinned learned sigmoid scores bounded
  evidence/anchor/dynamics features before state mutation; below-threshold
  proposals are receipted and discarded while the exact prior state persists.
- **Divergence guard**: NaN or norm-ratio blowout ⇒ halt, revert to best state.
- **Fixed-point halting**: relative residual ‖Zₜ₊₁−Zₜ‖/‖Zₜ‖ < ε ⇒ converged.
- **Overthinking guard**: track best-scoring state; revert on halt if the
  trajectory peaked early (recurrent-depth literature's overthinking failure).

### Layer-schedule programs
A schedule is a validated program `[(start, end, repeats, α), ...]` over the
middle region — each transformer block becomes an instruction. Canonical
serialization + content hash; per-domain reliability tracked with Wilson
bounds (same math as the Verifier Foundry). `ScheduleSearch` (evolutionary,
budgeted, deterministic seeds) may only promote a schedule on **verified**
task improvements, and the library stores provenance receipts. Whatever the
program did, one clean final pass guarantees coherent slot KV.

### Virtual width (branches)
K workspaces over the SAME weights: different role anchors (constructor,
counterexample-hunter, checker, simplifier, …), same prompt KV (read-only).
Exchange: consensus of branch summaries blended into a designated
communication slot. Anti-collapse: decorrelation jitter when pairwise branch
cosine exceeds threshold. Selection at halt: verifier score, else convergence
quality. Equal-FLOP accounting (token-layer applications) is first-class so
Experiment 4 can honestly compare against self-consistency sampling.

### Latent optimization (gradient descent over thoughts)
Differentiable proxy that cannot leak answers:
`S(Z) = λ_r·R(Z) − λ_d·D(Z, Z₀)` where **R** = teacher-forced logprob of the
prompt's own tokens decoded from Z through the coda (the document's
"reconstruct the problem" term) and **D** = manifold distance (RMS drift +
cosine drift from Z₀). Non-differentiable verifier signal enters via
accept/reject hill-climbing on decoded probes. `control_step()` applies a
matched-magnitude **random** perturbation — the Experiment-5 control is part
of the API, not an afterthought.

### Episode fast weights (test-time self-configuration)
Low-rank ΔW = s·U Vᵀ on selected window-layer linears (o_proj / down_proj),
V initialized to zero ⇒ **exact identity at attach**. U,V are optimized
during the episode by the same proxy/verifier loop (test-time training with
frozen base; seeded from slot statistics). Lifecycle is a ratchet:
`ATTACHED → EVALUATED → ERASED`, and erase is **proven** by unwrapping and
asserting probe-batch output equality with the baseline. Candidates that
repeatedly win go to `data/latent_cortex/consolidation_queue/` (governed
writes) for the existing LoRA-compounding loop — permanent learning stays
behind the existing regression gates.

### Compute economy
Budget currency is **token-layer applications** (prefill = L·64; a recurrence
step = M·window). The Will/metabolic layer allocates an episode budget from
stakes, uncertainty, and BodyState pressure; hard caps + wall-clock deadline
guard the worker. Cap hits are info-level backpressure, not failures.

### Fail-honest contract
Any divergence, invariant breach, or budget exhaustion mid-episode ⇒ the
engine returns the best verified state it has, or falls back to the vanilla
path, **with a receipt saying exactly what happened**. No silent fallback,
no theatrical success.

## Falsification harness (`experiments.py`)

| # | Experiment | Verdict it can earn |
|---|------------|---------------------|
| 1 | Recurrence utility sweep (windows × repeats × α, vs vanilla / longer-CoT / best-of-N at equal FLOPs) | positive ∂accuracy/∂steps curve |
| 2 | Depth extrapolation (k-hop reachability, nested boolean, modular chains; deterministic generators) | T_required ∝ problem depth |
| 3 | Slot causality (ablate → specific loss; restore → recovery) | workspace carries computation |
| 4 | Virtual width vs equal-FLOP self-consistency | branches beat sampling or they don't |
| 5 | Latent opt vs matched-magnitude random perturbation | gradient direction matters or it doesn't |
| 6 | Frontier comparison (equal information/tools/compute; blind fresh tasks) | the only claim that counts |

Results are graded claims — PROVEN / SUPPORTED / CONJECTURE / REFUTED — and
verifier verdicts are recorded to the Verifier Foundry reliability ledger.

Experiment 6 evidence is certified by a standalone verification kernel
(`frontier_verifier.py` + `tools/verify_latent_cortex_frontier.py`) that
recomputes every raw binding from disk. Two comparison kinds are supported:
`resident_32b_vs_vanilla_same_checkpoint` (same-checkpoint superiority) and
`resident_32b_vs_external_frontier` (frontier comparison). External-frontier
evidence must pin the control model/build/provider in the preregistration,
bind every trial's control receipt to those pins, and ship the raw provider
responses in a `provider_receipts` store whose per-trial SHA-256 is
recomputable — otherwise the package is rejected. Supporting a comparison
kind is evidence machinery, not a capability claim: no external-frontier
campaign has run yet.
Task generators are seeded and self-verifying (graph reachability, boolean
evaluation, modular-arithmetic composition), so Experiments 1–5 run offline
on any checkpoint, including the tiny random-weight Qwen2 used by the test
suite (mechanics proof) and the real 32B (capability measurement, operator-
launched, bounded).

## Runtime wiring

- **Worker** (`mlx_worker.py`): action `latent_reason` — runs the engine on
  the resident model under the metal semaphore; refuses while generation is
  in flight; returns `{text, receipts}`. KV/prompt caches cleared after
  episodes that attached fast weights (weights changed ⇒ caches invalid).
- **Client** (`mlx_client.py`): `latent_reason_async(...)` mirroring the
  `set_expert_adapter` request/response pattern.
- **Service** (`core/brain/latent_cortex_service.py`): resolves budgets from
  the Will/metabolic state, exposes `deep_reason(...)`, registers under
  `ServiceNames.LATENT_CORTEX`, participates in the health contract.
- **Causal path**: deep-deliberation/cognitive-engine routes depth-worthy
  problems through the latent cortex when the Will allocates depth.
  Kill switch: `AURA_LATENT_CORTEX=0`. Budgets conservative by default.

## Honest claims ladder (current state)

- **PROVEN (test suite, real mlx_lm Qwen2 architecture):** mechanics —
  KV rewind correctness, RMSMatch stability bounds (anchored trust band; the
  moving-reference ratchet failure is regression-tested), schedule
  validation, identity-at-attach and proven-erase for fast weights,
  checkpoint invariant, slot-ablation causality on the answer distribution,
  matched-magnitude control arm, equal-FLOP accounting, grader
  conservatism (underpowered ⇒ CONJECTURE, compute mismatch ⇒ voided).
- **PROVEN (real trained checkpoint, Qwen2.5-1.5B-Instruct-4bit — the same
  quantization format as the resident 32B):** the full episode pipeline runs
  end to end in ~1.3s: contracting residuals (0.95 → 0.10, a genuine fixed
  point on trained weights), branch exchange + selection, invariant clean,
  coherent chain-of-thought text decoded through persisted thought slots —
  with a visible qualitative effect (latent-conditioned decode skips
  preamble and starts computing).
- **CONJECTURE (until Experiments 1–5 run on the 32B via
  tools/latent_cortex_lab.py):** capability gains. Frozen-loop literature
  says expect small broad gains; the integrated machine (workspace +
  schedules + width + optimization + fast weights) is the untested
  combination the spec argues could be qualitatively more. The harness is
  built so this question gets ANSWERED, not vibed.
- **Not claimed:** new world knowledge from recurrence. That comes from the
  memory/tool organs, per the spec's own boundary.

## Operational notes

- Live path: restart Aura ⇒ worker gains the `latent_reason` action; deep
  deliberation routes DEEP passes through latent episodes automatically.
  Kill switch `AURA_LATENT_CORTEX=0`. Budgets damped by body pressure.
- Lab runs (operator-launched, bounded, memory-safe — 1.5B/7B only while
  the live 32B is resident):
  `caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py --model <mlx-dir> --experiments 1,2,3,5 --max-minutes 30`
- Consolidation candidates land in `data/latent_cortex/consolidation_queue/`
  for the existing LoRA-compounding regression gates; nothing consolidates
  from inside an episode.

## Closeout state (2026-07-17, head 86d27cf0+)

**Live resident-32B evidence through the signed installed app:**
`artifacts/current/cp106_live_latent_turn.json` remains the first authentic
full-stack answer. `artifacts/current/cp118_live_latent_turn.json` retains
mechanics credit for the complete organ head: verifier-guided branch
selection, typed cognitive ingress, execution-controller observation, live
consolidation export, EOS floor, repetition guard, newline discipline,
sentence grace, time-aware wall wind-down, accepted fast-weight descent,
proven erase, and unchanged base parameters. Its product-quality PASS is
revoked by `artifacts/current/cp118_live_latent_turn_review.json`: the exact
public reply leaked request/protocol text, never selected an architecture,
never supplied the requested cancellation/timeout/worker-restart verification
plan, and did not match the text hash that the original receipt graded. It is
mechanism evidence only, not a complete-answer, capability, frontier, or
release certificate.

**Organ inventory (the "Core architecture" additions):** all seven runtime
organs landed - recurrence-native training objective (+ recurrent-depth
curriculum loss, train/inference norm parity with the engine's anchored
trust band), learned per-problem execution controller (evidence-gated
contextual bandit, verified-outcome rewards), full neural bytecode with
verifier-guided backtracking, role lesion/swap causality (Experiment R),
GWT↔RLC bidirectional coupling, continuous pre-action cortex loop,
retrieval→ΔW compilation; plus fast-weight capability canaries, latent
safety telemetry, attractor escape ladder, vector organ ingress,
held-out facet grading, verifier arbitration over ΔW, and durable
adapter distillation with the anti-interference battery.

**Training programs (items 10–16) — machinery landed 2026-07-18:** the six
programs now exist as real modules in `core/learning/` alongside the
already-landed recurrence-native objective and depth curriculum:
`transition_grading` (every consequential step scored on named dimensions;
reliability compounds multiplicatively; verified failures never train
positive), `on_policy_repair` (earliest CAUSAL error by replay bisect from
the agent's own trajectory; corrections from the exact reached state;
retained only on rerun success + transfer majority; emits the
(state, operations, best operation, verified outcome) unit),
`teacher_federation` (verifier has the last word, Wilson-bounded
reliability ledgers break ties — never prestige; unverifiable agreement is
tiered consensus_unverified; verified failures kept as negatives),
`minimax_curriculum` (P(d) ∝ (1−S_d/S_ref)^γ toward the weakest measured
domain, replay floor, explicit exploration share for unmeasurable domains),
`social_outcome_learning` (delayed relational outcomes price the reward;
the manipulation guard zeroes dishonestly-won gains; untracked
theory-of-mind caps credit), `robustness_families` (structured slots +
alternative templates generate paraphrase/reorder/rename/value-change/
distractor/mislead/missing/contradiction variants with truthfully
recomputed answers; grading rewards invariance AND correct movement).
These are program machinery with contract tests; no training run under
them has produced capability evidence yet — runs remain operator-launched
under the single-owner resident protocol.

**Probe memoization (item 20):** decode probes are memoized per episode on
the exact latent state (`probe_cache.py`); a hit costs the budget nothing,
savings are receipted, and every fast-weight lifecycle transition flushes
the cache via the `on_function_change` hook — a probe memoized under a
different model function is a lie, and the invalidation trail proves the
boundary held. Prompt-KV sharing across branches and O(1) reference
snapshots were already in place.

With these, all 22 items of the RSL addendum are implemented: runtime
organs live in `core/brain/llm/latent_cortex/`, ingress/coupling seams in
`core/brain/`, training programs in `core/learning/`. Implementation is
not capability: the preregistered campaigns above still hold the honest
capability verdicts, and the recurrence-native resident training (CP139+)
is the arm expected to move them.

The in-episode fast-weight boundary now has two independent guards. The
behavioral battery measures protected continuations, while an exact structural
check computes the RMS of each effective `scale * U @ V.T` update from
rank-sized Gram matrices. A destructive update that happened to improve every
fixed continuation score exposed the behavioral battery's blind spot during
the CP119 broad gate; it now deterministically walks the bounded rescale ladder
and erases before decode when its magnitude remains non-finite or above the
configured ceiling. This hardens temporary adaptation but is not evidence of a
capability gain.

**Capability evidence (honest):** template-parity 32B sweep - latent
accuracy scales 0.167→0.375 from 1→2 recurrent steps (the live
profile's setting) then plateaus; vanilla 0.417 leads on point estimate
with fully overlapping Wilson intervals at n=24. Statistical parity,
graded CONJECTURE. The mechanism is proven causal, cheap (~3.5s of a
~110s episode; decode dominates), and live; the intelligence dividend
has not yet appeared in the data. The recurrence-native objective exists
precisely because the frozen-loop ceiling is real — the next capability
move is training, not more runtime machinery.

## Preregistered offline campaign (2026-07-17, seed committed first)

`artifacts/current/latent_campaign_prereg_20260717.json` pinned the fresh
task seed (20260717), power (n=24/family, 72/arm), and hypotheses BEFORE
any campaign task was generated. Vehicle: Qwen2.5-1.5B-Instruct-4bit
(the resident 32B stayed live). Reports: `latent_campaign_1p5b_run{1,2}.json`.

| Experiment | Verdict | The data |
|---|---|---|
| 1 recurrence sweep | CONJECTURE | no monotone step curve |
| 2 depth extrapolation | CONJECTURE ×3 families | no T∝depth signal |
| 3 slot causality | **REFUTED** | no ablated slot caused specific loss (n=72) |
| 4 virtual width | CONJECTURE (negative point) | self-consistency beat branches: boolean 12v4, khop 4v0 |
| 5 latent opt | CONJECTURE | gradient == random control == off, exactly |
| A factorial ablations | **all 7 arms REFUTED** | vanilla 21/72 beats every latent arm (7–13/72) |

The honest headline: **on an untrained-for-recurrence checkpoint at this
scale, the frozen-loop RLC does not merely fail to help — it hurts.**
Plain decoding wins at matched budgets. Combined with the 32B parity
sweep, the campaign converts "the intelligence dividend has not appeared"
from an impression into a preregistered, adequately-powered, Holm-corrected
result. The runtime machinery is causal, governed, and live; the dividend
must come from recurrence-native training (the objective + curriculum
losses are the entry point) — exactly what this harness was built to be
able to say without flinching.

## First resident-32B consolidation execution

`artifacts/current/latent_consolidation_train_32b_first.json` records the first
real fused-32B execution of the durable-learning transaction. Seven candidates
passed provenance and honesty screening, one domain adapter was distilled over
layers 16-17, the 11-probe anti-interference battery passed at 1.0 stable
fraction, activation succeeded, and exact rollback restored both layers. Two
invalid candidates were rejected for explicit honest flags. This proves the
candidate-to-adapter-to-gate-to-activation-to-rollback machinery on the resident
checkpoint. The run intentionally ended in proven rollback and did not measure
held-out reasoning improvement, so it is not evidence that a retained adapter
improves capability or reverses the preregistered null/regression result.
