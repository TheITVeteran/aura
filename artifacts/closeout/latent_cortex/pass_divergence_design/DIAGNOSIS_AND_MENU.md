# Pass-divergence design: diagnosis and training menu (Fable lane, 2026-07-24)

Claimed in `docs/RLC_SPARK_EXECUTION_LEDGER.md` (Fable-lane record,
commit 8eb2b0eb). Scope: diagnosis, code, tests, small-model measurement.
**No training is launched from this lane** — SPARK-069's admission
preflight, campaign protocol, and every 32B run stay bound to the march.

## 1. Why fifteen recurrent-GRPO campaigns produced no learning signal

Two independent failures stack. Either alone is fatal; both are present.

### Failure A — the reward channel is flat (upstream of recurrence)

`cp305_resident_32b_recurrent_grpo/training/grpo_receipt.json`: the
baseline eval is 0.0 at every depth, and every episode reads
`contract.reason: no_marker` with `decode_termination: token_limit`. With
`cot: true` and `max_tokens: 320` the checkpoint never reaches the answer
marker, so every sample is unparseable, every reward identical
(`format_credit: 0.0`), and GRPO's group advantage is exactly zero.
Terminal diagnosis agrees: `uniform_partial_reward`. **No architecture
can learn through a reward with zero variance.** The march already owns
this half (CP309–313 answer-channel curriculum + source-bound admission
preflight); it is listed here because any divergence intervention will be
invisible until admission passes.

### Failure B — the architecture has no mechanism for passes to differ

The slot-path update is `Z_{t+1} = (1−α)·Z_t + α·RMSMatch(Window(Z_t))`
with a **step-invariant** Window and `alpha_schedule: "constant"`
(α = 0.5) in every campaign protocol. A damped iteration of one fixed
map is a local power iteration: successive increments align with the
dominant eigendirection of `(1−α)I + αJ`, so pass t+1 re-computes pass
t's step instead of taking a new one. cos(pass1, pass2) = 0.9994 is the
signature of that dynamic, not a training deficiency — **no amount of
GRPO on the fixed architecture can remove it**, because nothing in the
architecture is capable of expressing "do something different this pass":

* `wrap_depth_conditioned` (per-step operator deltas, CP219) has **no
  call site in the campaign path** — only `tools/train_intrinsic_recurrence.py`
  attaches banks. The forward seam is live (`recurrence_step` publishes
  the depth index; `ScopedLoRALinear` consults `depth_bank` when
  present), so the campaign trains a mechanism-in-name-only.
* `jitter_scale` perturbs **branches**, not passes. `collapse_cos_threshold`
  deduplicates **branches**, not passes. Neither touches the contraction.
* On the intrinsic path the levers exist but were disengaged:
  CP227 trained at `rotation_weight = 0.0` (its own RESULTS.md calls
  this the floor), `anchor_injection = 0.0`.

### Instrument defect — the accuracy gate compared base against base

`tools/eval_intrinsic_accuracy.py::_decode` ran `recurrent_logits`
outside `recurrence_adapter_scope`, where `ScopedLoRALinear` is dark, so
BOTH arms of `cp227_accuracy_gate/` decoded the bare base model — hence
on@d == off@d exactly (6/2/0 in both arms at every depth). **The gate's
negative verdict is VOID.** Whether CP227's CE crossover converts to
accuracy is unknown, not refuted. Repaired in commit eb3735c7 (scope
entered, activation counts recorded, FATAL on a dark block;
`tests/test_eval_intrinsic_accuracy_instrument.py` pins it). The gate
must be re-run before any conclusion about the CP227 adapter is drawn.

## 2. Measured lever menu (1.5B, T=4, receipt `measurement_1p5b.json`)

mean cos(Δt, Δt+1) across 6 prompts, pairs (1→2, 2→3); lower = each pass
computes a more distinct step. All arms finite, zero divergence.

| condition | cos pair 1 | cos pair 2 | reading |
|---|---:|---:|---|
| baseline (plain loop) | 0.746 | 0.944 | power-iteration capture; alignment GROWS with depth |
| renormalize | 0.202 | 0.278 | removing shared magnitude growth is the single biggest lever |
| anchor 0.1 (+renorm) | 0.151 | 0.215 | input re-injection decorrelates further, dose-dependent |
| anchor 0.3 (+renorm) | 0.118 | 0.113 | |
| noise 0.05 (+renorm) | 0.063 | 0.077 | near-orthogonal; exploration lever, not computation |
| noise 0.1 (+renorm) | 0.013 | 0.018 | |
| step-ops δ=0.005 | 0.689 | 0.869 | below threshold — inert |
| step-ops δ=0.02 | −0.266 | −0.194 | above threshold — overshoots into anti-alignment |
| all levers stacked | −0.467 | −0.318 | toward the period-2 failure; do NOT stack blindly |

Honest boundary: this is geometry on an untrained model with random
step-operator deltas. Rotated increments are **necessary** for depth to
compute anything new, not **sufficient** for it to compute something
useful. Only trained runs answer usefulness — that is the march's lane.

## 3. Recommended order of operations

1. **Re-run the accuracy gate with the repaired instrument** on the
   existing CP227 adapter (no new training needed). If accuracy now
   converts, the cheapest possible win was sitting behind a dark scope.
   Runbook: same command as the first run; the tool now hard-fails if
   the adapter never fires.
2. **Admission before architecture** (march-owned): no campaign until
   the answer-channel preflight yields measured parseability > 0 and
   reward variance > 0 at the campaign's decode budget. cp305 shows 320
   tokens with `cot: true` does not reach the marker on this checkpoint.
3. **Next intrinsic training run** (the one CP227 already proved safe):
   `renormalize=True`, `anchor_injection=0.1`, `rotation_weight` in
   0.05–0.2 (the lever built for this exact obstacle, never engaged),
   depths (1,2,4), anchor_weight 1.0 as before. Success telemetry: the
   objective already emits per-depth `rotation.mean_cos`; the CP227
   grader already emits `depth_helps`/`collapse_repaired`.
4. **Campaign wiring for step-conditioned operators** (march-owned,
   ~3 lines): after the campaign attaches its ScopedLoRA adapters, call
   `wrap_depth_conditioned(model, depths=<max recurrent_steps>)` and add
   the bank tensors to the trainable set. The consuming seams already
   exist: `recurrence_step` publishes the index
   (`core/brain/llm/latent_cortex/recurrence.py`), `ScopedLoRALinear`
   consults `depth_bank`. Zero-init deltas keep attach identity-safe.
5. **Per-sample seeded inter-pass noise as GRPO exploration**
   (`RecurrentDepthPlan.interpass_noise`, new in eb3735c7): give each
   group sample a distinct `noise_seed` and the latent trajectory itself
   varies within a group — reward variance from the recurrence side,
   which the uniform-reward diagnosis says is exactly what the groups
   lack. Start at 0.05 with `renormalize=True`; anneal toward 0 as
   admission stabilizes. Deterministic per seed, so receipts replay.
6. **Alpha schedule**: `cosine` instead of `constant` in the execution
   spec (field already exists) — aggressive early steps, gentle late
   ones; constant 0.5 is the most contraction-friendly choice available.

## 4. What would falsify this diagnosis

* A campaign whose admission passes (nonzero reward variance) and whose
  per-step operators are attached, still showing cos(Δt,Δt+1) > 0.99
  after training — would mean the contraction is weight-dominated, not
  architecture-dominated, and the menu above is insufficient.
* A repaired-instrument accuracy-gate re-run showing on@d == off@d
  *with nonzero activation counts* — would mean CP227's deltas are
  genuinely sub-threshold at decode, and training pressure (rotation,
  larger delta_scale) matters more than instrumentation did.
