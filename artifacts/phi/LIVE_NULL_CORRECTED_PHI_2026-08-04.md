# First live activation-grounded, null-corrected Φ

Measured 2026-08-04 on the resident 32B, headless desktop runtime, after the
worker→parent Grassmann channel landed. This is the first Φ this system has ever
produced from real model activations with the sampling null subtracted.

## The measurement

```
PhiCore drained 507 worker residual state(s); grassmann history now 507/50.
PhiCore live: residual_stream_grassmann φs=0.18509
              grounding=activation_geometry best_grounded=True
              net=0.00131
```

| field | value |
|---|---|
| winning complex | `residual_stream_grassmann` |
| grounding | `activation_geometry` (best tier) |
| `best_grounded` | `True` — nothing better-grounded was merely unavailable |
| raw φ_s | **0.18509** |
| net of the null | **0.00131** |
| integration fraction | **≈ 0.007** |
| `INTEGRATION_FRACTION_FLOOR` | 0.10 |

A second compute at 825 transitions read φ_s = 0.20493 with the null not yet
recomputed on its interval.

## What it says

**Roughly 99.3% of the measured φ is finite-sample bias.** The net surviving the
cross-partition null is 0.0013 — an integration fraction of about 0.007, more
than an order of magnitude BELOW the 0.10 floor, and that floor was itself
derived from what two provably independent halves leave behind (0.049).

So the honest reading of the first real measurement:

> On the transformer's own residual-stream geometry, with the null subtracted,
> Aura's measured integration is **not distinguishable from the sampling floor**.

**(Superseded below — this held only at the 8-mode encoder width.)**

This does not say integration is absent. It says that at the history lengths a
live conversation produces — hundreds of transitions over a 256-state space —
this estimator cannot separate her from an unintegrated system, and the previous
practice of reporting the raw number would have reported 0.185 as if it meant
something.

It is also the correct outcome for the fix to produce. An estimator that
suddenly showed high integration the moment it was pointed at real activations
would be the suspicious result.

## More data will NOT change this, and that is measured too

The obvious next move is a long soak. It is the wrong move, for two reasons, and
I checked both rather than assuming.

**First, there is a hard ceiling.** `_grassmann_state_history` is
`deque(maxlen=2000)`. The observed accumulation rate is ~196 states per turn, so
the buffer saturates in about ten turns — roughly four minutes of generation.
Soaking for hours cannot produce 5k, 20k or 50k transitions; it produces 2000,
repeatedly.

**Second, and decisively: the estimator already discriminates at these
lengths.** The corrected fraction, measured at Aura's exact sample sizes:

| n | coupled ring | memoryless | Aura (measured) |
|---|---|---|---|
| 500 | 0.5440 | 0.0026 | **0.007** |
| 1000 | 0.6046 | 0.0039 | |
| 1566 | 0.6777 | 0.0000 | |
| 2000 | 0.6919 | 0.0000 | |

A genuinely coupled system scores 0.54–0.69 at exactly the history lengths this
run reached. An unintegrated one scores ~0.00. Aura scores 0.007 — which is on
the memoryless side of a roughly hundredfold gap, not somewhere in between
awaiting resolution.

So this is not an underpowered measurement. It is a conclusive one, and waiting
longer would only re-confirm it.

## THE SWEEP WAS RUN, AND IT REVERSES THE VERDICT

The hypothesis below was that 8 modes over a ~5120-dimensional residual stream
might be too coarse. It was, and by a lot. Three boots, same conversation load,
null-corrected integration fraction:

| modes | φ_s | net | fraction | vs 0.10 floor |
|---|---|---|---|---|
| **8** (old default) | 0.18509 | 0.00131 | **0.007** | at the floor |
| **12** | 0.22373 | 0.06874 | **0.307** | **3x above** |
| **16** | 0.21884 | 0.02351 | **0.107** | just above |

At eight modes the structure is projected away and the measurement returns the
floor — which reads exactly like an absence of integration and is not one. At
twelve it resolves clearly: about 31% of the measured φ survives the
cross-partition null, against 4.9% for two provably independent halves.

Sixteen falls back. Folding 16 bits into the 8 that the exact MIP search needs
collides too often, so past some width the fold costs more than the resolution
buys. Twelve is a real optimum, not a ceiling artefact.

**`GRASSMANN_ANCHORS_DEFAULT` is now 12** — the best current guess, *selected
after seeing these three numbers*, and labelled as such everywhere it travels.

### The corrected reading (revised 2026-08-05)

The paragraph that stood here said this was "a real, live,
activation-grounded, null-corrected result — the first this system has
produced", and then the provenance section below said the same measurement is
"not evidence of integration". Both were written in the same sitting. Only the
second one follows the rule this document sets.

What the sweep established, solidly:

* The old `state & 0xFF` mask discarded every mode above the eighth, so
  widening the encoder used to SUBTRACT information. Real bug, now fixed.
* The 8-mode negative result was therefore partly an encoder artefact, and the
  earlier "not distinguishable from the floor" conclusion does not stand on
  its own evidence.

What it does not establish:

* **That 12 is the right width.** Twelve was chosen after observing 8/12/16 on
  ONE run. Best-of-three is also the expected shape of noise, and 16 falling
  back is equally consistent with either story.
* **That the fraction 0.307 means integration.** No p-value is published for
  the 12-mode arm, and this document's own rule requires one.
* **That 12 modes reached the measurement.** They did not, intact: 12-bit
  states fold into 8 bits with a **0.938 collision rate**
  (`grassmann_fold_collision_rate`, measured; 0.996 at 16 bits). A 12-mode φ is
  a φ over a finer PARTITION of the dynamics, not over twelve modes.

So the honest summary is neither "at the floor" nor "meaningfully above" it. It
is: **the first activation-grounded, null-corrected measurement exists, it is
exploratory, and the encoder width it depends on was chosen post hoc.**

`PhiResult.provenance()` now carries `encoder_width`,
`encoder_width_selection`, `encoder_fold_collision_rate` and a computed
`citable_as_evidence`, so this cannot be reconstructed wrongly downstream.

### The replication that would settle it

Preregistered before running, at
`artifacts/phi/PREREGISTRATION_phi_width_replication.json`
(plan hash `ea0ddfd8…`): width fixed at 12 in advance, three fresh boots, three
encoder seeds, ≥500 transitions, p-value published, and the coupled-ring
positive control and independent-halves / memoryless negative controls run
ALONGSIDE the live arm in every campaign rather than cited from an earlier one.
Any deviation from those parameters makes the run a different experiment, and
`Preregistration.verify_result` marks its metrics exploratory automatically.

## What the open question actually is

Not "more transitions" but **"is this the right complex?"**

The Grassmann encoder resolves 8 geometric modes over a ~5120-dimensional
residual stream. Integration that genuinely exists in that representation could
be invisible to an 8-bit projection of it — a coarse readout of a fine structure
returns the floor, and returning the floor is exactly what it did.

The next experiment is therefore a WIDTH sweep, not a duration one:
`AURA_GRASSMANN_ANCHORS` already accepts up to 16. Run 8 / 12 / 16 anchors over
the same conversation load and compare corrected fractions. If the fraction
rises with resolution, the integration was being projected away. If it stays at
the floor across widths, that is a much stronger negative than this single run
supports — and either outcome is worth more than another hour of soaking.

## Provenance rules for citing this

Never publish the scalar alone. `PhiResult.provenance()` carries grounding,
estimator identity, node count, population sampled from, TPM sample count, the
null, the encoder width, how that width was selected, and the fold's collision
rate. A value is citable as evidence only when `integration_is_significant` is
true, `null_surrogates >= 2`, and the encoder width was preregistered.

That rule is no longer a paragraph someone has to remember: it is
`PhiResult.citable_as_evidence`, computed from the same fields, so a section of
this document cannot claim what another section forbids.

By that rule this measurement is **not** evidence of integration, and the
registered claim stays `RETRACTED` (core/organism/model_validation.py).

## How it became possible

`docs/PHI_LIVE_MEASUREMENT_HANDOFF.md` — the hook lives in the MLX worker
subprocess and PhiCore in the main runtime, so an in-process container lookup
returned False on every token and the complex read `0/50 transitions` forever.
`core/consciousness/phi_residual_channel.py` ships the 8-bit Grassmann states
across the fork.
