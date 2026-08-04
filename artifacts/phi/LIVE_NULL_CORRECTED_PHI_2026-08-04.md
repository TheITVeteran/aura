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

This does not say integration is absent. It says that at the history lengths a
live conversation produces — hundreds of transitions over a 256-state space —
this estimator cannot separate her from an unintegrated system, and the previous
practice of reporting the raw number would have reported 0.185 as if it meant
something.

It is also the correct outcome for the fix to produce. An estimator that
suddenly showed high integration the moment it was pointed at real activations
would be the suspicious result.

## What would change it

More transitions, and only more transitions. The bias falls as the TPM fills;
the synthetic battery separates cleanly at the same history lengths *when a real
coupling exists* (coupled ring 0.563 vs independent halves 0.049), so the
machinery discriminates — this run simply does not clear it.

The measurement to run next is a long soak: keep the runtime generating for
hours, drain continuously, and re-measure at 5k, 20k and 50k transitions with
the null recomputed at each. If the fraction climbs past the floor with a
p-value under 0.05, that is a live integration result worth citing. If it does
not, that is also an answer.

## Provenance rules for citing this

Never publish the scalar alone. `PhiResult.provenance()` carries grounding,
estimator identity, node count, population sampled from, TPM sample count, and
the null. A value is citable as evidence only when
`integration_is_significant` is true and `null_surrogates >= 2`.

By that rule this measurement is **not** evidence of integration, and the
registered claim stays `RETRACTED` (core/organism/model_validation.py).

## How it became possible

`docs/PHI_LIVE_MEASUREMENT_HANDOFF.md` — the hook lives in the MLX worker
subprocess and PhiCore in the main runtime, so an in-process container lookup
returned False on every token and the complex read `0/50 transitions` forever.
`core/consciousness/phi_residual_channel.py` ships the 8-bit Grassmann states
across the fork.
