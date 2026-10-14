# Current material marks feed facing, October 14, 2026

Presentation readiness remains open. These measurements exercise pixel
observation and direction binding with a supplied body identity. They do
not prove control discovery, ordinary app entry, game completion or the
56-game campaign.

## Connection and custody

The current point pose now retains its visible named marks. Proper rigid
correspondence yields a relative angular phase. A phase is image geometry;
it receives no facing authority from a character label or game identity.

Owned travel pairs a position-window chord with the phase observed at
its midpoint. The window starts after the delivered input settles and
after overlapping input ends. It cannot cross a different input receipt,
body, pose basis or an observation gap above 0.35 seconds. The bounded
window extends to 1.2 seconds and rejects an arc wider than a quarter turn.
Candidate windows compete on measured uncertainty. Three agreeing
observations with at least 30 degrees of phase spread qualify the relation.
The 0.12-radian qualification limits remain unchanged.

A calibrated relation can follow a current measured phase while the body
stops translating. Both `facing_now` and the travel vector use that current
heading. A later picture cannot borrow the old phase. Unknown velocity
error supplies no calibration sample; measured point positions have their
own error estimate and can supply a resolved chord.

Position and angle keep separate reference histories. An existing support
bridge preserves position through matching pixels and exterior edges. It
provides no rotation measurement. A new angular basis can retain its old
reference only through proper rigid correspondence on at least three
continuing visible marks. Missing, reflected, deformed or stale matches
start a new basis. Bridge errors accumulate as RMS estimates; these are
not bounds on correlated tracking errors or drift.

New marks are back-projected through the complete reference transform,
including when the current frame has no separate local rotation receipt.
An all-consensus fit avoids the hypothesis sweep only when every residual
meets the existing rigid-fit limit. Other cases retain the bounded sweep.
This uses the existing CPU point observer without model weights.

## Falsification and measurements

The first 36-frame probe yielded 29 phases and no calibrated direction.
Velocity uncertainty blocked its samples; its shorter moving interval
also limits the midpoint-phase spread. The later 60-frame probe is not a
same-input comparison with that first run.

The first extended pixel probe calibrated one texture and followed all
20 stopped-turn frames. A second texture then failed the same threshold.
The first attempted correction calibrated it but admitted a wrong angle
after a reference reset: maximum error was 1.348 radians. Splitting the
references and requiring a measured angular bridge removed that error.
Its first conservative repeat supplied 16 of 20 stopped-turn frames and
still failed the unchanged test threshold.

Angular RMS error now uses the actual consensus residuals and a nonzero
quarter-pixel resolution term. Position error retains the maximum fitted
residual. This distinguishes an RMS angular estimate from a maximum
positional error; neither is a formal guarantee.

Final separate repeats, using the default classical observer:

| Texture seed | Frames | Current phases | Calibrated stopped-turn frames | Median angle error | Maximum angle error | Observer and binding time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 762 | 60 | 53 | 20 of 20 | 0.02356 rad | 0.03031 rad | 1.39885 s |
| 831 | 60 | 53 | 20 of 20 | 0.01621 rad | 0.02853 rad | 1.39420 s |

The private probes use an oracle only to select the rendered singleton.
The regression tests supply entity 1 explicitly and feed the controller
only observations and input receipts. The renderer's position and angle
remain in the renderer and assertions. Both textures pass stopped-turn
coverage and error checks; rotating animation during straight travel
supplies no calibrated direction in either texture.

One scoped run passed 201 tests in 23.27 seconds. Another passed 196 tests
in 15.83 seconds. These suites overlap; their counts must not be added as
unique coverage. Registered canaries cover current geometric evidence
and the qualification boundary.

- Final seed 762: `/private/tmp/codex-steer/pixel-direction-final-762-1014/summary.json`
- Final seed 831: `/private/tmp/codex-steer/pixel-direction-final-831-1014/summary.json`
- Each final directory contains source hashes taken before measurement.
- Counterexample before the reference correction: `/private/tmp/codex-steer/pixel-phase-direction-texture-831-1014/summary.json`
- Gate log: `/private/tmp/codex-steer/measured-angular-direction-gates-1014.log`

Checkpoint gates passed Ruff, writing, changed-method size, layering,
dependency consistency, compile, lint and smoke. Smoke passed 165 tests
with one skip in 45.16 seconds. The global size ratchet reports inherited
debt: `ResponseGenerationPhase` has 22 methods against its 20-method
baseline, and oversize exceeds the budget by 191 lines. Those paths were
not changed here. Governance lint reports 123 existing regressions; its
offender list contains none of this checkpoint's changed modules. No
baseline was widened. Governance log:
`/private/tmp/codex-steer/qualified-angle-governance-1014.log`.
