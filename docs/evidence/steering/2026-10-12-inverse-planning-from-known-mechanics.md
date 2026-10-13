# Inverse planning from known mechanics, 2026-10-12

The user asked for executable mathematics from the available game logic,
rather than waiting for a model to discover a strategy. The new bounded
inverse planner accepts a forward transition, explicit outcome constraints,
an ordering cost, and a complete state identity. It searches input paths
and returns the predicted states with its plan. Failed intermediate states
are removed before ranking; later recovery cannot erase a failure. Search
budget exhaustion and beam pruning remain explicit. The planner returns a
prediction, not an observed outcome.

The supplied state may include position, velocity, posture, stored control
response, and remaining supply. No names, keys, colours, or archived-game
parameters appear in the planner. A numerical reference module accepts
channel rules and output matrices as data. Its rules represent gradual
build-up, released decay, passive drift, and ordered channel updates.

## Measurements

The invented-example suite passed 38 tests in 4.51 seconds. These cover
inverse arithmetic, motion in both directions ending at rest, posture and
supply constraints, continuing motion after release, typed outcome
requirements, finite costs, allocation bounds, and an exact transition
budget. Returned input sequences are replayed by independent arithmetic
for the inertial and stored-response cases. The registered claim executes
the adjacent failure-boundary invariant.

Ruff, writing, changed-method size, module size, layering, dependency
checks, compile, and lint passed. Smoke passed 165 tests with one skip in
105.11 seconds. Logs under `/private/tmp/codex-steer/`:
`required-state-final-tests.log`, `required-state-gates-summary.log`,
and `required-state-checkpoint-gates.log`.

## Inspected source reference

The authorized offline program remains SHA-256
`a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
The source reader was instrumented externally to retain bytecode branch
targets during inspection. The saved output is
`/private/tmp/codex-steer/landing-branch-targets-18.json`.

A separate translation of the interior movement recurrence was compared
with the parameterized forward model over 2,400 ticks from 200 seeded
initial states and varying controls. Maximum numeric disagreement was
`1.1368683772161603e-13`. This comparison covers build-up, decay, drift,
lateral movement, and tilt. Boundary resets, contact, rendering, and other
objects were excluded from that numerical equivalence check.

The source-reference search then included the inspected movement and
landing checks, including the near-top thrust reset, horizontal wrapping,
and a ground failure. Its first cost penalized early movement and failed
the long transfer. Projecting the stopping place and remaining tilt solved
16 of 18 starting-position and target pairs. The two long transfers still
descended before alignment could finish. Adding the neutral-channel motion
over the time needed to align solved all 18 tested pairs. The final search
took 1.82 to 3.27 seconds per case on this host, including process startup.
Those timings are too slow to run at the picture rate.

The first batch and its failures remain in
`source-landing-batch-18-v2.json`; the final batch is
`source-landing-batch-18.json`, with `source-landing-batch-18-v3.log`.
Each reference uses direct full numerical state and a fixed destination
height. It is not a run of the original emulator. Obstacles, assets,
animation, observation uncertainty, and live input delivery are not modeled
by that reference. Beam pruning means the search is not exhaustive even
when it finds a path satisfying its supplied constraints.

## Remaining integration

The inverse planner is a reusable core function, exercised by the numerical
reference. The normal visual play loop does not yet receive a qualified
source transition or full state for this solver. The reference's stopping
and readiness costs are still an external experiment. They must become
parameterized controller mathematics, with observation and model-validity
contracts, before the reference can justify a live-path change. No model
prompt, game-specific production branch, or privileged visual identity has
been added. Actual run 18w remains lost, and presentation readiness remains
unproven.
