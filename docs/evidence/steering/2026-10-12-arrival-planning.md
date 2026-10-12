# Arrival planning, 12 October 2026

This checkpoint concerns translating bodies with measured acceleration.
It does not establish game or presentation readiness.

## What the comparison found

An exact-state probe of eight invented inertial bodies separated steering
from visual identification. A stopping-distance reference reached a small
destination region at low speed in eight cases; the existing planner did
so in five. Its failures included stopping short and braking late.

That success condition was insufficient for a landing: it let a body pass
through the destination fast, then return slowly. Reviewing the first
contact showed that the reference was unsafe for that stricter task.
Neither its eight passes nor the planner's five are landing proof.

The stricter test stops at first contact. It samples inside each delivered
input interval, rather than checking only the final frame. It learns each
control's push through `WhichIsHers` before asking the real planner to steer.
No archived game's frames, names, constants or input meanings supply these
plants.

## What changed

- A feedback path is an additional candidate in the existing planner. It
  chooses a speed from the room left to brake and the measured opposing
  push, then recomputes its input once per control turn.
- A distant arrival's distance score stays responsive to progress. The old
  capped score made different faraway endpoints equivalent.
- Terminal speed has a stopping-room cost beyond the finite horizon.
  Closing too fast to brake before the contact boundary also costs the
  plan, even when that contact falls beyond the horizon.
- First-contact speed is bounded by both endpoints of the imagined step.
  A slow final sample cannot erase the faster approach immediately before it.
- A body crossing a destination quickly keeps that destination as its
  approach. Merely overlapping it does not establish a settled arrival.
- The old push continues during command delivery. Other bodies advance
  through that same delay. An unmeasured old push cannot supply that prediction.

The feedback path is evaluated with the other paths against the observed
world. It requires finite state, a goal, a neutral measurement and opposing
pushes on an axis. It supplies no identity or authority to unmeasured motion.

## Measurements so far

The new file's 28 tests passed in 18.35 seconds. They include 16 bounded
learned-control arrivals across horizontal and vertical axes, starting
velocities, forces, directions and delivery delay. Two more arrivals use
both axes together, with and without a passive downward pull. The first
contact in each must be below the stated speed limit.

The earlier related run passed 37 tests in 9.66 seconds. Expanded regression
and repository gate results will be appended before this checkpoint is saved.

These tests cover a measured constant-push model. Rotation, deformation,
contact ambiguity, unmodelled forces and missing body identity require their
own evidence. A coarse input cadence may leave no gentle arrival among the
available commands; a marker revisited later is not a substitute.

## Research used

[MIT's trajectory-optimization notes](https://underactuated.mit.edu/trajopt.html)
describe terminal costs, dynamic constraints and input limits.
[Its feedback-planning notes](https://underactuated.mit.edu/feedback_motion_planning.html)
describe simulating a feedback policy into a trajectory and replanning it.
Those principles informed the additional candidate and terminal braking check.
The implementation here was written independently for measured screen motion.

[Gymnasium's landing environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
offers a heuristic demonstration with privileged state, including angle,
velocity and contact. That is a useful reference design; it has not been
installed, run or used as proof that Aura can play an archived game.

## Archived-game evidence

`/private/tmp/codex-steer/run-18r` ran the pushed point-path revision
`66347532dd5445e318b3c421a7cae0de1f57dd6d`. It completed normally in
193.31 seconds, with no win and no controlled-body identification.
Its final visible counters were zero landings and two crashes.

The Play click was logged at 16:17:41.132. The first real-time control
picture arrived at 16:17:45.369, 4.237 seconds later, with the craft already
near the ground. The play handoff began at 16:17:44.680. This early loss of
observation time remains a separate unresolved integration problem.
The run does not contain the arrival-planning changes in this checkpoint.

## Checkpoint validation

The same first-contact fixtures were run against the clean pushed baseline
and the changed planner. The baseline passed 7 of 18 arrivals; the changed
planner passed all 18. Both reports identify the same fixture SHA-256,
`ec2b6c86fc373c2f7be1fcb540667d3554d59211a12a08e880f70d508bb9e589`,
and record controller source hashes. The harness extracts only the plant
and arrival fixtures; each process imports its own checkout's controller.

Reports: `/private/tmp/codex-steer/first-arrival-before.json` and
`/private/tmp/codex-steer/first-arrival-after.json`. The comparison harness
is `/private/tmp/codex-steer/measure_first_arrivals.py`.

Expanded regressions passed 191 tests in 69.39 seconds. Changed-file Ruff,
writing, function size, module size, layering, dependency graph, compile and
repository lint passed. Smoke passed 165 tests with one skip in 71.87 seconds.
Gate output is `/private/tmp/codex-steer/arrival-checkpoint-gates.log`.

The changed model remains unproved on an archived game. In particular,
these results do not resolve the late play handoff observed above.
