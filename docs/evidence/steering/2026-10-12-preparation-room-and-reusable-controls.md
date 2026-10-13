# Preparation room and reusable control equations — 12 October 2026

The general controller now has core mathematics for stored input response,
coupled changes during braking, and time needed to stop before transferring
to a required place. Parameters are supplied by a measured or inspected
model. The core contains no archived game's constants, names or coordinates.

`controls_that_take_time_to_build.py` moved from the reference tools into
`core/agency/`; its former location reexports the types. Affine channels can
forecast a bounded number of ticks without repeating each tick. Targeted
channels and near-unit decay use bounded iteration. An endpoint forecast
does not prove that intermediate contacts are safe.

`room_to_finish_a_required_motion.py` projects a constant opposing rate until
movement stops, including other state changes under that input. A rest to
rest transfer accounts for a speed cap. Preparation combines the stop and
transfer durations. These assumptions must match the supplied model; the
result is a feasible duration estimate rather than a globally optimal path.
Unknown rates and nonfinite results remain unavailable.

Validation: 88 focused tests passed in 14.03 seconds. Independent physical
steps check the fast forecasts, braking and transfer distances. Allocation
bounds, overflow, missing measurements and registered claims are covered.
Log: `/private/tmp/codex-steer/preparation-math-tests-v2.log`.

The external movement/contact reference for assigned game 18 now uses these
core calculations instead of its handwritten preparation formula. All 18
start/target pairs still reached the numerical required state. Controls and
terminal states matched the retained prior batch in all 18 cases. Search
took 0.731–2.757 seconds per case, excluding process startup. This is too
slow to repeat for every control frame. Artifacts:
`/private/tmp/codex-steer/source-landing-batch-18-v3.json` (before),
`/private/tmp/codex-steer/source-landing-batch-18.json` (after), and
`/private/tmp/codex-steer/source-landing-batch-18-v4.log`.

Scope: direct numerical state and an inspected movement/contact recurrence,
with fixed target height. This does not include all source objects or prove
visual identification, original-emulator success, normal Aura runtime use,
or a presentation-ready game win. The latest original game run still lost.

## Existing software inspected for the next control step

[Mario AI Framework's AStarTree](https://github.com/amidos2006/Mario-AI-Framework/blob/master/src/agents/robinBaumgarten/AStarTree.java)
retains a current action plan, predicts the state after a short prefix, and
searches while the plan advances. Its search respects a timer. This suggests
retaining validated work between observations instead of rebuilding the
whole search. Its forward model and game-specific defaults cannot supply
Aura's visual identity or measured dynamics. No assets or code were copied.

[acados features](https://docs.acados.org/features/)
describe separate preparation and feedback phases to reduce control delay,
plus advanced-step real-time iterations. We will use the separation as an
architectural pattern. This checkpoint does not install acados or implement
its numerical solver.

[Gymnasium's LunarLander heuristic](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py)
combines position, velocity and posture errors, and changes response on
contact. It supports considering arrival state together with distance. Its
calibrated gains and privileged state are specific to that environment;
they were not copied into Aura. The general preparation equations above
are independently tested.

Checkpoint gates: changed Ruff, writing, method size, module size, layering,
dependency declarations, compile and lint passed. Smoke passed with 165 tests
and one skip in 103.12 seconds. Log:
`/private/tmp/codex-steer/preparation-room-gates.log`.
