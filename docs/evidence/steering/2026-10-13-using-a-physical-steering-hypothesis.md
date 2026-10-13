# Using a physical steering hypothesis, 13 October 2026

The owner asked to invent the steering model and build it into Aura.
This checkpoint supplies a usable structural model before calibration
finishes. Its forecasts stay provisional and must answer to observations.
It does not qualify the presentation.

## What reaches the actor

An analytic curved-motion model keeps position, heading and signed speed.
It considers moving turns, straight travel, no control effect, autonomous
curvature and a world-fixed driver. An unknown turn's initial radius is
the current body's extent, bounded by pixel uncertainty. Both signs remain
possible. Opaque key names supply no turn direction.

Current explicit turning instructions or an observed curved trial arm the
turn-role hypothesis. A current owned movement anchors its physical state.
Goal-directed bounded actions can then run between complete facing trials.
The actor calls this path before the facing-calibration scheduler. This
changes delivered behavior, rather than adding a disconnected planner.

Each action freezes its predicted paths, body number, birth, identification
epoch and input receipt. Current observations fit angular rates through
intersecting uncertainty bounds. Pixel position error and observation time
bound direction error; the fitter does not assume successive derived
velocities are independent. Neutral travel can contradict a supposed
control effect. Stops separate angular samples. Birth, epoch, coordinate
reference changes and missing observations invalidate pending forecasts.

The model's status remains provisional. It changes neither measured
`turns_her()` nor identity and control-readiness authority. Goal-selected
inputs are delivered with `trying=False`. If they reuse an independently
tested key, a physical release ends the old interval first.

## Time and ownership

Picture age and measured delivery delay participate in the forecast.
The acknowledgement binds the action to the input actually delivered.
Available field boundaries and known wall cells must admit every competing
path with the body's extent and prediction uncertainty.

An owned async deadline releases the provisional key without waiting for
another picture. Expiry and changed-input delivery share one serial control
boundary. An old deadline cannot release another input or a later press of
the same key. The control loop also checks elapsed wall time when a picture
is stale. This does not promise a hard real-time deadline on a blocked
event loop or unavailable browser.

Stationary-turn hypotheses ask for a subsequent drive observation. This
version does not yet fit a stationary angular rate from that sequence.
That remains a model limitation.

## Verification

The related model, actor, timing, input, calibration, heading, motion and
identity suite returned 234 passed in 27.00 seconds, with three warnings
reported. The new model and actor boundary have registered invariants and
measured organism claims. All nine source gates passed. Smoke returned
165 passed and 1 skipped in 73.41 seconds while pixel probes also ran.
Whole-suite chunk tests were not run.

The tests include one usable action before complete calibration, correction
of a wrong assumed sign, several invented rates and speeds, quantized
positions at several cadences and scales, autonomous and Cartesian
alternatives, current ownership, missing frames, delay, clearance and a
physical expiry with no subsequent picture.

Logs are `/private/tmp/codex-steer/invented-model-final-tests.log` and
`/private/tmp/codex-steer/invented-model-gates.log`.

## Independent pixel comparison

The private invented plant supplies only rendered RGB, timestamps and its
ordinary control instructions to the actor. A user-specified bright marker
is located from RGB. Plant position and physics remain scoring data outside
the observer. It uses the source `WhatMoves`, `WhichIsHers`, ordinary chooser
and `_act`, without the private rotation-aware tracking prototype.

Forty simulated seconds at the native 640-pixel width produced 226 identified
frames and fourteen frames carrying four distinct body/prediction action
identifiers. The closest independent plant distance to the goal was
50.19988053002429 pixels. It finished 202.88417970516792 pixels away, with
no current body or measured steering model.

The same fixture with this provisional-action path disabled produced
328 identified frames. It came within 103.6961935408252 pixels and finished
154.11248152017907 pixels away. This one comparison shows earlier approach
through delivered steering, alongside worse final distance and weaker
identity continuity. It does not show completed navigation or broad
performance improvement.

At the production 240-pixel working width, the fixture produced zero owned
body identifications and zero model actions. The source-path model cannot
repair perception by assuming a body identity.

Evidence:

- `/private/tmp/codex-steer/pixel-goal-native-model/`
- `/private/tmp/codex-steer/pixel-goal-native-baseline/`
- `/private/tmp/codex-steer/pixel-goal-small-model/`
- `/private/tmp/codex-steer/pixel_goal_model_probe.py`

The earlier actor fixtures lacked an explicit navigation goal and measured
calibration. This marker comparison measures provisional navigation input;
neither fixture establishes an original-game win or normal desktop proof.

## Latest reference

The general portions of `AURA56_Unified_Engineering_Report.zip` were read as
research. Its contention overlay was not applied. It reports no independent
original-game wins, and its browser measurements use synthetic CPU
contention rather than this Mac's model workload. Its caution about joining
existing mechanisms is relevant: this checkpoint integrates physical
prediction, delivery, observation and correction in the existing actor.
It does not adopt the supplied installer's instructions or rewrite the
resident model-serving lifecycle.

## Task cancellation follow-up

The three warnings were unawaited input-deadline coroutines. The canonical
task tracker can wrap a coroutine; cancelling that wrapper before it runs
left the coroutine unstarted and open. The shared ownership sink now closes
a native coroutine still in its created state when the owning task finishes.
Started work retains normal cancellation, cleanup and result semantics.
This has a registered invariant and four focused regressions.

The expanded related suite returned 245 passed in 35.83 seconds without
warnings. All nine source gates passed again. Smoke returned 165 passed and
1 skipped in 79.61 seconds. Final logs are
`/private/tmp/codex-steer/invented-model-task-cleanup-tests.log` and
`/private/tmp/codex-steer/invented-model-cleanup-gates.log`.
