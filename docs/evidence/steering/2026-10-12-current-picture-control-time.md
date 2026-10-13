# Control time from the current picture, 2026-10-12

The real-time loop used the median of earlier picture-to-input intervals
as the delay for its current picture. That interval includes processing
before dispatch. A fast earlier reading could understate the age of a slow
current frame; a slow earlier reading could overstate a fresh frame's age.
The planner was also constructed before more work on the frame and used
without refreshing that age.

`ControlTiming` separates the current monotonic picture age, measured input
delivery, and fitted response after acknowledgement. Steering refreshes
the age just before choosing. Physical input changes record dispatch and
acknowledgement separately from the original picture timestamp. Timed
clicks and presses record delivery without including their held duration.
Control discovery uses the current estimate when checking whether it can
afford another trial.

The existing velocity-transition fit now contributes only the response
after acknowledgement, rather than adding delivery twice. An effect during
delivery contributes zero extra delay. Acceleration controls do not supply
a constant-velocity transition fit. A secondary input cancels the pending
single-transition measurement. The old total intervals remain available
for existing callers and diagnostics.

An API acknowledgement bounds delivery; it does not timestamp the exact
instant the application consumed the input. Unknown delivery or physical
response contributes no measured estimate. This change uses the capture
timestamp supplied by the existing frame reader. It does not establish
capture freshness, bound future jitter, or remove the time spent computing
the choice itself. Those are separate measurement problems.

## Verification

- The expanded control suite passed 151 tests in 43.11 seconds, before the
  three final secondary-input and timed-delivery tests were added.
- The final focused suite passed 38 tests in 8.88 seconds, including those
  three tests and the input-isolation tests.
- Eight independent constant-speed fixtures cover both axes, both
  directions, and two speeds. A frame aged 40 ms plus 10 ms delivery reaches
  the destination under its old held input. The refreshed controller
  releases there; the earlier historical-delay choice holds on and
  overshoots by another 20 ms of travel. A separate fresh-frame test checks
  that an earlier slow frame does not cause an early release.
- The real `_act` path is exercised with extra work between construction
  and choice. The test checks its final prediction, delivered release, and
  the separate receipt; carried controls cannot supply a velocity fit.
- The registered claim runs the adjacent picture-age invariant.
- Ruff on changed files, writing, changed-method size, module size,
  layering, dependency checks, compile, and lint passed. Smoke passed
  165 tests with one skip in 90.50 seconds.

Logs: `/private/tmp/codex-steer/picture-age-tests.log`,
`picture-age-final-tests.log`, `picture-age-gates-summary.log`, and
`picture-age-checkpoint-gates.log` in the same directory.

These are deterministic control and integration results. No new archived
game result is claimed here. The latest completed assigned-game run remains
18v: zero landings and a crash. Presentation readiness remains unproven.
