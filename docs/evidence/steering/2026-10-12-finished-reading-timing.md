# Finished readings and optional inspection, 12 October 2026

This record concerns timing of observation and refinement. It does not certify
gameplay or presentation readiness.

## Software inspected

The user asked for software that handles timing well. Primary sources inspected:

| Software | Mechanism to study | Application to Aura |
| --- | --- | --- |
| [PsychoPy frame monitoring](https://psychopy.org/general/timing/detectingFrameDrops.html) and [keyboard capture](https://psychopy.org/api/hardware/keyboard.html) | Frame intervals and dropped-frame counts; keyboard events timestamped independently of the main thread | Preserve an observation or input event's time separately from when a worker processes it. Measure cadence and missed deadlines. |
| [PsychoPy non-slip timing](https://devdocs.psychopy.org/general/timing/nonSlipTiming.html) | Known-duration trials use an overall clock rather than accumulating relative waits | Schedule from absolute times when duration is known; do not apply that assumption to an interaction whose endpoint is unknown. |
| [Psychtoolbox Screen Flip](https://psychtoolbox.org/docs/Screen-Flip) and [keyboard queues](https://psychtoolbox.org/docs/KbQueueCreate) | Requested and completed presentation times, missed-deadline reporting, queued press and release times | Keep requested, dispatched, acknowledged and observed times distinct. A delivery acknowledgement alone does not prove a game reacted. |
| [ros2_control asynchronous controllers](https://control.ros.org/lyrical/doc/ros2_control/controller_manager/doc/running_controllers_asynchronously.html) | A slower controller can run independently of the main update loop | Separate urgent control from slow inspection, with explicit completion and freshness checks for the latter. |
| [osu-framework interpolated clock](https://github.com/ppy/osu-framework/blob/master/osu.Framework/Timing/InterpolatingFramedClock.cs) | Tracks drift against a source clock, smooths small differences, and bypasses interpolation when divergence is too large | Distinguish clock alignment from physical latency. A smoothed clock cannot turn old pixels into a fresh observation. |

These packages were not installed or benchmarked on this host. Display engines
control their own presentation surfaces; keyboard queues observe hardware input.
Neither facility directly guarantees the timing of an input sent to another
application. The architecture below is an independent implementation of
completed-stage retention, not a port of any package.

## Observed failure

The owned offline run is `/private/tmp/codex-steer/run-18u`, at
`da86b053757becae9649a92e3cf2036704b80d81`. Passive wrappers timed the ordinary
capture, reading and pointer-inspection functions without changing their results
or actions.

In the first attempt, capture and reading finished in 1.829 seconds. Optional
pointer inspection then occupied 6.195 seconds until the eight-second outer
timeout cancelled the entire read. Two later attempts followed the same pattern:
the pixel reading had completed, but a pointer sweep prevented its delivery.
The fourth attempt eventually returned in 6.774 seconds. This identifies a
completed result being discarded by later work, rather than a capture that
never answered.

The complete saved-frame reader also finished in 3.276 seconds. Its probe is
`/private/tmp/codex-steer/full-reading-probe.log`. The actual-page trace, rather
than that saved-frame result, identifies pointer inspection as the delayed stage.

The run was stopped by its owned 85-second process-group bound. It reached play
and a crash screen, but did not establish a win. No other runtime was stopped.

## Implementation

`a_reading_before_more_detail.py` keeps a completed, settled `ScreenReading`
inside the requesting call's context. Its private pixel snapshot, capture time,
surface, epoch and detached public fields are retained. A dictionary without
measured capture evidence cannot register a reading. Refused, occluded, hidden
or unsettled captures cannot register one either.

The page reader registers only after it has attached its scope and checked that
navigation did not occur. Normal completion returns the fully inspected result.
If optional work times out, that work is cancelled. A separate check, bounded
to one second, takes another picture and requires the same page identity, URL,
viewport dimensions and exact bounded pixel snapshot. Failure or an unavailable
check propagates the timeout. The retained reading is never given a new capture
time. Its returned fields identify the incomplete extra inspection.

Contexts are private to each read, including parallel and nested calls. Closed
contexts cannot accept late publications. External cancellation propagates.
There is no background pointer sweep competing with subsequent input.

The integration covers pursuits on browser pages, including ordinary sites and
editors. It contains no game names, controls, geometry or rules. The retention
interface can accept another capture source with its own independent freshness
check; native-window integration has not been added in this checkpoint.

## Verification status

The first related run passed 32 tests. A wider run and repository gates follow.
Tests use invented observations and pages. They cover detached fields, private
pixels, complete and incomplete reads, changed captures, scope changes, resize,
refusal, unavailable validation, validation timeout, cancellation and concurrent
read ownership. Real-game improvement remains to be measured after the save.

The expanded run passed 184 tests in 110.46 seconds. Its log is
`/private/tmp/codex-steer/finished-reading-expanded-tests.log`. Ruff passed for
the five changed Python files. The expanded run includes the real page-reader
and pursuit integrations on invented pages, alongside existing screen-scope,
capture, locking and pointer-discovery tests.

All checkpoint gates passed: Ruff, writing, changed-function size, module size,
layering, dependency graph, compile and lint. Smoke passed 165 tests with one
skip in 57.74 seconds. Logs are
`/private/tmp/codex-steer/finished-reading-checkpoint-gates.log` and
`/private/tmp/codex-steer/finished-reading-gates-summary.log`.

## Saved revision and offline rerun

The implementation was rebased over intervening changes to research tools and
reference runners. There was no overlap in the changed production files. A
focused post-rebase run passed 29 tests in 10.99 seconds, recorded at
`/private/tmp/codex-steer/finished-reading-rebase-tests.log`.

Commit `26548a8afa27ac483a24215283627d69d32baa6a` was pushed and verified as
`origin/main`. The clean proof worktree then ran that revision in
`/private/tmp/codex-steer/run-18v`. It completed in 77.136 seconds without its
process-group timeout. The campaign result was `won: false`; the final reading
reported zero landings, one crash and zero percent success. Its `goal_reached`
line denotes ending detection, not a successful landing.

There were no observation-timeout messages in this rerun. There were also no
retained-reading fallback messages. The rerun therefore demonstrates ordinary
execution of the changed path, not a causal before-and-after improvement of
fallback behavior. The tests establish that behavior under controlled delay.
The previous and current runs did not hold machine load constant.

The run reached control stretches at 22.9 and 19.7 pictures per second. It did
not establish reliable control of the falling body. Attempts to use model-based
visual identification also reported a missing configured vision-model path and
admission refused because another campaign owned the exclusive model lane.
That campaign and its server were left running.

Presentation readiness remains unproven. The next timing work is to measure
capture age, input delivery and release jitter separately, and keep initial
perception and control from waiting behind optional inspection.
