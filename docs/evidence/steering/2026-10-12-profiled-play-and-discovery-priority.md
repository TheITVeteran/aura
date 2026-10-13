# Profiled play and the discovery phase

Original archived game 9 remained unwon at
`9129f8f57562bd0b029baf092112a8a626d77d13`. The owned process completed
after 173.08862258301815 s with return code 0, no timeout and a result file.
Its result was `unread`, with `won: false` and no harness error.
The main stretch lasted 81.4 s, read 2.1 pictures per second and established
no controlled body. It reported 75 down inputs; its trace shows two held
down presses followed by repeated down taps. This is not presentation proof.

The retained result, frames, exact input receipts and stage measurements are
under `/private/tmp/codex-steer/run-9-stage-costs-1012/`.
The read-only instrumentation timed ordinary methods and recorded input
receipts without changing their decisions. Its bounded histories retain
at most 4096 calls per stage. Rows below retained every call for that stage.

| Stage | Calls | Total seconds | Median milliseconds | 95th percentile milliseconds |
| --- | --- | --- | --- | --- |
| Page picture acquisition | 572 | 122.376638 | 183.211938 | 552.275125 |
| Control actor | 170 | 36.197974 | 260.723375 | 495.889691 |
| Object tracking | 170 | 18.586802 | 104.275625 | 165.409086 |
| Object proposals within tracking | 167 | 12.119871 | 70.395875 | 109.638162 |
| Point tracking within tracking | 170 | 1.250602 | 6.262834 | 13.610150 |
| Control identification | 170 | 0.673838 | 3.168479 | 8.932275 |

These are elapsed call times, including awaits for asynchronous methods.
Acquisition includes background readers and setup. Nested and overlapping
calls cannot be summed into a runtime budget. Other campaigns were active;
the figures do not isolate machine contention or establish a performance
comparison with earlier attempts. They identify costs in this attempt.

## Reproduced scheduling defect

The ordinary actor's favored-action branch could interrupt an active body
trial after the first pass through the inputs. Its priority check reserved
one press and rest for each input; the separate trigger check reserved two.
Causal identification requires repeated independent presses. A delivered
trial's observation deadline was also lost at that count boundary.

The retained reproduction, `/private/tmp/codex-steer/probe-priority-before.log`,
called the actual actor with two unknown controls, a count of four turns,
a delivered trial ending at 11.2 s and a favored second input. At 10.7 s
the actor tapped the second input and changed its discovery clock to 10.7 s.
The live trace's repeated taps and input accounting are consistent with
this favored-action branch; the earlier live attempt did not record branch
identities. This defect is reproduced independently of the game.

## Change

Control discovery and secondary-trigger selection now use one finite phase
predicate. Two isolated press and rest cycles precede ordinary favored actions.
Each delivered body trial, including the last rest, retains its finite
observation interval after the count limit. A changed input receipt or
identification epoch cannot own that interval.

Favored taps use their own spacing clock. The control-recheck clock is not
renewed by a favored action. A due discovery retry runs before those taps;
the new pass again owns its finite measurement phase. Explicit current-screen
requests keep their existing ability to preempt the experiment. Known body
controls proceed to normal use. No game names, key assignments or source
constants were added.

## Validation

The related suite passed 139 tests in 35.97 s. Its log is
`/private/tmp/codex-steer/probe-priority-tests-v2.log`. It covers the actual
actor's second discovery cycle, an active final rest, favored-action spacing,
rechecking despite repeated favored taps, changed receipts and experiments,
explicit screen preemption and normal use of known controls.
The real actor still identifies an invented inertial body and learns all its
accelerations across both axes, both signs and 7.5, 10, 15 and 60 observations
per second, including a misleading favored-action prior and an independently
moving distractor. Those observations supply direct state; they do not prove
original-game pixel tracking or a live win.

Changed-file Ruff, writing, method-size, module-size, layering, dependency
checks, compile and repository lint passed. Smoke passed 165 tests with one
skip in 163.39 s. Gate output is retained as
`/private/tmp/codex-steer/probe-priority-gates.log`. The full offline suite
was not run for this checkpoint.
