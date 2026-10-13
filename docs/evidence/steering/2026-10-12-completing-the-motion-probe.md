# Complete the motion probe before using its body

The full control actor could identify an invented turning body but fail to
learn its steering even with perfect state observations. In three of four
cases, planning kept the delivered forward trial held. The facing observer
learns a press's speed when it is let go. Without that completion it had no
ahead control, and could not request its turning experiment. Increasing the
run to 30 simulated seconds did not resolve those three cases.

The retained experiment is `/private/tmp/codex-steer/check_turning_actor.py`.
Before output is `/private/tmp/codex-steer/turning-actor-v2.log`.
The first setup attempt omitted the runtime's pressures component and failed
before exercising the actor; it supplies no product outcome.

## Change

A delivered body probe remains owed its rest when its press interval ends,
even if the body and controls have been identified during the press. The
ordinary trial actor releases that input and reserves the rest's finite
observation interval. Planning resumes once the rest has completed.
Receipt identity, experiment identity and finite timing remain required.
Explicit screen requests retain their existing preemption priority.

`ControlTrial.belongs_to_input` supplies the common receipt check used both
while waiting for pictures and while completing the trial. A registered
invariant and measured claim cover the rest obligation. The fast actor's
existing trial path performs the release; no alternate solver was added.

## Direct-state observations

The same full actor and invented plant were exercised at two observation
rates and two initial headings. The plant's forward speed, reverse speed
and turn rates are test values; none enter production.

| Observations/s | Initial heading | Before: turning known within 30 s | After: turning known at simulated seconds |
| --- | --- | --- | --- |
| 10 | 0 | no | 11.6 |
| 10 | 1.2 | no | 11.6 |
| 30 | 0 | yes | 9.366667 |
| 30 | 1.2 | no | 9.366667 |

After output is `/private/tmp/codex-steer/turning-actor-v3.log`.
All four cases identify the body, learn its forward control and measure a
turn rate of -2.8 radians/s. The prior successful case completed at
8.933333 simulated seconds. The changed procedure spends time on its rest.
This proves neither pixel tracking nor an original-game win.

The related suite passed 140 tests in 37.98 s. Its log is
`/private/tmp/codex-steer/trial-rest-tests-v1.log`. It includes the actual
turning actor, misleading favored-action tests, cadence tests, facing
evidence, finite-view checks and independent input receipts.

## Original-game result at the preceding checkpoint

Archived game 9 at `9508740a574f47083a0fdfa57936b203aa5808bd` completed
after 153.95884341700003 s with return code 0, no timeout and a result file.
It remained `unread`, with `won: false` and no harness error. Its 57.1 s main
stretch read 2.7 pictures per second and established no controlled body.
The repeated favored taps were absent; the trace shows a fresh discovery
retry. It reported up: 3, down: 3, left: 3 and right: 2 held inputs.

The actor's 156 calls had median elapsed time 0.454792 ms and total time
9.370172 s. Tracking's median was 122.404375 ms; page acquisition's median
was 200.1815 ms and its 95th percentile was 1000.446391 ms. Other campaigns
were active. This run has a different setup time and gameplay duration
from the earlier profile, so it is not an isolated speed comparison.

Evidence is under `/private/tmp/codex-steer/run-9-discovery-phase-1012/`.
Its input-origin record contains only the final release outside a trial.
This observation verifies the absence of those interfering taps in the
rerun. Missing visual motion still prevents controlled gameplay, and
presentation qualification remains open.

## Repository validation

Smoke passed 165 tests with one skip in 117.82 s. Module size, layering,
dependency declarations, compilation, writing and repository lint passed.
The first changed-file gates found an import formatting error and a
401-line claim installer. The discovery claims were extracted into a
separate bounded function. Changed-file Ruff and method-size checks then
passed. The extracted claim registrations and actor tests were rerun;
their result is retained in
`/private/tmp/codex-steer/trial-rest-extracted-tests.log`.
The full suite was not run. The initial gate log is
`/private/tmp/codex-steer/trial-rest-gates.log`.
