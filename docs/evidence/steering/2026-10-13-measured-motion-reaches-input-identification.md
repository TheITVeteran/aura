# Measured motion reaches input identification, 13 October 2026

The state estimator previously supplied an owned controller but not the
experiment that found its owner. Input identification still consumed the
older two-picture velocity. One policy now supplies a current measured
velocity to both consumers. Owner, birth, reference and capture time must
agree; excessive velocity uncertainty leaves the older measured reading in
place. A forecast cannot enter this boundary.

An input may change speed or curve a body's travel without choosing one
screen direction. Identification can now compare acceleration along the
trajectory and signed lateral acceleration across different initial
headings. Each response needs four finite, ordered measurements within one
isolated press. Repeated neutral and active trials are required. Irregular
speed, angular aliasing, interrupted input and missing repeats abstain.
Existing identity thresholds and observed displacement requirements remain
in force. No pooled class history supplies these trials.

## Evidence and limits

The connected suites passed 167 tests in 40.13 s. Further ownership,
dispatch, trial-cadence, trajectory and causal-identification suites passed
180 tests in 34.24 s. One falsification gives raw velocities that agree
with each input while measured positions follow an autonomous trajectory.
The measured state reaches identification, and no control owner is selected.
The existing pixel-to-identification-to-steering test also passes.

Retrospective analysis of the previous original-program traces did not find
a qualifying motion-coordinate response. This mathematical feature alone
has not repaired those runs. The new state connection needs a fresh run.

At 7acf7b50e the original game 9 test lasted 193.553856 s without a win.
It recorded 919 actor observations, 237 with control identity, 196 qualified
state estimates and 207 controller poses. Its provisional steering actions
were absent. One brief drive response appeared but did not produce an
action. Later observations were mostly stationary. Original game 18 lasted
195.401762 s without a win: 463 observations, no owned body, no controller
pose. Each run reported the same final loss more than once; these labels
are not independent episodes.

Traces are under `/private/tmp/codex-steer/run-9-retained-anchor-1013/`
and `/private/tmp/codex-steer/run-18-retained-motion-1013/`.
Original-program wins and an ordinary desktop demonstration remain open.

All nine gates passed. Smoke passed 165 tests with one skip in 107.42 s.
Gate log: `/private/tmp/codex-steer/relative-control-final-gates-1013.log`.
