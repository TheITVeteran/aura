# Steering keeps the state estimate, 13 October 2026

Presentation readiness remains unproved. The original game 9 run at
e08b5e1b5 lasted 203.935760 s, returned without an infrastructure error and
did not win. Its three loss labels do not prove independent episodes.

Of 676 actor observations, 316 named an owned body. There were 281 current
qualified state estimates and 284 observed controller poses. The first
control attribution came from independent repeated trials, with statistic
51.977046 and effect 36.638251. The first track retained its birth through
visual changes for approximately 42 seconds. Retained picture 000170 places
that track's box on the moving car's visible patch. Later track replacements
still reset control experiments and remain a continuity limitation.

No provisional steering action was delivered. The trace identifies two
further broken connections:

- Curved paths under acceleration controls admitted those controls as turn
  keys. Arming a turn key removes its driver role. At several observations
  every movement control was armed as a turn and the driver set was empty.
- The fitted visual velocity reached the controller, but its uncertainty
  did not. The controller recomputed uncertainty as two position errors
  divided by the last picture interval, even though the supplied velocity
  was already estimated from a measured history.

Trace: `/private/tmp/codex-steer/run-9-upstream-correspondence-1013/`.
The trial histories and state estimates are retained there. Median visual
observation cost was 98.16 ms and frame supply 5.75 ms. The machine was
also running another session's tests; no isolated speed claim is made.

## General changes

The steering bank accepts the uncertainty belonging to a current measured
velocity. Missing uncertainty retains the existing position-difference
path. The supplied uncertainty controls heading resolution and the forecast
envelope, including the first state. Nonfinite, negative or boolean values
retire the current pose. Input interval, observation, owner and reference
requirements remain in force.

The actor passes this uncertainty only with the qualified current fitted
velocity. It does not substitute a prediction for an observation.

A resolved controlled-receiver method with an explicit acceleration,
thrust, brake or drive role is not classified as a turn solely because its
trajectory curved. Explicit turning descriptions still arm turns, as do
unclassified controls whose measured paths curve. Queries, unrelated
receivers and combined drive-and-turn methods cannot exclude a turn prior.
These method roles remain structural priors, not calibrated control facts.

## Validation

Invented tests demonstrate that a 30 px/s fitted velocity with 3 px/s
uncertainty remains usable at .01 s capture intervals instead of acquiring
a spurious 400 px/s derivative uncertainty. They cover initial uncertainty,
an unanchored input interval, unresolved velocity, invalid uncertainties and
role conflicts. The existing pixel-to-controller test also checks that the
controller retains the fitted state's velocity error.

The first focused run caught a boolean interpreted as zero uncertainty.
That boundary was repaired. The rerun passed 136 tests in 31.06 s; the
registered uncertainty claim and its focused tests passed 11 tests in 5.09 s.
Original-program action delivery, successful play and desktop demonstration
remain to be verified.

Checkpoint gates: `/private/tmp/codex-steer/confidence-final-gates-1013.log`.
All nine gates passed. Smoke passed 165 tests with one skip in 93.90 s.
