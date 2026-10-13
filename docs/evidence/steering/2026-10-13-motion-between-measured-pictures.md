# Motion between measured pictures, 13 October 2026

Ordinary desktop demo readiness remains unproved. This work adds a general
state estimator and connects it to existing visual steering. No character
catalogue, game identifier, learned game-specific weights or prompt was added.

## The original program after capture repair

At checkpoint 2554ef567, the original game 9 ran for 192.403913 s and did
not win. It produced 1,309 actor observations; 267 named a controlled body.
Body 124 accounted for 231 of those. Retained pictures 000332, 000441 and
000603 show the moving car at its measured places. This supports the visual
interpretation, without independently certifying every control attribution.

There were 11 distinct provisional steering predictions and 28 feedback
records marked partially consistent. Neither internal feedback nor repeated
loss labels establish a completed independent episode or a successful task.
The capture supplier's median cost was 3.612 ms and visual observation's was
55.810 ms. The final screen said "time's up! QUIT".

The objective required moving other objects into destinations. Steering the
owned body toward a feature did not solve that manipulation objective. The
trace also labelled the same terminal screen as several subsequent losses.
Those planning and physical restart defects remain open.

Trace: `/private/tmp/codex-steer/run-9-adaptive-capture-1013/`.

## Position measurements and competing motion laws

Six small Gaussian estimators compete: constant velocity and constant
acceleration, each with three process-noise priors normalized to view size.
Measured position corrects each prediction. Gaussian innovation evidence
updates model weights; disagreement contributes to reported uncertainty.
Position and a velocity derived from it are not counted as two observations.

The implementation uses Kalman's prediction and correction equations and
the Joseph covariance update. The implementation is independent and adds no
external filtering dependency. Mathematical and implementation references:

- https://www.cs.cmu.edu/~./motionplanning/papers/sbp_papers/k/Kalman1960.pdf
- https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py
- https://github.com/rlabbe/filterpy/blob/master/filterpy/common/kinematic.py

Forecasts retain the last observation time and grow uncertainty. They never
qualify control identity, renew observations or train the steering model.
Birth, view geometry, coordinate reference changes and excessive gaps reset
derivative history. Innovations incompatible with every motion law restart
motion estimation without asserting that the body changed identity.

Point paths now carry an explicit reference epoch. Independently matching
pixels and exterior edges can carry it into replacement marks. Without that
bridge, new marks start a new reference. Unknown references cannot supply a
current state estimate. The existing controller receives fitted velocity
only when current measured evidence and bounded uncertainty qualify it.

## Tests and negative results

Twelve irregular noisy-motion cases cover three seeds and four acceleration
values. Estimated velocity error must be less than half the two-picture
difference error; forecast error must be below 65 percent of that baseline.
A rendered rotating and accelerating textured body passes through the
ordinary detector, point tracker, estimator and steering-model observer.
Ground truth selects the synthetic body only in the component test; it is
not supplied to perception or estimation. This is not an original game win.

The first broad regression exposed address-dependent representations across
the observer process boundary. Numeric dataclass representations repaired
that parity defect. A second run exposed cold SciPy imports and numerical
compilation consuming approximately half a second on a blank observation.
Preparation now happens during worker startup without submitting a picture
or establishing a body. The observation deadline was not relaxed.

A bounded broad run passed 563 tests before its last test exceeded the
360-second group budget. That remaining test passed separately in 85.17 s.
This is recorded as incomplete group validation, rather than a completed
green group run. After the reference-epoch repair, 82 connected tests passed
in 14.98 s. After moving startup preparation out of the observer class and
separating the existing want-record readouts, 111 integration tests passed
in 20.15 s; seven observer-process parity tests passed in 9.37 s.

The want-record class already exceeded its method ceiling at the starting
checkpoint. Its retained readouts now live in a small base class with the
same public interface, lock and method bodies. Startup preparation is a
module function so the visual observer does not acquire another method.

The controller also retires its physical forecast when the measured point
reference changes. That boundary, motion estimation and observer-process
parity passed 99 connected tests in 15.18 s.

Smoke exposed an order-dependence defect in the existing interiority test
fixture. A fresh service did not own its transmission state or random
stream. Giving it a fresh cleft and receptor bank produced a passing smoke
run of 165 tests and one skip in 54.89 s, but the failure recurred after
integration with main. That first repair was insufficient.

A diagnostic run showed two intended activations, four attempted quanta,
zero released quanta and zero eligibility traces. The fresh receptor bank
was not depleted. A nonempty dictionary of zero transmissions did not mean
something had crossed. Seeding the process-wide generator did not isolate
its draws from other work. The fixture now supplies the cleft's supported
explicit random generator as well as fresh transmission state, restoring
the resident singleton afterwards. No production transmission rule changed.
Diagnostic state: `/private/tmp/codex-steer/credit-smoke-diagnosis-1013.json`.
With its owned stream the same intended activations produced three releases
on the pursuit channel, and credit assignment passed. Full smoke passed
165 tests and skipped one in 52.18 s; the diagnostic records that transmission
in `/private/tmp/codex-steer/credit-smoke-owned-stream-1013.json`.

Module size, method size, layering, dependency consistency, compilation,
production lint and writing lint passed. Changed-file lint identified two
import formatting issues and an existing unused loop variable in the
touched test file; those were corrected before final lint validation.
Gates are in `/private/tmp/codex-steer/motion-final-gates3-1013.log`.
Original-program reruns and desktop demonstration remain necessary before
any presentation-readiness claim.
