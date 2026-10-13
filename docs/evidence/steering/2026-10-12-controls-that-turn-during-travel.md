# Controls that turn during travel

## Boundary

This checkpoint addresses a conditional control response and the consumers
of that response. It does not qualify the presentation or certify a game.
The most recent original-game attempt at 3cc9f77d0 ended with TIME'S UP.
No original-game win has been established in this continuation.

## Source and research

The assigned, seen program 9 was parsed with Aura's Flash reader. Its
932113-byte SWF has SHA256
ef2498224df3a9bfdea35a1441e5b42ad06538998624e5b4155313595095b93d.
The parsed turning method returns when the body is not moving. This
motivated a test of a general conditional response; no game identity,
character, key mapping or extracted physics coefficient enters production.

ROS's [steering controller library](https://docs.ros.org/en/kilted/p/bicycle_steering_controller/)
and [steering kinematics implementation](https://docs.ros.org/en/ros2_packages/kilted/api/steering_controllers_library/generated/program_listing_file_include_steering_controllers_library_steering_kinematics.hpp.html)
separate traction and steering. This checkpoint implements observed angular
response during travel. It does not implement the library's full vehicle
model or a speed-dependent steering law. No third-party implementation was
copied. The user's research files remain suggestions and reference data.

## Repairs

TurningWhileTravelling observes the unwrapped velocity direction under each
delivered input. Two coherent presses, each with four settled observations,
must contrast with two observed neutral intervals. Autonomous curves,
missing measurements, jumps, gaps and unstable angular rates supply no
turning authority. A stopped body supplies no moving-turn prediction.

Facing measurements belong to the identified body, its kind and the current
identification epoch. Changing that binding clears the measurements. Earlier
control trials on that same body can request new heading experiments; they
do not supply heading measurements.

Readiness and identity consistency use the appropriate response model. A
moving turn predicts the next tangent from the preceding measured travel.
Identity checking still rejects two controls that stop answering. The
available combination of turning and driving exposes two movement axes.

Observed heading dependence survives a window of subsequent straight
presses. The recent direction window previously forgot that the body could
turn, and planning stopped while the target remained in reach. Current
turning responses are still required. A new body or experiment clears this
qualification.

## Offline evidence

The invented rolling plant initially identified its body but learned no
steering in four cases after 40 simulated seconds. The repaired actor learns
steering with stationary or rolling turns at 10 and 30 Hz, from two initial
headings. These are direct state observations, with perfect geometry and an
invented plant.

A stronger test uses the actual actor, planner, delivered-input receipts and
identity observer. It requires contact with four sequential targets in
different quadrants and preserves identity on every observation after the
body is first identified. All eight cases passed. The contact oracle checks
the independent plant's bounding boxes; it does not use Aura's success flag.
These cases have no obstacles, visual tracking errors or browser delays.

Earlier checks that credited contacts despite a lost identity are not proof.
The first strict route run exposed the axis defect and a forgotten turning
qualification. Its failures were retained at
/private/tmp/codex-steer/travel-navigation-tests-v1.log. The repaired combined
run passed 95 tests in 12.30 seconds at
/private/tmp/codex-steer/travel-navigation-tests-v3.log.

The broader related run passed 233 tests and failed one new test whose
expected value was wrong: a known steering-only input supplies no translation
axis. That expectation was corrected without changing production behavior.
Further validation and the original-program attempt belong below.

## Validation before integration

The corrected ownership and navigation files passed all 18 tests in 5.03
seconds. The other 216 related tests had passed in the preceding broader run.
The changed-file Ruff gate required one import-spacing correction. After
that correction, Ruff, writing, method size, module size, layering, generated
dependencies, compilation and repository lint all passed. Smoke passed 165
tests with one skip in 50.09 seconds. The full chunked suite was not run.
The gate log is /private/tmp/codex-steer/travel-turn-gates-v2.log.

## Integrated validation

The change was rebased onto b9af446ac, including Claude's intervening
perception, capture and sign-control changes. All 234 related tests passed
in 49.50 seconds. All gates passed again; smoke passed 165 tests with one
skip in 49.39 seconds. Logs:
/private/tmp/codex-steer/travel-integrated-tests.log and
/private/tmp/codex-steer/travel-integrated-gates.log.
