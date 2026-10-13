# Points that turn

This checkpoint preserves a measured reference through a rigid turn. It does
not qualify the demonstration. The original program at 099589b59195ae25f69376315c26dd1c6114eba6
was lost, and the invented pixel actor still fails to learn steering.

## Mechanism and reference

The old point observer required most marks to share one translation. Marks
on a turning patch do not. A bounded consensus now fits a proper rotation
and translation when translation alone fails. At least three noncollinear
continuing marks and 70 percent agreement are required. The hypothesis
budget is 120 pairs over at most 256 correspondences. Reflection, large
scaling, deformation and missing coordinates supply no pose.

The retained origin is transformed with its original marks. New marks join
through that transform in a minority. A changing detection box cannot
replace the reference. A current named rotation also supplies the expected
aspect when associating a body whose tall outline becomes wide.

[OpenCV's partial affine estimator](https://docs.opencv.org/4.13.0/d2/d28/calib3d_8hpp.html)
is a reference for robust geometric consensus. This implementation fits
rotation and translation only; it includes no scale parameter and copies no
third-party implementation. No game identity, character or extracted physics
coefficient enters this change.

## Evidence before integration

164 related tests passed in 15.36 seconds. Three cases obtain their point
correspondences from independently rotated pixel images at 5, 8 and 12
degrees. They require a measured pose, position error below 0.6 pixels and
angular error below 0.02 radians. Other cases check the retained origin over
20 turns, minority outliers, changing outline aspect, invalid receipts and
rejected geometry. The measured claim and invariant are registered.

The actual actor was also driven by rendered pixels of an invented rolling
body. Its internal physics and true centre were withheld from the actor.
Neither the normal 240-pixel working width nor a diagnostic 480-pixel width
learned steering in 40 simulated seconds. At 480, 34 observations contain an
identified body and 19 contain point support. Those figures do not establish
continuous identity or a working control model.

The diagnostic run recorded 374 matched-body updates. In 16 updates a prior
point path ended while independent patch comparison reported measured
translation; the consumer marked that motion unavailable. Other missing
paths had changed or missing support. This identifies a reference handoff
to investigate; it does not authorize a fallback across a changing outline.
Overlapping old and fresh tracks also remain in the final observation.

Logs and diagnostic artifacts:

- /private/tmp/codex-steer/rigid-final-tests-v2.log
- /private/tmp/codex-steer/pixel-actor-rigid-240.log
- /private/tmp/codex-steer/pixel-actor-rigid-480.log
- /private/tmp/codex-steer/pixel-actor-rigid-diagnosed/bridges.json
- /private/tmp/codex-steer/pixel-actor-rigid-diagnosed/observations.json

The full chunked suite and an original-program attempt with this change have
not been run. Gate and integration results belong below.

## Gates and integration

The first gate run rejected import formatting, one-line test branches and a
missing narrow organism dependency. These were corrected. The second gate
run passed all nine checks; smoke passed 165 tests with one skip in 44.59
seconds. The complete log is /private/tmp/codex-steer/rigid-gates-v2.log.

The change was rebased onto 802d2606d. All 164 related tests passed again in
16.97 seconds. All nine gates passed again; smoke passed 165 tests with one
skip in 45.05 seconds. Integrated logs:
/private/tmp/codex-steer/rigid-integrated-tests.log and
/private/tmp/codex-steer/rigid-integrated-gates.log.
