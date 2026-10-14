# Measured rotation reaches texture proposals, October 14, 2026

Presentation readiness remains open. The assigned driving game still had
no controlled-body attribution at checkpoint `3283da356`. Its browser
measurements are recorded in `2026-10-14-current-material-marks-feed-facing.md`.

The continuing-texture proposal used translated old pixels even when its
current named-point receipt supplied a proper rigid rotation. A rotated
body could lose its visible support as its outline joined another patch.
The proposal now projects its measured corners through the rigid transform
and samples its old texture through the inverse transform. Translation
continues to use the existing path when no rigid pose is supplied.

At least three continuing owned marks and matching current foreground
pixels remain required. Invalid, stale or incomplete pose receipts cannot
create a proposal. A generated translation receipt cannot substitute for
the missing rotation. The remaining outline keeps all its pixels.

The isolated counterexample supplied exact pose receipts for three generated
rotations. Before this change each case returned the joined outline. The
new path separated the rotated body. An initial half-turn retained 591 of
600 body pixels because floating-point coordinates fell just outside image
edges. Coordinates within `1e-9` of an integer now retain their exact grid
value. The strengthened regression requires every actual body pixel in
these grid-rotation cases, rather than a coverage allowance.

The normal `WhatMoves._blobs` path retains a multicoloured rotated body and
a differently coloured touching outline. The focused suites passed 105
tests in 16.28 seconds; a smaller proposal and contact suite passed 41
tests in 5.55 seconds. Registered validation includes a rigid-texture canary.

These component tests supply the object's earlier support and exact
kinematic receipts. They do not establish cold-scene object discovery,
control attribution or game completion. The fix has not yet improved a
measured browser run.

- Initial counterexample: `/private/tmp/codex-steer/rigid-texture-counterexample-1014/summary.json`
- Gates: `/private/tmp/codex-steer/rigid-visible-patch-gates-1014.log`

Checkpoint gates passed Ruff, writing, changed-method size, layering,
dependency consistency, compile, lint and smoke. Smoke passed 165 tests
with one skip in 42.63 seconds. The inherited global size debt remains;
this checkpoint changes no over-limit module or class. Governance still
reports 123 existing regressions, with none in this checkpoint's changed
modules. No baseline was changed. Governance log:
`/private/tmp/codex-steer/rigid-patch-governance-1014.log`.
