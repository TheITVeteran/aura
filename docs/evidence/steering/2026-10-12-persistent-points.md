# Persistent visual points, 12 October 2026

This checkpoint adds point tracking alongside foreground and object tracking.
It does not establish demo readiness or a completed game.

## Cause and change

Seen offline, a changing foreground patch could inherit a small fragment's
identity. Its changing centre was then available as motion evidence. The
previous point experiment renewed points from each object's box and could
borrow scenery inside that box.

The new tracker keeps bounded point identities across consecutive pictures,
using corners and pyramidal Lucas-Kanade with reverse tracking and patch
comparison. Missing frames, occlusion, resizing and scene resets retire
point identities. Spatial sharing lets new objects acquire points even
when old scenery fills the budget.

Object matching can use a common point displacement only from the object's
measured foreground pixels. A proposed outline must still contain the
tracked support. Shape changes have an explicit geometric allowance.
Internal texture motion does not replace the body's measured velocity.
A spatial index reduces candidate pairs while preserving the exhaustive
assignment, checked on randomized invented scenes.

References: [OpenCV optical flow tutorial](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html),
Shi and Tomasi, Good Features to Track (1994).

## Validation

- 96 related point, object, control identity and trajectory tests passed.
- Smoke: 165 passed, 1 skipped.
- Changed-file ruff, writing, method size, module size, compile, lint,
  layering and dependency checks passed. The first layering run identified
  the new validation import; its explicit DEPS entry passed the rerun.
- A separate synthetic probe measured a newly arriving object receiving
  four points, preserved five measurements of a moving patch beside dense
  scenery, and excluded nine background points from foreground ownership.
- At 240 by 180 pixels and 192 points, the isolated point tracker measured
  2.40 ms median / 2.58 ms p95 on stationary frames and 13.85 ms median /
  14.42 ms p95 on translating frames. These are component measurements,
  not end-to-end play rates.

Three broader failures reproduce on the preceding c071466af checkpoint:
scenery appearance count, and two cases of a bobbing figure being treated
as a moving world. Their diagnosis remains active. No assertion was relaxed.

Own local evidence: `/private/tmp/codex-steer/point-final-focused.log`,
`persistent-point-gates.log`, `point-layering-recheck.log`,
`point-baseline-tests.log`, and `point-review/first-review.json`.
The authorized seen-game run in `run-27c` remains unread and won=false.
Its exact tracker inputs were saved for controlled perception replay.
No held-out game was inspected, and no live instance was restarted.
