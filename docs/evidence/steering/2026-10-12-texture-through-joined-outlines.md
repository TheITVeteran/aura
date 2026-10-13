# Continuing texture through joined outlines — 12 October 2026

The assigned saved run still lost its body when foreground touched scenery,
although continuing point motion remained. The joined outline had a much
larger box and could not qualify as the same body. Colour-only separation
also failed to supply a distinct body part.

`texture_that_continues_when_outlines_join.py` supplies a bounded proposal
from the prior observed texture under current measured translation. Colours
must match inside current foreground, with a one-pixel sampling neighbourhood.
At least three unique, current, spread-out landmarks must own the proposal.
More than half the old support must remain visible. All unassigned pixels
stay available to the other foreground methods. Normal motion eyes use the
proposals before their existing colour separation and identity matching.

Work is confined to the projected texture region, with a bounded count of
proposals. Missing or stale motion, a changed birth, mismatched geometry,
duplicate or collinear landmarks, unowned pixels and insufficient matching
texture cannot supply a part. This observes a visible part; it does not
certify an occluded body's full extent or solve deformation and rotation.

Validation: 63 initial tests passed in 13.70 seconds. After adding birth,
geometry and proposal-budget cases, the related point, foreground, motion
support and texture group passed 148 tests in 17.72 seconds. The tests use
invented scenes, including the normal motion-eye path with touching objects.
Log: `/private/tmp/codex-steer/continuing-texture-related-tests.log`.

The same 46 saved observations were replayed before and after, using the
frozen prior revision and the current implementation. Track 1 had current
motion evidence in 25 frames before and 33 after. Its last current frame
changed from 000077 to 000088. The blob-stage median time changed from
2.547 to 7.231 milliseconds; its greatest time changed from 10.520 to 12.101
milliseconds. These timings cover this stage and this recorded sequence,
not end-to-end input latency. Artifacts:
`/private/tmp/codex-steer/continuing-texture-before-18w.jsonl` and
`/private/tmp/codex-steer/continuing-texture-core-18w.jsonl`.

Tracking is still interrupted in the replay. Saved actions cannot change,
so this does not establish control identification or a game win. The latest
original-game attempt remained unread on its title screen. Presentation
readiness and in-app proof remain open. No game names, colours, coordinates
or source-specific rules enter the production code.

The support-count condition is exact: the matching current proposal must
contain at least 60% as many pixels as the prior support. It does not measure
visibility of each prior pixel or certify complete object extent.

Checkpoint gates: changed Ruff, writing, method size, module size, layering,
dependency declarations, compile and lint passed. Smoke passed with 165 tests
and one skip in 44.71 seconds. Log:
`/private/tmp/codex-steer/continuing-texture-gates.log`.
