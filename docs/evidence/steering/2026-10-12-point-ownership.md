# Point ownership — 12 October 2026

Persistent points now also supply approach estimates and continuity across
changed outlines. Those uses previously accepted every point in an object's
bounding box. Scenery visible through the box could therefore belong to the
object, and an old approach estimate could remain after its points vanished.

Only points on current measured object support now contribute. Missing points
clear approach history. Predicted boxes cannot renew ownership. A new identity
can inherit a previous birth time only from one unexpired owner; competing
owners remain unresolved. Approach samples must come from consecutive frames.

Validation: 34 focused tests passed, including moving and expanding invented
textures, missing support, stale predictions, expired owners and ambiguity.
Ruff, writing, method size, module size, layering, dependency consistency,
compilation and lint passed. Smoke: 165 passed, 1 skipped (41.74 seconds).
Logs: /private/tmp/codex-steer/point-meaning-gates.log.

Two gameplay runs on the preceding checkpoint 4bf1f6042 remain failures to
qualify the demo. Run 18f recorded one landing followed by one crash and no
identified controlled object. Run 12e continued assigning control to scenery
fragments and completed no verified win. These results do not measure the
ownership changes above and do not establish presentation readiness.
