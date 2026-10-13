# Exposed background after motion — 12 October 2026

The saved assigned-game run showed a moving body joined to its original
silhouette. The patch grew from 21 by 24 to 21 by 62 working pixels while
its top stayed at the old position. Background correction required the
patch's median to match its surroundings. A body that occupied most of the
patch prevented correction of the exposed background minority.

`background_visible_beside_a_patch.py` checks local samples on opposite sides
of each row or column. Both samples must agree and have little texture, and
the foreground pixel must match both. Correction then requires an open path
to the patch exterior through matching or already nonforeground pixels.
Enclosed details remain. Missing sides, conflicting samples and texture do
not supply a correction. The normal motion eyes use this check when their
existing majority-background correction cannot apply.

Validation: 26 focused tests passed in 6.36 seconds; the broader foreground,
point identity, visible support and motion-eye group passed 128 tests in
45.94 seconds. Invented scenes vary motion direction, axis and background
gradient. Other tests cover enclosed details, strict-mask padding, textured
and disagreeing surroundings, absent measurements and the real blob path.
Logs: `/private/tmp/codex-steer/exposed-background-tests.log` and
`/private/tmp/codex-steer/exposed-background-related-tests.log`.

The unchanged saved pictures from run 18w were replayed through the actual
motion-eye implementation. Across 46 recorded observations, initial track 1
had current motion evidence in 13 frames before and 24 after. Its last
current motion frame changed from 000069 to 000077. At 000065 its box changed
from `[20,68,41,130]` to `[20,83,41,107]`; at 000073 it retained its identity
where the earlier implementation had lost it. Baseline:
`/private/tmp/codex-steer/track-interruptions-18w.jsonl`; current:
`/private/tmp/codex-steer/exposed-background-core-18w.jsonl`.

Limits: the tracker still loses the body when it joins another large patch
later in these pictures. Saved-frame replay cannot change the recorded
actions or establish a game win. This checkpoint improves one measured
foreground failure; it does not prove continuous control or presentation
readiness. No game colours, positions, names or source constants enter the
production change.

Checkpoint gates: changed Ruff, writing, method size, module size, layering,
dependency declarations, compile and lint passed. Smoke passed with 165 tests
and one skip in 74.54 seconds. Log:
`/private/tmp/codex-steer/exposed-background-gates.log`.
