# Visible detail before point tracking

The original-program retest at `890370087` lasted 193.2686864589923
seconds and returned no win. The input-selection repair reached its
intended boundary: 416 of 1,244 actor frames held one of the four movement
keys. There were zero claimed bodies and zero steering-model actions.
The run is `/private/tmp/codex-steer/run-9-current-code-inputs-1013/`.

A private point-resolution ablation used the same frozen actor with a
prototype tracker retaining source pixels. It lasted 194.28298600000562
seconds. It recorded 360 claimed-body frames among 973 actor frames and
30 model-action frames, representing four distinct owned model actions.
It also returned no win. This is an original-program observation of a
missing connection becoming active; it does not establish independent
identity or task completion. Its path is
`/private/tmp/codex-steer/run-9-native-point-ablation-1013/`.

The production implementation tracks a bounded source picture before
reducing its coordinates. Local translation and rotation are refined with
RGB residuals in both directions. A forward match needs its backward
match, colour support and consistent rotation. Corner slots remain bounded
and spread by working-picture cells. All exposed positions use the
existing working geometry. Gaps, resets, repeated timestamps and changes
to source or working dimensions retire the old point names.

This implementation uses full RGB residuals rather than the prototype's
colour projection. It bounds tracking to 256,000 pixels and 192 default
marks. It does not grant body identity or calibrated control. Existing
visible-support and causal control checks still determine what the marks
can establish. No game, character or environment identifier is used.

The expanded control and perception suite returned 550 passed in 34.57
seconds. Additional motion/support tests returned 80 passed in 27.20
seconds. All nine source gates passed; smoke returned 165 passed and
1 skipped in 40.53 seconds. The geometry tests cover independent rendered
translation and rotation, grayscale-preserving colour replacement, absence,
capture gaps and geometry changes. Whole-suite chunks were not run.

A private 12-picture cost experiment measured median point-tracking costs
of 16.21, 28.41 and 22.82 milliseconds for source pictures of 320×200,
640×400 and 960×600. The two larger source cases both used at most 256,000
tracking pixels. Working-only medians were 3.36, 2.12 and 2.08 milliseconds.
The additional work retained more marks on the small rotating patch; this
fixture does not measure whole-loop latency or game success. Raw records
are `/private/tmp/codex-steer/source-points-cost.json`.

The production implementation still requires a fresh original-program
retest. The private prototype result cannot be substituted for that test,
and neither result qualifies a normal desktop prompt or presentation.
