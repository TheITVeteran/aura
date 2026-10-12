# Foreground identity during transfers, 11 October 2026

The current transfer verifier compared whole rectangular RGB patches. A
transparent object carried from a pale palette to a darker canvas could be
rejected despite retaining its own appearance. A new regression reproduced
that limitation: seven cases failed and two negative cases passed before
the repair.

Appearance witnesses can now exclude a consistent, flat exterior connected
to the patch boundary. Enclosed highlights remain part of the witness.
Foreground support is bounded; missing or ambiguous exterior evidence keeps
the existing full-patch comparison. Source ownership, delivery, capture
freshness, surface, destination locality, distinctive appearance and unique
new occupancy remain required. Comparison thresholds were not loosened.

The mask follows the placement receipt through bounded storage and later
independent reobservation. Malformed masks cannot restore success. Repeated
effect observations also log a bounded receipt identity, state and reason,
so the next live failure can be attributed without guessing from activity.

The focused effect-path suite passed 209 tests in 40.00 seconds. Smoke,
compile, lint, layering, module-size and writing passed. The synthetic mask
invariant is registered with model validation. No baseline was increased.

The method follows the use of masks to isolate relevant template pixels in
[OpenCV's template-matching documentation](https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html).
It does not add character names or game-specific rules. It currently covers
bounded translation appearance with a measured flat exterior, not arbitrary
rotation, deformation or occlusion.

The previous normal-app attempt is retained separately. The new verifier has
not received an app retest: the Mac locked before the idle app could be quit.
No game pass or presentation qualification is claimed.

Logs: `/Users/bryan/.aura/flash-proof-2026-10-11/foreground-transfer-before.log`,
`foreground-transfer-after.log`, `foreground-transfer-owned-effects.log`
and `foreground-checkpoint-gates.log` in that directory.
