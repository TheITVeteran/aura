# Object identity when points disappear — 12 October 2026

An observed track began on a moving white figure, then retained its identity
as its box collapsed into a small brown or black scenery fragment. The normal
association path limited abrupt size changes only while point support was
available. Losing points relaxed that condition. Reappearance also checked a
shared kind without checking the earlier object's measured shape.

The same extent bound now applies with or without points. Without current
point support, appearance must meet the existing same-look threshold used for
sudden jumps. Reappearance checks the former object's shape and appearance.
Point motion still constrains association and does not replace body velocity.

The separation of appearance evidence from motion prediction is consistent
with the tracking problem described by [Wojke et al.](https://arxiv.org/abs/1703.07402).
No model, benchmark weights or game-specific recognition were added here.
[CoTracker3](https://arxiv.org/abs/2410.11831) was also reviewed as a reference
for persistent point tracking; its trained tracker has not been adopted.

All 66 related tests passed on invented scenes. Ruff, writing, method size,
module size, layering, dependency consistency, compilation and lint passed.
Smoke: 165 passed, 1 skipped, 48.05 seconds.

Replaying the first 1,105 measured frames from run 12e gave these results:

| Measure | Recorded preceding run | Candidate replay |
| --- | ---: | ---: |
| Box-area changes greater than fourfold | 1,282 | 0 |
| RGB median changes greater than 160 | 380 | 160 |
| Mean current observed tracks | 63.1 | 71.3 |
| Frames with a rough white-body proxy | 863 | 826 |
| Identities in that proxy | 102 | 96 |

The proxy is not labelled object accuracy. Its reduced coverage is a concern
for gameplay verification. Candidate perception took median 43.0 ms and 95th
percentile 74.7 ms on this replay. These measurements do not establish control
identity or presentation readiness.

Logs: /private/tmp/codex-steer/identity-continuity-tests.log,
identity-continuity-gates.log and replay-continuity-final.log.
