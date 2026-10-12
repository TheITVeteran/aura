# Overlay integration check — 12 October 2026

The identity continuity checkpoint was rebased over 11236324e, which restores
the previous scene after a temporary overlay clears. Both changes are kept.

The overlay test selected the first red track, including an older predicted
fragment whose last observation was 0.75 seconds. The current red square was
another track seen at 0.95 seconds and survived the overlay. The test now
selects exactly that current measured square, checks it is present when the
view returns, and still checks its identity four subsequent frames later.

The integrated focused run passed 42 tests. All required code gates passed;
smoke passed 165 tests with 1 skipped in 46.15 seconds. The earlier replay
report describes the pre-rebase candidate, not this merged scene handling.
Logs: /private/tmp/codex-steer/continuity-after-rebase-fixed.log and
identity-continuity-gates.log.
