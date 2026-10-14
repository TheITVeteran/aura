# Original point correspondence and shared host capacity

CoTracker's adapter and checkpoint registry entry were removed from active
source. Its published license is CC-BY-NC-4.0. Historical measurements and
external cached reference files were preserved. No license notice was
removed from reused material. The replacement uses original RGB correlation
and reverse correspondence code, NumPy and SciPy; it imports no CoTracker
implementation and uses no CoTracker weights, training data or distillation.
This is implementation provenance, not a legal certification.

`continuity_points` is selectable through the actual motion observer. It
retains a fixed appearance anchor, a recently checked patch and a bounded
hidden interval. Current competing matches abstain. Rotation uses surrounding
material pixels and the same low-pass scale in both directions. Named
object groups may narrow a search through a measured rigid fit, but a group
prediction cannot supply visibility. Reacquisition requires current pixels.
Only consecutive visible frames supply public point motion. The default
remains the classical tracker while further pixel and whole-app comparisons
are incomplete.

The initial 64-point RGB translation measurement took 280.624 to 289.385 ms
per frame. Removing repeated window sums and unnecessary rotation passes
reduced that same measurement to 27.991 to 29.486 ms, with all 64 points
visible in each of four frames. That measurement preceded the padded
rotation support change and is a component result, not a gameplay win or
proof of general superiority to a learned tracker.

TAPIR and SAM 2 now enter the existing fenced host model-memory ledger before
real weights can allocate. Direct real-file adapter construction without
a live memory fence refuses. Their persistent weights and streaming-state
estimates remain counted until actual cleanup. Optional operations take
a shared OS permit without waiting; unknown or insufficient memory headroom
defers them. Device work synchronizes before the permit is returned.
Model peak reservations are estimates and still need actual peak measurements.
No resident process was stopped or restarted.

A real TAPIR load attempt was refused by the existing exclusive G09 owner,
PID 10553, before the tracking model loaded. It returned in 1.218737 seconds;
swap remained 4,798,611,456 bytes before and after. This verifies refusal under
actual contention, not model throughput under contention. Artifact:
`/private/tmp/codex-steer/admitted-tapir-1013/result.json`.

Refusal is expected backpressure. Background corrections retry after 500 ms
and retain current fast tracking. Direct selection preserves a fast observer
when startup admission refuses; a running direct tracker also keeps current
fast points when compute becomes unavailable. Fallback and model point names
occupy separate namespaces. Late startup or frame completion closes the actual
observer after its worker finishes. Memory ownership is not returned while
an abandoned thread can still use the weights. Retained correction sources
are reduced to their declared pixel budget before they enter the history.

The focused tracking, model adapter, point geometry, ownership and deadline
suites passed 189 tests in 20.62 seconds. Repository gates and further pixel
measurements are recorded below. The original 56-game campaign and ordinary
request-to-win presentation proof remain open. A saved source checkpoint
does not mean the resident runtime has imported it.

References:

- https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md
- https://docs.pytorch.org/docs/2.14/mps.html

### Gates before checkpoint

Smoke passed 165 tests with one skip in 41.58 seconds. Focused resource
checks also passed after the lock directory moved to the canonical file
gateway. Ruff, writing, changed-method size, layering, dependency graph,
compile and lint passed. The size ratchet still fails on the unchanged
main-branch ResponseGenerationPhase (21 methods against 20 recorded) and
101 lines of aggregate oversize debt. These tracking changes contribute no
new oversize module. Governance lint still finds existing regressions;
the final offender list has no overlap with these changed paths. Neither
baseline was relaxed. Logs: `/private/tmp/codex-steer/continuity-capacity-final-gates-1013.log`
and `/private/tmp/codex-steer/continuity-governance-final-1013.log`.

### Correction to the preceding governance note, October 14

The checkpoint writer did not stop on its offender-overlap result. The
preceding statement that the final offender list had no overlap was wrong:
`optional_compute.py` still had a new gateway directory-creation call.
That call was removed. Optional work now requires the runtime directory
to be initialized by its existing owner; absence defers the operation
without creating a directory. A regression test checks that no directory
appears. The dated record above is retained with this correction.

After that correction, 19 resource tests passed in 3.32 seconds. Governance
lint returned to its 116 existing regressions; `optional_compute.py` is absent
from the offender list. Log: `/private/tmp/codex-steer/optional-permit-gov-1014.log`.

### Recorded Flash pixels and query allocation, October 14

The first committed replacement at `48c35d520` failed an owned original-game
9 pixel replay. At 192 queries it supplied player pose in zero of 96 frames;
the existing source tracker supplied 59. Median whole-observer times were
142.986 and 52.706 ms respectively. The generated 64-point translation
result did not carry to that scene. The default stayed classical.

The replacement now retains Aura's original RGB subpixel correspondence
checks for uninterrupted motion and uses anchor matching for recovery.
A common coarser scale covers sampling blur with the same forward/reverse
checks. Continuing visible identity can survive a duplicate elsewhere;
reacquisition after absence still rejects competing current matches.

The point bridge allocated corners in source cells while object support
was measured in working cells. Query allocation now uses the working
geometry for every numerical backend. A seeded or warming query set gets
its first observation before replacement. Replenishment uses one eighth
of the actual query count, rather than one eighth of the configured ceiling.
That distinction matters when a 192-query request finds a small patch's
small query set. It has real pixel and lifecycle regression tests.

Recovery starts further patch searches only within a 12 ms allowance and
rotates pending queries between frames. One admitted operation may finish
after the allowance; its actual elapsed time is still returned. This is
bounded admission, not a hard real-time guarantee. The source observer
and actual play-control deadlines remain separate.

With the corrected working geometry, the same replay supplied spread player
point support in 94 of 96 frames, against 73 for the existing tracker, and
matched its 59 pose-supported frames. The final budgeted repeat retained
those counts. At 192 points its median whole-observer cost was 91.846 ms
against 63.667 ms for the existing path. The additional continuity path
is therefore optional while its cost is improved. The original unweighted
selection uses 192 slots; learned backends retain their measured 64-slot
setting and now share the same coordinate allocation rules.

These are recorded JPEG frames with a diagnostic source/human colour label
used only after observation. No tracker received that label. This replay
measures support and cost; it tests no decisions, physical inputs or wins.
The reference cadence is subsampled. Other model work was present on the
host, so these measurements do not establish an idle-host timing limit.

The final focused suites passed 196 tests in 24.25 seconds. A registered
canary covers first observation and the actual live-query replacement
budget. Neither this replay nor that canary closes presentation readiness.

- Initial replacement: `/private/tmp/codex-steer/original-continuity-pixel-comparison-1014/summary.json`
- Subpixel-only attempt: `/private/tmp/codex-steer/named-subpixel-pixel-comparison-1014/summary.json`
- Working geometry: `/private/tmp/codex-steer/working-basis-point-pixel-comparison-1014/summary.json`
- Query lifecycle: `/private/tmp/codex-steer/query-lifecycle-pixel-comparison-1014/summary.json`
- Bounded recovery: `/private/tmp/codex-steer/budgeted-recovery-pixel-comparison-1014/summary.json`

Final query checkpoint gates: Ruff passed after import formatting. Smoke
passed 165 tests with one skip in 48.08 seconds, and the focused query
fixtures passed 23 tests in 4.39 seconds. Writing, changed-method size,
layering, dependency graph, compile and lint passed. The unchanged global
size-ratchet debt remains open. Gate log:
`/private/tmp/codex-steer/named-query-final-gates-1014.log`.
