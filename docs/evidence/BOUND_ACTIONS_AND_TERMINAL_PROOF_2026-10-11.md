# Bound actions and terminal proof, 11 October 2026

Base revision: `6229f3dad`. Presentation qualification remains **unproved**.
The 56-game campaign has no qualified passes at this checkpoint.

The integration retains the newer board, motion and situation work on main.
A choice obligation and its next placement obligation can now share one
physical action, provided both refer to the same currently owned revealed
object. The input owner checks private source pixels and freezes the transfer
identity before pickup. Independent placement evidence must satisfy both
obligations and the exact bound transfer. Failed delivery, stale captures,
replacement plans and unrelated successful transfers give no success credit.

Selector navigation retains a measured container and its object witnesses.
Same-word palettes with different contents have different identities. Missing
or conflicting observations stay unknown. A confirmed named dismissal can
retire a view only when the exact input received an owned response and its
previous object witnesses are absent from the fresh private capture.

Review found a real caller ordering defect: procedure settlement ran before
the action inventory included newly revealed objects. Settlement and decision
now share the same verified current inventory. The physical pipeline test
calls the actual `decide_the_next_move` path at this boundary; it no longer
supplies the missing object inventory itself. The planner's admitted action
and the body dispatch check also use the same earliest held obligation.

Current-turn writing vetoes a predicted win or loss. The terminal reader
abstains on unnamed roles, unknown named winners, conflicting assertions,
conditions and instructions such as “defeat the robots.” Standalone terminal
labels and explicit player/opponent result assertions remain supported.
This repairs false completion claims; it does not prove completed gameplay.

Validation used the shared Python and isolated logs under
`~/.aura/flash-proof-2026-10-11/`:

- Actual decision/body inventory and delivered-evidence checks: 23 passed
  in 20.79 seconds. An earlier overlapping refinement set passed 66 tests.
- Terminal and actual instruction-stop checks: 153 passed in 9.48 seconds.
- Earlier combined control/completion set: 301 passed in 22.92 seconds,
  before the final instruction and caller ordering repairs.
- Smoke: 165 passed, one skipped, in 104.11 seconds after those repairs.
- Compile, Ruff, layering, writing and module-size gates passed. No limits
  or baselines were relaxed. Selection and progress extraction retained the
  original function bodies; the affected deterministic sweep reached 623
  passing tests and exposed two defects subsequently repaired here.

A bounded broader play sweep was stopped at 403 passed and five failures.
The instruction false positive is repaired. Needle timing, pixel paddle
description, charge survival and held-key narration still need causal
reproduction and qualification. Unchanged extracted bodies do not dismiss
those failures. Full-suite and aggregate quality checks are not complete.

Governance lint reports 103 existing regressions on both the exact base and
candidate scans, with zero changed effect buckets. The full gate remains red;
no governance baseline was refreshed.

The supplied recording shows an unbuilt scene and then an emulator reload;
it supplies no successful construction or completed play proof. The later
ordinary checkers delivery reported a win while the screen still asked for
a move. Those observations remain failures. Bryan has requested an offline
campaign for all 56 games before further presentation qualification. Original
SWF assets are being retained with hashes; missing files and dependency or
emulator failures must remain unresolved rows rather than omitted cases.
