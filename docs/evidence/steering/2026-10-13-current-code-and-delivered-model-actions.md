# Current code controls and delivered prepared actions

The original-program retest at
`6033196238c964ee9e0689ac853518cdba8ad0a2` returned no win. It lasted
193.25192279100884 seconds, completed without a harness error, and recorded
1,565 actor frames with zero claimed bodies or model actions. None of
those frames held a key. The capture stream published 3,923 pictures with
browser-wall metadata. The actor was repeatedly entered with only the
reset key and pointer despite its spoken account naming movement keys.
The trace is `/private/tmp/codex-steer/run-9-offloop-slow-1013/`.

An independent reproduction on that frozen revision showed why known
controls could disappear. The parser marks body controls as `its code`.
The guide's source-priority table omitted that source. A later reading
could therefore downgrade those controls to `its program`. Remaining
source-derived cheat and quit controls kept the guide in its worded-control
branch, whose allowed list was empty. This reproduction used invented
manual text with the same control meanings; the exact cached manual used
by the original run was not recovered. Raw and cached source reads both
reproduced the discard. Artifacts are under
`/private/tmp/codex-steer/discarded-controls-study/`.

The guide now gives the parsed-code source its declared priority. Current
source methods bound to a resolved controlled class also supply finite
body experiments when the caller omitted their keys. These experiments
leave measured role and body identity unchanged. Unresolved methods and
other classes do not create keys. A rotation method on that controlled
class can arm a provisional turning family even when the broader control
role is only `move`. Method names do not grant measured turning authority.

The stored-response planner now has an actual actor seam. A typed provider
can supply a prepared path once the current observation owner is qualified.
Each action binds an immutable key set to a physical duration. Every down
and up acknowledgement is recorded. The hold starts after the final down
acknowledgement; its finite timer releases the exact owned inputs without
waiting for another picture. Partial delivery, cancellation, changed owner
or an external input invalidates the retained path. Unconfirmed releases
block further dispatch until cleanup succeeds. Combined inputs close
single-key experiments rather than supplying them with mixed evidence.

Dispatch freshness and physical completion are separate receipts in
`FollowingPreparedPath`. An acknowledged action may finish after its
observation's dispatch deadline, but early completion cannot advance the
path. A later action requires a capture newer than actual completion.

Tests include the real pixel play loop trying an omitted source-bound key,
the actual actor delivering combined keys, and the full play loop obtaining
a current-owner provider and completing two physical input windows. That
provider fixture supplies direct public state; it does not estimate hidden
channels from pixels. No automatic stored-state estimator or automatic
programme-to-dynamics binding is supplied by this milestone. Original-game
improvement and normal desktop prompt qualification require fresh evidence.

## Pre-integration validation

The expanded suite returned 481 passed in 38.29 seconds. All nine source
gates passed; smoke returned 165 passed and 1 skipped in 41.44 seconds.
Logs are `/private/tmp/codex-steer/source-prepared-verified-tests.log` and
`/private/tmp/codex-steer/source-prepared-serial-gates.log`.

Earlier tests exposed a neutral watcher that remained active after primary
release, and incomplete fixture state in the full-loop test. Those were
corrected. One smoke pass returned an empty attribution result in
`tests/interiority/test_wiring.py`; a later complete smoke pass succeeded.
That failed run remains in the overwritten-gate predecessor's captured
tool output; it was outside the control tests. Its order or load dependence
has not been isolated and must not be represented as a confirmed flake.
Whole-suite chunk tests were not run.
