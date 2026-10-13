# One current motion state for facing and feedback, 13 October 2026

Facing, the check that an owned body still answers, and the remembered
continuity velocity now use the same current measured-state policy as input
identification and provisional steering. Older public velocity fields stay
available for their existing callers. A replacement, departed reference,
missing measurement or stale capture cannot borrow the earlier state.

The new tests give an observed rightward trajectory a false raw velocity
pointing left and down. The actual facing consumer reads rightward motion.
A continuing measured trajectory under a known input retains its control
identity through forty further observations despite the false raw velocity.
Missing or stale motion supplies no heading, and a new birth cannot borrow
the previous estimate. The registered facing canary also exercises a current
state whose measured direction differs from its raw velocity.

Connected heading, ownership, steering and motion-model suites passed 177
tests in 37.74 s. After adding the registered canary's state boundary,
43 focused tests passed in 9.81 s. The first new-test run had two fixture
failures: an unjustified two-pixel velocity-error assertion and expected
control data placed in a cache that is correctly cleared on model change.
The repaired tests use the production uncertainty criterion and actual
continuing observations with established control history.

The original game 9 test at f1b6fa67b lasted 197.953932 s without a win.
It recorded 758 observations, 41 with control identity, 21 current estimates
and 21 controller poses. No provisional steering action was delivered.
The first selection used repeated input trials (F = 40.945343,
effect = 49.402583). Its facing trace never acquired a forward input; this
upstream state connection remains necessary but has not proven success.

Trace: `/private/tmp/codex-steer/run-9-measured-identification-1013/`.
Ordinary desktop play and presentation qualification remain open.

All nine gates passed. Smoke passed 165 tests with one skip in 132.56 s.
Gate log: `/private/tmp/codex-steer/facing-state-final-gates-1013.log`.
