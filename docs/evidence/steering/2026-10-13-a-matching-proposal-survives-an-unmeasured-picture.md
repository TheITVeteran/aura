# A matching proposal survives an unmeasured picture, 13 October 2026

The original game 9 run at 80192670b lasted 193.418574 s without a win or
an infrastructure error. All 780 actor observations lacked an owned body,
so this run could not test the newly connected controller uncertainty or
drive-role repair. Four loss labels included the same ending and are not
independent episode proof. This negative result remains part of the record.

Trace: `/private/tmp/codex-steer/run-9-steering-confidence-1013/`.

## Remaining continuity boundary

The previous matching proposal required the latest detection's motion to
be measured. A doubtful detection could therefore discard the earlier
trajectory proposal while that trajectory still had a valid bounded
forecast. Missing evidence and contradictory evidence were being given the
same consequence for correspondence.

Each observed state now keeps the visible box belonging to that state.
A correspondence proposal uses that immutable measured anchor rather than
an unmeasured detection or a projected public box. A doubtful later picture
may retain the bounded proposal, but cannot renew its observation time.
Birth, belief owner, reference and view geometry must still agree. A current
measured reference change invalidates the old trajectory. Excessive gaps and
uncertainty still make the proposal unavailable.

This data supplies candidate matching only. A current controller estimate
still requires current measured motion, and a forecast supplies no control
identity or calibration authority. Visible return matching retains both
the belief and its original measured box without declaring the returning
picture's motion measured.

## Validation

New tests simulate an unmeasured detection displaced arbitrarily from the
last measured box. Matching retains exactly the earlier bounded proposal;
the current controller estimate remains absent, both observation clocks
remain unchanged, and the proposal expires. A replacement birth cannot use
that anchor. The connected correspondence, estimation, process-observer,
input-deadline, heading and reference suites passed 92 tests in 31.37 s.

All nine gates passed. Smoke passed 165 tests with one skip in 52.35 s.
Gate log: `/private/tmp/codex-steer/retained-proposal-final-gates-1013.log`.
Original-program reruns and desktop demonstration remain necessary.
