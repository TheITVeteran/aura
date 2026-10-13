# A direction experiment needs motion and a new receipt, 13 October 2026

An original driving run found its controlled body but never established a
forward response. It had one or fewer usable current direction pictures
under each active input. A direction averaged across rotated presses may be
small even when each press moved the body.

Repeated measured movement on the current entity now requests a fresh
direction experiment. Opposed trial vectors do not erase those requests.
The history needs separate presses with finite, ordered capture times;
interrupted or missing repeats abstain. These requests supply no heading,
driver calibration or identity. Existing request and per-body calibration
budgets remain bounded.

A new direction experiment that asks for an already held active input now
physically releases and presses it again. Rechoosing a held key otherwise
retains its old delivery receipt, including an almost expired deadline.
The new measurement window starts at actual delivery. Subsequent pictures
keep that experiment intact. An initial neutral observation phase also
handles an absent input timestamp without claiming a physical press.

Related suites passed 97 tests in 30.55 s. New tests show that opposite
directions request an experiment while the heading remains unknown;
missing repeats, interrupted inputs and reversed clocks do not. The actor
test witnesses release and repress of the same requested input, a future
deadline and preservation of the active trial. The existing no-renewal
input-lease tests still pass. The registered fresh-heading canary includes
opposed motion trials.

Original game 18 at f1b6fa67b lasted 200.510683 s without a win. All 431
observations lacked an owned body. Its last view was a crash summary with
zero landings. This checkpoint does not claim to repair that run.
Trace: `/private/tmp/codex-steer/run-18-measured-identification-1013/`.

Original-program and desktop qualification remain open.

All nine gates passed. Smoke passed 165 tests with one skip in 96.17 s.
Gate log: `/private/tmp/codex-steer/direction-trial-final-gates-1013.log`.
