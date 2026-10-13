# A retry needs a new observed attempt, 13 October 2026

The original-program traces repeatedly returned one final screen as several
lost runs. Between runs the caller changed its remembered attempt but did
not restart the application. A completed run now performs the page's
physical restart before resetting remembered attempt state. The same applies
to requested score attempts and a page that went blank. Restart failure
returns no new band and preserves the prior state.

A new attempt records the boundary after restart. Its page reader must see
active play in a capture made after that boundary before another ending
counts. Old captures, missing capture times, future times and a new reading
of the same ending cannot establish activity. A current responsive play
record can also supply activity. Callers lacking both a boundary receipt
and measured activity return unknown rather than passing. An ending with
no observed new activity stops with that reason and supplies neither a win
nor another completed run.

The new claim is registered beside its offline canary. Tests exercise the
actual page reader and outer play loop: one observed loss, a physical
restart, then the same ending without new activity. Exactly one attempt
counts; completion remains false. Failure to reload changes no remembered
state. The existing three-score and fourteen-run completion tests now
witness respectively two and thirteen physical restarts. Related completion,
blank-page, current-word and ending suites passed 66 tests in 15.85 s.

At f892d03fd the original game 9 run lasted 192.846787 s without a win.
All 718 observations lacked an owned body, so the direction-trial change
was not exercised. Original game 9 at 37d38ea38 also failed to own a body:
369 observations over 193.804423 s. Neither run is presentation proof.

Traces: `/private/tmp/codex-steer/run-9-fresh-direction-trials-1013/`
and `/private/tmp/codex-steer/run-9-shared-current-motion-1013/`.
Fresh original-program and ordinary desktop verification remain open.

All nine gates passed. Smoke passed 165 tests with one skip in 102.72 s.
Gate log: `/private/tmp/codex-steer/observed-retry-final-gates-1013.log`.
