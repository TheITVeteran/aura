# Optional geometry leaves control time, 13 October 2026

The original game 18 launch at fef54cc41 exposed a synchronous event-loop
stall before continuous play. The first active capture at 429851.330 arrived
1.667 s late. The board and well routes on that retained picture returned
false after 1.477 s and 0.591 s; a concurrent 10 ms heartbeat stopped for
2.078 s. The observer became ready 23 ms after continuous play began.
This measured case implicates optional geometry, not observer startup.

Shared activity understanding now records whether a board or falling-piece
check fits the described activity. Known continuous steering reaches
control immediately. Arrows, gravity, grids and real time alone do not
exclude discovery. Explicit turn and falling-piece rules retain it.
The actual route uses the retained instruction history; its last few score
screens cannot hide an earlier steering instruction. Instruction-cache
identity includes content, so replacing a reading without changing its
count cannot preserve an earlier activity.

Pure board and well picture work now runs on a worker with detached inputs
and output. Each probe owns one raw future. Its caller has a finite wait;
timeout returns unmeasured, never a visual miss. Running work retains its
capacity until actual completion. Cancellation and closure forbid late
publication. Browser capture and temporal well history remain on their
original loop. The existing three/eight-second well watch remains for
unresolved scenes; this checkpoint does not make that watch incremental.

The lattice search groups phases by their observed tooth counts rather
than scanning each phase in Python. A private benchmark matched 2,088
profile cases and 62 invented/random/blank pictures. On a 960 by 576 random
RGB picture, lattice search fell from a median 556.61 ms to 20.64 ms,
26.97 times faster. Ties, tail-neighbor zero, nonfinite rejection and the
smallest nearly-best spacing remain unchanged. Existing geometry misses
also remain unchanged. The first suite found an invalid extra assertion
that one legacy fixture must be detected; the equivalence result was already
correct. Independent eight-by-eight successful-detection canaries remain.

The final activity, worker ownership, lattice, actual route, board/well,
negative-cache and shared-understanding suites passed 153 tests in 80.90 s.
New claims are registered with their tests; organism dependencies name the
two new canary modules precisely.

The preceding original game 9 run at 20613535ba lasted 136.120760 s without
an error or a win. All 602 actor observations lacked an owned body. The
main observer retained 400 frames, median 0.167 s apart; only one gap exceeded
0.35 s. Track 237 rotated into a stationary arrow and disappeared while
the vehicle remained visible. Its short lifetime supplied one acceleration
trial, one reverse trial and two neutral trials. No moving point support
continued that player fragment. These timing fixes do not establish repair
of that separate perception failure.

Traces: `/private/tmp/codex-steer/run-18-observed-retries-1013/` and
`/private/tmp/codex-steer/run-9-travel-speed-witness-1013/`.
Benchmark: `/private/tmp/codex-steer/lattice_period_benchmark_1013.json`.
Gates: `/private/tmp/codex-steer/probe-routing-final-gates-1013.log`.
Fresh original-program and ordinary desktop verification remain open.

All nine gates passed. Smoke passed 165 tests with one skip in 57.95 s.
