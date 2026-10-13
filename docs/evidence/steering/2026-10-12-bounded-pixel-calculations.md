# Quicker calculations on the same measured pixels

An original-game profile at `3cc9f77d0804e1bd77e5967c22380423cacd56e7`
measured a median 84.737666 ms for tracking and 53.30425 ms for foreground
proposals. Exterior edge sampling and background correction repeatedly
used general median routines on only two or three pixels.

The edge calculation now sorts its three depths and selects the observed
middle sample or middle pair. Holes remain missing. An empty sample cannot
make an exterior boundary visible. Two background-side pixels use their
mean, which equals their median. The support criteria are unchanged.

## Same-input replay

The replay used 156 lossless working pictures from the previous assigned
game-9 run, with their original timestamps and tracker resets. Pictures
were loaded before timing. Both paths were warmed, then their order was
alternated over three measured pairs.

| Pair | Before total s | After total s | Before median ms | After median ms |
| --- | --- | --- | --- | --- |
| 0 | 4.361042 | 3.925004 | 26.490896 | 24.013333 |
| 1 | 4.426917 | 3.959737 | 27.086625 | 24.143437 |
| 2 | 4.332552 | 3.854148 | 26.688688 | 23.684584 |

Each pair had zero differing frames in the recorded fingerprints. Those
cover detected identities, geometry, velocities, paths, motion evidence,
support masks, patches, colour readings, exterior-edge samples, events and
background arrays. This is an output comparison, not a gameplay score or
an exhaustive comparison of all program state. Other campaigns were on
the same machine; these elapsed times are bounded replay measurements.

The comparison and outputs are
`/private/tmp/codex-steer/compare_saved_tracking.py`,
`/private/tmp/codex-steer/paired-tracking-costs.json` and
`/private/tmp/codex-steer/paired-tracking-costs.log`.
The detailed profiles are under
`/private/tmp/codex-steer/tracking-profile-before-v2.*` and
`/private/tmp/codex-steer/tracking-profile-after.*`.
One first profiling attempt referred to a nonexistent diagnostic field;
it stopped before producing a profile and is not product evidence.

## Validation

Related tests passed 121 in 11.08 s. They cover all seven nonempty depth
combinations, missing samples, a large finite sample, observed exterior
boundaries, exposed background and measured translation. The first suite
had a bad test fixture: it repainted the quiet side samples along with the
patch and then expected those sides to disagree. The fixture was corrected.

Changed-file Ruff, writing, method size, module size, layering, dependency
declarations, compilation and repository lint passed. Smoke passed; its
count and duration are retained in
`/private/tmp/codex-steer/bounded-edge-gates.log`.
Test output is `/private/tmp/codex-steer/bounded-edge-tests-v2.log`.
The full suite was not run.

## Original-game result before this speed change

Game 9 at the preceding pushed checkpoint completed after
167.4465632920037 s with return code 0, no timeout and a result file.
It ended `lost`, with `won: false` and no harness error. The first stretch
selected body candidate 83 and measured forward/reverse responses, but
learned no usable steering. It read 4.3 pictures/s for 81.7 s. A later
24.5 s stretch read 5.4 pictures/s and ended at the time limit.
The selected identity has not been independently qualified.

Evidence is `/private/tmp/codex-steer/run-9-trial-rest-1012/`.
This rerun does not prove a general gameplay improvement. The game did not
finish successfully, and presentation qualification remains open.
