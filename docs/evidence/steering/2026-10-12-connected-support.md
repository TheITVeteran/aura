# Connected foreground support, 12 October 2026

The point-tracking checkpoint remains insufficient for a demo. Its next
seen platforming run, `/private/tmp/codex-steer/run-12d`, ended at a visible
Game Over screen; the harness recorded won=false.

Two older perception defects were reproduced and repaired:

1. Initial background estimation kept a solid object's middle but removed
   its corners. Slight bobbing then moved small fragments far relative to
   their invented size. Existing foreground seeds now extend through a
   bounded connected colour patch when its measured surrounding floor is
   different. Separate patches without their own seeds remain background.
2. A search window cropped part of a larger stationary colour patch. As the
   window moved, the crop's centre supplied false travel. Prior-object
   extraction now requires one connected component whose support does not
   meet an interior search-window edge. A visible component at the actual
   view edge remains available.

No game identifiers, assets, coordinate maps or control scripts enter these
mechanisms. Regression fixtures are invented shapes and textures.

## Evidence

The three previously failing bobbing/scenery tests pass. The connected
foreground, whole-patch, point and object suite passed 59 tests. The control
suite passed 69 tests after reconciling two scenery fixtures with the
concurrent upstream change: scrolling scenery moves opposite its input in
those fixtures, while unrelated entities moving along that input must not
be rejected merely for moving together. The production change to that
control classifier belongs to upstream commit 745e3e78a.

All prescribed gates passed, including smoke (165 passed, 1 skipped).

A controlled replay of 376 lossless inputs from the authorized run-27c
compared c071466af with this foreground change plus persistent points:

| Measure | Earlier checkpoint | Candidate |
|---|---:|---:|
| Object births | 2,272 | 1,244 |
| Box-area changes greater than fourfold | 2,444 | 286 |
| Mean active observed tracks | 208.1 | 114.1 |
| Median perception milliseconds | 56.4 | 55.6 |
| 95th-percentile milliseconds | 123.3 | 87.5 |

These measure tracker fragmentation and cost, not gameplay success. A rough
colour-based body-presence proxy fell from 302 to 288 frames and is not a
labelled accuracy metric. The replay does not certify body identity or a win.

Local logs: `/private/tmp/codex-steer/whole-connected-tests.log`,
`whole-support-control-tests.log`, `whole-support-gates.log`,
`replay-original-checkpoint.log`, `replay-whole-support.log`.
