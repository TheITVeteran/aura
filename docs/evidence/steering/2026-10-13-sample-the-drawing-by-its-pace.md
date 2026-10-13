# Sampling a drawing by its measured pace, 13 October 2026

The goal remains ordinary desktop play of unfamiliar games. No original
program win or desktop prompt demonstration has been shown in this work.

## What the previous original runs showed

At revision 37d6ec47c, game 9 ran for 194.690901 s without a win. Of 668
actor observations, none acquired a controlled body and none delivered a
model-based steering action. The frame supplier took a median 232.584 ms;
vision took a median 58.130 ms. Game 18 ran for 197.613670 s without a win:
951 actor observations, no controlled body, no model actions, and its final
screen reported zero landings and eight crashes. Loss labels do not prove
independent completed rounds.

Traces:

- `/private/tmp/codex-steer/run-9-heading-lineage-repaired-1013/`
- `/private/tmp/codex-steer/run-18-current-ordinary-actor-1013/`

## Experiments that did not establish an improvement

On 140 retained native pictures from the first trace, using a synthetic
0.2 s replay interval, correcting pixel-centre mapping yielded 532 point
path observations against 512, but only one moving path observation in
either case. A pretrained general optical-flow model used only to seed
strict forward/backward RGB correspondence recovered 63 of 896 attempted
point matches; the same replay still had 512 path observations and one
moving path observation. Reserving marks for observed moving regions gave
583 path observations against 512, still one moving path observation, and
increased median vision cost from 50.71 to 56.83 ms. None was adopted.
These replay totals are diagnostics, not original gameplay outcomes.

The official RAFT-small weights were downloaded only to the private study
folder. No trained model or per-game reference was added to production.

## The measured capture bottleneck

A separate browser probe ran the original program from the verified local
bundle and entered its intro through the visible launch button. There was
no Aura desktop runtime and no language-model call. A WebGL canvas read
outside rendering was blank; the existing screen comparison rejected it.
The browser stream remained the supplier.

Thirty reads with `everyNthFrame=4` took a median 223.624 ms. Thirty reads
with `everyNthFrame=1` took 56.214 ms. Their capture clocks, rather than
arrival times, showed the corresponding source intervals. A private
prototype changed the stride to two after three measured slow intervals;
its median over thirty reads was 109.915 ms.

The protocol counts rendered frames, not a fixed display frequency.
Chromium's implementation requires a stopped stream before another starts:

- https://chromium.googlesource.com/chromium/src/+/refs/heads/main/content/browser/devtools/protocol/page_handler.cc
- https://registry.khronos.org/webgl/specs/latest/1.0/

Raw probes are in `/private/tmp/codex-steer/capture-started-1013.txt`,
`capture-full-rate-1013.txt`, and `capture-adaptive-1013.txt`.

## Production change and its limits

`DrawingCadence` reduces frame thinning only after three consecutive
measured capture intervals all exceed half the supplier's 0.25 s fallback
window. It uses the shortest interval allowed by each capture's uncertainty.
Missing clocks, repeated frames, reversed times and changed clock epochs
break the series. Queued arrival bursts cannot teach capture cadence.

`PageFrames` stops and starts the screencast on the same producing session,
retaining its calibrated clock, original frame receipt and consumed time.
Closing during the stop prevents a replacement stream from starting.
Failure retires this stream for fresh screenshot fallback. It never changes
application state, game timing, control receipts or identity thresholds.

The focused capture suites passed 79 tests in 14.12 s. The broader
integration suites passed 794 tests and skipped one, with one known failure:
a steering path dereferenced the number of an absent body description.
That boundary was repaired to abstain, and the observer now invalidates its
model without reading coordinates from an unowned description.

Cloud changes also exceeded three size limits. Record persistence and
valuation helpers were moved out of the record class; five action-record
helpers were moved out of the execution class; and game record hooks went
to a separate module. The executable bodies of all fifteen moved record
helpers matched the former methods after translating their calls. Existing
public record operations remain in the record class.

Faster pictures alone do not establish body identity or successful steering.
A complete original-program rerun and ordinary desktop proof remain due.

## Checkpoint validation

After the missing-description repair, 149 boundary and record tests passed
in 13.04 s. After extracting the game memory helpers, 39 affected game and
record tests passed in 89.37 s. All nine final gates passed; smoke passed
165 tests with one skip in 46.49 s. The gates are recorded in
`/private/tmp/codex-steer/drawing-cadence-final-gates-1013.log`.
