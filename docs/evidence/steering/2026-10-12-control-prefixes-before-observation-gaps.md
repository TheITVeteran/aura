# Control prefixes before observation gaps, 12 October 2026

Ballet Parking remains unread and unwon. This checkpoint repairs the loss of
measured control evidence; it does not qualify a presentation.

## Original browser attempts

Both attempts used frozen revision `1351e3cee74a6df4c06d49cbe41a4cb65bbc430a`,
the authorized original asset and Ruffle, and Aura's ordinary pursuit and
control paths. Neither opened or restarted the desktop app or model server.
The resident G12 experiment retained its exclusive model lane. Avatar model
fallbacks were refused. These are browser harness attempts, not desktop demos.

- `/private/tmp/codex-steer/run-9-controls-1012`: normal return in 208.097 s;
  no timeout; `won: false`, `runs: [unread]`. The two play stretches reported
  4.2 and 3.6 pictures per second and no established body.
- `/private/tmp/codex-steer/run-9-trial-losses-1012`: normal return in
  125.055 s; no timeout; the same unread and unwon result. The play stretch
  reported 10.1 pictures per second and eventually selected a body, but
  changed tracked identity and learned only one movement key.

Read-only instrumentation in the second attempt recorded 45 deletions of
prefixes containing at least two measured velocity samples. All were caused
by unavailable motion, rather than contradictory motion. Eighteen contained
an absolute velocity component over 5 working pixels per second. The first
such deletion discarded two moving samples under one press. These counts
include scenery and neutral intervals; they do not identify 45 controlled
body trials.

The first attempt retained 439 lossless tracker inputs and their original
monotonic capture times. Replaying those inputs took 30.992 ms per picture at
the median, 34.224 ms at the 95th percentile, and 204.410 ms at the maximum,
including image opening. Blob extraction took 18.853 ms at the median. This
saved-input measurement does not isolate the browser, action-delivery or
other live-loop costs and cannot explain its full cadence.

## General repair

An unavailable reading ends an observed interval without disproving what
was measured within it. `_Speeds.unseen` now closes a prefix containing two
or more finite, chronologically ordered, timed settled samples. Later
observations of the same input cannot extend it across the gap or create a
second trial. An incomplete interval can still restart from new observations.

Contradiction before input completion retires a prefix closed by a gap.
Completion of the delivered input closes its ownership; observations after
that completion cannot revise the earlier interval. Pixel displacement,
entity continuity, independent repeated presses, contradiction and world-motion
exclusions still apply before the ordinary body-discovery path selects control.
Pinned samples retain their constraints. The retained interval sets are bounded
with the existing 80-input history.

Replaying the 45 recorded deletion cases through the frozen code preserved
zero prefixes. The changed code preserves all 45 with their exact sample
values and times. This measures evidence custody only; it does not establish
identity, strategy or victory. Results are in
`/private/tmp/codex-steer/prefix-replay-before.json` and
`/private/tmp/codex-steer/prefix-replay-after.json`.

The related suite passed 127 tests in 10.25 s, including an invented example
through `WhichIsHers.saw` that requires independent repeated presses, observed
pixel excursion and a current measured body. Tests also cover contradiction,
input completion, malformed timing, duplicate timestamps, pinned samples and
bounded history. No archived game or game constants occur in these tests.

## Existing software considered

[OpenCV's optical-flow API](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
supports initial flow for iterative registration. A separate, bounded local
texture-search experiment on the saved inputs did not establish a clear
tracking improvement and was not added to Aura. No OpenCV dependency was added.

[Duckietown's lane-following pipeline](https://docs.duckietown.com/ente/duckietown-manual/40-demonstrations/lane-following-lf.html)
combines measured pose, uncertainty, a kinematic prediction and feedback
control. Its calibrated geometry and controller gains are not transferable to
an unknown screen by themselves. Its runner was not installed or represented
as an Aura capability.

[OpenCV's CSRT tracker](https://docs.opencv.org/5.0/extra_modules/classcv_1_1TrackerCSRT.html)
is another reference for appearance and spatial reliability. It was researched,
not implemented or validated here. The adopted change is the measured-prefix
repair described above. Further body tracking, turning, planning and live
performance remain open.

Validation completed before checkpoint: changed-file Ruff, writing,
method-size, module-size, layering, dependency checks, compile and repository
lint passed. Smoke passed 165 tests with one skip in 49.26 s. Ruff import
formatting was corrected after that sweep and checked again; the imported
symbols and executable statements were unchanged. The full offline suite
was not run.
