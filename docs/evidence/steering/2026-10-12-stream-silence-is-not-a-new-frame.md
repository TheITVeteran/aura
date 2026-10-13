# Stream silence is not a new frame, 2026-10-12

The streamed-frame reader returned its last image with the current time
when no new frame arrived. That made a stalled stream indistinguishable
from a newly observed still page and erased the age needed for control.

The reader now consumes each arrival once. After its bounded wait, silence
returns unavailable and the existing play reader takes a fresh screenshot.
A new arrival with identical pixels remains a valid new observation. Slow
decoding retains the original arrival timestamp.

Canvas and fallback screenshot requests now retain their start time
through transport and decoding. This is a conservative lower bound on
capture time, rather than a timestamp of the exact physical capture.
Stream arrivals still use local receipt time; this change does not map the
browser's capture clock or measure transport latency before receipt.

The new tests exercise repeated consumption, an identical new arrival,
an arrival during the wait, slow decoding, the real play-reader screenshot
fallback, slow canvas reply and decoding, failed fresh capture, and the
registered adjacent invariant. The combined frame-time, control-time,
and completed-reading suite passed 56 tests in 12.43 seconds.

Ruff on changed files, writing, changed-method size, module size, layering,
dependency checks, compile, and lint passed. Smoke passed 165 tests with
one skip in 48.79 seconds. Logs are under `/private/tmp/codex-steer/`:
`frame-time-tests.log`, `frame-time-gates-summary.log`, and
`frame-time-checkpoint-gates.log`.

This closes a frame-age accounting error. It supplies no new game win or
presentation qualification. The next owned game run must test the combined
current-frame timing and capture changes through the ordinary control path.
