# Reading again after loss — 12 October 2026

The body-locating vision request ran once per stretch. A returned box was tied
to its original frame, correctly, but loss of that frame's track left the
request complete forever. The run could continue for minutes without another
look. Run 12f received one body reading and established no reliable body.

A reusable ReadingAgain schedule now requests another reading after measured
loss when it is still needed. It allows one active request, tolerates brief
occlusion and waits at least two seconds after completion. Longer requests
space retries by their measured duration, up to fifteen seconds. Unread
pictures cannot establish loss. Existing body ownership suppresses renewal.
Cancellation resets the schedule with the request's guidance lifetime.

The returned box remains tied to its captured frame and continuing track.
The prompts and visual model are unchanged. A fresh request does not grant
body identity by itself.

Validation: 53 focused tests passed, including delayed publication,
cancellation, fresh-frame renewal and preservation of established control.
Writing, method size, module size, layering, dependency consistency,
compilation, and standard lint passed. Changed-file Ruff reported the same
two existing import-order findings in test_guidance_lifetime.py as HEAD;
there are no new Ruff findings. Smoke: 165 passed, 1 skipped, 58.02 seconds.

Runs 12f and 27d on f8735b6b4, preceding this change, both finished without a
verified win. Run 27d attributed control to a building and showed score zero.
Run 12f also attributed control to scenery. The next investigation concerns
box movement being learned as physical movement without agreeing support.

Logs: /private/tmp/codex-steer/reading-renewal-tests.log,
reading-renewal-gates.log and renewal-ruff-baseline.log.
