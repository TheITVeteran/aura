# A fresh heading after reacquisition

The pixel actor at source-image resolution identified a body in 328 of 400
observations but learned no steering. After a track change it retained known
movement keys, correctly discarded its old heading, and repeatedly tried an
unresolved turn while stopped. The turn required travel. The current heading
learner could not request a driver until it already had a heading.

## Change

Earlier owned movement responses can now request bounded current direction
experiments. Each candidate is offered at most twice, only if it exists among
the current controls. The held experiment ends with a rest so its observed
direction can be retained. These hints supply no heading, turn rate or control
authority. A changed binding still clears all earlier heading measurements.

No game identity, character or extracted physics enters the change.

## Validation

The four new tests initially failed three times and passed once. The related
run, after the repair and measured-claim registration, passed all 144 tests in
16.63 seconds. Existing stationary and rolling navigation cases still require
four contacts with continuous identity. The claim and invariant are registered.
Logs:
/private/tmp/codex-steer/fresh-heading-red.log and
/private/tmp/codex-steer/fresh-heading-related-tests-v2.log.

The source-resolution pixel actor was rerun. It identifies a body in 327 of
400 observations and now records a current heading and one driver observation
after reacquisition. Steering remains unqualified; the facing-trial budget was
exhausted. This is a repair of the bootstrap request, not a gameplay win.
Before and after logs:
/private/tmp/codex-steer/pixel-actor-native-modes.log and
/private/tmp/codex-steer/pixel-actor-fresh-heading-native.log.

A separate bounded native-crop prototype at the normal 240-pixel working
width identified no body and learned no steering. It has not been adopted.
The log is /private/tmp/codex-steer/pixel-actor-local-point-probe.log.

The full chunked suite and an original-program attempt with this change have
not been run. Gate and integration results belong below.

## Gate result

All nine gates passed on the e41895d27 base. Smoke passed 165 tests with one
skip in 58.04 seconds. The gate log is
/private/tmp/codex-steer/fresh-heading-gates.log. The related run remains the
144-case result above. The full chunked suite has not been run.

## Original-program result at the pushed checkpoint

A clean detached proof checkout loaded
2590e77133f5900147af683eb88be3642a8ea6a3. The assigned original program 9
ran through the normal archived-program harness with the existing model
service. No desktop runtime or model service was restarted.

The owned process completed after 251.85628820897546 seconds with exit zero
and no timeout. won was false; the last words were time's up and QUIT.
The harness reported 11 lost runs. Its later entries repeatedly read the
ending screen, so they are not evidence of 11 independently started rounds.

The 388 bounded heading snapshots contain no identified body, no current
heading and no qualified steering model. The offline repairs have not
produced an original-program win or presentation qualification.

Evidence root: /private/tmp/codex-steer/run-9-reference-heading-1013.
owner.json and completion.json retain process custody and the exact source
revision. results.json retains the outcome; heading-models.jsonl retains the
bounded learner observations. Frame captures and input receipts remain in
that root. No continuous body identity was independently established.
