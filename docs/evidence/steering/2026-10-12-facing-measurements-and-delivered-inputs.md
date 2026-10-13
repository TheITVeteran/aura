# Facing measurements under delivered inputs

The actual `_what_the_place_does` observer produced a false stationary turn
from predicted motion. An invented body was observed travelling east, its
motion was unavailable during a held stationary control, and it was then
observed travelling south. The prior observer recorded
`({'yaw': 0.5}, 1.5707963267948966)` despite the missing motion evidence.
The reproduction called the ordinary runtime observer with explicit pixel
motion timestamps; it supplied direct test state, not original-game pixels.

## Change

`observe_facing` binds the captured picture to the earlier delivered input
receipt. The current input can have been delivered after that picture.
The receipt's delivery time supplies the press clock, rather than the time
its first picture was processed. Current measured motion is required.
An absent body, absent receipt, stale motion timestamp or overlapping input
censors the response measurement. The physical input clock still advances.

`WhichWaySheFaces` rejects duplicate, out-of-order and nonfinite picture
times. Missing or nonfinite velocity clears the current observation segment.
A gap longer than 0.35 s breaks continuous turning evidence. Known turn rates
can still predict the result of an unobserved input, but the prediction cannot
teach a new rate. Fresh travel anchors a subsequent observed turn.
A new receipt for the same named input starts a distinct press.

The observer is wired into the ordinary fast play path. A registered
invariant and measured claim cover input and visual observation custody.
No game identities, keys, source constants or model instructions were added.
This fixes false learning; it does not supply missing pixels or establish
a controlled body when tracking has failed.

## Validation

The related suite passed 116 tests in 28.60 s. Its log is
`/private/tmp/codex-steer/facing-tests-v3.log`. Tests use the actual runtime
observer to reject stationary and moving predictions, bind an older picture
to its input, preserve receipt duration under delayed first observations,
and censor missing bodies and overlapping inputs. They also cover valid
stationary turning, invalid times and velocities, observation gaps, fresh
anchors and prediction through gaps using already measured rates.

The existing turning-body test still learns the controls and reaches at
least six invented targets for each of its six seeds. These are direct
state tests; they do not prove original-game tracking or presentation readiness.
The cadence, causal identity, finite-view and input contract suites also pass.

Changed-file Ruff initially rejected the test's `NS` import alias. The alias
was renamed and Ruff passed. Writing, method-size, module-size, layering,
dependency checks, compile and repository lint passed. Smoke passed 165 tests
with one skip in 108.16 s. Gate output is retained as
`/private/tmp/codex-steer/facing-gates.log`, with the corrected Ruff result in
`/private/tmp/codex-steer/facing-ruff-fixed.log`. The full offline suite was
not run for this checkpoint.
