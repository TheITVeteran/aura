# Control trials at the observed picture cadence

The real control actor could not learn an invented inertial body's controls
at 7.5 or 10 observations per second. Each short trial supplied at most three
settled samples. `_Speeds.pushes()` requires four samples spanning at least
0.15 s. The same actor and plant learned the controls at higher picture rates.

The original parking run at `213ace28a` read 7.1 and 8.6 pictures per second
in its two stretches. Its tracking failures remain separate from this
sampling defect. The invented plant receives direct state observations;
these results do not establish visual tracking or an original-game win.

## Change

`time_to_observe_a_control.py` examines at most twelve recent picture intervals.
Three valid intervals are required; unavailable cadence remains unknown.
The upper quartile supplies a planning cadence. The interval reserved after
input settlement includes the estimator's required sample count and time span.
Ordinary body trials reserve at least 0.45 s and at most four times that value.
Facing trials reserve two settled observations within the existing turn limit.

The actual actor uses this interval for each delivered trial and its rest.
The deadline stays anchored to the delivery receipt. A changed picture rate
cannot renew it. A same-named input delivered again cannot own an earlier
trial, and an older picture cannot interrupt an input awaiting observation.
Counter experiments retain the phase duration chosen when they began.
The no-response search allowance includes the longest trial actually used.
The existing measured braking check still runs before a trial is retained.

Time alone does not establish a response. Current measured motion, contrasting
independent presses, entity continuity and the estimator's evidence requirements
still apply. No game names, source constants or model instructions were added.

## Measurements

The retained external comparison uses the actual `_act` and control learner
with an invented constant-acceleration plant and synthetic direct observations.
It stops after 16 simulated seconds or when all controls are measured.

| Observations/s | Before: all controls known | After: all controls known | After: elapsed simulated seconds |
| --- | --- | --- | --- |
| 7.5 | no | yes | 10.933333 |
| 10 | no | yes | 7.5 |
| 15 | yes | yes | 5.666667 |
| 30 | yes | yes | 5.533333 |
| 60 | yes | yes | 5.333333 |

Before: `/private/tmp/codex-steer/trial-cadence-before.log`.
After: `/private/tmp/codex-steer/trial-cadence-after-v2.log`.
Comparison script: `/private/tmp/codex-steer/check_trial_cadence.py`.

The repository tests exercise the actual actor across both axes, both signs
and four picture rates, with an independently moving distractor. They check
the learned acceleration values and controlled entity. Additional tests cover
bounded history, unavailable cadence, time limits, input delivery delay,
changing cadence and retaining an input without a new receipt.

The related suite passed 125 tests in 29.75 s. Its log is
`/private/tmp/codex-steer/cadence-tests-v3.log`. It includes the existing finite
view, control identity, single-input receipt and facing suites. A registered
invariant checks bounded observation-time reservation. Full offline and
desktop presentation qualification remain open.

The additional no-response and executor contract suite passed 46 tests in
11.24 s. Changed-file Ruff, writing, method-size, module-size, layering,
dependency checks, compile and repository lint passed. Smoke passed 165 tests
with one skip in 90.54 s. Gate output was retained as
`/private/tmp/codex-steer/cadence-gates.log`; contract output is
`/private/tmp/codex-steer/cadence-contracts.log`. No full offline suite was run.
