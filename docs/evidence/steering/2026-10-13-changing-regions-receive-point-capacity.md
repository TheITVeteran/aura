# Changing regions receive point capacity, 13 October 2026

The fresh original-program run at d9fc8dddf used grouped MPS tracking and
the explicit 1,000,000 pixel budget. It finished in 255.740555 s without
a harness error, without a win and without an owned body. It retained
552 measured frames. Its two main observers had median capture gaps of
0.395677 and 0.275901 s, with no moving per-object point receipts.
The former exceeds the existing 0.35 s continuity bound. WhatMoves.see
cost 229.163 ms at its median and 436.574 ms at its 95th percentile.
Component tracking support did not establish live control. This run
remains negative evidence.

The fast source tracker also allocated its point capacity to long-lived
scenery. New texture in a changing region competed with those older marks
before acquiring enough observations to describe a moving object.
Source corner admission now considers actual local pixel change before
spending its fixed budget. Continuing moving marks retain capacity;
new changing regions share the remaining active slots, while quiet
reference marks retain a third when available. Unused reservations
return to other visible marks. Large whole-image change retains ordinary
spatial sharing. Nothing in this selection knows the player's identity,
the task, or which kind of object should be approached.

Every new corner starts a new point identity. Activity supplies no
velocity, identity transfer or control response. Subsequent matched
pixels, forward/reverse registration, color residuals and the existing
object-support checks still adjudicate its path. Disappearance retires
its marks. The observation and continuity deadlines are unchanged.

RGB planes now use one shared spatial filter. Tests preserve each old
filtered color plane exactly; deriving brightness from those planes
agrees with the previous brightness filter within 2e-7. This removes
duplicate filtering while retaining the measured color comparisons.

The paired generated-pixel replay uses the preceding experiment's same
240 images and actual capture times. At the existing 256,000 pixel budget,
three continuing moving patch marks appeared on 101 frames, compared
with zero before this change. Median point observation cost was
0.041140 s, maximum 0.050280 s. At 1,000,000 pixels, support appeared on
238 frames versus zero; median cost was 0.079555 s, maximum 0.127923 s.
The higher cost reflects more actual changing texture being followed.
These are point correspondence counts, not control attribution or wins.

The related source-detail, identity, reference, motion, learned-correction
and steering suites passed 274 tests in 34.63 s. All nine repository gates
passed; smoke passed 165 tests with one skip in 56.00 s. The capacity
claim is registered with its invariant and test through a narrow organism
dependency. New original-program and ordinary desktop proof remain open.

Records:

- Original run: `/private/tmp/codex-steer/run-9-grouped-native-correspondence-1013/`.
- Paired pixel replay: `/private/tmp/codex-steer/changing-point-capacity-benchmark-1013/`.
- Earlier paired baselines: `/private/tmp/codex-steer/grouped-pixel-benchmark-retained-visible-marks-1013/`.
- Gates: `/private/tmp/codex-steer/changing-point-capacity-final-gates-1013.log`.
