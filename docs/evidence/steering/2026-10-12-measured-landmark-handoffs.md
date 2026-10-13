# Measured landmark handoffs — 12 October 2026

The motion tracker replenishes its feature points, but a retained point
origin previously kept only its first landmarks. At frame 000075 in the
assigned saved run, three established marks still located the body and a
fourth owned mark appeared. At 000076 one old mark was unavailable. The
fourth had never entered the retained origin, so current motion was refused.

`position_from_the_same_points.py` now admits new marks only after at least
three previously admitted marks pass the existing spread, coherence, current
pixel ownership and measured-step checks. Each new mark is back-projected
into the established origin. The box centre cannot replace that origin.
Admissions are bounded to half the count of continuing qualified marks;
missing and incoherent marks retire. A replacement without three qualified
prior marks still ends the path.

[OpenCV's optical-flow tutorial](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html)
discusses periodic feature detection and backward checks because a vanished
point can be matched incorrectly. Aura already has its own point tracker.
This change concerns the separate admission of a newly detected mark to an
entity's retained origin. No OpenCV implementation or fixed detection
interval was copied.

Validation: 73 point-path, landmark-admission, exposed-background and point
motion tests passed in 11.38 seconds. An invented body moves along an
independently specified path while its initial landmarks gradually retire;
the origin and displacement remain exact. Tests also reject two-mark
handoffs, collinear prior marks, unowned pixels and a large new set attempting
to take over. Existing replacement, stale support and causal input-curve
checks remain passing. Log: `/private/tmp/codex-steer/point-bridge-tests.log`.

Saved-frame effect is limited: current motion for initial track 1 rises from
24 to 25 frames in the same 46-observation replay. The previously unavailable
000076 frame now has point evidence. Track loss on the later joined patch
remains. Replay artifact: `/private/tmp/codex-steer/point-bridge-core-18w.jsonl`.
This is a general continuity capability and a measured one-frame improvement,
not proof of repaired gameplay or presentation readiness.

Checkpoint gates: changed Ruff, writing, method size, module size, layering,
dependency declarations, compile and lint passed. Smoke passed with 165 tests
and one skip in 114.51 seconds. Log:
`/private/tmp/codex-steer/point-bridge-gates.log`.
