# Translation evidence — 12 October 2026

Run 12g on 49b8ef9dd requested fresh visual readings after losing a body
track, but still established no reliable control and finished without a
verified win. A detection box could shift across scenery while its pixels
stayed put. Learning from that centre shift assigned movement to scenery.

Translation now has an observation status: measured, unavailable or
contradicted. Current owned points must agree with the box. Stationary points
cannot let pixel tolerance accumulate into box drift without independently
observed edges. The fallback for a plain patch requires actual exterior
boundaries on all four sides, stable extent and matching owned interior
pixels. A window sliding along a uniform wall has no such boundaries.

Interior comparison excludes the uncertain outer pixel. A bounded invented
browser diagnostic reproduced a regression: 25 valid movements failed the
old comparison because antialiased edges changed across eleven- and
twelve-pixel crops. Their eroded interiors agreed at 0.977 or better.

Unmeasured boxes do not supply velocity or baseline drift for learning.
Stationary observations start a separate measured path. A missing interval
starts a new control-response segment; contradictory motion retires that
input trial. A press curve can recover against its original visible support
for discovery. Identity receipts retain only continuous measured segments
at their original elapsed times; comparisons use their common observed
window. A contradiction after another delivered input cannot revoke an
already ended receipt. Historical trials cannot assign a new body while
its current pixel movement is unavailable, contradicted or stale. Physics
accepts current measured motion only.

An entity's own learned curve precedes a pooled kind history. Both remain
discovery aids; assigning control still requires that entity's independent
receipts. A policy change while holding the same input does not deliver a
new intervention. Its physical start and experiment status remain fixed.

Settled behavior no longer ends independent observation. In the invented
browser runner, short no-op curves ended near 0.15 seconds while recovered
jump measurements began later. The updated observation interval keeps the
measured stationary response until the delivered trial ends. A later
measured departure revises that same behavior curve and retains its
latency, without adding another trial. A bounded
diagnostic then preserved one body throughout, recorded 939 measured and
14 unavailable translations, and learned its jump. Three jump segments
overlapped the stationary receipts lasting about 0.47 seconds. It did not
establish sustained avoidance, and supplied no evidence for relaxing point
containment during association.

Focused validation: 165 tests passed, including the browser regression,
in 67.18 seconds. The preceding browser regression passed in 37.07 seconds.
After adding the current-body invariant, 66 related tests passed in 10.94
seconds. Checkpoint gates passed: changed-file Ruff, writing, method and
module limits, layering, dependency graph, compilation, lint and smoke.
Smoke measured 165 passed and one skipped in 58.37 seconds.
Boundary construction passed 17 tests with warnings treated as errors and
measured 0.102 ms median per plain proposal, down from 0.428 ms. Earlier
browser regressions during this change exposed fragmented tracks and
insufficient trial coverage; a single earlier passing run was not reliable
qualification. No prompts or game-specific constants changed.

A replay of 700 earlier frames measured 25.68 ms median and 106.11 ms p95
per perception update. It accepted 1,765 of 11,625 current box intervals.
These are observation and timing counts; they do not measure body accuracy
or establish gameplay success. Demo readiness remains unproven.

Evidence: /private/tmp/codex-steer/translation-current-unit.log,
translation-current-browser.log, replay-translation-boundary.log,
boundary-profile.txt, and runner-review/diagnostic-association.json.
