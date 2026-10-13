# Observation work and slow steering anchors

An original-program run claimed a body in 358 actor frames and delivered
zero model actions. A capture-receipt subtype failed the model's birth
check, and slow measured movement could not select a calibrated driver.
Image observation also ran synchronously on the input event loop.

The play actor now awaits a worker-owned observer. Its mutable tracker
stays with that worker; completed detached snapshots update the public
view only on the event loop. A timeout leaves the measurement unobserved,
keeps its capacity permit until actual completion, and queues no further
frame behind it. Closing forbids late publication; late exceptions are
consumed. A pending observation yields so physical input deadlines run.
Measured picture activity keeps its change and quiet evidence without
sharing the worker's preceding pixels.

A slow steering anchor accumulates displacement from one current input
receipt. Endpoint pixel error and chord-heading uncertainty remain in
the forecast and its clearance checks. Pixel jitter below those bounds
does not select a driver. Gaps, absent measurements, new owners and input
boundaries break accumulation. The bank remains a provisional finite
curvature family; it does not grant identity or a calibrated control.

Frame callbacks and asynchronous decoding retain their producing session
generation. An old callback acknowledges its original session; an old
waiter or decoder cannot consume a replacement session's picture. Model
deadlines also retain their absolute physical end and exact press receipt.

The expanded suite returned 367 passed in 30.87 seconds. It includes an
actual play-loop test whose real eyes are delayed while the canonical
input lease releases its key, and an actor test using slow displacement
with no calibrated driver. All nine source gates passed. Smoke returned
165 passed and 1 skipped. Logs are
`/private/tmp/codex-steer/async-slow-motion-clean-tests.log` and
`/private/tmp/codex-steer/async-slow-motion-clean-gates.log`.

The first source-gate pass rejected import formatting and a claim
registration function grown past 400 lines. The registration was moved
into the existing claim helper and imports corrected before final checks.
Whole-suite chunks were not run. Original-game improvement and normal
desktop prompt performance remain unproved by this checkpoint.
