# Short responses before slower planning

The play loop can now give a fresh measured control response a short
physical lease before reading counters and choosing a task action. The
circuit is arithmetic over typed observations, response costs, context
identities and acknowledged inputs. It has no game vocabulary and does
not call an LLM. Its current play adapter handles measured direct motion.
Carried motion, turning controls, lifting presses and pointer motion
remain with their existing controllers.

The circuit rejects stale, future and repeated observations. It requires
an alternative with less than half the current predicted cost. A response
holds for 120 ms, with an independent release task; that task can release
only its exact delivered key and timestamp. Discovery trials, prepared
input sets, charged inputs and displayed key requests retain their owner.
Two non-improving later risk observations suspend a response in the same
model context. Missing outcomes stay unknown. Small speed-estimate changes
do not clear that history; an owner or substantial response-model change
creates another context.

The current feedback cost is a risk estimate from later observed geometry.
It is not a measured damage counter or proof that the action caused the
risk to fall. Independent confirmation of the physical response is the
next connection to build. Neither component tests nor this pilot certify
presentation readiness.

The first whole-app dodge pilot used the normal game entry, public frames
and normal hands. Its independent final tally was still playing, score
42, two lives, one loss and no completed game. Won was false. It returned
normally after 67.202324 seconds. Six reflex leases started with pictures
52.824 to 81.676 ms old. The trace exposed model-context churn from small
speed-estimate changes; the subsequent stability fix has a regression test.
The pilot preceded that fix and the displayed-request arbitration change.

The final focused suites passed 86 tests in 13.91 seconds, including
physical ownership, release while a slower task waits, simulated cursor,
obstacle and backlog dynamics, source-model qualification and prepared
input ownership. All nine gates passed. The canary is registered with its
validation test.

Research basis: Brooks's layered reactive control separates low-level
responses from more complex behavior, and Simplex separates an advanced
controller from a monitored fallback. This implementation borrows the
control separation; it makes no formal safety guarantee from those papers.

- https://people.csail.mit.edu/brooks/papers/AIM-864.pdf
- https://arxiv.org/abs/2102.12981
- Pilot: `/private/tmp/codex-steer/reflex-dodge-whole-app-seed1-1013/`
- Gates: `/private/tmp/codex-steer/reflex-final-gates-1013.log`

Original Flash wins, the 56-game campaign and ordinary desktop request
proof remain open. A fresh whole-app repeat and response confirmation are
next.

### Later response confirmation and bounded observation gaps

The response circuit now checks independent later motion against the
measured input model and the alternative of continuing the previous input.
Risk falling alone cannot verify the response. Contradictory motion
suspends the response; missing or ambiguous motion remains unknown.
Actual request-to-acknowledgement intervals bound clock uncertainty.
A brief missing measurement does not cut an owned pulse short. After
release, a 150 ms feedback window expires without retaining input ownership.

Four further ordinary-entry private dodge runs did not establish a verified
physical reflex improvement. The bounded-gap candidate finished still
playing with 61 points and three lives, but its two settled risk outcomes
had one unknown physical response and one contradiction. The current
clock-envelope candidate returned in 65.880976 seconds with 14 points,
three lives and no completed game. Startup and menu work consumed most
of that budget: none of its 640 cycles started a reflex. Neither survival
nor absence of a loss field is evidence of a causal improvement.

The clock-envelope focused suites passed 112 tests in 13.33 seconds.
The final gates and any further results are recorded below. The repository
governance lint has 116 existing regressions in 30 other files; these
changed files do not occur in that offender set. No baseline was relaxed.

- Bounded-gap pilot: `/private/tmp/codex-steer/reflex-bounded-observation-gap-dodge-seed1-1013/`
- Current pilot: `/private/tmp/codex-steer/reflex-clock-envelope-dodge-seed1-1013/`

Final clock-envelope gates: all nine passed. Smoke passed 165 tests,
with one skip, in 43.37 seconds. Log:
`/private/tmp/codex-steer/reflex-clock-final-gates-1013.log`.
