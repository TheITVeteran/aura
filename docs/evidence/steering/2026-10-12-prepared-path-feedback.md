# Prepared paths with current feedback — 12 October 2026

`core/agency/following_a_prepared_path.py` separates a prepared inverse plan
from each control decision. It replays the remaining path from the current
observed state, under the same identity for model, body and goal. It expands
no alternative actions. Changed state may continue when the remaining path
still reaches the required state; an intermediate failure refuses it.

The executor receives one input with the observation's deadline. It checks
that deadline before acting and acknowledges the completed transition,
including its duration. Dispatch alone is insufficient. Only that input's
own timely receipt advances the cursor. Feedback then requires a newer
observation. Missing receipts, expired observations, changed model identity,
and exhausted replay budgets cannot advance the path.

This is a clean-room adoption of retaining planned work between observations
and separating preparation from feedback. The primary implementations
inspected were [Mario AI Framework's AStarTree](https://github.com/amidos2006/Mario-AI-Framework/blob/master/src/agents/robinBaumgarten/AStarTree.java)
and [acados real-time iteration features](https://docs.acados.org/features/).
No source code, game assets or calibrated controller gains were copied.
Neither package is installed or called by this module.

Validation: 114 focused tests passed in 16.53 seconds, including independent
movement and prerequisite-workflow loops. Tests cover changed constraints,
future and stale observations, duplicated and forged receipts, failed or
expired inputs, slow feedback and bounded replay. Log:
`/private/tmp/codex-steer/prepared-feedback-tests-v2.log`.

The 18 assigned source-reference cases were prepared once, then followed
through current direct numerical observations. Execution used a separate
implementation of the inspected physical recurrence rather than the
planner's channel model. All 18 reached the numerical required state, with
772 input transitions and 17,694 checked suffix transitions. Per-case median
feedback times were 0.331–0.818 milliseconds; the greatest individual
feedback call was 3.196 milliseconds. Preparation still takes seconds.
Artifact: `/private/tmp/codex-steer/source-landing-batch-18.json`; log:
`/private/tmp/codex-steer/source-landing-batch-18-feedback.log`.

Limits: direct full numerical state, fixed target height, movement/contact
subset, and virtual action receipts. This is not an original game run.
Visual state estimation, model acquisition, real input duration, and planning
while a live world advances remain caller responsibilities. The normal
visual play loop does not yet supply this known-state path. These results
do not prove presentation readiness; the latest original game run lost.

Checkpoint gates: changed Ruff, writing, method size, module size, layering,
dependency declarations, compile and lint passed. Smoke passed with 165 tests
and one skip in 107.10 seconds. Log:
`/private/tmp/codex-steer/prepared-feedback-gates.log`.
