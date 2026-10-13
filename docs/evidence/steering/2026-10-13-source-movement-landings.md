# Original movement equations and direct-state landing proof

The production dynamics model is executable code. `GradualControls`
advances stored channels from supplied rules and maps them into movement
and posture. `solve_for_a_required_state` searches bounded transitions;
`FollowingPreparedPath` checks its retained plan against newer observations
and physical action-completion receipts. None of these three calls an LLM.

The independent private oracle extracts and executes the movement function
from the permitted official programme with SHA-256
`a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
Its AVM1 interpreter executes the original branches and operators. The
controller uses the production code at
`0a92d2be26677b7adb6a1085ce513c55c6617767`; source-derived coefficients
and contact geometry are private fixture data. No game identity or rules
were added to production modules.

Six source spawn positions and three pad positions, including both pad
extremes, produced 18 source-observed landings out of 18 cases. Every one
of 3,162 executed native ticks matched its predicted state exactly. Each
terminal source sign was `good`, and every feedback follower finished at
`observed_goal`. The oracle also executed 2,391 original collision checks
with the level-zero empty obstacle array. The batch took 24.45 seconds;
the slowest case took 2.05 seconds. The largest feedback call took about
1.19 milliseconds.

The exact channels include vertical gravity, stored thrust, lateral
response and visual angle. Thrust is not rotated by visual angle. Input
release has a continuing response; landing checks updated speed and angle
against pre-movement contact geometry. Simultaneous thrust and lateral
inputs are required by some prepared paths.

This is a direct-full-state proof of isolated level-zero motion and
contact. It supplies known coefficients and internal numerical channels.
It excludes pixels, state estimation, later-level obstacles, actual input
transport and whole-game completion. The production play actor does not
yet call this stored-response planner or expose its simultaneous-input
actions. Numerical reachability therefore does not qualify a live demo.

Private artifacts are under
`/private/tmp/codex-steer/lander-source-study/`: `source_oracle.py`,
`program_landings.py`, `original-program-landing-batch-v3.json` and
`original-program-landing-summary-v3.json`. Earlier failed private harness
batches remain recorded: the first reused a completion timestamp as a new
observation; the second lacked AVM1 array-length support. Both failed
before this proof and supplied no gameplay pass.
