# Arrival qualification and run 18w, 2026-10-12

The arrival planner enabled its terminal braking costs before it established
a usable feedback arrival path. Missing neutral input or opposing pushes
could therefore change ordinary plan scoring even though arrival control
remained unavailable. It now enables those costs only after the path exists.

Three integration tests remove neutral, first, or second push evidence in
turn. Each checks that the unavailable arrival leaves the ordinary plan,
worth, and decision unchanged. The related arrival and carried-control
suite passed 54 tests in 11.76 seconds. Ruff, writing, changed-method size,
module size, layering, dependency checks, compile, and lint passed. Smoke
passed 165 tests with one skip in 73.15 seconds.

## Actual run after current-frame timing and capture changes

Owned run `/private/tmp/codex-steer/run-18w` used frozen source revision
`5604456d19b70511ba46e0f5edcf4e8cc57ab68b`. It completed normally in
95.284734 seconds with return code zero and no harness timeout. Its result
records two losses, `won: false`, zero landings and one crash in the final
screen. The actual harness records 1.5 minutes, although the dispatched
command requested 1.3. Both values remain in the original artifacts.

The only real-time stretch lasted 7.3 seconds at 13.5 pictures per second.
It never assigned a controlled body or learned movement keys. Inputs were
four up presses, one left, and one right. The later slow pursuit remained
on the result screen. No completed landing validates the steering changes.

The separate shared model campaign still owned its exclusive lane; scene
and body interpretation requests were refused. Its process and the resident
server were left running. No fallback capability is inferred from a health
endpoint or from normal harness completion.

A diagnostic replay of the saved tracker pixels shows an initial body box
growing from 21 by 24 to 21 by 58 pixels as its upper edge remains at the
original location. The original track disappears at frame 000073 despite a
compact body-sized proposal remaining nearby. This identifies an observation
and continuity problem to investigate; it does not authorize treating the
proposal as the body without evidence. Replay is diagnostic and uses the
saved working-size pixels, not the original full-size captures.

## Source inspection following the user's request

The authorized offline program is `programs/18.swf`, SHA-256
`a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
Aura's existing static reader extracts 707 statements. The extracted code
is saved at `/private/tmp/codex-steer/landing-source-18.json`. It was parsed,
not executed, and no external or leaked game files were used.

The statements show accumulating and decaying thrust, gradual lateral
control, tilt, position-dependent landing checks, and a descent-speed
threshold. These call for a general actuator-response model and arrival
constraints beyond position alone. The reader writes conditional bytecode
branches as offsets; exact branch semantics still need independent checking
before a source-derived reference is called faithful. A named sprite in the
source cannot establish that Aura has identified it from the picture.

Logs under `/private/tmp/codex-steer/`: `arrival-qualification-tests.log`,
`arrival-qualification-gates-summary.log`, `arrival-qualification-checkpoint-gates.log`.
The run's `completion.json`, `results.json`, `game18.log`, and measured
frames retain the actual experiment. Presentation readiness remains unproven.
