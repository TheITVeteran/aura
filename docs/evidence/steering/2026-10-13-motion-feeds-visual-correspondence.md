# Measured motion feeds visual correspondence, 13 October 2026

Checkpoint a69f07f3c added state estimation for objects. An original game 9
rerun lasted 192.643283 s, produced 986 actor observations, and established
no controlled body. Consequently no qualified body estimate reached the
steering controller. It did not win. Four loss labels included repeated
readings of the same terminal screen; they are not four verified episodes.

Trace: `/private/tmp/codex-steer/run-9-motion-state-1013/`.

The visual observer still used raw velocity differences in its association
and candidate search. Its new motion belief could feed an owned controller,
but was not connected to the upstream task of keeping that body's identity.
The original run's median visual cost was 72.279 ms against 4.190 ms for
frame supply. Faster capture did not repair attribution on this run.

## General connection

A current measured motion history now proposes where the last measured box
should appear. Candidate search, association cost and the sudden-movement
fallback use that forecast and its uncertainty. A missing picture's public
projection does not become the next forecast's origin. Predictions expire
with their observation interval; uncertainty greater than the body's extent
does not qualify a prediction. Near-stationary histories keep the existing
matching path rather than paying for an unnecessary numerical forecast.
Each object's proposal is cached for one observation and measured state.

Current owned points still take precedence. Appearance, visible support,
size and aspect checks remain independent requirements. Predictions cannot
establish control identity or mark unseen motion as measured.

A visible return after a short absence may keep the original number and
birth across a change of visual classification when a bounded trajectory
and the visible appearance agree. Two indistinguishable predicted returns
leave identity unknown. The recovered motion belief keeps its observation
time; replacement point origins still reset it without a visible bridge.

## Validation and limits

Invented component tests cover corrupted raw derivatives, candidate search,
leap fallback, an already projected public box, expired predictions, changed
references, contradictory shape and appearance, reclassification and
ambiguous returns. The first ambiguity fixture incorrectly changed an
object's number without changing the belief's owner. That mismatch correctly
invalidated its forecast. The corrected fixture independently establishes
both objects' histories and confirms that ambiguity causes abstention.

A pixel-only integration test renders two textured moving bodies and sends
independent repeated trials through the ordinary observer and causal
identifier. No oracle assigns the controlled body. Its own trial receipts
identify the controlled object; current state then reaches steering. Known
coordinates are used only to adjudicate the test result.

The first connected correspondence suites passed 117 tests in 21.87 s.
After adding the registered invariant and the complete pixel-to-controller
test, 76 connected tests passed in 28.21 s. These invented tests do not
establish successful original-program play. Original reruns and ordinary
desktop demonstration remain due.

Integration with newer main commits exposed a prompt-assembler size
regression already present at this checkpoint's starting revision. Five
existing helpers were moved into its existing block base. Their executable
AST bodies matched the originals apart from call-time module imports. The
public method paths remain available through inheritance. The size ratchet
was tightened to the resulting smaller class and module; it was not relaxed.
The casual-interaction helper remains on the assembler, as its import-time
ownership contract requires.

The final related suites passed 80 tests in 29.74 s. All nine checkpoint
gates passed; smoke passed 165 tests with one skip in 78.79 s. Initial gate
failures are retained in the earlier logs. Final checkpoint gate log:
`/private/tmp/codex-steer/upstream-final-integrated-gates-1013.log`.
