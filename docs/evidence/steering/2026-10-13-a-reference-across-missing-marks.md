# A reference across missing marks

The rotating-mark checkpoint 9b630dd7b passed its geometry tests but did not
make the pixel actor learn steering. This change addresses two measured
discard points in that actor. It does not qualify the presentation.

## Observed failures

At a normal working width of 240 pixels, 12 matched updates discarded
independently measured translation when a point path ended. At a diagnostic
width of 480, 16 did. Other ended paths had changed or missing support and
remain unavailable.

In a second diagnostic, the first five rejected associations at normal
resolution all had matching local pixels and visible exterior edges. Their
corner coordinates lost sufficient current-mask ownership to qualify a
point path. The estimated displacement still agreed with the independently
measured whole-patch displacement. Rejecting the whole association created
duplicate tracks.

## Mechanism

A retained reference now belongs to one birth and one measured observation.
Matching local pixels, stable extent and corresponding visible exterior
edges can carry that reference by their independently observed translation
over at most 0.35 seconds. Fresh owned marks can start in that continued
origin. New, stale, changed or unobserved support cannot bridge it.

Off-patch point receipts yield only when independent whole-patch evidence
agrees with their displacement within the existing two-pixel uncertainty.
Missing point ownership alone supplies no authority. Disagreement, missing
edges and changed pixels still block the association. The existing size,
aspect, position and appearance limits remain in the association cost.

No game name, character, asset or extracted physics coefficient enters the
change. It uses the existing pixel and exterior-edge evidence rather than a
privileged state observation.

## Validation

The new handoff tests first failed twice and passed seven rejection cases
against the preceding implementation. After the repair, all 178 related
tests passed in 19.83 seconds. These include the previous rule that losing
marks across a changing outline ends the path, and that an unbridged new set
of marks cannot inherit an origin. The new invariant and measured claim are
registered. Logs:
/private/tmp/codex-steer/reference-bridge-red.log and
/private/tmp/codex-steer/reference-bridge-combined-tests-v2.log.

The full pixel actor was rerun at both 240 and 480 working pixels. Neither
learned steering within 40 simulated seconds. The renderer supplies only
pixels to the actor; its internal physics and true centre are withheld.
The final normal-width observation contains track numbers 2 and 4, compared
with 9, 21 and 28 before these changes. This narrower diagnostic is not a
certification of identity, control or gameplay success. Logs:

- /private/tmp/codex-steer/pixel-actor-first-drops/assignments.json
- /private/tmp/codex-steer/pixel-actor-support-at-drops/assignments.json
- /private/tmp/codex-steer/pixel-actor-whole-patch-240.log
- /private/tmp/codex-steer/pixel-actor-whole-patch-480.log

The original program and full chunked suite have not been run with this
change. Gate and integration results belong below.

## Gate result

All nine gates passed before integration. Smoke passed 165 tests with one
skip in 94.33 seconds. The log is
/private/tmp/codex-steer/reference-bridge-gates.log. Related tests remain the
178-case run recorded above. The full chunked suite has not been run.

## Integrated validation

The change was rebased onto 6e639b088. All 178 related tests passed again in
19.58 seconds. All nine gates passed again; smoke passed 165 tests with one
skip in 48.54 seconds. Logs:
/private/tmp/codex-steer/reference-bridge-integrated-tests.log and
/private/tmp/codex-steer/reference-bridge-integrated-gates.log.
