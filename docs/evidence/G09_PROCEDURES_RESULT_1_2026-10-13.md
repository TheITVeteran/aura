# G09 kept procedures, first attempt: a gain in transfer, and not yet broad

The first confirmatory run of [the preregistered plan](G09_PROCEDURES_PREREGISTRATION_2026-10-08.md),
read against its own criteria. G09 stays open: the gain is real and confined
to one domain, and the plan says a gain in one domain is reported as that.

## What ran

Held-out induction (`~/.aura/rlc-evidence/g09-induce-heldout-20261008`):
code `3d34c1109` with the fixed-group decoder on top (`19b9c7e6d`), her
persona 27B, private channel open, 20,480 tokens a proposal, one line a
kind, three rounds, three shown problems. 58 proposals were decoded
(920,976 tokens). 38 reached the cap still thinking, so ten of the 20 kinds
never produced a function.

Kept, each agreeing with all 180 of its known answers (90 shown pool, 90
sealed) after one round of repair from its own counterexamples:

| Kind | Rounds | Known answers agreed |
| --- | ---: | ---: |
| BBEH dyck_languages | 2 | 180 of 180 |
| BBEH shuffled_objects | 2 | 180 of 180 |

Closest without being kept: CRUXEval at 199 of 200 in its pool after its
third and last round (it raised on the 200th), BBEH word_sorting at 17 of the
first 20, trip planning at 9 of the first 12.

## Paired result

Drafts were decoded for every request a kept procedure answered: the 40
BBEH Mini problems of the two kept kinds (`g12-bbeh-her-20261008`, her
ordinary protocol, fixed groups of eight). The kept arm answered all 40
(`tools/g09_kept_procedure_arm.py`); `tools/g09_analysis.py` reads:

| | Requests | Draft right | Kept right | Gained | Lost | Two-sided p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dyck_languages | 20 | 14 | 20 | 6 | 0 | 0.031 |
| shuffled_objects | 20 | 19 | 20 | 1 | 0 | 1.0 |
| **transfer** | 40 | 33 | 40 | 7 | 0 | 0.016 |

No other request reached a kind with a kept procedure: 0 of 4,500 MATH
problems, 60 AIME, 164 HumanEval+, 7,405 HotpotQA, 6,511 BBH, and the
CRUXEval, calendar and trip test halves (400, 500, 800); of BBEH Mini's
460, exactly the 40 above. Those requests keep their drafts in both arms.

## Against the criteria

* Primary, pooled over five domains: 7 gained, 0 lost, p = 0.016. Holds.
* Breadth, at least two domains on held-out kinds: one domain (transfer).
  Does not hold.
* No harm: no domain lost a request. Holds.

G09 does not close on this attempt.

## What it says about the mechanism

Where her model wrote a function, the loop worked as designed: two
functions that disagreed with known answers were repaired from their
counterexamples into procedures that then answered every unseen test
problem of their kinds correctly. The limit was earlier, at writing a
function at all. That is what the
[second attempt](G09_PROCEDURES_PREREGISTRATION_2_2026-10-12.md) changes.
