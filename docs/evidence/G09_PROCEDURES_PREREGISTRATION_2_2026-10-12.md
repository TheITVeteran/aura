# G09 kept procedures, second attempt: preregistered before its development run

Written on 12 October 2026, before any development run for this attempt and
before any request of its held-out kinds has been decoded. The first attempt
([preregistration](G09_PROCEDURES_PREREGISTRATION_2026-10-08.md)) has its own
result page; nothing here changes how that one is read.

## What the first attempt showed about the mechanism

Its held-out induction decoded 58 proposals at a 20,480-token cap. 38 of them
reached the cap with her private channel still open, so ten of its 20 kinds
never produced a function in three rounds. Where a function existed, repair
from counterexamples worked: two kinds were kept after one round
(BBEH dyck_languages and shuffled_objects, each agreeing with all 180 known
answers and then with all 20 of its test problems), and CRUXEval reached 199
of 200 shown known answers in its third and last round. Proposals that did
write a function thought for at most 18,369 tokens and replied in at most
3,492.

## What changes

* Budget forcing (`decode_with_thinking_budget`, 208098332): a proposal
  still thinking when 3,492 of its 20,480 tokens remain has its private
  channel closed by her template's own marker and writes its reply in the
  rest. 3,492 is the longest reply a function was written in.
* Rounds and lines are fixed after the development run below and appended
  here, dated, before the held-out run.
* Everything else is the first attempt's mechanism: three shown problems,
  the sealed half, repair from at most three counterexamples, the routing
  rule, decoding in fixed groups of eight, no mechanisms module.

## Kinds

Development: every kind the first attempt touched (the 23 BBEH tasks,
Natural Plan calendar and trip, CRUXEval output prediction).

Held out, untouched by any decision so far:

* code: LiveCodeBench's execution set (`livecodebench/execution-v2`);
* planning: Natural Plan meeting planning; PlanBench Blocksworld and
  Mystery Blocksworld plan generation;
* math: DeepMind mathematics `algebra__linear_1d`,
  `algebra__linear_2d_composed`, `comparison__sort_composed`,
  `polynomials__evaluate` (drawn by `random.Random(20261012)` from 56).

Each kind's problems are split in halves by a hash of their id; the first
half is the only material the induction run sees.

## Requests

For each held-out kind, 150 test problems drawn by `random.Random(20261012)`
from its test half (all of them where there are fewer). At a 10% draft error
rate that is about 15 expected gains; six gains with no loss already reject
at 0.05. Every request of the routing-only sets (MATH outside MATH-500,
AIME 2024 and 2025, HumanEval+, HotpotQA, BBH, BBEH Mini) is routed too. As
before, a draft is decoded for every request a kept procedure answers,
wherever it comes from; a request no kept procedure answers keeps its draft
in both arms and cannot be discordant.

## Claims, fixed before the run

As in the first attempt:

* Primary: pooled over the five domains, kept answers gain more requests
  than they lose; exact two-sided McNemar, alpha 0.05.
* Breadth: in at least two domains, counting held-out kinds only, gains
  exceed losses at two-sided alpha 0.05 within the domain. Transfer has no
  held-out kind in this attempt, so breadth must come from code, planning
  and math.
* No harm: no domain has losses exceeding gains at two-sided alpha 0.05,
  counting every request a kept procedure answered in that domain, held out
  or not.
* G09 closes only if all three hold.

## What this does not show

* Anything yet: this page is the plan.
