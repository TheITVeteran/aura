# General reasoning: settled where it can be, grounded where it can't, stable where neither applies

A plan, written 14 October 2026 while G09 attempt 2 drafts its test requests.
Bryan's request: take the machinery that made checkable problems reliable and
raise open reasoning, knowledge questions and judgment with it.

## What the G09 work showed

Her kept procedures beat her own reasoning for one reason: an answer that can
be checked is trusted over an answer that was only thought. Open questions
have no answer to check against. Their parts often do:

* a quantity can be computed;
* a constraint problem inside a story can be solved by an engine;
* a fact can be found, or not found, in her corpus;
* two independent derivations from the same question either agree, or part at
  a point that can be named.

A wrong answer to an open question usually turns on one such point: a fact
misremembered, a number slipped, a case missed, a word read two ways. The
program is to find that point and settle it with the strongest check that
applies, and to let her answer with what was settled in front of her.

## What exists, and what does not (survey of 14 October)

* The reasoning amplifier runs only for code, math, repository and
  architecture questions, and for logic or planning questions it can execute.
  Explanatory and open questions skip it.
* Self-consistency exists as an exact-match vote
  (`core/brain/reasoning_amplifier.py`, `reasoning_strategies._consistency`).
  Nothing finds where disagreeing derivations part, and nothing settles that
  point with a check.
* Her corpus is a 37 GB full-text index of about seven million Wikipedia pages
  (`core/knowledge/local_corpus.py`), and grounding from it is live. Nothing
  checks her draft's claims against it, and nothing lets her decline when the
  claim has no support and her derivations disagree.
* One formalizer exists: her model fills a game specification and
  `finite_game.solve_game` solves it (`core/reasoning/game_answer.py`). The
  constraint engine (`settled_by_neighbours`), the positional engine and the
  STRIPS simulator have no such front end.
* The paths that would turn verified answers into training for her weights
  have never fed anything (`reasoning_self_improvement`: 51 traces, none fed).

## The four mechanisms

**A. Where derivations part.** Her greedy derivation and one sampled
derivation answer the question independently. When they agree, that is the
answer. When they do not, her model names the crux: the one claim each side
asserts that the other denies. The crux is settled by the first check that
applies: a program in the sandbox for a quantity, her corpus for a fact, an
engine for a constraint, and otherwise two derivations of the crux alone,
which is a smaller question than the whole. Her final derivation is written
with the settled crux in front of her as runtime evidence, the same channel a
kept answer uses.

**B. Claims her corpus can speak to.** After her draft to a knowledge
question, its decisive claim is looked up in her corpus. Support is evidence;
contradiction sends the question back with the passages in front of her; no
support, with derivations that disagree, lets her say she does not know
instead of guessing. SimpleQA scores a declined answer above a wrong one.

**C. Formalize, then solve.** Constraint problems written as stories (who sits
where, who did what, which team gets whom) go to her model to be written as a
specification for an engine: variables, domains, constraints. The engine
returns how many solutions there are. None means the specification is wrong
and goes back with that; more than one means a constraint is missing; exactly
one is the answer. This generalizes the game formalizer to the constraint and
positional engines.

**D. Make it hers.** Answers that A, B and C settled, and that a known answer
confirms, become training data for her own weights through the existing
governed fine-tune pipe. The gain then costs nothing at answer time, and is
measured with the machinery switched off.

## How it is measured

Every claim is paired, request by request, against her ordinary answer (one
greedy decode, her serving effort, the G12 protocol), with an exact two-sided
McNemar test. Development halves tune the mechanisms; test halves, sealed
until a plan for them is committed, decide the claims.

| Benchmark | What it asks | Split |
| --- | --- | --- |
| MMLU-Pro (TIGER-Lab, 12,032 questions, 14 subjects, ten options) | knowledge and reasoning across subjects | halves by hash of question id |
| MuSR (TAUR-Lab, murder mysteries, object placements, team allocation) | reasoning through long narratives | halves by hash |
| BBEH's tasks no procedure was kept for (causal understanding, disambiguation, sarcasm, humour, sport, films, hyperbaton, spatial, board games, zebra puzzles, Linguini) | hard reasoning that is not mechanical | 180 development, BBEH Mini 20 test |
| SimpleQA (OpenAI, 4,326 questions) | short facts; correct, wrong or declined | halves by hash |

Two comparisons, both reported:

* against her ordinary answer: what the machinery adds;
* against plain self-consistency at the same number of decoded tokens: what
  the machinery adds beyond spending more compute. A mechanism that only
  matches majority voting at equal cost is not reported as a gain.

Tokens per answer are reported for every arm. GPQA is left out: its data is
gated behind terms only Bryan can accept.

## What this cannot show

Writing quality and new ideas have no answer to check against, and this
program does not claim them. Judgment is measured only where it shows: on
SimpleQA, as declining when she would be wrong, and everywhere, as whether her
stated confidence matches how often she is right.

## Order of work

1. A, with its harness and the equal-compute baseline, on CPU now; a
   development run of about 40 questions per benchmark when the model is free.
2. C, on BBEH's zebra puzzles, spatial reasoning and board games, MuSR's
   object placements and team allocation.
3. B, on SimpleQA, using the corpus that is already live.
4. A plan for the test halves, committed before they are run.
5. D, once A to C have settled answers worth learning from.

The model is busy with G09 and G12 for days. Development runs take the gaps;
nothing here delays either item.
