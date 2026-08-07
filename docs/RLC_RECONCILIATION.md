# RLC reconciliation — what the two negative runs actually showed

Started 2026-08-07. This document is the resume point: read it, then read
`/Users/bryan/.aura/rlc-reconcile-20260807/DECISION.md` if it exists.

## The question

Two resident-32B recurrence training runs came back negative. Retraining costs
about two days each, so the question was whether anything could be resolved
without a third run. Most of it could: the decisive measurement never needed an
optimizer at all.

## What the retained evidence says

The 2026-08-06 four-arm directional campaign, on identical frozen weights:

| arm | correct / 28 | median response |
| --- | ---: | ---: |
| base vanilla | 13 | 1,502 chars |
| adapter vanilla | 13 | 1,502 chars |
| base RLC | 5 | 1,681 chars |
| adapter RLC | 3 | 80 chars |

Three separate defects sit underneath those numbers.

**1. The recurrent arms were instructed to answer partially.** Every recurrent
episode injected a terminal-disposition block ahead of the answer decode; no
vanilla episode did. On 24 of 28 tasks that block read *"give only the best
bounded answer … disclose the unresolved part."* The retained failures match
the instruction: five of the seven losses are well-formed answers carrying
truncated values, including 4- and 5-element sequences where the graded answer
has 7.

That block fired because a fixed-depth schedule reports `max_steps` when it runs
the steps it was configured to run, and the classifier read that as
`recurrence_budget_exhausted`. Completing a plan is not exhausting a budget.

**2. The bridge receipt was not answering for the bridge.** The disposition
tokens were appended to the same list the decode bridge used, so all 56
recurrent episodes published `decode_bridge_policy="none"` beside
`decode_bridge_applied=True` and a 43-token count.

**3. The trained adapter learned to stop reasoning.** Every SFT target was
literally `FINAL_ANSWER: {json}` + EOS — `resident_recurrent_sft_bootstrap_execution.py`
requires it. No reasoning tokens were ever inside the loss, so the cheapest way
to lower it is to emit the answer immediately, and that is exactly what happened:
median generated tokens 28 against 452 for the untrained path. Validation
cross-entropy fell smoothly and monotonically the entire way (cp796: 3.347 →
2.072 over 96 steps). **No teacher-forced loss on an answer-only target can see
this failure**, which is why neither run warned anyone.

## What was fixed

- `terminal_disposition.py` — a completed fixed-depth schedule classifies as
  `planned_depth_complete`, not budget exhaustion.
- `types.py` / `engine.py` — `terminal_instruction_policy` lets a research arm
  decode from exactly the context an ordinary decode sees; the bridge receipt
  answers for the configured policy alone, and `decode_prefix_composition`
  attributes every injected token to its source.
- `recurrent_checkpoint_admission.py` — the **vanilla floor**: an admission
  built without an `ordinary_decode` control is sealed
  `reject_no_ordinary_decode_control`, and a trained arm that does not
  out-answer that control fails `beats_ordinary_decode`. Plus
  `no_answer_only_collapse`, structural rather than a length threshold: a
  response with nothing before its `FINAL_ANSWER` marker answered without
  working, and the trained arm may not do that more often than the ordinary
  path does.

## What is running

Run root: `/Users/bryan/.aura/rlc-reconcile-20260807/`

Two detached processes, both reparented to launchd, both sleep-inhibited:

- `launch_sweep.sh` → `sweep/` — the frozen execution-spec sweep. Five arms ×
  28 tasks, one model load. Crosses terminal-disposition injection against
  recurrent depth on frozen weights: `vanilla`, `rlc_asrun` (4 steps,
  disposition applied — reproduces the 5/28), `rlc_nodisp` (4 steps,
  suppressed), `rlc_shallow` (1 step), `rlc_shallow_nodisp`. **No optimizer
  runs.** Resumable: re-running the script skips every committed cell.
- `launch_pipeline.sh` → `DECISION.md` — waits on the sweep, then gates each
  later phase on the previous one's evidence.

The decision rule, in order:

1. If no recurrent arm reaches the vanilla score → `no_fusion_recurrent_path_below_ordinary_decode`. Stop.
2. Else bisect the 97 retained cp796 generations for the one that still
   reasons. If none out-answers ordinary decode → `no_fusion_no_checkpoint_beats_ordinary_decode`. Stop.
3. Else fuse it and re-validate on a seed the candidate has never seen. If
   ordinary decode regresses or the recurrent gain does not reproduce →
   `no_activation_fused_candidate_regressed`. Stop.
4. Else → `fused_candidate_passed_staged_for_activation`, with the resident
   preserved byte-for-byte as the rollback target.

Fusion is separate from activation on purpose. The recurrence adapter is a
`ScopedLoRALinear`: its delta applies at latent slot positions and nowhere
else. Folding it into the linear weights removes that scoping, so a fused
model is a different function on every ordinary token too — it has to re-earn
ordinary decode, and the pipeline refuses to activate one that does not.

## Honest expectation

Low. The frozen path starts 8 points behind ordinary decode, and the
disposition confound explains some of that gap but is not guaranteed to
explain all of it. The value of the run is that it converts "two negatives" into
a measured attribution, at a cost of hours rather than another two days — and
that the `rlc_nodisp` arm is the first clean measurement of the recurrent path
this program has ever taken.

## Resume

```bash
cat /Users/bryan/.aura/rlc-reconcile-20260807/DECISION.md
```

If it is absent, check `pipeline_status.json` and `sweep/status.json`. A stalled
sweep is restarted by re-running `launch_sweep.sh` — it skips committed cells.
