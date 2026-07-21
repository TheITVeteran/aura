# CP238 GRPO result — the method works, this run was underpowered

## Verified numbers (from grpo_receipt.json, not the log)

Held-out accuracy (68 disjoint tasks, exact-match):
    step  40: 38.2%
    step 120: 47.1%  (peak)
    step 240: 47.1%
    step 280: 45.6%  (final)
    net +7.4 points, +9 at peak, non-monotonic (rose then eased)

Honesty instrumentation REFUSED the triumphant reading:
    had_signal: False
    degenerate_fraction: 0.86  (41 of 300 groups usable)
    diagnosis: "tasks_too_easy: the model always succeeds"

## Why underpowered — the curriculum pass rates

    arithmetic_chain (math)  @2=1.00 @4=0.58 @8=0.05
    constraint_order (logic) @2=1.00 @4=0.95 @8=0.95
    program_trace    (code)  @2=0.05 @4=0.05 @8=0.05

Only math@4 (0.58) was solidly in the learnable band. Logic was already
mastered (no room to gain); program_trace was impossible at every depth;
math@8 impossible. The gain came from essentially ONE cell, and once the
model improved on it the band was exhausted -- hence the plateau.

## The program_trace=0.05 finding (not a bug)

Tasks and grading verified correct (depth 2: 17 -> 55 -> 131, expected 131).
The model genuinely fails them, because they require COMPOUNDING
multiplication (x*2+c) held in one shot, while the arithmetic it aced is
small add/subtract. This is Bryan's own thesis in the data: the tasks the
model fails are exactly the ones that need it to THINK before answering,
and the terse FINAL_ANSWER format denies it that. A one-shot 32B cannot
trace a multiply-loop; a deliberating one might.

## What this establishes and does not

ESTABLISHES: verifier-driven RL converts to real held-out ACCURACY where
CE-trained intrinsic recurrence (CP227) converted to nothing. The engine
turns over. The anti-self-deception machinery (curriculum + telemetry)
worked -- it reported +7 AND why not to trust it.

DOES NOT establish a capability magnitude: the run was signal-starved, the
gain sits on ~1 cell with small n, and it is on synthetic arithmetic with
no transfer shown.

## The powered rerun (what makes the next number real)

The trainer is sound; the TASK SET needs a difficulty gradient with more
cells in the 0.05-0.95 band for THIS model. Options, in order:
1. Finer depth granularity (depths 3,4,5,6 not 2,4,8) so more cells land in
   the band -- pure config, no code change.
2. Allow chain-of-thought room (larger max-tokens, a "reason step by step"
   framing) so program_trace becomes learnable rather than impossible --
   this also tests whether reasoning room is what the model lacks.
3. warm_start_pass_rates (already built, CP237) to drop hopeless+saturated
   cells before training so no step is wasted.

Recommended rerun: depths 3,4,5, add CoT room, calibrate first. Queue when
the integrated eval frees the GPU.
