# RLC reconciliation — state of the campaign

Last updated 2026-08-07 16:50 PDT. Supersedes every earlier plan in this
file's history.

## The one thing to understand

**A win was structurally impossible until 2026-08-07.** Every negative result
this program has produced — the 2026-08-06 campaign's 13-vs-5, and its 9-vs-4
reproduction — measured a system that either was not switched on or had no
code path by which it could exceed ordinary decode.

That is not a face-saving reading. It is three specific defects:

- the promotion gate was wired to `decode_incumbent_policy == "latent"`, the
  very policy that removes the floor. Under `latent` the recurrent path owned
  the answer outright and could score far below vanilla. Under
  `vanilla_incumbent` the floor held and replacement was **force-disabled**, so
  the episode was exactly ordinary decode at several times the cost. No
  configuration could both keep the floor and gain.
- promotable rows were built only from `local_repair` requests, so branch
  answers — the entire product of the workspace, branches and recurrence — had
  no route to the output under the safe policy.
- that coupling existed in **three** places (engine gate, receipt authority,
  service validator). Fixing fewer than all three makes every receipt report
  `answer_replacement_unproven`.

## The invariant this is now built against

Bryan's contract, and the right one:

> **≥ vanilla always. No improvement is neutral. Improvement is gain. It must
> never return a lower-quality answer.**

Enforced by `tests/test_rlc_never_worse_than_vanilla.py`, which enumerates the
decode contract rather than trusting it, requires ordinary decode to own the
answer until a gain gate promotes something, and requires every arm to declare
which side of the floor it sits on. The mechanism ablation is the only arm
permitted below it.

## Ten defects fixed today, in dependency order

Each masked the next; none was visible until its predecessor was fixed.

1. **Verifier never admitted.** Fast weights took 0 optimization attempts, both
   verifiers reported `admitted_task_verifier_unavailable`, latent optimization
   ran with `verifier policy: off`, the controller abstained at the floor.
   Admission is not passing a callable — `blind_review.run_decoy_preflight`
   requires separating correct from incorrect arithmetic by ≥0.05 *and*
   bit-identical scores on identical input. An answer-key oracle fails it by
   design. Use `EpisodeTaskVerifier`; it already implements the whole contract
   including `fast_weight_learning_evidence`.
2. **`±inf` sentinels** ("no verified score yet") leaking into a causal receipt
   canonicalized with `allow_nan=False`, destroying episodes that had already
   answered. Fixed at the one serialization boundary they all cross
   (`_finite_record`), not field by field.
3. **Verification objective empty** — passing `token_ids` alone leaves it
   blank and both verifiers refuse regardless of admission. Pass `messages`.
4. **Controller quitting at floor depth.** Terminal actions cost 0.01 to
   execute but END the episode, so `gain/cost` inflated them ~100×; at step 2
   of 8 abstain scored 3.1 against check_assumption's 2.75. Pricing the
   forfeited budget into cost was WRONG — `gain/cost` is not monotonic in cost
   once gain can be negative. Correct rule: keep going while any continuing
   action has positive expected value.
5. **Workspace at effective rank 1.** Every slot seeded from the same global
   mean prompt embedding: slot-to-slot cosine **0.9993** against 0.0419 for the
   prompt's own tokens. Sixteen slots held one direction sixteen times. This is
   the `cos(pass1,pass2) = 0.9994` obstacle chased since CP226 — it came from
   the seed, not the recurrence. Slots now pool disjoint spans of the prompt
   (mean of token embeddings stays in their convex hull, so seeds remain
   in-manifold). After: mean 0.4277, min 0.0282.
6. **Ordinary decode excluded from the candidate pool** (`incumbent_policy`).
7. **Decode parameters diverging from the control** — a 1.25 repetition penalty
   the deployed system does not use, hostile to arithmetic that repeats digits
   and phrasing by construction.
8. **Promotion gate coupled to the floor-removing policy** (above).
9. **Branch answers structurally unpromotable** (above). They now win on the
   same lower-bound-dominance rule repairs use.
10. **That coupling in three places, only one fixed** — caught by the wiring
    and verified-best tests.

## Current status

Working: verifier admitted; both verifiers emitting real receipts; latent
optimization on `strict_task_score_improvement_v1`; branch scores real; the
controller spending depth (`steps_taken` 2 → 5, `halting_reason`
`value_controller_abstain` → `schedule_complete`); ordinary decode as
incumbent; branch candidates promotable.

Believed correct, do NOT "fix" without evidence: fast weights reporting
`not_admitted_high_confidence_evidence_absent` — TheSpark specifies adaptation
only on high-confidence evidence.

Open empirical questions the battery answers: does a branch ever dominate on
real 32B output, and does the system beat 11/28.

## Measured costs (32B, 2026-08-07)

| arm | correct | finished | median latency |
|---|---|---|---|
| `vanilla` | 9/28 | 12/28 | 60s |
| `vanilla_equal_compute` (best-of-3 + vote) | 11/28 | 22/28 | 103s |
| `full_stack` (pre-fix, latent-owned) | 2/28 | ~90% | 135–164s |

**11/28 is the bar.** Beating plain greedy decode while costing more proves
nothing — Anima Rationis: *"otherwise it is just expensive self-consistency."*

## Running it

`chain.sh` gates the battery on a clean regression, publishes the verified
source into the capsule, purges stale latent cells (ordinary-decode cells stay
valid — that path never touches the workspace), and launches. Every terminal
state lands in `chain_status.json`:

```bash
cat /Users/bryan/.aura/rlc-full-20260807/chain_status.json
```

Reclaim the GPU at a cost of one cell:

```bash
touch /Users/bryan/.aura/rlc-full-20260807/sweep/YIELD
```

**Only the `launch_*.sh` / `chain.sh` scripts detach properly** (nohup +
caffeinate, reparent to PID 1). Anything started with an agent harness's own
background runner dies with that session.

## If you are a new session

1. `git pull` — everything is on origin/main.
2. Read the invariant above. It is the design contract.
3. Validate on the 1.5B rig before spending 32B time — a `--model` swap to
   `~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/*/`.
   Same `qwen2` architecture and tokenizer as the fused 32B, ~2 min for a full
   protocol run, ~1GB. It found most of the ten defects above in minutes. Its
   known limit: it validates plumbing, not capability — it never reaches the
   token cap, and its output contains nothing a deterministic router can check.
4. Launch detached via `chain.sh`.

## Standing rule

An arm whose subsystems report `unavailable` has not measured the thing its
name claims. Check the receipt before believing the number.
