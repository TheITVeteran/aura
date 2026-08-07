# RLC reconciliation — state of the campaign

Last updated 2026-08-07 12:35 PDT. Read this first; it supersedes every
earlier plan in this file's history.

## The one thing to understand

**Every negative RLC result this program has produced measured a system that
was not switched on.** Not "failed" — never ran.

Everything above the recurrence loop is gated on an *admitted task verifier*.
Nothing ever supplied one, so on every prior run:

- fast weights took **0** optimization attempts (`verifier_unavailable`)
- the generative and counterfactual verifiers reported
  `admitted_task_verifier_unavailable`
- latent optimization descended a proxy with `verifier policy: off` — its loss
  moved 0.001 across 4 steps
- the value-of-computation controller, with no evidence to act on, chose
  `ABSTAIN` and halted at the minimum step

The 2026-08-06 campaign's 13-vs-5, and this morning's 9-vs-4 reproduction of
it, are both measurements of that state. They say nothing about recurrence,
the terminal disposition, or the unified architecture. `RLC Proofs` named this
exact gap as an open item before any of it was run.

## What is fixed and pushed

Admission is not granted by passing a callable. `blind_review.run_decoy_preflight`
scores four synthetic controls — correct arithmetic + valid code, wrong
arithmetic + invalid code, and two byte-identical texts — and admits only a
reviewer that separates correct from incorrect by ≥0.05 **and** returns
bit-identical scores for identical input. That deliberately refuses an
answer-key oracle (which scores every control alike) and admits an executable
verifier.

`EpisodeTaskVerifier` (core/brain/llm/latent_cortex/task_verifiers.py) already
implemented the whole contract, including the `fast_weight_learning_evidence`
provider that fast-weight attachment requires. Use it. Do not write another
one — that mistake was already made and reverted today.

With it admitted, previously unreachable code ran for the first time and
immediately crashed: `±inf` sentinels meaning "no verified score yet" leaked
into a causal receipt canonicalized with `allow_nan=False`, raising inside
receipt construction and destroying episodes that had already produced their
answers. Fixed at the single serialization boundary they all cross
(`_finite_record` in types.py), not field by field.

Also fixed: the harness passed `token_ids` alone, leaving the verification
objective empty, so both verifiers refused with
`verification_objective_unavailable` however well the verifier was admitted.

## Current live status

Working: verifier admitted, both verifiers emitting real receipts, latent opt
on `strict_task_score_improvement_v1`, branch scores real for the first time
(`[None, 0.558594]` — one branch scored on evidence, one unscored).

**Open, and blocking the battery:** `value_controller_abstain`. On both the
1.5B and the 32B the controller takes `steps_taken = 2` — the floor — against
`max_steps = 8`, every single task. The full stack therefore pays ~2.6×
ordinary decode for two recurrent steps. Running the battery before this is
fixed would produce a third "it didn't help" result that again means "it
didn't run."

Fast weights report `not_admitted_high_confidence_evidence_absent`. That is
believed **correct** — TheSpark specifies adaptation only on high-confidence
evidence — and should not be "fixed" without evidence it is wrong.

## Measured costs (32B, 2026-08-07)

| arm | median latency | note |
|---|---|---|
| vanilla | 62–66s | ordinary decode, the deployability bar |
| full_stack | 164s | ~2.6× ordinary decode |

## The battery to run once the controller is fixed

One run answers both "does it work" and "is it just compute" — the
equal-compute control is an arm, and the depth-scaling curve comes free from
`steps_taken`, which is recorded per cell.

| arm | purpose | est. |
|---|---|---|
| `vanilla` | baseline | 29 min |
| `vanilla_equal_compute` | best-of-N at matched FLOPs — **the real bar** | ~75 min |
| `full_stack` | the unified system | ~77 min |
| `full_stack_oracle` | selection vs generation ceiling; never promotable | ~77 min |

≈4.3h. Anima Rationis sets the standard: *"the latent system wins only if…
otherwise it is just expensive self-consistency."* A win over plain vanilla
that is not also a win over equal-compute vanilla is not an architectural
result.

## Taking the machine back

The host cannot hold two 32B models, so the campaign and the live instance are
exclusive — but the campaign leaves on request. See `YIELD.md` in the run
directory:

```bash
touch /Users/bryan/.aura/rlc-reconcile-20260807/sweep/YIELD
```

Stops at the next cell boundary (≤165s). Resume by deleting the file and
re-running `launch_sweep.sh`. Cells carry the sha256 of the decode
configuration that produced them, so a resume can never mix configurations.

## If you are a new session picking this up

1. `git pull` — everything is on origin/main.
2. Read the "Open" item above. Fix the controller first.
3. Validate on the 1.5B rig before spending 32B time. It is a `--model` swap:
   `~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/*/`
   — same `qwen2` architecture and tokenizer as the fused 32B, ~2 min for a
   full 7-arm protocol run, ~1GB. It found four live defects today in minutes.
4. Then launch the battery detached: `cd /Users/bryan/.aura/rlc-reconcile-20260807
   && nohup ./launch_sweep.sh >> runner.log 2>&1 &` — nohup + caffeinate
   reparents it to PID 1, so it survives the session that started it.

**Anything launched with the harness's own background runner does NOT survive
a session.** Only the `launch_*.sh` scripts detach properly.

## Standing rule

Nothing here awards a reasoning gain, a frontier result, or a promotion. An
arm whose subsystems report `unavailable` has not measured the thing its name
claims. Check the receipt before believing the number.
