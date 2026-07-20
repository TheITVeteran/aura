# RLC wiring handoff (CP234)

State as of Jul 20 2026. All seven Anima Rationis components exist with
tests. **None except the halting bridge have a seam into the live engine.**
This document is the map for finishing that.

## What is done

| component | module | live seam |
|---|---|---|
| 1 recurrent depth | `core/learning/intrinsic_recurrence.py` | **none** |
| 2 writable slots | `latent_cortex/engine.py` | live |
| 3 schedule search | `core/learning/schedule_search.py` | engine has `_resolve_schedule` + `LayerSchedule`; search is not connected to it |
| 4 virtual width | `core/consciousness/parallel_branches.py` | live |
| 5 latent optimization | `latent_cortex/latent_opt.py` | **ALREADY LIVE** — engine.py:1372, with matched-random control AND manifold drift. `core/learning/latent_optimization.py` (CP231) DUPLICATES this; prefer the live one. |
| 6 fast weights | `latent_cortex/fast_weights.py` | live |
| 7 adaptive halting | `core/learning/adaptive_halting.py` | **LIVE** — `HaltingController.halting_head` (recurrence.py); None = old policy |
| RLVR | `core/learning/grpo.py` + `tools/train_grpo.py` | n/a (training) |
| verifiable tasks | `core/learning/verifiable_tasks.py` | n/a (data) |

## Next: call the halting bridge from the engine

The engine halts on residual convergence. `HaltingState` lives in
`latent_cortex/types.py:576` (`steps_taken`, `residual_trail`). The
ensemble check is `engine.py:1236` (`ensemble.all_halted()`).

1. Add `halting_mode: str = "residual"` to `RLCExecutionSpec`
   (`execution_spec.py`). Note `adaptive_halting: bool = False` already
   exists and is a **v2-training constraint validator**, not a dead flag --
   do not repurpose it.
2. In the per-branch step loop, replace the residual comparison with
   `should_halt(step=..., residual_trail=branch.halting.residual_trail,
   config=..., head=..., state=branch.z)`.
3. Collect verdicts per branch; attach `bridge_receipt(...)` to the episode
   receipt. **Check `head_was_causal`** -- a learned run whose every stop
   came from the residual floor is the old policy under a new name.
4. Default MUST stay `residual` until a trained head beats it offline.

## Component 5 needs NO wiring — it was already live

`latent_cortex/latent_opt.py` is wired at `engine.py:1372` and already has
both honesty controls: `control_mode` applies matched-magnitude random
perturbations sized from the true gradient step, and the objective carries
a manifold term (RMS + cosine drift from the post-prelude seed).

`core/learning/latent_optimization.py` (CP231) was written without finding
this and duplicates it. It is not wrong, but it is redundant — an audit
that searched `latent_optim` missed a module named `latent_opt`. Prefer the
live one; treat CP231 as a standalone reference implementation or delete
it. **Search by capability, not by guessed filename.**

`spec.latent_opt_mode` is validated to `"disabled"` for v2 TRAINING only;
that is a training constraint, not the live default.

## Then: schedule search (component 3)

`engine._resolve_schedule(domain)` already consults `self.library`. Feed
`evolve_schedules` results into that library. The search REFUSES a single
scorer for search and held-out -- keep it that way.

## Then: intrinsic recurrence (component 1)

`intrinsic_recurrence` currently only runs in `tools/`. Deciding whether the
live engine uses slot-recurrence, intrinsic recurrence, or both is an
architecture decision, not a wiring task. CP227's result should inform it.

## Standing hazards in this codebase

Six bugs on Jul 20, all one species: **a mechanism that appears present but
does not fire.**

- checkpointing delegated to a scope bound to a different module (no-op)
- params snapshotted outside the trace -> all-zero gradients (looks converged)
- `adaptive_depth_loss` returns a DEPTH; treated as an index (silent mislabel)
- metadata `{"ordered": True}` beside a set-equality grader
- held-out eval after a `continue` -> a run never measured itself
- `sqrt(0)` NaN gradient at the optimizer's own starting point

**Test that the mechanism FIRES, not that it exists.** Every bridge here
carries a `*_was_causal` field for that reason.

## Bar for success

Anima Rationis line 658, and do not soften it: +5 broad unseen reasoning,
+15 hard verifiable, 2x at the failure frontier, <=2 point decline
elsewhere, positive d(accuracy)/d(steps), transfer to unseen families,
causal evidence.
