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
| 5 latent optimization | `core/learning/latent_optimization.py` | **none** (`spec.latent_opt_mode` exists, "disabled") |
| 6 fast weights | `latent_cortex/fast_weights.py` | live |
| 7 adaptive halting | `core/learning/adaptive_halting.py` | **bridge built** (`latent_cortex/learned_halting_bridge.py`), not called |
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

## Then: latent optimization (component 5)

`spec.latent_opt_mode` is validated to `"disabled"` for v2 training. Wire
`optimize_latent` into the engine between recurrence and fast-weights (the
docstring order at `engine.py:2` already names this slot). Two
non-negotiables from Anima Rationis:

* the score fn must be verifier-grounded (line 220 -- optimizing confidence
  strengthens confident mistakes); `LatentObjective` refuses an objective
  with neither verifier nor consistency term.
* `matched_random_control` must run on every claim (line 453). It ships in
  the module, not the tests, for this reason.

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
