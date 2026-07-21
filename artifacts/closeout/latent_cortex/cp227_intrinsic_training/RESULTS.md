# CP227 intrinsic-recurrence training — results (Jul 20 2026)

**Run:** resident fused 32B (`Aura-32B-crsm-closeout-jul1`), 387 steps, 180.2 min,
clean exit. Encode 0–16 / recurrent core 16–48 / coda 48–64. LoRA r8 on
o_proj+v_proj across the window (48 layers, 96 projections, 96 depth-banks).
Curriculum depths T=1/2/4, task-depth 8, families khop/modular/register_trace.
192 train / 24 held-out tasks. `rotation_weight = 0.0` (the anti-cosine lever
was OFF — this is the floor).

## Verdict (the trainer's own held-out grader)

```
collapse_repaired   = true
depth_helps_heldout = true
claimable           = "depth improves held-out reasoning"
```

## Held-out CE by depth (24 tasks/depth)

| step | CE d1 | CE d2 | CE d4 | d1→d4 Δ | depth_helps |
|---:|---:|---:|---:|---:|:---:|
| 50  | 0.49792 | 0.50220 | 0.55758 | −0.060 | ❌ |
| 100 | 0.46157 | 0.46384 | 0.47048 | −0.009 | ❌ |
| 150 | 0.45570 | 0.44811 | 0.45674 | −0.001 | ✅ |
| 200 | 0.49873 | 0.50436 | 0.50041 | −0.002 | ❌ |
| 250 | 0.47765 | 0.47370 | 0.47753 | +0.0001 | ✅ |
| 300 | 0.43595 | 0.43500 | **0.43095** | **+0.005** | ✅ |
| 350 | 0.44945 | 0.44524 | 0.44712 | +0.002 | ✅ |

The sign flip is the result: at step 50 deeper recurrence was 0.06 nats WORSE
(the CP226 collapse); by step 300 it is 0.005 nats BETTER, and the ordering
held across the last three evals. Best checkpoint was step 300 (lowest CE,
strictly monotone); the saved `adapters.safetensors` is the final (step ~380).

## What this is NOT (do not overclaim)

- **CE ordering, not accuracy.** No exact-answer capability shown yet.
- **+0.005 nats is small** — directional, not a leap. `worst_relative_ce = 1.0`
  (deepest pass at parity-or-better with the anchor).
- **Floor result** — `rotation_weight = 0`, the lever built for CP226's
  cos(pass1,pass2)=0.9994 obstacle was disengaged.
- **Structured families only**, zero organs. Not Anima line-658's bar
  (+5 broad / +15 hard / 2× frontier / ≤2pt regression / transfer / causal).

## Live status: NOT DEPLOYED

- `adapters.safetensors` is on disk (gitignored 25 MB), NOT loaded by the worker.
- The live engine does not run the intrinsic recurrent forward
  (`docs/RLC_WIRING_HANDOFF.md`: component-1 live seam = `none`).
- Opening the app = base checkpoint, unchanged. Deploying requires a proven
  accuracy gain (gate eval) → engine wiring → adapter load behind a flag.

## Next: the accuracy gate

`tools/eval_intrinsic_accuracy.py` — decode through the intrinsic path at each
depth, exact-answer grade, paired arms (vanilla / adapter-off / adapter-on),
held-out tasks. Does the CE crossover convert to accuracy that climbs with
depth? That result gates the frontier battery and any live wiring.
