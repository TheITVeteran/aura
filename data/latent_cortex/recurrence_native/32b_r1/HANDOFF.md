# Recurrence-native 32B training — run 32b_r1 (HANDOFF)

Detached run started 2026-07-17 ~05:15 PT by the RLC finish-line session.
Any session can check in, resume, or evaluate from this file alone.

## Why this run exists
The preregistered campaign (seed 20260717, artifacts/current/) REFUTED the
frozen-loop RLC: on a checkpoint never trained for recurrence, vanilla
decoding beats every latent arm, and depth 4 recurrence is WORSE than
depth 1 (untrained loss trail 3.35 / 3.28 / 4.17). This run trains LoRA
(rank 8, o_proj+v_proj) on the recurrent-window layers of the RESIDENT
fused 32B under the depth-curriculum objective
(core/learning/recurrence_native_objective.py): answer-span CE through the
exact anchored recurrent forward the live engine executes, plus a
monotonicity hinge that fires whenever more recurrence raises loss.

## The run
- model: training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118
  (THE resident checkpoint, fingerprint 8eae71e7…)
- train seed 777 (disjoint from every eval seed), families khop/boolean/
  modular, task depths 2/4/8, per-cell 64 (576 tasks), curriculum depths
  1/2/4, lr 1e-4, bounded 180 min, ~12-40 s/step.
- process: fully detached (setsid + nohup + caffeinate). PID in
  `train.pid`, live log in `train.log`, loss trail + config in
  `receipt.json`, checkpoints `adapter_latest.safetensors` every 40 steps,
  final in `adapter_final.safetensors`.

## Check in
    tail -5 data/latent_cortex/recurrence_native/32b_r1/train.log
    cat data/latent_cortex/recurrence_native/32b_r1/receipt.json | python -m json.tool | tail -20
    ps -p $(cat data/latent_cortex/recurrence_native/32b_r1/train.pid)

Healthy = mean_loss decreasing, steps advancing. Non-finite loss halts the
run honestly (halt_reason in receipt.json).

## Resume after a crash / extend after the bound
    AURA_LOG_DIR=~/.aura/lab-logs setsid nohup caffeinate -dims \
      .venv/bin/python tools/recurrence_native_train.py \
      --model training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118 \
      --out-dir data/latent_cortex/recurrence_native/32b_r1 --resume \
      --train-seed 777 --families khop,boolean,modular --depths 2,4,8 \
      --per-cell 64 --curriculum-depths 1,2,4 --lora-rank 8 \
      --learning-rate 1e-4 --max-minutes 120 --log-every 10 \
      --checkpoint-every 40 >> data/latent_cortex/recurrence_native/32b_r1/train.log 2>&1 &

MEMORY SAFETY: the live Aura app must be DOWN while this runs (the 32B
cannot be loaded twice on 64GB). Quit via `osascript -e 'quit app "Aura"'`.

## Evaluate (the ONLY way to claim anything)
Preregistered BEFORE any eval: artifacts/current/latent_posttrain_eval_prereg_20260718.json
(eval seed 20260718; hypotheses h1-h3; grader conservatism unchanged).

    # adapter ON
    AURA_LOG_DIR=~/.aura/lab-logs caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py \
      --model training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118 \
      --adapter data/latent_cortex/recurrence_native/32b_r1 \
      --experiments 1,A --task-seed 20260718 --per-cell 8 --max-minutes 60 \
      --out artifacts/current/latent_posttrain_32b_adapter_on.json
    # adapter OFF (control)
    AURA_LOG_DIR=~/.aura/lab-logs caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py \
      --model training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118 \
      --experiments 1,A --task-seed 20260718 --per-cell 8 --max-minutes 60 \
      --out artifacts/current/latent_posttrain_32b_adapter_off.json

## After a positive eval: live wiring
The engine recurs through the SAME module objects, so wrapping the resident
worker's window projections with this LoRA makes live episodes run under
the trained operator. Path: worker-side attach at boot behind a flag (e.g.
AURA_RECURRENCE_ADAPTER=<run-dir>, declared via core/runtime/flags.py), or
the durable-adapter activation seam. Then relaunch the app and drive
tools/drive_live_latent_certificate.py --checkpoint 120.

## Session state pointer
Remaining non-training work when this was written: ~12 full-suite failures
(parallel-session surfaces: chat_social_authority ×2, feedback_audit ×2,
server/system route hardening ×4, expressive_affordances, personality
latch, mind_visualizer, settings UI, legacy shell, chat human contract,
crsm dataset gate), app relaunch + RENDER THIS live proof (governed-scope
fix landed, needs restarted app), CP119/120 live latent certificate, and
the final memory/summary. Everything else on the 17-task board is done
and pushed (HEAD f4966c87+).
