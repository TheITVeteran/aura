#!/usr/bin/env python3
"""Train a checkpoint to accept — and then exploit — its own recurrence (CP227).

CP226 established the target this run has to hit. Untrained, the resident
32B collapses under intrinsic recurrence:

    T=1  12% reasoning / 79% answered   (== vanilla, identity gap 0.0)
    T=2   8% / 71%      T=4  0% / 4%      T=8  0% / 0%

Nothing diverged. The coda simply fails on states it was never trained to
receive. So success here is staged, and the stages are not interchangeable:

  1. COLLAPSE REPAIRED -- CE at T>1 within ~1.5x of the T=1 anchor. The
     model tolerates its own depth. Necessary, and not yet interesting.
  2. DEPTH HELPS -- some T>1 beats the anchor on HELD-OUT tasks. This is
     the claim the whole arc exists to test, and the only one worth
     reporting as a reasoning gain.

Stage 1 without stage 2 is a real but modest result and must be reported as
such. The run also reports base-ability drift, because a model that buys
depth tolerance by getting worse at T=1 has not improved at anything.

Bounded by wall clock and step count; writes a receipt whether it succeeds
or fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.learning.intrinsic_recurrence import checkpointed_window  # noqa: E402
from core.learning.intrinsic_recurrence_objective import (  # noqa: E402
    IntrinsicTrainingSpec,
    adapted_layer_indices,
    answer_cross_entropy,
    depth_tolerance,
    intrinsic_depth_loss,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402

TRAIN_SCHEMA = "aura.intrinsic_recurrence_training.v1"


def attach_adapters(model, spec, *, rank, targets, depth_conditioned):
    """LoRA on the window AND the coda, optionally depth-conditioned."""
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    total = len(model.model.layers)
    indices = adapted_layer_indices(spec, total)
    attached = 0
    for index in indices:
        layer = model.model.layers[index]
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in targets:
                projection = getattr(parent, target, None)
                if projection is None or isinstance(projection, ScopedLoRALinear):
                    continue
                setattr(
                    parent, target, ScopedLoRALinear.from_base(projection, r=rank)
                )
                attached += 1
    if not attached:
        raise RuntimeError("no projections were adapted; check --lora-targets")

    banks = {}
    if depth_conditioned:
        from core.learning.depth_conditioned_lora import wrap_depth_conditioned

        # Per-iteration weight deltas are the mechanism that makes pass 4 a
        # different function from pass 1. Measured on the 1.5B:
        # cos(pass1, pass2) = 0.9994 -- the same map applied twice barely
        # rotates the state, so without this depth is repetition.
        banks = wrap_depth_conditioned(model, depths=max(spec.depths))
    return {
        "adapted_layers": len(indices),
        "adapted_projections": attached,
        "depth_banks": len(banks),
        "window": [spec.prelude_end, spec.coda_start],
        "coda": [spec.coda_start, total],
    }


def trainable_parameters(model):
    """Only LoRA factors and depth deltas train; the checkpoint is frozen."""
    from mlx.utils import tree_flatten

    return {
        name: value
        for name, value in tree_flatten(model.trainable_parameters())
        if "lora" in name or "delta" in name
    }


def encode_example(tokenizer, task, bridge):
    """Prompt + bridge as context; the JSON completion as the target.

    ``task.answer`` is the full 'FINAL_ANSWER: {...}' string while the
    bridge already emits that prefix, so training on the raw answer would
    teach the model to say FINAL_ANSWER twice -- and the contract parser
    would then reject its own training signal.
    """
    import mlx.core as mx

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    ) + bridge
    target = task.answer
    marker = "FINAL_ANSWER:"
    if bridge.strip().endswith(marker) and marker in target:
        target = target.split(marker, 1)[1].lstrip()
    if not target:
        raise ValueError(f"empty training target for {task.family}")
    answer_ids = tokenizer.encode(target, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        # Without EOS the model never learns to stop, and decode runs to the
        # token budget emitting trailing garbage after a valid answer.
        answer_ids = answer_ids + [eos]
    return mx.array([tokenizer.encode(rendered)]), mx.array([answer_ids])


def evaluate(model, tokenizer, tasks, spec, bridge, *, envelope):
    """Held-out CE by depth -- the signal that says which stage was reached."""
    totals = {f"T{d}": 0.0 for d in spec.depths}
    for task in tasks:
        prompt, answer = encode_example(tokenizer, task, bridge)
        for depth in spec.depths:
            loss, _ = answer_cross_entropy(
                model, prompt, answer, spec.plan_at(depth)
            )
            totals[f"T{depth}"] += float(loss)
        if envelope is not None:
            envelope.reclaim(force=True)
    count = max(len(tasks), 1)
    return depth_tolerance({k: v / count for k, v in totals.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prelude-end", type=int, default=16)
    parser.add_argument("--coda-start", type=int, default=48)
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=int, default=8)
    parser.add_argument("--per-cell", type=int, default=64)
    parser.add_argument("--holdout-per-cell", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    # Direct pressure on the CP226 obstacle (cos(pass1,pass2)=0.9994):
    # penalize same-ray consecutive increments so repetition must become
    # computation. 0.0 keeps the objective byte-identical to CP227's.
    parser.add_argument("--rotation-weight", type=float, default=0.0)
    # Re-inject the post-prelude state at every window RE-entry (driven
    # recurrence, the retrofitted-recurrence stabilizer). 0.0 = free-running.
    parser.add_argument("--anchor-injection", type=float, default=0.0)
    parser.add_argument("--no-depth-conditioned", action="store_true")
    parser.add_argument("--checkpoint-group", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260721)
    # CP223's bridge, so downstream numbers are comparable. CP226 used a
    # different cue and its absolute accuracies could not be quoted against
    # the earlier ladder; that confound is not worth repeating.
    parser.add_argument("--bridge", default="assistant_answer")
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    from core.learning import recurrence_curriculum as curriculum

    spec = IntrinsicTrainingSpec(
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        depths=tuple(int(d) for d in args.depths.split(",")),
        anchor_weight=args.anchor_weight,
        rotation_weight=args.rotation_weight,
        anchor_injection=args.anchor_injection,
    )
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(args.bridge, args.bridge)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())

    train_tasks = curriculum.task_battery(
        families, [args.task_depth], args.per_cell, seed=args.seed
    )
    # task_battery groups by family, so stepping through it in order trains
    # on 64 consecutive khop examples, then 64 modular, and so on -- the
    # model would fit each family and forget the previous one. Shuffle once,
    # deterministically, so a resumed run sees the same sequence.
    import random as _random

    _random.Random(args.seed).shuffle(train_tasks)
    holdout = curriculum.task_battery(
        families, [args.task_depth], args.holdout_per_cell, seed=args.seed + 9973
    )
    # Held-out tasks are generated from a different seed; overlap would make
    # the only number that matters a training-set number.
    train_prompts = {t.prompt for t in train_tasks}
    holdout = [t for t in holdout if t.prompt not in train_prompts]
    if not holdout:
        raise RuntimeError("held-out split is empty after de-overlap")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_minutes * 60.0

    from mlx_lm import load
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"train-intrinsic-recurrence:{Path(args.out_dir).name}",
        model_path=args.model,
        purpose="training",
        preemptible=False,
        metadata={"tool": "train_intrinsic_recurrence", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        model.freeze()
        wiring = attach_adapters(
            model, spec, rank=args.lora_rank, targets=targets,
            depth_conditioned=not args.no_depth_conditioned,
        )
        print(f"[wiring] {wiring}", flush=True)
        trainable = trainable_parameters(model)
        print(
            f"[trainable] {len(trainable)} tensors, "
            f"{sum(v.size for v in trainable.values()):,} parameters",
            flush=True,
        )
        if not trainable:
            raise RuntimeError("nothing is trainable; adapters did not attach")

        optimizer = optim.Adam(learning_rate=args.learning_rate)
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        baseline = None
        history = []
        step = 0
        loss_and_grad = nn.value_and_grad(
            model,
            lambda m, p, a: intrinsic_depth_loss(m, p, a, spec)[0],
        )
        # ONE checkpoint scope for the whole run. Entering it per step
        # rebuilt the mx.checkpoint closures every iteration, so MLX
        # recompiled them and its graph cache grew without bound --
        # measured as 13.8s -> 18s -> 24s per step over 40 steps.
        checkpoint_scope = checkpointed_window(
            model, group_size=args.checkpoint_group
        )
        with checkpoint_scope:
          while step < args.max_steps and time.time() < deadline:
              task = train_tasks[step % len(train_tasks)]
              prompt, answer = encode_example(tokenizer, task, bridge)
              # start=None adapts EVERY position: the whole stream recurs
              # here, unlike the slot architecture where only four positions
              # were ever adapted.
              with recurrence_adapter_scope(start=None, stop=None):
                  loss, grads = loss_and_grad(model, prompt, answer)
                  optimizer.update(model, grads)
                  mx.eval(model.parameters(), optimizer.state)
              step += 1
              if step % 10 == 0:
                  envelope.reclaim(force=True)
                  print(
                      f"[step {step}] loss={float(loss):.4f} "
                      f"({(time.time()-started)/60:.1f}m)",
                      flush=True,
                  )
              if step % args.eval_every == 0 or step == args.max_steps:
                  with recurrence_adapter_scope(start=None, stop=None):
                      report = evaluate(
                          model, tokenizer, holdout, spec, bridge, envelope=envelope
                      )
                  if baseline is None:
                      baseline = report
                  report["step"] = step
                  history.append(report)
                  print(
                      f"[eval {step}] ce={report['ce']} "
                      f"collapse_repaired={report['collapse_repaired']} "
                      f"depth_helps={report['depth_helps']}",
                      flush=True,
                  )

        adapter_path = out_dir / "adapters.safetensors"
        mx.save_safetensors(
            str(adapter_path),
            {k: v for k, v in trainable_parameters(model).items()},
        )

    final = history[-1] if history else None
    receipt = {
        "schema": TRAIN_SCHEMA,
        "model": args.model,
        "spec": spec.to_receipt(),
        "wiring": wiring,
        "bridge": args.bridge,
        "steps": step,
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout),
        "history": history,
        "final": final,
        # Stated in the receipt so a reader cannot mistake stage 1 for the
        # claim this arc is actually testing.
        "verdict": {
            "collapse_repaired": bool(final and final["collapse_repaired"]),
            "depth_helps_heldout": bool(final and final["depth_helps"]),
            "claimable": (
                "depth improves held-out reasoning"
                if final and final["depth_helps"]
                else "model tolerates its own recurrence"
                if final and final["collapse_repaired"]
                else "collapse not repaired"
            ),
        },
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    (out_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[verdict] {receipt['verdict']}", flush=True)
    print(f"[receipt] {out_dir / 'training_receipt.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
