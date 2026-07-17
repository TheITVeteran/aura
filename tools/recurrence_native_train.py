#!/usr/bin/env python
"""Recurrence-native post-training — teach the window layers to think in loops.

The preregistered campaign proved the frozen-loop ceiling: on a checkpoint
never trained for recurrence, the RLC is parity-to-negative. This tool is
the answer: LoRA adapters on the RECURRENT WINDOW layers, trained under the
depth-curriculum objective (answer-span CE through the exact recurrent
forward the engine executes, plus the monotonicity hinge that punishes
"more thought made it worse"). Gradients flow through every recurrent
application, so descent shapes fixed-point behavior, not one-shot output.

Training data: the self-verifying task families (khop / boolean / modular)
at a TRAIN seed disjoint from the preregistered eval seed. Validation is
NOT this tool's claim to make — rerun the falsification harness
(tools/latent_cortex_lab.py --adapter <dir>) and let the graders speak.

Bounded, operator-launched, honest. Set AURA_LOG_DIR away from live logs:

  AURA_LOG_DIR=~/.aura/lab-logs caffeinate -dims \
      .venv/bin/python tools/recurrence_native_train.py \
      --model <mlx-model-dir> --max-minutes 90 \
      --out-dir data/latent_cortex/recurrence_native/<run-id>

MEMORY SAFETY: 32B runs require the live instance DOWN.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRAIN_SCHEMA = "aura.recurrence_native_train.v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="mlx model directory")
    parser.add_argument("--out-dir", required=True, help="adapter + receipt output dir")
    parser.add_argument("--train-seed", type=_positive_int, default=777)
    parser.add_argument("--families", default="khop,boolean,modular")
    parser.add_argument("--depths", default="2,4,8", help="task difficulty depths")
    parser.add_argument("--per-cell", type=_positive_int, default=64)
    parser.add_argument(
        "--curriculum-depths", default="1,2,4", help="recurrent-step ladder"
    )
    parser.add_argument("--lora-rank", type=_positive_int, default=8)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--monotonicity-weight", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--max-steps", type=_positive_int, default=100_000)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=100)
    parser.add_argument("--log-every", type=_positive_int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load adapter_latest.safetensors from --out-dir before training "
        "(step counter continues from the saved receipt)",
    )
    args = parser.parse_args()

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"recurrence-native-train:{Path(args.out_dir).name}",
        model_path=args.model,
        purpose="training",
        preemptible=False,
        metadata={"tool": "recurrence_native_train", "operator_launched": True},
    ) as model_lane_lease:
        return _run(args, model_lane_lease=model_lane_lease)


def _wrap_window_layers(model, *, rank: int, targets: tuple[str, ...]) -> int:
    """Freeze everything, then LoRA-wrap the recurrent window's projections."""
    from mlx_lm.tuner.lora import LoRALinear

    inner = model.model
    n_layers = len(inner.layers)
    prelude_end = max(1, int(n_layers * 0.25))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * 0.25))
    model.freeze()
    wrapped = 0
    for layer in inner.layers[prelude_end:coda_start]:
        for target in targets:
            parent = layer.self_attn if target.endswith("proj") and hasattr(
                layer.self_attn, target
            ) else layer.mlp if hasattr(layer.mlp, target) else None
            if parent is None:
                continue
            base = getattr(parent, target)
            setattr(parent, target, LoRALinear.from_base(base, r=rank))
            wrapped += 1
    return wrapped


def _render_example(tokenizer, task) -> tuple[list[int], int]:
    """Chat-templated prompt + answer tokens; returns (tokens, answer_start)."""
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    try:
        answer_tokens = tokenizer.encode(str(task.answer), add_special_tokens=False)
    except TypeError:
        answer_tokens = tokenizer.encode(str(task.answer))
    eos = getattr(tokenizer, "eos_token_id", None)
    tail = list(answer_tokens) + ([int(eos)] if eos is not None else [])
    return list(prompt_tokens) + tail, len(prompt_tokens)


def _run(args: argparse.Namespace, *, model_lane_lease: object) -> int:
    if getattr(model_lane_lease, "active", False) is not True:
        raise RuntimeError(
            "recurrence-native model load requires an active standalone "
            "model-lane lease"
        )
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from core.brain.llm.latent_cortex.experiments import task_battery
    from core.brain.llm.latent_cortex.governance import checkpoint_file_fingerprint
    from core.learning.recurrence_native_objective import (
        RECURRENCE_NATIVE_SCHEMA,
        depth_curriculum_loss,
    )

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    task_depths = [int(v) for v in args.depths.split(",") if v.strip()]
    ladder = tuple(int(v) for v in args.curriculum_depths.split(",") if v.strip())
    targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.max_minutes * 60.0
    print(f"loading {args.model} …", flush=True)
    model, tokenizer = load(args.model)
    fingerprint = checkpoint_file_fingerprint(args.model)

    wrapped = _wrap_window_layers(model, rank=args.lora_rank, targets=targets)
    resumed_from_step = 0
    if args.resume:
        latest = out_dir / "adapter_latest.safetensors"
        prior_receipt_path = out_dir / "receipt.json"
        if latest.is_file():
            model.load_weights(list(mx.load(str(latest)).items()), strict=False)
            if prior_receipt_path.is_file():
                try:
                    resumed_from_step = int(
                        json.loads(prior_receipt_path.read_text()).get("steps") or 0
                    )
                except (ValueError, OSError):
                    resumed_from_step = 0
            print(
                f"resumed adapter from {latest.name} (prior steps: {resumed_from_step})",
                flush=True,
            )
        else:
            print("--resume set but no adapter_latest.safetensors; starting fresh", flush=True)
    trainable = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    print(
        f"LoRA: {wrapped} projections wrapped (rank {args.lora_rank}, "
        f"targets {targets}) — {trainable:,} trainable params",
        flush=True,
    )
    if wrapped == 0:
        print("nothing wrapped — aborting")
        return 1

    battery = task_battery(families, task_depths, args.per_cell, seed=args.train_seed)
    examples = [_render_example(tokenizer, task) for task in battery]
    print(
        f"train battery: {len(examples)} tasks "
        f"(families {families}, task depths {task_depths}, seed {args.train_seed})",
        flush=True,
    )

    def loss_fn(mdl, tokens: list[int], answer_start: int):
        return depth_curriculum_loss(
            mdl,
            tokens,
            answer_start,
            depths=ladder,
            monotonicity_weight=args.monotonicity_weight,
            alpha=args.alpha,
        )

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    receipt: dict = {
        "schema": TRAIN_SCHEMA,
        "objective_schema": RECURRENCE_NATIVE_SCHEMA,
        "model": args.model,
        "checkpoint": fingerprint,
        "train_seed": args.train_seed,
        "families": families,
        "task_depths": task_depths,
        "curriculum_depths": list(ladder),
        "lora": {
            "rank": args.lora_rank,
            "targets": list(targets),
            "wrapped_projections": wrapped,
            "trainable_params": int(trainable),
        },
        "learning_rate": args.learning_rate,
        "monotonicity_weight": args.monotonicity_weight,
        "alpha": args.alpha,
        "started_at": time.time(),
        "loss_trail": [],
        "steps": 0,
    }

    def save_adapter(tag: str) -> None:
        flat = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(out_dir / f"adapter_{tag}.safetensors"), flat)
        receipt["saved_at"] = time.time()
        (out_dir / "receipt.json").write_text(
            json.dumps(receipt, indent=1, sort_keys=True)
        )

    step = resumed_from_step
    receipt["resumed_from_step"] = resumed_from_step
    window_losses: list[float] = []
    order = list(range(len(examples)))
    epoch = 0
    try:
        while step < args.max_steps and time.monotonic() < deadline:
            if not order:
                epoch += 1
                order = list(range(len(examples)))
            index = order.pop(
                (step * 2654435761) % len(order)  # deterministic shuffle-free pick
            )
            tokens, answer_start = examples[index]
            loss, grads = value_and_grad(model, tokens, answer_start)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state)
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                print(f"step {step}: non-finite loss — halting", flush=True)
                receipt["halt_reason"] = "non_finite_loss"
                break
            window_losses.append(loss_value)
            step += 1
            receipt["steps"] = step
            if step % args.log_every == 0:
                mean_loss = sum(window_losses) / len(window_losses)
                receipt["loss_trail"].append(
                    {"step": step, "mean_loss": round(mean_loss, 5)}
                )
                remaining_min = max(0.0, (deadline - time.monotonic()) / 60.0)
                print(
                    f"step {step} epoch {epoch} mean_loss {mean_loss:.4f} "
                    f"({remaining_min:.1f} min left)",
                    flush=True,
                )
                window_losses.clear()
            if step % args.checkpoint_every == 0:
                save_adapter("latest")
    except KeyboardInterrupt:
        receipt["halt_reason"] = "interrupted"

    receipt["finished_at"] = time.time()
    receipt.setdefault(
        "halt_reason",
        "max_steps" if step >= args.max_steps else "wall_clock",
    )
    save_adapter("latest")
    save_adapter("final")
    print(
        f"done: {step} steps, halt={receipt['halt_reason']}; "
        f"adapter + receipt → {out_dir}",
        flush=True,
    )
    print(
        "validate with: tools/latent_cortex_lab.py --adapter "
        f"{out_dir} --experiments 1,A --task-seed <fresh preregistered seed>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
