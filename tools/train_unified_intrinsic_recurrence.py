#!/usr/bin/env python3
"""Train the unified intrinsic recurrent controller on a bounded checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from core.learning.intrinsic_recurrence import checkpointed_window  # noqa: E402
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    unified_answer_trajectory,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402

TRAINING_SCHEMA = "aura.unified_intrinsic_training.v1"


class UnifiedTrainingBundle(nn.Module):
    def __init__(self, model: Any, controller: UnifiedRecurrentController) -> None:
        super().__init__()
        self.model = model
        self.controller = controller


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    scratch.write_text(encoded, encoding="utf-8")
    with scratch.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(scratch, path)


def _attach_window_adapters(
    model: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    rank: int,
    targets: tuple[str, ...],
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    sites = []
    for layer_index in range(spec.prelude_end, spec.coda_start):
        layer = model.model.layers[layer_index]
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in targets:
                projection = getattr(parent, target, None)
                if projection is None or isinstance(projection, ScopedLoRALinear):
                    continue
                site = f"model.layers.{layer_index}.{parent_name}.{target}"
                setattr(
                    parent,
                    target,
                    ScopedLoRALinear.from_base(
                        projection,
                        r=rank,
                        block_index=layer_index,
                        site=site,
                    ),
                )
                sites.append(site)
    if not sites:
        raise RuntimeError("unified recurrence attached no window projections")
    return {
        "window": [spec.prelude_end, spec.coda_start],
        "adapted_sites": sorted(sites),
        "adapted_projection_count": len(sites),
        "coda_adapted": False,
        "readout_adapted": False,
    }


def _trainable(bundle: UnifiedTrainingBundle) -> dict[str, Any]:
    return dict(tree_flatten(bundle.trainable_parameters()))


def _residual_hidden_size(model: Any) -> int:
    """Infer model width from an unquantized residual-space parameter.

    Quantized embedding weights expose their packed storage width, not the
    transformer hidden width. RMSNorm always has one scalar per residual
    channel and therefore remains representation independent.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    weight = (
        getattr(getattr(layers[0], "input_layernorm", None), "weight", None)
        if layers
        else None
    )
    if weight is None or len(weight.shape) != 1 or int(weight.shape[0]) < 1:
        raise ValueError("model residual hidden size is unavailable")
    return int(weight.shape[0])


def _save_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    *,
    step: int,
    history: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    tensors = {
        f"bundle.{name}": value for name, value in _trainable(bundle).items()
    }
    tensors.update(
        {
            f"optimizer.{name}": value
            for name, value in tree_flatten(optimizer.state)
        }
    )
    scratch = out_dir / f".checkpoint.{os.getpid()}.safetensors"
    mx.save_safetensors(str(scratch), tensors)
    os.replace(scratch, out_dir / "checkpoint_latest.safetensors")
    body = {
        "schema": TRAINING_SCHEMA,
        "step": step,
        "history": history,
        "identity": identity,
        "checkpoint_sha256": hashlib.sha256(
            (out_dir / "checkpoint_latest.safetensors").read_bytes()
        ).hexdigest(),
    }
    _atomic_json(
        out_dir / "checkpoint_latest.json",
        {**body, "receipt_sha256": _canonical_sha256(body)},
    )


def _restore_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    identity: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    receipt_path = out_dir / "checkpoint_latest.json"
    weights_path = out_dir / "checkpoint_latest.safetensors"
    if not receipt_path.is_file() or not weights_path.is_file():
        return 0, []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    stored_identity = receipt.get("identity")
    if not isinstance(stored_identity, dict):
        stored_identity = {}
    stored_identity_body = {
        key: value
        for key, value in stored_identity.items()
        if key != "identity_sha256"
    }
    if (
        receipt.get("receipt_sha256") != _canonical_sha256(body)
        or _canonical_sha256(stored_identity) != _canonical_sha256(identity)
        or stored_identity.get("identity_sha256")
        != _canonical_sha256(stored_identity_body)
        or receipt.get("checkpoint_sha256")
        != hashlib.sha256(weights_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("unified recurrence checkpoint identity differs")
    tensors = mx.load(str(weights_path))
    bundle_values = {
        name.removeprefix("bundle."): value
        for name, value in tensors.items()
        if name.startswith("bundle.")
    }
    expected = set(_trainable(bundle))
    if set(bundle_values) != expected:
        raise RuntimeError("unified recurrence checkpoint tensor inventory differs")
    bundle.update(tree_unflatten(list(bundle_values.items())))
    optimizer_values = [
        (name.removeprefix("optimizer."), value)
        for name, value in tensors.items()
        if name.startswith("optimizer.")
    ]
    if optimizer_values:
        optimizer.state = tree_unflatten(optimizer_values)
    mx.eval(bundle.parameters(), optimizer.state)
    return int(receipt["step"]), list(receipt.get("history", []))


def _evaluate(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    depths: tuple[int, ...],
    *,
    envelope: Any,
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    totals = {depth: 0.0 for depth in depths}
    with recurrence_adapter_scope(start=None, stop=None):
        for task in tasks:
            prompt, answer = encode_example(tokenizer, task, bridge)
            for depth in depths:
                _states, losses = unified_answer_trajectory(
                    bundle.model,
                    prompt,
                    answer,
                    spec.plan_at(depth),
                    bundle.controller,
                )
                totals[depth] += float(losses[-1].item())
            envelope.reclaim(force=True)
    count = len(tasks)
    ce = {f"T{depth}": totals[depth] / count for depth in depths}
    anchor = ce["T1"]
    deeper = [value for key, value in ce.items() if key != "T1"]
    return {
        "examples": count,
        "ce": ce,
        "best_depth": min(ce, key=ce.__getitem__),
        "best_deep_relative_gain": (
            (anchor - min(deeper)) / max(anchor, 1e-9) if deeper else 0.0
        ),
        "depth_helps": bool(deeper and min(deeper) < anchor),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prelude-end", type=int, default=7)
    parser.add_argument("--coda-start", type=int, default=21)
    parser.add_argument("--train-depths", default="1,2,4")
    parser.add_argument("--heldout-depths", default="8,16")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=int, default=8)
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--holdout-per-cell", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--controller-rank", type=int, default=16)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810198)
    parser.add_argument("--bridge", default="assistant_answer")
    parser.add_argument("--memory-fraction", type=float, default=0.48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from mlx_lm import load

    from core.learning import recurrence_curriculum as curriculum
    from core.runtime.model_lane_control import standalone_model_lane

    train_depths = tuple(int(value) for value in args.train_depths.split(","))
    heldout_depths = tuple(
        int(value) for value in args.heldout_depths.split(",")
    )
    spec = UnifiedIntrinsicTrainingSpec(
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        train_depths=train_depths,
        heldout_depths=heldout_depths,
    )
    families = tuple(
        value.strip() for value in args.families.split(",") if value.strip()
    )
    targets = tuple(
        value.strip() for value in args.lora_targets.split(",") if value.strip()
    )
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        args.bridge,
        args.bridge,
    )
    train_tasks = curriculum.task_battery(
        families,
        [args.task_depth],
        args.per_cell,
        seed=args.seed,
    )
    random.Random(args.seed).shuffle(train_tasks)
    holdout = curriculum.task_battery(
        families,
        [args.task_depth],
        args.holdout_per_cell,
        seed=args.seed + 9_973,
    )
    train_prompts = {task.prompt for task in train_tasks}
    holdout = [task for task in holdout if task.prompt not in train_prompts]
    if not holdout:
        raise RuntimeError("unified recurrence holdout is empty")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_minutes * 60.0
    with standalone_model_lane(
        owner_id=f"train-unified-intrinsic:{out_dir.name}",
        model_path=args.model,
        purpose="training",
        preemptible=False,
        metadata={
            "tool": "train_unified_intrinsic_recurrence",
            "operator_launched": True,
        },
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        model, tokenizer = load(args.model)
        model.freeze()
        wiring = _attach_window_adapters(
            model,
            spec,
            rank=args.lora_rank,
            targets=targets,
        )
        hidden_size = _residual_hidden_size(model)
        controller = UnifiedRecurrentController(
            UnifiedRecurrenceConfig(
                hidden_size=hidden_size,
                correction_rank=args.controller_rank,
                minimum_iterations=1,
            )
        )
        bundle = UnifiedTrainingBundle(model, controller)
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
        identity = {
            "schema": TRAINING_SCHEMA,
            "model": args.model,
            "spec": spec.to_dict(),
            "families": list(families),
            "task_depth": args.task_depth,
            "seed": args.seed,
            "wiring": wiring,
            "readout_sha256": readout_sha256,
        }
        identity["identity_sha256"] = _canonical_sha256(identity)
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        step, history = (
            _restore_checkpoint(out_dir, bundle, optimizer, identity)
            if args.resume
            else (0, [])
        )
        print(
            f"[unified] step={step} trainable={sum(v.size for v in _trainable(bundle).values()):,} "
            f"readout={readout_sha256[:12]}",
            flush=True,
        )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        def objective(candidate: UnifiedTrainingBundle, prompt: Any, answer: Any):
            return unified_intrinsic_training_loss(
                candidate.model,
                prompt,
                answer,
                candidate.controller,
                spec,
                readout_sha256=readout_sha256,
            )[0]

        loss_and_grad = nn.value_and_grad(bundle, objective)
        with checkpointed_window(model, group_size=args.checkpoint_group):
            while step < args.max_steps and time.time() < deadline:
                task = train_tasks[step % len(train_tasks)]
                prompt, answer = encode_example(tokenizer, task, bridge)
                with recurrence_adapter_scope(start=None, stop=None):
                    loss, gradients = loss_and_grad(bundle, prompt, answer)
                    optimizer.update(bundle, gradients)
                    mx.eval(bundle.parameters(), optimizer.state, loss)
                step += 1
                if step % 5 == 0:
                    print(
                        f"[step {step}] loss={float(loss.item()):.5f} "
                        f"elapsed_min={(time.time() - started) / 60.0:.1f}",
                        flush=True,
                    )
                if step % args.eval_every == 0 or step == args.max_steps:
                    report = _evaluate(
                        bundle,
                        tokenizer,
                        holdout,
                        spec,
                        bridge,
                        (1, *heldout_depths),
                        envelope=envelope,
                    )
                    report["step"] = step
                    history.append(report)
                    print(f"[eval {step}] {report}", flush=True)
                if step % args.checkpoint_every == 0 or step == args.max_steps:
                    _save_checkpoint(
                        out_dir,
                        bundle,
                        optimizer,
                        step=step,
                        history=history,
                        identity=identity,
                    )
                envelope.reclaim(force=True)

        final_readout = readout_fingerprint(model, spec.coda_start)
        if final_readout != readout_sha256:
            raise RuntimeError("unified training changed the frozen readout")
        final = history[-1] if history else None
        body = {
            "schema": TRAINING_SCHEMA,
            "identity": identity,
            "steps": step,
            "history": history,
            "final": final,
            "readout_sha256_before": readout_sha256,
            "readout_sha256_after": final_readout,
            "readout_frozen": True,
            "elapsed_minutes": round((time.time() - started) / 60.0, 3),
            "verdict": (
                "heldout_depth_gain"
                if final and final["depth_helps"]
                else "no_heldout_depth_gain"
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        _atomic_json(out_dir / "training_receipt.json", receipt)
        print(f"[verdict] {receipt['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
