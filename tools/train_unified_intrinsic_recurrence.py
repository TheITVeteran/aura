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
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from core.learning.intrinsic_recurrence import _run, checkpointed_window  # noqa: E402
from core.learning.recurrent_state_schema import (  # noqa: E402
    STATE_SLOT_NAMES,
    state_targets_from_trace,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    structured_state_loss,
    unified_answer_and_recurrent_trajectory,
    unified_answer_trajectory,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    unified_recurrent_logits,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402

TRAINING_SCHEMA = "aura.unified_intrinsic_training.v1"
TRAINING_SOURCE_FILES = (
    "core/learning/depth_conditioned_lora.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/unified_intrinsic_objective.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "tools/train_unified_intrinsic_recurrence.py",
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path: str) -> dict[str, Any]:
    directory = Path(model_path).expanduser().resolve(strict=True)
    config = directory / "config.json"
    weights = sorted(directory.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise ValueError("model checkpoint identity is incomplete")
    rows = []
    for path in weights:
        resolved = path.resolve(strict=True)
        rows.append(
            {
                "name": path.name,
                "size": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    body = {
        "canonical_path": str(directory),
        "config_sha256": _file_sha256(config.resolve(strict=True)),
        "weights": rows,
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


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
    depth_basis_size: int,
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
    from core.learning.depth_conditioned_lora import (
        wrap_continuous_depth_conditioned,
    )

    depth_operators = wrap_continuous_depth_conditioned(
        model,
        basis_size=depth_basis_size,
    )
    if set(depth_operators) != set(sites):
        raise RuntimeError("continuous depth operator inventory differs from adapters")
    return {
        "window": [spec.prelude_end, spec.coda_start],
        "adapted_sites": sorted(sites),
        "adapted_projection_count": len(sites),
        "continuous_depth_operator_count": len(depth_operators),
        "continuous_depth_basis_size": depth_basis_size,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": True,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_bridge": "continuous_depth_residual_preserves_t1",
    }


def _trainable(bundle: UnifiedTrainingBundle) -> dict[str, Any]:
    return dict(tree_flatten(bundle.trainable_parameters()))


def _ground_state_value_embeddings(
    model: Any,
    tokenizer: Any,
    controller: UnifiedRecurrentController,
    *,
    prelude_end: int,
) -> str:
    """Initialize typed values on the frozen model's native prelude manifold."""

    rows = []
    for slot_name in STATE_SLOT_NAMES:
        values = []
        for value in range(controller.config.state_cardinality):
            label = f"Internal state {slot_name}={value}"
            try:
                token_ids = tokenizer.encode(label, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(label)
            if not token_ids:
                raise RuntimeError("state codebook label encoded to no tokens")
            tokens = mx.array([token_ids], dtype=mx.int32)
            hidden = model.model.embed_tokens(tokens)
            hidden = _run(model.model.layers[:prelude_end], hidden)
            values.append(hidden[0, -1, :].astype(mx.float32))
        rows.append(mx.stack(values))
    grounded = mx.stack(rows)
    if grounded.shape != controller.state_value_embeddings.shape:
        raise RuntimeError("grounded state codebook shape differs from controller")
    controller.state_value_embeddings = grounded
    mx.eval(controller.state_value_embeddings)
    digest = hashlib.sha256(
        bytes(memoryview(controller.state_value_embeddings.astype(mx.float32)))
    ).hexdigest()
    return digest


def _optimization_phase(
    step: int,
    semantic_warmup_steps: int,
    state_warmup_steps: int = 0,
) -> str:
    if type(step) is not int or step < 0:
        raise ValueError("optimization step must be non-negative")
    if type(semantic_warmup_steps) is not int or semantic_warmup_steps < 0:
        raise ValueError("semantic warmup steps must be non-negative")
    if type(state_warmup_steps) is not int or state_warmup_steps < 0:
        raise ValueError("state warmup steps must be non-negative")
    if step < semantic_warmup_steps:
        return "semantic_anchor"
    if step < semantic_warmup_steps + state_warmup_steps:
        return "state_transition"
    return "recurrence"


def _phase_gradients(gradients: Any, phase: str) -> Any:
    """Keep the T1 semantic anchor fixed while training residual recurrence.

    Shared adapters learn the scoped T1 anchor.  Typed-state interpretation is
    owned by the continuous depth residuals, so a joint update cannot erase a
    useful shallow candidate or alter ordinary model inference.
    """

    if phase not in {"semantic_anchor", "state_transition", "recurrence"}:
        raise ValueError("unified optimization phase is invalid")
    masked = []
    for name, value in tree_flatten(gradients):
        shared_adapter = (
            name.startswith("model.")
            and "continuous_depth_" not in name
            and (name.endswith(".lora_a") or name.endswith(".lora_b"))
        )
        keep = shared_adapter if phase == "semantic_anchor" else not shared_adapter
        masked.append((name, value if keep else mx.zeros_like(value)))
    return tree_unflatten(masked)


def _clip_gradient_norm(gradients: Any, max_norm: float) -> tuple[Any, Any]:
    if (
        isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not 0.0 < float(max_norm)
    ):
        raise ValueError("maximum gradient norm must be positive")
    flattened = tree_flatten(gradients)
    if not flattened:
        raise ValueError("gradient tree must not be empty")
    norm = mx.sqrt(
        mx.sum(
            mx.stack(
                [
                    mx.sum(value.astype(mx.float32) ** 2)
                    for _name, value in flattened
                ]
            )
        )
    )
    scale = mx.minimum(1.0, float(max_norm) / mx.maximum(norm, 1e-12))
    return tree_unflatten(
        [(name, value * scale.astype(value.dtype)) for name, value in flattened]
    ), norm


def _gradient_ownership_group(name: str) -> str:
    if name.startswith("model."):
        return "scoped_transformer_bridge"
    if name.startswith(("controller.state_transition_", "controller.state_readout_")):
        return "typed_state_transition"
    if name.startswith(
        ("controller.state_value_embeddings", "controller.state_slot_embeddings")
    ):
        return "typed_state_codebook"
    return "recurrent_controller"


def _clip_gradient_groups(
    gradients: Any,
    max_norm: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Clip independent mechanisms without letting one starve the others."""

    if (
        isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not 0.0 < float(max_norm)
    ):
        raise ValueError("maximum gradient norm must be positive")
    flattened = tree_flatten(gradients)
    if not flattened:
        raise ValueError("gradient tree must not be empty")
    grouped: dict[str, list[Any]] = {}
    for name, value in flattened:
        grouped.setdefault(_gradient_ownership_group(name), []).append(value)
    group_norms = {
        group: mx.sqrt(
            mx.sum(
                mx.stack(
                    [mx.sum(value.astype(mx.float32) ** 2) for value in values]
                )
            )
        )
        for group, values in grouped.items()
    }
    scales = {
        group: mx.minimum(1.0, float(max_norm) / mx.maximum(norm, 1e-12))
        for group, norm in group_norms.items()
    }
    clipped = tree_unflatten(
        [
            (
                name,
                value * scales[_gradient_ownership_group(name)].astype(value.dtype),
            )
            for name, value in flattened
        ]
    )
    global_norm = mx.sqrt(
        mx.sum(mx.stack([norm.astype(mx.float32) ** 2 for norm in group_norms.values()]))
    )
    return clipped, global_norm, group_norms


def _student_rollin_probability(
    step: int,
    *,
    semantic_warmup_steps: int,
    max_steps: int,
    initial: float,
    final: float,
) -> float:
    if not semantic_warmup_steps <= step < max_steps:
        raise ValueError("student roll-in schedule step is outside recurrent phase")
    recurrent_steps = max_steps - semantic_warmup_steps
    progress = (
        (step - semantic_warmup_steps) / (recurrent_steps - 1)
        if recurrent_steps > 1
        else 1.0
    )
    return float(initial + progress * (final - initial))


def _sha256_tokens(tokens: Any) -> str:
    values = [int(value) for value in tokens.tolist()[0]]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _deterministic_student_mix(
    answer_tokens: Any,
    generated_tokens: Any,
    *,
    probability: float,
    seed: int,
) -> tuple[Any, tuple[int, ...]]:
    """Use generated history without allowing it to become a label."""

    if answer_tokens.shape != generated_tokens.shape:
        raise ValueError("generated roll-in must be answer-aligned")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("student roll-in probability must be inside [0, 1]")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("student roll-in seed must be inside [0, 2^64)")
    answer = [int(value) for value in answer_tokens.tolist()[0]]
    generated = [int(value) for value in generated_tokens.tolist()[0]]
    threshold = int(float(probability) * (1 << 64))
    effective: list[int] = []
    selected: list[int] = []
    for position, (target, produced) in enumerate(
        zip(answer, generated, strict=True)
    ):
        # The final decoder input has no successor label and cannot influence
        # this sequence loss, so leave it canonical.
        digest = hashlib.sha256(
            b"aura.unified.student-rollin.v1\0"
            + seed.to_bytes(8, "big")
            + position.to_bytes(8, "big")
        ).digest()
        use_generated = (
            position + 1 < len(answer)
            and int.from_bytes(digest[:8], "big") < threshold
        )
        effective.append(produced if use_generated else target)
        if use_generated:
            selected.append(position)
    return mx.array([effective], dtype=answer_tokens.dtype), tuple(selected)


def _generate_student_rollin(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer_tokens: Any,
    plan: Any,
    *,
    eos_token_id: int | None,
    state_slot_start: int | None = None,
) -> Any:
    """Greedily materialize a fixed-length deep-policy decoder history."""

    token_count = int(answer_tokens.shape[-1])
    if token_count < 1:
        raise ValueError("student roll-in target must not be empty")
    tokens = prompt
    generated: list[int] = []
    stopped = False
    for _position in range(token_count):
        if stopped and eos_token_id is not None:
            token = int(eos_token_id)
        else:
            logits, _telemetry = unified_recurrent_logits(
                bundle.model,
                tokens,
                plan,
                bundle.controller,
                state_slot_start=state_slot_start,
            )
            token = int(mx.argmax(logits[0, -1]).item())
            stopped = eos_token_id is not None and token == eos_token_id
        generated.append(token)
        tokens = mx.concatenate(
            [tokens, mx.array([[token]], dtype=tokens.dtype)],
            axis=1,
        )
    return mx.array([generated], dtype=answer_tokens.dtype)


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
    stem: str = "checkpoint_latest",
    optimization_phase: str = "recurrence",
) -> None:
    if not stem.startswith("checkpoint_") or not stem.replace("_", "").isalnum():
        raise ValueError("unified recurrence checkpoint stem is invalid")
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
    weights_path = out_dir / f"{stem}.safetensors"
    os.replace(scratch, weights_path)
    body = {
        "schema": TRAINING_SCHEMA,
        "step": step,
        "optimization_phase": optimization_phase,
        "history": history,
        "identity": identity,
        "checkpoint_sha256": hashlib.sha256(
            weights_path.read_bytes()
        ).hexdigest(),
    }
    _atomic_json(
        out_dir / f"{stem}.json",
        {**body, "receipt_sha256": _canonical_sha256(body)},
    )


def _restore_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    identity: dict[str, Any],
    *,
    semantic_warmup_steps: int = 0,
    state_warmup_steps: int = 0,
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
    expected_phase = _optimization_phase(
        int(receipt["step"]),
        semantic_warmup_steps,
        state_warmup_steps,
    )
    if receipt.get("optimization_phase") != expected_phase:
        raise RuntimeError("unified recurrence checkpoint phase differs")
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
    state_totals = {depth: 0.0 for depth in depths}
    state_counts = {depth: 0 for depth in depths}
    with recurrence_adapter_scope(start=None, stop=None):
        for task in tasks:
            prompt, answer = encode_example(tokenizer, task, bridge)
            for depth in depths:
                recurrent_states, _states, losses, state_logits = (
                    unified_answer_and_recurrent_trajectory(
                        bundle.model,
                        prompt,
                        answer,
                        spec.plan_at(depth),
                        bundle.controller,
                        use_state_slots=(
                            getattr(task, "transition_trace", None) is not None
                        ),
                    )
                )
                totals[depth] += float(losses[-1].item())
                trace = getattr(task, "transition_trace", None)
                if trace is not None:
                    targets = state_targets_from_trace(trace, depth)
                    _state_loss, state_accuracy, _step_accuracy = structured_state_loss(
                        bundle.controller,
                        recurrent_states,
                        targets,
                        public_token_count=int(prompt.shape[-1]),
                        state_slot_start=int(prompt.shape[-1]),
                        state_logits=state_logits,
                    )
                    state_totals[depth] += state_accuracy
                    state_counts[depth] += 1
            envelope.reclaim(force=True)
    count = len(tasks)
    ce = {f"T{depth}": totals[depth] / count for depth in depths}
    state_accuracy = {
        f"T{depth}": (
            state_totals[depth] / state_counts[depth]
            if state_counts[depth]
            else None
        )
        for depth in depths
    }
    anchor = ce["T1"]
    trained_deeper = [
        ce[f"T{depth}"] for depth in spec.train_depths if depth != 1
    ]
    heldout = [ce[f"T{depth}"] for depth in spec.heldout_depths]
    all_deeper = trained_deeper + heldout
    return {
        "examples": count,
        "ce": ce,
        "state_accuracy": state_accuracy,
        "best_depth": min(ce, key=ce.__getitem__),
        "best_deep_relative_gain": (
            (anchor - min(all_deeper)) / max(anchor, 1e-9)
            if all_deeper
            else 0.0
        ),
        "best_heldout_relative_gain": (
            (anchor - min(heldout)) / max(anchor, 1e-9) if heldout else 0.0
        ),
        "trained_depth_helps": bool(
            trained_deeper and min(trained_deeper) < anchor
        ),
        "heldout_depth_helps": bool(heldout and min(heldout) < anchor),
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
    parser.add_argument(
        "--task-depth",
        type=int,
        help="legacy single task depth; overrides --task-depths when supplied",
    )
    parser.add_argument("--task-depths", default="1,2,3,4")
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--holdout-per-cell", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--controller-rank", type=int, default=16)
    parser.add_argument("--state-weight", type=float, default=2.0)
    parser.add_argument("--stutter-weight", type=float, default=0.1)
    parser.add_argument("--depth-basis-size", type=int, default=4)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--recurrent-learning-rate",
        type=float,
        help="recurrent-phase rate; defaults to --learning-rate",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--semantic-warmup-steps", type=int, default=0)
    parser.add_argument("--state-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--state-learning-rate",
        type=float,
        help="state-transition phase rate; defaults to recurrent learning rate",
    )
    parser.add_argument(
        "--student-rollin-probability",
        type=float,
        default=0.0,
        help="initial fraction of recurrent-phase history taken from the deep policy",
    )
    parser.add_argument(
        "--student-rollin-final-probability",
        type=float,
        help="final generated-history fraction; defaults to the initial fraction",
    )
    parser.add_argument(
        "--state-teacher-forcing-probability",
        type=float,
        default=1.0,
        help="initial training-only exact-state roll-in probability",
    )
    parser.add_argument(
        "--state-teacher-forcing-final-probability",
        type=float,
        default=0.25,
        help="final exact-state roll-in probability; inference is always zero",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="global norm trust bound applied after phase masking",
    )
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810198)
    parser.add_argument("--init-seed", type=int, default=20260810198)
    parser.add_argument("--bridge", default="assistant_answer")
    parser.add_argument("--memory-fraction", type=float, default=0.48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (
        0 <= args.semantic_warmup_steps < args.max_steps
        and args.state_warmup_steps >= 0
        and args.semantic_warmup_steps + args.state_warmup_steps < args.max_steps
    ):
        raise ValueError("warmup phases must leave at least one recurrent step")
    rollin_final_probability = (
        args.student_rollin_probability
        if args.student_rollin_final_probability is None
        else args.student_rollin_final_probability
    )
    if not (
        0.0 <= args.student_rollin_probability <= 1.0
        and 0.0 <= rollin_final_probability <= 1.0
        and args.student_rollin_probability <= rollin_final_probability
    ):
        raise ValueError("student roll-in probability must be inside [0, 1]")
    if not (
        0.0 <= args.state_teacher_forcing_final_probability
        <= args.state_teacher_forcing_probability
        <= 1.0
    ):
        raise ValueError(
            "state teacher-forcing schedule must decrease inside [0, 1]"
        )
    if args.max_gradient_norm <= 0.0:
        raise ValueError("maximum gradient norm must be positive")
    recurrent_learning_rate = (
        args.learning_rate
        if args.recurrent_learning_rate is None
        else args.recurrent_learning_rate
    )
    state_learning_rate = (
        recurrent_learning_rate
        if args.state_learning_rate is None
        else args.state_learning_rate
    )
    if (
        args.learning_rate <= 0.0
        or recurrent_learning_rate <= 0.0
        or state_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if args.state_weight <= 0.0 or args.stutter_weight < 0.0:
        raise ValueError("state weight must be positive and stutter weight non-negative")

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
        state_weight=args.state_weight,
        stutter_weight=args.stutter_weight,
    )
    state_spec = replace(
        spec,
        answer_weight=0.0,
        anchor_weight=0.0,
        trajectory_weight=0.0,
        halt_weight=0.0,
        stutter_weight=0.0,
    )
    task_depths = (
        (args.task_depth,)
        if args.task_depth is not None
        else tuple(int(value) for value in args.task_depths.split(","))
    )
    if (
        not task_depths
        or any(depth < 1 for depth in task_depths)
        or max(task_depths) > max(spec.train_depths)
    ):
        raise ValueError("task depths must be positive and inside the trained recurrence horizon")
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
        task_depths,
        args.per_cell,
        seed=args.seed,
    )
    random.Random(args.seed).shuffle(train_tasks)
    holdout = curriculum.task_battery(
        families,
        task_depths,
        args.holdout_per_cell,
        seed=args.seed + 9_973,
    )
    train_prompts = {task.prompt for task in train_tasks}
    holdout = [task for task in holdout if task.prompt not in train_prompts]
    if not holdout:
        raise RuntimeError("unified recurrence holdout is empty")
    missing_traces = [
        task.task_id
        for task in train_tasks + holdout
        if task.transition_trace is None
    ]
    if missing_traces:
        raise RuntimeError(
            "state-supervised curriculum contains tasks without exact traces: "
            + ",".join(missing_traces[:5])
        )

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
        mx.random.seed(args.init_seed)
        model, tokenizer = load(args.model)
        model.freeze()
        wiring = _attach_window_adapters(
            model,
            spec,
            rank=args.lora_rank,
            targets=targets,
            depth_basis_size=args.depth_basis_size,
        )
        hidden_size = _residual_hidden_size(model)
        controller = UnifiedRecurrentController(
            UnifiedRecurrenceConfig(
                hidden_size=hidden_size,
                correction_rank=args.controller_rank,
                minimum_iterations=1,
                initialization_seed=args.init_seed,
            )
        )
        state_codebook_sha256 = _ground_state_value_embeddings(
            model,
            tokenizer,
            controller,
            prelude_end=spec.prelude_end,
        )
        bundle = UnifiedTrainingBundle(model, controller)
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
        identity = {
            "schema": TRAINING_SCHEMA,
            "model": _model_identity(args.model),
            "spec": spec.to_dict(),
            "families": list(families),
            "task_depths": list(task_depths),
            "per_cell": args.per_cell,
            "holdout_per_cell": args.holdout_per_cell,
            "seed": args.seed,
            "init_seed": args.init_seed,
            "semantic_warmup_steps": args.semantic_warmup_steps,
            "state_warmup_steps": args.state_warmup_steps,
            "student_rollin_probability": args.student_rollin_probability,
            "student_rollin_final_probability": rollin_final_probability,
            "state_teacher_forcing_probability": (
                args.state_teacher_forcing_probability
            ),
            "state_teacher_forcing_final_probability": (
                args.state_teacher_forcing_final_probability
            ),
            "max_gradient_norm": args.max_gradient_norm,
            "semantic_learning_rate": args.learning_rate,
            "recurrent_learning_rate": recurrent_learning_rate,
            "state_learning_rate": state_learning_rate,
            "bridge": args.bridge,
            "lora_rank": args.lora_rank,
            "controller_rank": args.controller_rank,
            "state_weight": args.state_weight,
            "stutter_weight": args.stutter_weight,
            "state_codebook": "frozen_prelude_semantic_labels",
            "state_codebook_sha256": state_codebook_sha256,
            "depth_basis_size": args.depth_basis_size,
            "lora_targets": list(targets),
            "wiring": wiring,
            "readout_sha256": readout_sha256,
            "source_sha256s": {
                relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
                for relative in TRAINING_SOURCE_FILES
            },
        }
        identity["identity_sha256"] = _canonical_sha256(identity)
        def phase_learning_rate(phase: str) -> float:
            return {
                "semantic_anchor": args.learning_rate,
                "state_transition": state_learning_rate,
                "recurrence": recurrent_learning_rate,
            }[phase]

        optimizer = optim.Adam(
            learning_rate=phase_learning_rate(
                _optimization_phase(
                    0,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                )
            )
        )
        step, history = (
            _restore_checkpoint(
                out_dir,
                bundle,
                optimizer,
                identity,
                semantic_warmup_steps=args.semantic_warmup_steps,
                state_warmup_steps=args.state_warmup_steps,
            )
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

        def semantic_objective(
            candidate: UnifiedTrainingBundle,
            prompt: Any,
            answer: Any,
        ):
            _states, losses = unified_answer_trajectory(
                candidate.model,
                prompt,
                answer,
                spec.plan_at(1),
                candidate.controller,
            )
            return losses[-1]

        semantic_loss_and_grad = nn.value_and_grad(bundle, semantic_objective)
        rollin_totals = {
            "examples": 0,
            "answer_tokens": 0,
            "generated_positions": 0,
            "generated_matches": 0,
            "last_generated_sha256": None,
            "last_effective_sha256": None,
            "max_preclip_gradient_norm": 0.0,
            "max_preclip_gradient_norms": {},
            "last_probability": None,
            "last_state_teacher_forcing_probability": None,
        }
        with checkpointed_window(model, group_size=args.checkpoint_group):
            while step < args.max_steps and time.time() < deadline:
                phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                )
                task = train_tasks[step % len(train_tasks)]
                prompt, answer = encode_example(tokenizer, task, bridge)
                with recurrence_adapter_scope(start=None, stop=None):
                    if phase == "semantic_anchor":
                        loss, gradients = semantic_loss_and_grad(
                            bundle,
                            prompt,
                            answer,
                        )
                    else:
                        if phase == "state_transition":
                            effective = answer
                            objective_spec = state_spec
                            state_teacher_probability = 1.0
                        else:
                            recurrent_start = (
                                args.semantic_warmup_steps + args.state_warmup_steps
                            )
                            rollin_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=recurrent_start,
                                max_steps=args.max_steps,
                                initial=args.student_rollin_probability,
                                final=rollin_final_probability,
                            )
                            state_teacher_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=recurrent_start,
                                max_steps=args.max_steps,
                                initial=args.state_teacher_forcing_probability,
                                final=args.state_teacher_forcing_final_probability,
                            )
                            generated = _generate_student_rollin(
                                bundle,
                                prompt,
                                answer,
                                spec.plan_at(max(spec.train_depths)),
                                eos_token_id=tokenizer.eos_token_id,
                                state_slot_start=int(prompt.shape[-1]),
                            )
                            effective, selected = _deterministic_student_mix(
                                answer,
                                generated,
                                probability=rollin_probability,
                                seed=args.seed * 1_000_003 + step,
                            )
                            answer_values = [
                                int(value) for value in answer.tolist()[0]
                            ]
                            generated_values = [
                                int(value) for value in generated.tolist()[0]
                            ]
                            rollin_totals["examples"] += 1
                            rollin_totals["answer_tokens"] += len(answer_values)
                            rollin_totals["generated_positions"] += len(selected)
                            rollin_totals["generated_matches"] += sum(
                                generated_values[index] == answer_values[index]
                                for index in selected
                            )
                            rollin_totals["last_generated_sha256"] = (
                                _sha256_tokens(generated)
                            )
                            rollin_totals["last_effective_sha256"] = (
                                _sha256_tokens(effective)
                            )
                            rollin_totals["last_probability"] = rollin_probability
                            objective_spec = spec
                        rollin_totals[
                            "last_state_teacher_forcing_probability"
                        ] = state_teacher_probability
                        def recurrent_objective(
                            candidate: UnifiedTrainingBundle,
                            objective_prompt: Any,
                            objective_answer: Any,
                            objective_rollin: Any,
                            transition_trace: Any = task.transition_trace,
                            state_teacher_forcing_probability: float = (
                                state_teacher_probability
                            ),
                            training_spec: UnifiedIntrinsicTrainingSpec = (
                                objective_spec
                            ),
                        ):
                            return unified_intrinsic_training_loss(
                                candidate.model,
                                objective_prompt,
                                objective_answer,
                                candidate.controller,
                                training_spec,
                                readout_sha256=readout_sha256,
                                decoder_input_tokens=objective_rollin,
                                transition_trace=transition_trace,
                                state_teacher_forcing_probability=(
                                    state_teacher_forcing_probability
                                ),
                            )[0]

                        loss, gradients = nn.value_and_grad(
                            bundle,
                            recurrent_objective,
                        )(
                            bundle,
                            prompt,
                            answer,
                            effective,
                        )
                    gradients = _phase_gradients(gradients, phase)
                    gradients, gradient_norm, gradient_group_norms = (
                        _clip_gradient_groups(
                            gradients,
                            args.max_gradient_norm,
                        )
                    )
                    mx.eval(gradient_norm, *gradient_group_norms.values())
                    rollin_totals["max_preclip_gradient_norm"] = max(
                        float(rollin_totals["max_preclip_gradient_norm"]),
                        float(gradient_norm.item()),
                    )
                    prior_group_norms = rollin_totals["max_preclip_gradient_norms"]
                    for group, group_norm in gradient_group_norms.items():
                        prior_group_norms[group] = max(
                            float(prior_group_norms.get(group, 0.0)),
                            float(group_norm.item()),
                        )
                    optimizer.update(bundle, gradients)
                    mx.eval(bundle.parameters(), optimizer.state, loss)
                step += 1
                next_phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                )
                if next_phase != phase:
                    optimizer = optim.Adam(
                        learning_rate=phase_learning_rate(next_phase)
                    )
                if step % 5 == 0:
                    print(
                        f"[step {step}] phase={phase} "
                        f"loss={float(loss.item()):.5f} "
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
                        spec.depths,
                        envelope=envelope,
                    )
                    report["step"] = step
                    report["optimization_phase"] = next_phase
                    generated_positions = int(rollin_totals["generated_positions"])
                    report["student_rollin"] = {
                        **rollin_totals,
                        "initial_probability": args.student_rollin_probability,
                        "final_probability": rollin_final_probability,
                        "generated_match_rate": (
                            int(rollin_totals["generated_matches"])
                            / generated_positions
                            if generated_positions
                            else None
                        ),
                        "labels_from_generated_tokens": False,
                    }
                    history.append(report)
                    print(f"[eval {step}] {report}", flush=True)
                    prior = history[:-1]
                    if report["heldout_depth_helps"] and (
                        not prior or report["best_heldout_relative_gain"] > max(
                            row.get("best_heldout_relative_gain", float("-inf"))
                            for row in prior
                        )
                    ):
                        _save_checkpoint(
                            out_dir,
                            bundle,
                            optimizer,
                            step=step,
                            history=history,
                            identity=identity,
                            stem="checkpoint_best_heldout",
                            optimization_phase=next_phase,
                        )
                    if report["trained_depth_helps"] and (
                        not prior or report["best_deep_relative_gain"] > max(
                            row.get("best_deep_relative_gain", float("-inf"))
                            for row in prior
                        )
                    ):
                        _save_checkpoint(
                            out_dir,
                            bundle,
                            optimizer,
                            step=step,
                            history=history,
                            identity=identity,
                            stem="checkpoint_best_trained",
                            optimization_phase=next_phase,
                        )
                if step % args.checkpoint_every == 0 or step == args.max_steps:
                    _save_checkpoint(
                        out_dir,
                        bundle,
                        optimizer,
                        step=step,
                        history=history,
                        identity=identity,
                        optimization_phase=next_phase,
                    )
                envelope.reclaim(force=True)

        if (
            not history
            or set(history[-1].get("ce", {}))
            != {f"T{depth}" for depth in spec.depths}
            or "heldout_depth_helps" not in history[-1]
        ):
            final_ladder = _evaluate(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                spec.depths,
                envelope=envelope,
            )
            final_ladder["step"] = step
            final_ladder["optimization_phase"] = _optimization_phase(
                step,
                args.semantic_warmup_steps,
                args.state_warmup_steps,
            )
            final_ladder["full_depth_ladder"] = True
            history.append(final_ladder)
            _save_checkpoint(
                out_dir,
                bundle,
                optimizer,
                step=step,
                history=history,
                identity=identity,
                optimization_phase=_optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                ),
            )
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
                if final and final["heldout_depth_helps"]
                else "trained_depth_gain_only"
                if final and final["trained_depth_helps"]
                else "no_heldout_depth_gain"
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        _atomic_json(out_dir / "training_receipt.json", receipt)
        print(f"[verdict] {receipt['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
