#!/usr/bin/env python
"""Train a recurrence-only adapter on the live latent-slot execution graph.

This is the v2 replacement for ``recurrence_native_train.py``. It binds the
effective base model, optional personality adapter, exact synthetic task bytes,
execution spec, objective/trainer sources, optimizer state, and sample cursor.
Checkpoints are immutable generations; ``latest.json`` advances only after the
adapter, optimizer, and completion record are durable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRAIN_SCHEMA_V2 = "aura.recurrence_native_train.v2"
ADAPTER_MANIFEST_SCHEMA_V2 = "aura.recurrence_adapter_manifest.v2"
GRADIENT_EXECUTION_SCHEMA = "aura.recurrence_streamed_depth_gradient.v1"
TASK_GENERATOR_SOURCE = REPO_ROOT / "core/learning/recurrence_curriculum.py"
MAX_CURRICULUM_SOURCE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class FrozenCurriculum:
    source_bytes: bytes
    binding: dict[str, Any]
    families: tuple[str, ...]
    task_battery: Callable[..., list[Any]]


def _execute_frozen_curriculum(source_bytes: bytes, *, origin: Path) -> FrozenCurriculum:
    if not source_bytes or len(source_bytes) > MAX_CURRICULUM_SOURCE_BYTES:
        raise RuntimeError("curriculum source size is invalid")
    digest = hashlib.sha256(source_bytes).hexdigest()
    module_name = f"_aura_frozen_recurrence_curriculum_{digest}"
    module = types.ModuleType(module_name)
    module.__file__ = str(origin)
    module.__package__ = "core.learning"
    sys.modules[module_name] = module
    try:
        exec(compile(source_bytes, str(origin), "exec"), module.__dict__)
    except BaseException:  # noqa: BLE001 - module-registry cleanup on any exit; original re-raised
        sys.modules.pop(module_name, None)
        raise
    families = getattr(module, "RECURRENCE_TRAINING_FAMILIES", None)
    battery = getattr(module, "task_battery", None)
    generators = getattr(module, "TASK_GENERATORS", None)
    if (
        not isinstance(families, tuple)
        or not families
        or any(not isinstance(family, str) or not family for family in families)
        or len(set(families)) != len(families)
        or not isinstance(generators, Mapping)
        or tuple(generators) != families
        or not callable(battery)
    ):
        sys.modules.pop(module_name, None)
        raise RuntimeError("curriculum source contract is invalid")
    try:
        relative_path = str(origin.resolve(strict=True).relative_to(REPO_ROOT))
    except ValueError as exc:
        sys.modules.pop(module_name, None)
        raise RuntimeError("curriculum source is outside the repository") from exc
    return FrozenCurriculum(
        source_bytes=source_bytes,
        binding={
            "path": relative_path,
            "sha256": digest,
            "size_bytes": len(source_bytes),
        },
        families=families,
        task_battery=battery,
    )


def _capture_frozen_curriculum(path: Path = TASK_GENERATOR_SOURCE) -> FrozenCurriculum:
    resolved = path.resolve(strict=True)
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_CURRICULUM_SOURCE_BYTES:
            raise RuntimeError("curriculum source is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            source_bytes = handle.read(MAX_CURRICULUM_SOURCE_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(source_bytes) != before.st_size
        or len(source_bytes) > MAX_CURRICULUM_SOURCE_BYTES
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise RuntimeError("curriculum source changed while it was captured")
    return _execute_frozen_curriculum(source_bytes, origin=resolved)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return parsed


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must contain comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _await_resource_guard(
    marker_path: Path,
    *,
    trainer_sha256: str,
    startup_lethal_mb: float,
    steady_lethal_mb: float,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Block the first training graph until the external guard is steady."""

    from core.runtime.resource_stage_guard import (
        ResourceStageGuardError,
        ack_path,
        publish_ready_marker,
        read_armed_ack,
        sha256_bytes,
    )

    acknowledgement = ack_path(marker_path)
    if acknowledgement.exists():
        raise ResourceStageGuardError("resource guard acknowledgement exists before trainer marker")
    marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256=trainer_sha256,
    )
    print(f"resource guard marker published: {marker_path}", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if acknowledgement.exists():
            acknowledgement_payload, ack_raw = read_armed_ack(
                marker_path,
                marker_raw=marker_raw,
                expected_target_pid=os.getpid(),
                startup_lethal_mb=startup_lethal_mb,
                steady_lethal_mb=steady_lethal_mb,
            )
            print(
                f"resource guard steady-stage acknowledgement accepted: {acknowledgement}",
                flush=True,
            )
            return {
                "marker_sha256": sha256_bytes(marker_raw),
                "ack_sha256": sha256_bytes(ack_raw),
                "marker": marker,
                "ack": acknowledgement_payload,
                "marker_raw": marker_raw,
                "ack_raw": ack_raw,
            }
        time.sleep(0.25)
    raise ResourceStageGuardError(
        "external sentinel did not acknowledge the steady memory guard in time"
    )


@dataclass(slots=True)
class _ResourceComputeGuard:
    marker_path: Path
    marker_raw: bytes
    predecessor_ack_raw: bytes
    target_pid: int
    compute_lethal_mb: float
    steady_lethal_mb: float
    timeout_s: float = 120.0
    sequence: int = 0

    def _await_ack(
        self,
        request_path: Path,
        *,
        request_raw: bytes,
        sequence: int,
        workload: str,
        action: str,
        active_lethal_mb: float,
    ) -> bytes:
        from core.runtime.resource_stage_guard import (
            ResourceStageGuardError,
            lease_ack_path,
            read_compute_lease_ack,
        )

        acknowledgement = lease_ack_path(request_path)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if acknowledgement.exists():
                _payload, raw = read_compute_lease_ack(
                    request_path,
                    request_raw=request_raw,
                    expected_target_pid=self.target_pid,
                    sequence=sequence,
                    workload=workload,
                    action=action,
                    active_lethal_mb=active_lethal_mb,
                )
                return raw
            time.sleep(0.1)
        raise ResourceStageGuardError(
            f"external sentinel did not acknowledge compute lease {sequence} {action}"
        )

    def acquire(self, workload: str) -> tuple[int, str, bytes]:
        from core.runtime.resource_stage_guard import publish_compute_lease_request

        self.sequence += 1
        request_path, _request, request_raw = publish_compute_lease_request(
            self.marker_path,
            marker_raw=self.marker_raw,
            target_pid=self.target_pid,
            sequence=self.sequence,
            workload=workload,
            action="acquire",
            predecessor_ack_raw=self.predecessor_ack_raw,
        )
        acknowledgement_raw = self._await_ack(
            request_path,
            request_raw=request_raw,
            sequence=self.sequence,
            workload=workload,
            action="acquire",
            active_lethal_mb=self.compute_lethal_mb,
        )
        print(
            f"resource compute lease acquired: sequence={self.sequence} workload={workload}",
            flush=True,
        )
        return self.sequence, workload, acknowledgement_raw

    def release(self, lease: tuple[int, str, bytes]) -> None:
        from core.runtime.resource_stage_guard import publish_compute_lease_request

        sequence, workload, acquire_ack_raw = lease
        if sequence != self.sequence:
            raise RuntimeError("resource compute lease release order is invalid")
        request_path, _request, request_raw = publish_compute_lease_request(
            self.marker_path,
            marker_raw=self.marker_raw,
            target_pid=self.target_pid,
            sequence=sequence,
            workload=workload,
            action="release",
            predecessor_ack_raw=acquire_ack_raw,
        )
        self.predecessor_ack_raw = self._await_ack(
            request_path,
            request_raw=request_raw,
            sequence=sequence,
            workload=workload,
            action="release",
            active_lethal_mb=self.steady_lethal_mb,
        )
        print(
            f"resource compute lease released: sequence={sequence} workload={workload}",
            flush=True,
        )


def _source_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _deterministic_order(size: int, seed: int, epoch: int) -> list[int]:
    """Stateless epoch permutation; no hidden PRNG state exists to lose."""

    if size <= 0:
        raise ValueError("training set must not be empty")
    return sorted(
        range(size),
        key=lambda index: hashlib.sha256(f"{seed}:{epoch}:{index}".encode("ascii")).digest(),
    )


def _wrap_window_layers(
    model: Any,
    *,
    rank: int,
    targets: tuple[str, ...],
    prelude_frac: float,
    coda_frac: float,
) -> list[str]:
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    inner = model.model
    n_layers = len(inner.layers)
    prelude_end = max(1, int(n_layers * prelude_frac))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * coda_frac))
    model.freeze()
    wrapped: list[str] = []
    for layer_index in range(prelude_end, coda_start):
        layer = inner.layers[layer_index]
        for target in targets:
            parent = (
                layer.self_attn
                if hasattr(layer.self_attn, target)
                else layer.mlp
                if hasattr(layer.mlp, target)
                else None
            )
            if parent is None:
                continue
            base = getattr(parent, target)
            setattr(parent, target, ScopedLoRALinear.from_base(base, r=rank))
            wrapped.append(
                f"model.layers.{layer_index}."
                f"{'self_attn' if parent is layer.self_attn else 'mlp'}.{target}"
            )
    return wrapped


def _streamed_depth_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: Any,
    depths: tuple[int, ...],
    monotonicity_weight: float,
    depth_margin: float = 0.0,
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    bridge_tokens: Sequence[int] = (),
    activation_checkpointing: bool = False,
) -> tuple[float, Any, dict[str, Any]]:
    """Evaluate one depth graph at a time while preserving the exact gradient.

    v2 hinge: ``relu(deep - stop_gradient(shallow))`` — its derivative only
    changes the coefficient of the deeper loss, so no cross-depth activation
    graph is required. v3 (CP181) generalizes the hinge with a positive
    margin — the coefficient engages while ``deep > shallow - margin`` — and
    adds a differentiable branch-diversity penalty INSIDE each depth's graph
    (the hinge comparison stays on the pure answer CE so diversity pressure
    cannot fake a depth advantage). Defaults reproduce v2 bit-for-bit.
    Returns ``(objective_value, gradients, telemetry)`` where telemetry
    carries per-depth answer CE and post-exchange pairwise cosines.
    """

    if (
        len(depths) < 2
        or tuple(sorted(set(depths))) != depths
        or any(type(depth) is not int or depth < 1 for depth in depths)
    ):
        raise ValueError("depths must be a strictly increasing tuple")
    if (
        isinstance(monotonicity_weight, bool)
        or not isinstance(monotonicity_weight, (int, float))
        or not math.isfinite(float(monotonicity_weight))
        or not 0.0 <= float(monotonicity_weight) <= 10.0
    ):
        raise ValueError("monotonicity_weight must be inside [0, 10]")

    if (
        isinstance(depth_margin, bool)
        or not isinstance(depth_margin, (int, float))
        or not math.isfinite(float(depth_margin))
        or not 0.0 <= float(depth_margin) <= 2.0
    ):
        raise ValueError("depth_margin must be inside [0, 2]")
    if (
        isinstance(diversity_weight, bool)
        or not isinstance(diversity_weight, (int, float))
        or not math.isfinite(float(diversity_weight))
        or not 0.0 <= float(diversity_weight) <= 10.0
    ):
        raise ValueError("diversity_weight must be inside [0, 10]")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    from core.learning.recurrence_native_objective_v2 import (
        branch_mean_answer_loss,
        live_path_forward,
        live_path_loss,
    )

    accumulated: Any | None = None
    loss_values: list[float] = []
    base_values: list[float] = []
    pairwise_cosines: dict[str, list[float]] = {}
    base_coefficient = 1.0 / len(depths)
    for depth in depths:
        depth_spec = spec.with_depth(depth)
        captured: dict[str, Any] = {}

        def depth_loss(
            mdl: Any,
            prompt: Sequence[int],
            answer: Sequence[int],
            _depth_spec: Any = depth_spec,
            _captured: dict[str, Any] = captured,
        ) -> Any:
            if float(diversity_weight) > 0.0:
                from core.learning.recurrence_native_objective_v3 import (
                    branch_diversity_penalty,
                )

                forward = live_path_forward(
                    mdl,
                    prompt,
                    answer,
                    spec=_depth_spec,
                    bridge_tokens=tuple(bridge_tokens),
                )
                base = branch_mean_answer_loss(forward, answer)
                penalty, cosines = branch_diversity_penalty(
                    forward, target_cos=float(diversity_target_cos)
                )
                _captured["base"] = base
                _captured["cosines"] = cosines
                return base + float(diversity_weight) * penalty
            return live_path_loss(
                mdl,
                prompt,
                answer,
                spec=_depth_spec,
                bridge_tokens=tuple(bridge_tokens),
            )

        if activation_checkpointing:
            from core.learning.recurrence_native_objective_v2 import (
                exact_adjoint_live_path_value_and_grad,
            )

            loss_value, gradients, base_value, cosines = (
                exact_adjoint_live_path_value_and_grad(
                    model,
                    prompt_tokens,
                    answer_tokens,
                    spec=depth_spec,
                    bridge_tokens=tuple(bridge_tokens),
                    diversity_weight=float(diversity_weight),
                    diversity_target_cos=float(diversity_target_cos),
                )
            )
            value = mx.array(loss_value)
            captured["base"] = mx.array(base_value)
            captured["cosines"] = cosines
        else:
            value, gradients = nn.value_and_grad(model, depth_loss)(
                model,
                prompt_tokens,
                answer_tokens,
            )
        finite_flags = [
            mx.all(mx.isfinite(gradient)) for _path, gradient in tree_flatten(gradients)
        ]
        mx.eval(value, gradients, finite_flags)
        loss_value = float(value)
        if not math.isfinite(loss_value) or not all(bool(flag) for flag in finite_flags):
            raise FloatingPointError("non_finite_streamed_depth_gradient")
        # The hinge compares pure answer CE across depths — diversity
        # pressure must never be able to fake (or hide) a depth advantage.
        base_value = float(captured["base"]) if "base" in captured else loss_value
        if not math.isfinite(base_value):
            raise FloatingPointError("non_finite_streamed_depth_gradient")
        pairwise_cosines[str(depth)] = [
            round(float(value_), 6) for value_ in captured.get("cosines", [])
        ]
        coefficient = base_coefficient
        if base_values and base_value > base_values[-1] - float(depth_margin):
            coefficient += float(monotonicity_weight)
        scaled = tree_map(
            lambda gradient, factor=coefficient: factor * gradient,
            gradients,
        )
        accumulated = (
            scaled
            if accumulated is None
            else tree_map(
                lambda previous, current: previous + current,
                accumulated,
                scaled,
            )
        )
        mx.eval(accumulated)
        loss_values.append(loss_value)
        base_values.append(base_value)
        del gradients, scaled, finite_flags
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("streamed gradient accumulator is empty")
    objective_value = sum(loss_values) / len(loss_values) + float(monotonicity_weight) * sum(
        max(deep - shallow + float(depth_margin), 0.0)
        for shallow, deep in zip(base_values, base_values[1:], strict=False)
    )
    telemetry = {
        "depth_base_losses": [round(value_, 6) for value_ in base_values],
        "pairwise_cos": pairwise_cosines,
    }
    return objective_value, accumulated, telemetry


def _render_example(tokenizer: Any, task: Any) -> dict[str, Any]:
    prompt_tokens = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    try:
        answer_tokens = list(tokenizer.encode(str(task.answer), add_special_tokens=False))
    except TypeError:
        answer_tokens = list(tokenizer.encode(str(task.answer)))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        answer_tokens.append(int(eos))
    if not prompt_tokens or not answer_tokens:
        raise RuntimeError("tokenizer produced an empty training example")
    return {
        "family": task.family,
        "depth": int(task.depth),
        "seed": int(task.seed),
        "prompt": str(task.prompt),
        "answer": str(task.answer),
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
    }


def _build_parser() -> argparse.ArgumentParser:
    curriculum = _capture_frozen_curriculum()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--adapter-id", default="resident-recurrence-v2")
    parser.add_argument(
        "--personality-adapter",
        default="auto",
        help="auto, none, or an explicit MLX adapter directory",
    )
    parser.add_argument("--train-seed", type=_positive_int, default=1777)
    parser.add_argument(
        "--families",
        type=_csv_strings,
        default=curriculum.families,
    )
    parser.add_argument("--task-depths", type=_csv_ints, default=(2, 4, 8))
    parser.add_argument("--per-cell", type=_positive_int, default=64)
    parser.add_argument("--curriculum-depths", type=_csv_ints, default=(1, 2, 4))
    parser.add_argument("--n-slots", type=_positive_int, default=16)
    parser.add_argument(
        "--branch-roles",
        type=_csv_strings,
        default=("constructive_solution", "counterexample_search"),
    )
    parser.add_argument("--exchange-interval", type=_positive_int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lora-rank", type=_positive_int, default=8)
    parser.add_argument("--lora-targets", type=_csv_strings, default=("o_proj", "v_proj"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--monotonicity-weight", type=float, default=0.5)
    # CP181 v3 objective surface. Defaults keep v2 behavior bit-for-bit;
    # --objective v3 engages the margin hinge, branch-diversity pressure,
    # live-bridge parity, and held-out validation.
    parser.add_argument("--objective", choices=("v2", "v3"), default="v2")
    parser.add_argument("--depth-margin", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--diversity-target-cos", type=float, default=0.98)
    parser.add_argument(
        "--bridge-policy",
        choices=("none", "assistant_answer"),
        default="none",
        help=(
            "assistant_answer trains through the SAME decode bridge tokens "
            "the live engine prepends (assistant_answer_v3), closing the "
            "training/live bridge-parity gap CP179 diagnosed"
        ),
    )
    parser.add_argument("--holdout-per-cell", type=int, default=0)
    parser.add_argument("--holdout-eval-samples", type=_positive_int, default=8)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--max-steps", type=_positive_int, default=100_000)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=25)
    parser.add_argument("--log-every", type=_positive_int, default=5)
    parser.add_argument("--resource-stage-path", type=Path)
    parser.add_argument("--resource-startup-lethal-mb", type=float)
    parser.add_argument("--resource-steady-lethal-mb", type=float)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="rematerialize each depth graph during backward to bound resident memory",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-migration-evidence",
        type=Path,
        help=(
            "one-time certified checkpoint migration; requires --resume and "
            "--activation-checkpointing"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"recurrence-native-v2:{Path(args.out_dir).name}",
        model_path=args.model,
        purpose="training",
        preemptible=False,
        metadata={"tool": "recurrence_native_train_v2", "operator_launched": True},
    ) as lease:
        return _run(args, model_lane_lease=lease)


def _resolve_personality_adapter(requested: str, model_path: str) -> str | None:
    value = str(requested or "auto").strip()
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        from core.brain.llm.model_registry import resolve_personality_adapter

        resolved = resolve_personality_adapter(model_path, backend="mlx")
        return str(resolved) if resolved else None
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("personality adapter must be a directory")
    return str(path)


def _checkpoint_payload(
    *,
    step: int,
    epoch: int,
    cursor: int,
    order: list[int],
    config_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    elapsed_training_s: float,
    invocation_count: int,
    loss_trail: list[dict[str, Any]],
    pending_window_losses: list[float],
    pending_window_cosines: list[float],
    holdout_trail: list[dict[str, Any]],
    holdout_eval_count: int,
) -> dict[str, Any]:
    return {
        "step": step,
        "epoch": epoch,
        "cursor": cursor,
        "order": list(order),
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "elapsed_training_s": round(elapsed_training_s, 6),
        "invocation_count": invocation_count,
        "loss_trail": list(loss_trail),
        "pending_window_losses": list(pending_window_losses),
        "pending_window_cosines": list(pending_window_cosines),
        "holdout_trail": list(holdout_trail),
        "holdout_eval_count": holdout_eval_count,
        "sampler": "sha256_stateless_epoch_permutation.v1",
        "stochastic_state": "none_all_keys_explicit",
    }


def _project_terminal_loss_trail(
    loss_trail: list[dict[str, Any]],
    *,
    step: int,
    pending_window_losses: list[float],
    pending_window_cosines: list[float],
) -> list[dict[str, Any]]:
    """Return receipt telemetry without mutating resumable window state."""

    projected = [dict(entry) for entry in loss_trail]
    if not pending_window_losses:
        return projected
    terminal: dict[str, Any] = {
        "step": step,
        "mean_loss": round(sum(pending_window_losses) / len(pending_window_losses), 6),
        "window_steps": len(pending_window_losses),
        "partial_window": True,
    }
    if pending_window_cosines:
        terminal["pairwise_cos_mean"] = round(
            sum(pending_window_cosines) / len(pending_window_cosines), 6
        )
    projected.append(terminal)
    return projected


def _run(args: argparse.Namespace, *, model_lane_lease: object) -> int:
    if getattr(model_lane_lease, "active", False) is not True:
        raise RuntimeError("v2 training requires an active model-lane lease")
    if not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        raise ValueError("max-minutes must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    resource_guard_values = (
        args.resource_stage_path,
        args.resource_startup_lethal_mb,
        args.resource_steady_lethal_mb,
    )
    resource_guard_enabled = all(value is not None for value in resource_guard_values)
    if any(value is not None for value in resource_guard_values) != resource_guard_enabled:
        raise ValueError("resource guard arguments must be supplied together")
    if resource_guard_enabled and not (
        math.isfinite(args.resource_startup_lethal_mb)
        and math.isfinite(args.resource_steady_lethal_mb)
        and args.resource_startup_lethal_mb > args.resource_steady_lethal_mb > 0.0
    ):
        raise ValueError("resource guard ceilings are invalid")
    if not math.isfinite(args.monotonicity_weight) or not 0.0 <= args.monotonicity_weight <= 10.0:
        raise ValueError("monotonicity-weight must be inside [0, 10]")
    objective_is_v3 = args.objective == "v3"
    resume_migration_path = getattr(args, "resume_migration_evidence", None)
    if resume_migration_path is not None and (
        not args.resume or not args.activation_checkpointing or not objective_is_v3
    ):
        raise ValueError(
            "resume migration requires --resume, --activation-checkpointing, and --objective v3"
        )
    if not objective_is_v3 and (args.bridge_policy != "none" or args.holdout_per_cell):
        raise ValueError("bridge-policy and holdout options require --objective v3")
    if not 0 <= args.holdout_per_cell < args.per_cell:
        raise ValueError("holdout-per-cell must be inside [0, per-cell)")
    depth_margin = float(args.depth_margin) if objective_is_v3 else 0.0
    diversity_weight = float(args.diversity_weight) if objective_is_v3 else 0.0
    if objective_is_v3 and not 0.0 <= depth_margin <= 2.0:
        raise ValueError("depth-margin must be inside [0, 2]")
    if objective_is_v3 and not 0.0 <= diversity_weight <= 10.0:
        raise ValueError("diversity-weight must be inside [0, 10]")
    if objective_is_v3 and not 0.0 <= float(args.diversity_target_cos) <= 1.0:
        raise ValueError("diversity-target-cos must be inside [0, 1]")
    curriculum = _capture_frozen_curriculum()

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        SOURCE_ROLES,
        full_weight_checkpoint_identity,
        model_behavior_bundle_identity,
        personality_bundle_identity,
        runtime_environment_identity,
    )
    from core.learning.recurrence_native_objective_v2 import (
        RECURRENCE_NATIVE_SCHEMA_V2,
    )
    from core.learning.recurrence_native_objective_v3 import (
        RECURRENCE_NATIVE_SCHEMA_V3,
    )

    objective_schema = (
        RECURRENCE_NATIVE_SCHEMA_V3 if args.objective == "v3" else RECURRENCE_NATIVE_SCHEMA_V2
    )
    from core.learning.recurrence_training_state import (
        canonical_json_bytes,
        load_recurrence_checkpoint,
        save_recurrence_checkpoint,
        sha256_bytes,
    )
    from core.runtime.atomic_writer import (
        atomic_write_bytes,
        atomic_write_text,
        ensure_private_directory,
    )

    out_dir = ensure_private_directory(Path(args.out_dir).expanduser())
    resume_migration: dict[str, Any] | None = None
    migration_bytes: bytes | None = None
    if resume_migration_path is not None:
        from core.learning.recurrence_checkpoint_migration import verify_migration

        resolved_migration = resume_migration_path.expanduser().resolve(strict=True)
        if resolved_migration != (out_dir / "checkpoint_migration.json").resolve(strict=True):
            raise ValueError("resume migration must be stored in the destination output root")
        resume_migration = verify_migration(
            resolved_migration,
            expected_destination_root=out_dir,
            expected_trainer_sha256=_sha256_file(Path(__file__).resolve(strict=True)),
        )
        migration_bytes = resolved_migration.read_bytes()

    families = tuple(args.families)
    unknown_families = sorted(set(families) - set(curriculum.families))
    if unknown_families:
        raise ValueError(f"unknown task families: {unknown_families}")
    ladder = tuple(args.curriculum_depths)
    if tuple(sorted(set(ladder))) != ladder:
        raise ValueError("curriculum-depths must be strictly increasing")
    spec = RLCExecutionSpec(
        n_slots=args.n_slots,
        branch_roles=tuple(args.branch_roles),
        exchange_interval=args.exchange_interval,
        recurrent_steps=max(ladder),
        alpha=args.alpha,
        alpha_schedule=args.alpha_schedule,
        decode_bridge_policy=args.bridge_policy,
    )
    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")

    model_path = str(Path(args.model).expanduser().resolve(strict=True))
    personality_adapter = _resolve_personality_adapter(args.personality_adapter, model_path)
    load_kwargs = {"adapter_path": personality_adapter} if personality_adapter else {}
    print(f"loading {model_path} personality={personality_adapter or 'none'}", flush=True)
    model, tokenizer = load(model_path, **load_kwargs)
    # Bridge parity (CP181): train through the SAME decode-bridge tokens the
    # live engine prepends before answers, so the trained operator meets the
    # exact conditioning it will run under.
    bridge_tokens: tuple[int, ...] = ()
    if args.bridge_policy == "assistant_answer":
        from core.brain.llm.latent_cortex.engine import (
            _ASSISTANT_ANSWER_BRIDGE_V3,
        )

        try:
            encoded_bridge = tokenizer.encode(_ASSISTANT_ANSWER_BRIDGE_V3, add_special_tokens=False)
        except TypeError:
            encoded_bridge = tokenizer.encode(_ASSISTANT_ANSWER_BRIDGE_V3)
        bridge_tokens = tuple(int(token) for token in encoded_bridge)
        if not bridge_tokens or any(token < 0 for token in bridge_tokens):
            raise RuntimeError("bridge policy produced invalid tokens")
    base_identity = full_weight_checkpoint_identity(model_path)
    model_behavior_identity = model_behavior_bundle_identity(model_path)
    personality_identity = personality_bundle_identity(personality_adapter)
    training_runtime_identity = runtime_environment_identity()
    mx.random.seed(args.train_seed)
    wrapped = _wrap_window_layers(
        model,
        rank=args.lora_rank,
        targets=tuple(args.lora_targets),
        prelude_frac=spec.prelude_frac,
        coda_frac=spec.coda_frac,
    )
    if not wrapped:
        raise RuntimeError("no recurrent projections were wrapped")
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not trainable or any(
        not (key.endswith(".lora_a") or key.endswith(".lora_b")) for key in trainable
    ):
        raise RuntimeError("trainable tree contains non-recurrence parameters")

    tasks = curriculum.task_battery(
        list(families), list(args.task_depths), args.per_cell, seed=args.train_seed
    )
    examples = [_render_example(tokenizer, task) for task in tasks]
    # Held-out validation split (CP181): the LAST holdout-per-cell samples
    # of every (family, depth) cell never receive a gradient; they are
    # evaluated at checkpoints so overfitting is visible in the receipt.
    holdout_indices: list[int] = []
    if args.holdout_per_cell:
        cell_members: dict[tuple[str, int], list[int]] = {}
        for index, example in enumerate(examples):
            cell_members.setdefault((str(example["family"]), int(example["depth"])), []).append(
                index
            )
        for members in cell_members.values():
            holdout_indices.extend(members[-args.holdout_per_cell :])
    holdout_set = frozenset(holdout_indices)
    holdout_examples = [examples[index] for index in sorted(holdout_set)]
    train_examples = [example for index, example in enumerate(examples) if index not in holdout_set]
    if not train_examples:
        raise RuntimeError("holdout split left no training examples")
    dataset_payload = {
        "schema": "aura.recurrence_native_dataset.v2",
        "generator": curriculum.binding,
        "train_seed": args.train_seed,
        "families": list(families),
        "task_depths": list(args.task_depths),
        "per_cell": args.per_cell,
        "examples": examples,
    }
    if objective_is_v3:
        dataset_payload["holdout_per_cell"] = args.holdout_per_cell
        dataset_payload["holdout_indices"] = sorted(holdout_set)
    dataset_bytes = canonical_json_bytes(dataset_payload)
    dataset_sha256 = sha256_bytes(dataset_bytes)
    if resume_migration is not None and (
        resume_migration["dataset_sha256"] != dataset_sha256
        or resume_migration["execution_spec_sha256"] != spec.sha256
    ):
        raise RuntimeError("migration dataset or execution spec differs from the source checkpoint")
    sources = {
        "trainer": _source_binding(Path(__file__)),
        # v3 binds the v3 objective source; its v2 live-path dependency stays
        # bound through the detached supervisor's git-tree execution manifest.
        "objective": _source_binding(
            REPO_ROOT
            / (
                "core/learning/recurrence_native_objective_v3.py"
                if objective_is_v3
                else "core/learning/recurrence_native_objective_v2.py"
            )
        ),
        "execution_spec": _source_binding(
            REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py"
        ),
        "recurrence_adapter": _source_binding(
            REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py"
        ),
        "workspace": _source_binding(REPO_ROOT / "core/brain/llm/latent_cortex/workspace.py"),
        "recurrence": _source_binding(REPO_ROOT / "core/brain/llm/latent_cortex/recurrence.py"),
        "branches": _source_binding(REPO_ROOT / "core/brain/llm/latent_cortex/branches.py"),
        "task_generator": curriculum.binding,
    }
    if set(sources) != set(SOURCE_ROLES):
        raise RuntimeError("training source inventory differs from v2 identity contract")
    source_snapshot_dir = ensure_private_directory(out_dir / "source_snapshots")
    source_artifacts: dict[str, dict[str, Any]] = {}
    for role, source in sorted(sources.items()):
        source_path = REPO_ROOT / source["path"]
        source_bytes = (
            curriculum.source_bytes if role == "task_generator" else source_path.read_bytes()
        )
        if (
            len(source_bytes) != source["size_bytes"]
            or hashlib.sha256(source_bytes).hexdigest() != source["sha256"]
        ):
            raise RuntimeError(f"training source changed while snapshotting: {role}")
        snapshot_name = f"{role}.py"
        atomic_write_bytes(source_snapshot_dir / snapshot_name, source_bytes)
        source_artifacts[role] = {
            "origin_path": source["path"],
            "snapshot_path": f"source_snapshots/{snapshot_name}",
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
        }
    config_payload = {
        "schema": "aura.recurrence_native_training_config.v2",
        "model_path": model_path,
        "base_checkpoint": base_identity,
        "model_behavior_bundle": model_behavior_identity,
        "personality_adapter_path": personality_adapter or "",
        "personality_adapter": personality_identity,
        "training_runtime": training_runtime_identity,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "dataset_sha256": dataset_sha256,
        "objective_schema": objective_schema,
        "curriculum_depths": list(ladder),
        "monotonicity_weight": args.monotonicity_weight,
        "lora": {
            "rank": args.lora_rank,
            "targets": list(args.lora_targets),
            "wrapped_projections": wrapped,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
        },
        "gradient_execution": {
            "schema": (
                "aura.recurrence_streamed_depth_gradient.v6"
                if args.activation_checkpointing
                else GRADIENT_EXECUTION_SCHEMA
            ),
            "mode": "depth_serial_exact_sum",
            "concurrent_depth_graphs": 1,
            "optimizer_updates_per_sample": 1,
            "finite_loss_and_gradient_required_before_update": True,
            **(
                {
                    "activation_rematerialization": "exact_discrete_adjoint",
                    "adjoint_schema": "aura.recurrence_exact_discrete_adjoint.v1",
                    "boundary_state_storage": "materialized_stop_gradient",
                    "terminal_branch_graphs_concurrent": 1,
                    "recurrent_transition_graphs_concurrent": 1,
                }
                if args.activation_checkpointing
                else {}
            ),
        },
        "train_seed": args.train_seed,
        "max_steps": args.max_steps,
        "sources": sources,
    }
    if objective_is_v3:
        config_payload["objective_options"] = {
            "depth_margin": depth_margin,
            "diversity_weight": diversity_weight,
            "diversity_target_cos": float(args.diversity_target_cos),
        }
        config_payload["bridge"] = {
            "policy": args.bridge_policy,
            "token_count": len(bridge_tokens),
            "tokens_sha256": sha256_bytes(canonical_json_bytes(list(bridge_tokens))),
        }
        config_payload["holdout"] = {
            "per_cell": args.holdout_per_cell,
            "count": len(holdout_examples),
            "eval_samples": args.holdout_eval_samples,
            "indices_sha256": sha256_bytes(canonical_json_bytes(sorted(holdout_set))),
        }
    if resume_migration is not None:
        config_payload["resume_migration"] = resume_migration
    config_bytes = canonical_json_bytes(config_payload)
    config_sha256 = sha256_bytes(config_bytes)
    execution_spec_bytes = canonical_json_bytes(spec.to_dict())
    atomic_write_bytes(out_dir / "dataset_manifest.json", dataset_bytes)
    atomic_write_bytes(out_dir / "execution_spec.json", execution_spec_bytes)
    atomic_write_bytes(out_dir / "training_config.json", config_bytes)

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    optimizer.init(model.trainable_parameters())
    step = 0
    epoch = 0
    cursor = 0
    order = _deterministic_order(len(train_examples), args.train_seed, epoch)
    prior_elapsed_s = 0.0
    invocation_count = 1
    loss_trail: list[dict[str, Any]] = []
    window_losses: list[float] = []
    window_cosines: list[float] = []
    holdout_trail: list[dict[str, Any]] = []
    holdout_eval_count = 0
    last_checkpoint_step = -1
    last_checkpoint_path: Path | None = None
    if args.resume:
        migration_source_load = False
        if resume_migration is not None:
            try:
                current_pointer = json.loads((out_dir / "latest.json").read_text(encoding="ascii"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("migration destination latest pointer is unreadable") from exc
            migration_source_load = (
                current_pointer.get("checkpoint") == resume_migration["source_checkpoint"]
            )
        loaded = load_recurrence_checkpoint(
            out_dir,
            expected_config_sha256=(
                resume_migration["source_config_sha256"]
                if migration_source_load
                else config_sha256
            ),
            expected_dataset_sha256=dataset_sha256,
            expected_execution_spec_sha256=spec.sha256,
        )
        if resume_migration is not None:
            if migration_source_load and (
                loaded.checkpoint_dir.name
                != Path(str(resume_migration["source_checkpoint"])).name
                or loaded.state.get("step") != resume_migration["source_step"]
            ):
                raise RuntimeError("loaded checkpoint differs from certified migration source")
            if not migration_source_load and loaded.state.get("step", 0) <= resume_migration[
                "source_step"
            ]:
                raise RuntimeError("post-migration checkpoint did not advance source state")
        model.load_weights(list(loaded.adapter_tensors.items()), strict=False)
        optimizer.state = loaded.optimizer_state
        optimizer.init(model.trainable_parameters())
        state = loaded.state
        step = int(state["step"])
        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        order = [int(value) for value in state["order"]]
        if order != _deterministic_order(len(train_examples), args.train_seed, epoch):
            raise RuntimeError("checkpoint sample order differs from deterministic epoch")
        if not 0 <= cursor <= len(order):
            raise RuntimeError("checkpoint sample cursor is invalid")
        prior_elapsed_s = float(state["elapsed_training_s"])
        invocation_count = int(state["invocation_count"]) + 1
        loss_trail = list(state["loss_trail"])
        window_losses = [float(value) for value in state["pending_window_losses"]]
        window_cosines = [float(value) for value in state["pending_window_cosines"]]
        holdout_trail = list(state["holdout_trail"])
        holdout_eval_count = int(state["holdout_eval_count"])
        last_checkpoint_step = step
        last_checkpoint_path = loaded.checkpoint_dir
        print(
            f"resumed exact checkpoint at step={step} epoch={epoch} cursor={cursor}",
            flush=True,
        )

    resource_compute_guard: _ResourceComputeGuard | None = None
    if resource_guard_enabled:
        handshake = _await_resource_guard(
            args.resource_stage_path.expanduser(),
            trainer_sha256=sources["trainer"]["sha256"],
            startup_lethal_mb=float(args.resource_startup_lethal_mb),
            steady_lethal_mb=float(args.resource_steady_lethal_mb),
        )
        resource_compute_guard = _ResourceComputeGuard(
            marker_path=args.resource_stage_path.expanduser(),
            marker_raw=handshake["marker_raw"],
            predecessor_ack_raw=handshake["ack_raw"],
            target_pid=os.getpid(),
            compute_lethal_mb=float(args.resource_startup_lethal_mb),
            steady_lethal_mb=float(args.resource_steady_lethal_mb),
        )

    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.max_minutes * 60.0

    def elapsed_training_s() -> float:
        return prior_elapsed_s + (time.monotonic() - started_monotonic)

    def checkpoint() -> Path:
        nonlocal last_checkpoint_path, last_checkpoint_step
        if (
            last_checkpoint_step == step
            and last_checkpoint_path is not None
            and last_checkpoint_path.is_dir()
        ):
            return last_checkpoint_path
        adapter = dict(tree_flatten(model.trainable_parameters()))
        optimizer_tensors = dict(tree_flatten(optimizer.state))
        last_checkpoint_path = save_recurrence_checkpoint(
            out_dir,
            adapter_tensors=adapter,
            optimizer_tensors=optimizer_tensors,
            state=_checkpoint_payload(
                step=step,
                epoch=epoch,
                cursor=cursor,
                order=order,
                config_sha256=config_sha256,
                dataset_sha256=dataset_sha256,
                execution_spec_sha256=spec.sha256,
                elapsed_training_s=elapsed_training_s(),
                invocation_count=invocation_count,
                loss_trail=loss_trail,
                pending_window_losses=window_losses,
                pending_window_cosines=window_cosines,
                holdout_trail=holdout_trail,
                holdout_eval_count=holdout_eval_count,
            ),
        )
        last_checkpoint_step = step
        return last_checkpoint_path

    def holdout_eval(*, durable: bool) -> dict[str, Any] | None:
        """Gradient-free held-out CE at max depth over a rotating window."""
        nonlocal holdout_eval_count
        if not holdout_examples:
            return None
        from core.learning.recurrence_native_objective_v2 import (
            live_path_loss,
        )

        count = min(int(args.holdout_eval_samples), len(holdout_examples))
        start = (holdout_eval_count * count) % len(holdout_examples)
        losses: list[float] = []
        compute_lease = (
            resource_compute_guard.acquire("holdout_eval")
            if resource_compute_guard is not None
            else None
        )
        try:
            for offset in range(count):
                example = holdout_examples[(start + offset) % len(holdout_examples)]
                value = live_path_loss(
                    model,
                    example["prompt_tokens"],
                    example["answer_tokens"],
                    spec=spec.with_depth(max(ladder)),
                    bridge_tokens=bridge_tokens,
                )
                try:
                    mx.eval(value)
                    loss = float(value)
                finally:
                    del value
                    mx.clear_cache()
                if not math.isfinite(loss):
                    raise FloatingPointError("non_finite_holdout_loss")
                losses.append(loss)
        except BaseException:
            mx.clear_cache()
            if compute_lease is not None:
                resource_compute_guard.release(compute_lease)
            raise
        if compute_lease is not None:
            resource_compute_guard.release(compute_lease)
        entry = {
            "step": step,
            "mean_loss": round(sum(losses) / len(losses), 6),
            "examples": count,
            "depth": max(ladder),
        }
        if durable:
            holdout_eval_count += 1
            holdout_trail.append(entry)
        print(
            f"holdout step={step} mean_loss={entry['mean_loss']:.5f} n={count}",
            flush=True,
        )
        return entry

    halt_reason = "wall_clock"
    interrupted = False
    try:
        while step < args.max_steps and time.monotonic() < deadline:
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                order = _deterministic_order(len(train_examples), args.train_seed, epoch)
            example = train_examples[order[cursor]]
            compute_lease = (
                resource_compute_guard.acquire("training_step")
                if resource_compute_guard is not None
                else None
            )
            try:
                loss_value, gradients, step_telemetry = _streamed_depth_value_and_grad(
                    model,
                    example["prompt_tokens"],
                    example["answer_tokens"],
                    spec=spec,
                    depths=ladder,
                    monotonicity_weight=args.monotonicity_weight,
                    depth_margin=depth_margin,
                    diversity_weight=diversity_weight,
                    diversity_target_cos=float(args.diversity_target_cos),
                    bridge_tokens=bridge_tokens,
                    activation_checkpointing=args.activation_checkpointing,
                )
            except FloatingPointError:
                mx.clear_cache()
                if compute_lease is not None:
                    resource_compute_guard.release(compute_lease)
                halt_reason = "non_finite_loss"
                break
            optimizer.update(model, gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            step_cosines = [
                cosine
                for cosines in step_telemetry.get("pairwise_cos", {}).values()
                for cosine in cosines
            ]
            # Do not retain the previous step's gradient graph while building
            # holdout/checkpoint work or the next RHS. On the resident 32B v3
            # path that overlap raised physical footprint above 66 GiB even
            # under the MLX wired limit.
            del gradients, step_telemetry
            mx.clear_cache()
            if compute_lease is not None:
                resource_compute_guard.release(compute_lease)
            step += 1
            cursor += 1
            window_losses.append(loss_value)
            if step_cosines:
                window_cosines.append(sum(step_cosines) / len(step_cosines))
            if step % args.log_every == 0:
                mean_loss = sum(window_losses) / len(window_losses)
                trail_entry: dict[str, Any] = {
                    "step": step,
                    "mean_loss": round(mean_loss, 6),
                }
                if window_cosines:
                    trail_entry["pairwise_cos_mean"] = round(
                        sum(window_cosines) / len(window_cosines), 6
                    )
                    window_cosines.clear()
                trail_entry.update(
                    {
                        "mlx_active_memory_bytes": int(mx.get_active_memory()),
                        "mlx_cache_memory_bytes": int(mx.get_cache_memory()),
                        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                    }
                )
                loss_trail.append(trail_entry)
                window_losses.clear()
                print(
                    f"step={step} epoch={epoch} cursor={cursor}/{len(order)} "
                    f"mean_loss={mean_loss:.5f}"
                    + (
                        f" cos={trail_entry['pairwise_cos_mean']:.4f}"
                        if "pairwise_cos_mean" in trail_entry
                        else ""
                    ),
                    flush=True,
                )
            if step % args.checkpoint_every == 0:
                holdout_eval(durable=True)
                published = checkpoint()
                print(f"checkpoint={published.name}", flush=True)
        if step >= args.max_steps:
            halt_reason = "max_steps"
    except KeyboardInterrupt:
        halt_reason = "interrupted"
        interrupted = True

    complete_run = step >= args.max_steps and halt_reason == "max_steps" and not interrupted
    if complete_run and window_losses:
        loss_trail = _project_terminal_loss_trail(
            loss_trail,
            step=step,
            pending_window_losses=window_losses,
            pending_window_cosines=window_cosines,
        )
        window_losses.clear()
        window_cosines.clear()
        last_checkpoint_step = -1
    receipt_holdout_trail = [dict(entry) for entry in holdout_trail]
    try:
        if not holdout_trail or holdout_trail[-1].get("step") != step:
            terminal_holdout = holdout_eval(durable=complete_run)
            if terminal_holdout is not None and not complete_run:
                receipt_holdout_trail.append(terminal_holdout)
    except FloatingPointError:
        # A non-finite held-out loss at shutdown must not cost the receipt;
        # the trail's absence for this step is itself honest evidence.
        print("holdout eval non-finite at shutdown; omitted", flush=True)
    receipt_loss_trail = _project_terminal_loss_trail(
        loss_trail,
        step=step,
        pending_window_losses=window_losses,
        pending_window_cosines=window_cosines,
    )
    final_checkpoint = checkpoint()
    adapter_bytes = (final_checkpoint / "adapter.safetensors").read_bytes()
    atomic_write_bytes(out_dir / "adapters.safetensors", adapter_bytes)
    atomic_write_bytes(out_dir / "adapter_final.safetensors", adapter_bytes)
    adapter_config = {
        "schema": "aura.recurrence_scoped_lora_config.v1",
        "fine_tune_type": "recurrence_scoped_lora",
        "loader": "aura_custom_loader_required",
        "model": model_path,
        "num_layers": len({int(path.split(".")[2]) for path in wrapped}),
        "wrapped_projection_count": len(wrapped),
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": list(args.lora_targets),
        },
        "execution_spec_sha256": spec.sha256,
    }
    adapter_config_text = json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    atomic_write_text(out_dir / "adapter_config.json", adapter_config_text)
    adapter_config_bytes = adapter_config_text.encode("utf-8")
    receipt = {
        "schema": TRAIN_SCHEMA_V2,
        "objective_schema": objective_schema,
        "objective_source_sha256": sources["objective"]["sha256"],
        "trainer_source_sha256": sources["trainer"]["sha256"],
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": spec.sha256,
        "base_checkpoint": base_identity,
        "model_behavior_bundle": model_behavior_identity,
        "personality_adapter": personality_identity,
        "training_runtime": training_runtime_identity,
        "lora": {
            "rank": args.lora_rank,
            "targets": list(args.lora_targets),
            "wrapped_projections": len(wrapped),
            "projection_paths": wrapped,
            "trainable_params": int(sum(value.size for value in trainable.values())),
        },
        "optimizer": config_payload["optimizer"],
        "gradient_execution": config_payload["gradient_execution"],
        "steps": step,
        "epoch": epoch,
        "cursor": cursor,
        "elapsed_training_s": round(elapsed_training_s(), 6),
        "invocation_count": invocation_count,
        "halt_reason": halt_reason,
        "complete": complete_run,
        "final_checkpoint": final_checkpoint.name,
        "loss_trail": receipt_loss_trail,
    }
    if objective_is_v3:
        receipt["objective_options"] = config_payload["objective_options"]
        receipt["holdout_trail"] = receipt_holdout_trail
    if resume_migration is not None:
        receipt["resume_migration"] = resume_migration
    receipt_bytes = canonical_json_bytes(receipt)
    atomic_write_bytes(out_dir / "receipt.json", receipt_bytes)
    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )

    tensor_metadata = [
        tensor.to_dict() for tensor in inspect_mlx_tensor_metadata(out_dir / "adapters.safetensors")
    ]

    def artifact_binding(path: str, payload: bytes) -> dict[str, Any]:
        return {
            "path": path,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }

    manifest = {
        "schema": ADAPTER_MANIFEST_SCHEMA_V2,
        "adapter_id": args.adapter_id,
        "base_checkpoint": base_identity,
        "model_behavior_bundle": model_behavior_identity,
        "personality_adapter": personality_identity,
        "training_runtime": training_runtime_identity,
        "adapter": artifact_binding("adapters.safetensors", adapter_bytes),
        "adapter_alias": artifact_binding("adapter_final.safetensors", adapter_bytes),
        "loader_config": artifact_binding("adapter_config.json", adapter_config_bytes),
        "training_receipt": artifact_binding("receipt.json", receipt_bytes),
        "training_config": artifact_binding("training_config.json", config_bytes),
        "dataset_manifest": artifact_binding("dataset_manifest.json", dataset_bytes),
        "execution_spec": artifact_binding("execution_spec.json", execution_spec_bytes),
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": spec.sha256,
        "sources": source_artifacts,
        "lora": receipt["lora"],
        "tensors": tensor_metadata,
    }
    if resume_migration is not None:
        if migration_bytes is None:
            raise RuntimeError("verified migration bytes are unavailable")
        manifest["checkpoint_migration"] = artifact_binding(
            "checkpoint_migration.json",
            migration_bytes,
        )
    manifest_bytes = canonical_json_bytes(manifest)
    atomic_write_bytes(out_dir / "recurrence_adapter_manifest.json", manifest_bytes)
    completion = {
        "schema": "aura.recurrence_native_training_completion.v1",
        "complete": receipt["complete"],
        "halt_reason": halt_reason,
        "step": step,
        "adapter_sha256": manifest["adapter"]["sha256"],
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }
    atomic_write_bytes(out_dir / "training_completion.json", canonical_json_bytes(completion))
    print(
        f"halt={halt_reason} step={step} adapter={manifest['adapter']['sha256']}",
        flush=True,
    )
    if interrupted:
        return 130
    if halt_reason == "non_finite_loss":
        return 1
    return 0 if receipt["complete"] else 75


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_MANIFEST_SCHEMA_V2",
    "FrozenCurriculum",
    "GRADIENT_EXECUTION_SCHEMA",
    "MAX_CURRICULUM_SOURCE_BYTES",
    "TASK_GENERATOR_SOURCE",
    "TRAIN_SCHEMA_V2",
    "_deterministic_order",
    "_capture_frozen_curriculum",
    "_execute_frozen_curriculum",
    "_run",
    "_streamed_depth_value_and_grad",
]
