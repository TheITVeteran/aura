#!/usr/bin/env python3
"""Train the unified intrinsic recurrent controller on a bounded checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
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
from core.learning.recurrent_action_schema import (  # noqa: E402
    ACTION_SLOT_NAMES,
    action_targets_from_program,
    action_value_semantic_label,
)
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import (  # noqa: E402
    LITERAL_MAX_VALUE,
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)
from core.learning.recurrent_opcode_grounding import (  # noqa: E402
    tokenizer_opcode_contract,
)
from core.learning.recurrent_state_schema import (  # noqa: E402
    STATE_SLOT_NAMES,
    state_targets_from_trace,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    structured_action_accuracy_breakdown,
    structured_action_loss,
    structured_initial_state_accuracy_breakdown,
    structured_initial_state_loss,
    structured_state_accuracy_breakdown,
    structured_state_loss,
    unified_answer_and_recurrent_trajectory,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    unified_recurrent_hidden_states,
    unified_recurrent_logits,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.mlx_memory_guard import host_pressure, mlx_memory_envelope  # noqa: E402
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    resident_bootstrap_tokenizer_identity,
)
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    CHECKPOINT_GENERATION_SCHEMA,
    CHECKPOINT_POINTER_SCHEMA,
    TRAINING_SCHEMA,
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_preload_barrier import verify_release  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    CAMPAIGN_BINDING_SCHEMA,
)
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    TOKENIZED_DATASET_FILENAME,
    freeze_source_dataset,
    freeze_tokenized_dataset,
    verify_tokenized_dataset,
)

TRAINING_SOURCE_FILES = (
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py",
    "core/learning/depth_conditioned_lora.py",
    "core/learning/intrinsic_recurrence.py",
    "core/learning/protected_memory.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrent_answer_emission.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_literal_grounding.py",
    "core/learning/recurrent_opcode_grounding.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/unified_intrinsic_objective.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/mlx_memory_guard.py",
    "core/runtime/model_lane_control.py",
    "pyproject.toml",
    "requirements_lock.txt",
    "tools/resident_recurrent_sft_bootstrap_identity.py",
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/evaluate_unified_intrinsic_decoding.py",
    "tools/train_intrinsic_recurrence.py",
    "tools/train_unified_intrinsic_recurrence.py",
    "tools/unified_intrinsic_checkpoint.py",
    "tools/unified_intrinsic_preload_barrier.py",
    "tools/unified_intrinsic_resident_identity.py",
    "tools/unified_intrinsic_tokenization_contract.py",
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


def _parse_campaign_binding(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        binding = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("campaign checkpoint binding is invalid JSON") from exc
    required = {
        "schema",
        "campaign_id",
        "campaign_config_sha256",
        "source_commit",
        "source_tree",
        "source_manifest_sha256",
        "model_manifest_sha256",
        "runtime_identity_sha256",
        "dataset_identity_sha256",
        "tokenizer_identity_sha256",
        "tokenized_dataset_identity_sha256",
        "training_profile_sha256",
        "binding_sha256",
    }
    body = (
        {key: value for key, value in binding.items() if key != "binding_sha256"}
        if isinstance(binding, dict)
        else {}
    )
    if (
        not isinstance(binding, dict)
        or set(binding) != required
        or binding.get("schema") != CAMPAIGN_BINDING_SCHEMA
        or binding.get("binding_sha256") != _canonical_sha256(body)
        or any(
            not isinstance(value, str) or not value
            for key, value in body.items()
            if key != "schema"
        )
    ):
        raise ValueError("campaign checkpoint binding differs")
    return binding


def _adam(learning_rate: float) -> Any:
    return optim.Adam(
        learning_rate=learning_rate,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=False,
    )


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
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "README.md"
    )
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("model checkpoint contains a symlinked artifact")
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        row = {
            "name": path.name,
            "size": before.st_size,
            "sha256": _file_sha256(resolved),
        }
        after = resolved.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("model checkpoint changed while hashing")
        rows.append(row)
        by_name[path.name] = row
    weight_rows = [by_name[path.name] for path in weights]
    behavior_rows = [
        row for row in rows if not row["name"].endswith(".safetensors")
    ]
    body = {
        "canonical_path": str(directory),
        "config_sha256": by_name[config.name]["sha256"],
        "weights": weight_rows,
        "behavior_files": behavior_rows,
        "behavior_sha256": _canonical_sha256(behavior_rows),
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _runtime_identity() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        runtime_environment_identity,
    )

    environment = runtime_environment_identity()
    executable = Path(os.path.abspath(sys.executable))
    real_executable = executable.resolve(strict=True)
    before = real_executable.stat()
    executable_sha256 = _file_sha256(real_executable)
    after = real_executable.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("Python interpreter changed while hashing")
    body = {
        "environment": environment,
        "interpreter": {
            "executable": str(executable),
            "real_executable": str(real_executable),
            "sys_prefix": str(Path(sys.prefix).resolve(strict=True)),
            "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
            "size_bytes": before.st_size,
            "sha256": executable_sha256,
        },
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _freeze_dataset(
    out_dir: Path,
    train_tasks: list[Any],
    holdout_tasks: list[Any],
) -> dict[str, Any]:
    from tools.unified_intrinsic_tokenization_contract import freeze_source_dataset

    return freeze_source_dataset(out_dir / "dataset.json", train_tasks, holdout_tasks)


def _load_frozen_dataset(path: Path) -> tuple[list[Any], list[Any]]:
    from tools.unified_intrinsic_tokenization_contract import load_source_dataset

    return load_source_dataset(path)


def _model_layer_count(model_path: str) -> int:
    config_path = Path(model_path).expanduser().resolve(strict=True) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError("model checkpoint config is unreadable") from exc
    if not isinstance(config, dict):
        raise ValueError("model checkpoint config differs")
    candidates = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.insert(0, text_config)
    for candidate in candidates:
        value = candidate.get("num_hidden_layers")
        if type(value) is int and value >= 3:
            return value
    raise ValueError("model checkpoint layer count is unavailable")


def _resolve_recurrent_window(
    model_path: str,
    *,
    prelude_end: int | None,
    coda_start: int | None,
    prelude_fraction: float,
    coda_fraction: float,
) -> tuple[int, int, dict[str, Any]]:
    layer_count = _model_layer_count(model_path)
    supplied = (prelude_end is not None, coda_start is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("explicit recurrent window requires both boundaries")
    if all(supplied):
        resolved_prelude = int(prelude_end)
        resolved_coda = int(coda_start)
        mode = "explicit"
    else:
        for name, value in (
            ("prelude", prelude_fraction),
            ("coda", coda_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 0.5
            ):
                raise ValueError(f"{name} fraction must be finite and inside (0, 0.5)")
        if float(prelude_fraction) + float(coda_fraction) >= 1.0:
            raise ValueError("recurrent window fractions leave no middle block")
        resolved_prelude = max(1, int(layer_count * float(prelude_fraction)))
        resolved_coda = min(
            layer_count - 1,
            layer_count - max(1, int(layer_count * float(coda_fraction))),
        )
        mode = "fractional"
    if not 0 < resolved_prelude < resolved_coda < layer_count:
        raise ValueError("resolved recurrent window is outside the model")
    body = {
        "mode": mode,
        "layer_count": layer_count,
        "prelude_end": resolved_prelude,
        "coda_start": resolved_coda,
        "prelude_fraction": (
            float(prelude_fraction) if mode == "fractional" else None
        ),
        "coda_fraction": float(coda_fraction) if mode == "fractional" else None,
    }
    return resolved_prelude, resolved_coda, {
        **body,
        "contract_sha256": _canonical_sha256(body),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, encoded, encoding="utf-8", mode=0o600)


def _atomic_canonical_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, encoded, encoding="ascii", mode=0o600)


def _await_resource_guard(
    marker_path: Path,
    *,
    trainer_sha256: str,
    startup_lethal_mb: float,
    steady_lethal_mb: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Refuse the first optimizer graph until an external sentinel is armed."""

    from core.runtime.resource_stage_guard import (
        ResourceStageGuardError,
        ack_path,
        publish_ready_marker,
        read_armed_ack,
        sha256_bytes,
    )

    acknowledgement = ack_path(marker_path)
    if marker_path.exists() or acknowledgement.exists():
        raise ResourceStageGuardError("resource guard attempt artifacts already exist")
    marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256=trainer_sha256,
    )
    print(f"resource guard marker published: {marker_path}", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if acknowledgement.exists():
            acknowledgement_payload, acknowledgement_raw = read_armed_ack(
                marker_path,
                marker_raw=marker_raw,
                expected_target_pid=os.getpid(),
                startup_lethal_mb=startup_lethal_mb,
                steady_lethal_mb=steady_lethal_mb,
            )
            return {
                "marker": marker,
                "marker_sha256": sha256_bytes(marker_raw),
                "ack": acknowledgement_payload,
                "ack_sha256": sha256_bytes(acknowledgement_raw),
            }
        time.sleep(0.25)
    raise ResourceStageGuardError(
        "external sentinel did not acknowledge unified training in time"
    )


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
        "window_tissue_mode": "scoped_lora",
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


def _configure_window_tissue(
    model: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    mode: str,
    rank: int,
    targets: tuple[str, ...],
    depth_basis_size: int,
) -> dict[str, Any]:
    """Build the declared recurrent tissue without silently adding base adapters."""

    if mode == "scoped_lora":
        return _attach_window_adapters(
            model,
            spec,
            rank=rank,
            targets=targets,
            depth_basis_size=depth_basis_size,
        )
    if mode != "controller_only":
        raise ValueError("unified recurrence window tissue mode is invalid")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not 0 <= spec.prelude_end < spec.coda_start <= len(layers):
        raise ValueError("controller-only recurrent window is outside the model")
    return {
        "window_tissue_mode": "controller_only",
        "window": [spec.prelude_end, spec.coda_start],
        "adapted_sites": [],
        "adapted_projection_count": 0,
        "continuous_depth_operator_count": 0,
        "continuous_depth_basis_size": 0,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": False,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_bridge": "typed_recurrent_controller_only",
    }


def _model_lane_purpose(window_tissue_mode: str) -> str:
    """Select the physical-memory envelope that matches the trainable tissue."""

    if window_tissue_mode == "controller_only":
        return "train_frozen_controller"
    if window_tissue_mode == "scoped_lora":
        return "train"
    raise ValueError("unified recurrence window tissue mode is invalid")


def _trainable(bundle: UnifiedTrainingBundle) -> dict[str, Any]:
    return dict(tree_flatten(bundle.trainable_parameters()))


def _ground_state_value_embeddings(
    model: Any,
    tokenizer: Any,
    controller: UnifiedRecurrentController,
    *,
    prelude_end: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Initialize typed values on the frozen model's native prelude manifold."""

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("grounding batch size must be a positive integer")

    def encode_label(label: str) -> list[int]:
        try:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(label)
        if not token_ids:
            raise RuntimeError("grounded semantic label encoded to no tokens")
        return [int(token_id) for token_id in token_ids]

    labels = [
        f"Internal state {slot_name}={value}"
        for slot_name in STATE_SLOT_NAMES
        for value in range(controller.config.state_cardinality)
    ]
    labels.extend(
        action_value_semantic_label(slot_name, value)
        for slot_name in ACTION_SLOT_NAMES
        for value in range(controller.config.action_cardinality)
    )
    labels.extend(str(value) for value in range(LITERAL_MAX_VALUE + 1))
    encoded = [encode_label(label) for label in labels]
    buckets: dict[int, list[tuple[int, list[int]]]] = {}
    for index, token_ids in enumerate(encoded):
        buckets.setdefault(len(token_ids), []).append((index, token_ids))

    grounded_rows: list[Any | None] = [None] * len(labels)
    forward_batches = 0
    for token_count in sorted(buckets):
        bucket = buckets[token_count]
        for offset in range(0, len(bucket), batch_size):
            batch = bucket[offset : offset + batch_size]
            tokens = mx.array(
                [token_ids for _index, token_ids in batch],
                dtype=mx.int32,
            )
            hidden = model.model.embed_tokens(tokens)
            terminal = _run(model.model.layers[:prelude_end], hidden)[:, -1, :].astype(
                mx.float32
            )
            mx.eval(terminal)
            for row, (index, _token_ids) in enumerate(batch):
                grounded_rows[index] = terminal[row]
            forward_batches += 1

    if any(value is None for value in grounded_rows):
        raise RuntimeError("grounded semantic label inventory is incomplete")
    concrete_rows = [value for value in grounded_rows if value is not None]
    cursor = 0

    def take(shape: tuple[int, int]) -> Any:
        nonlocal cursor
        row_count, cardinality = shape
        count = row_count * cardinality
        selected = concrete_rows[cursor : cursor + count]
        cursor += count
        return mx.stack(selected).reshape(row_count, cardinality, -1)

    grounded = take(
        (len(STATE_SLOT_NAMES), controller.config.state_cardinality)
    )
    grounded_actions = take(
        (len(ACTION_SLOT_NAMES), controller.config.action_cardinality)
    )
    literal_count = LITERAL_MAX_VALUE + 1
    grounded_literals = mx.stack(concrete_rows[cursor : cursor + literal_count])
    cursor += literal_count
    if cursor != len(labels):
        raise RuntimeError("grounded semantic label cursor differs")
    if grounded.shape != controller.state_value_embeddings.shape:
        raise RuntimeError("grounded state codebook shape differs from controller")
    controller.state_value_embeddings = grounded
    if grounded_actions.shape != controller.action_value_embeddings.shape:
        raise RuntimeError("grounded action codebook shape differs from controller")
    controller.action_value_embeddings = grounded_actions
    if grounded_literals.shape != controller.literal_value_embeddings.shape:
        raise RuntimeError("grounded literal codebook shape differs from controller")
    controller.literal_value_embeddings = grounded_literals
    mx.eval(
        controller.state_value_embeddings,
        controller.action_value_embeddings,
        controller.literal_value_embeddings,
        controller.state_slot_embeddings,
        controller.action_slot_embeddings,
    )
    digest = hashlib.sha256(
        bytes(memoryview(controller.state_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.action_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.literal_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.state_slot_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.action_slot_embeddings.astype(mx.float32)))
    ).hexdigest()
    return {
        "sha256": digest,
        "label_count": len(labels),
        "forward_batches": forward_batches,
        "batch_size": batch_size,
        "token_length_buckets": sorted(buckets),
        "max_token_length": max(buckets),
    }


def _optimization_phase(
    step: int,
    semantic_warmup_steps: int,
    state_warmup_steps: int = 0,
    answer_bridge_steps: int = 0,
) -> str:
    if type(step) is not int or step < 0:
        raise ValueError("optimization step must be non-negative")
    if type(semantic_warmup_steps) is not int or semantic_warmup_steps < 0:
        raise ValueError("semantic warmup steps must be non-negative")
    if type(state_warmup_steps) is not int or state_warmup_steps < 0:
        raise ValueError("state warmup steps must be non-negative")
    if type(answer_bridge_steps) is not int or answer_bridge_steps < 0:
        raise ValueError("answer bridge steps must be non-negative")
    # State parsing must exist before a semantic adapter is asked to decode a
    # typed path.  Training the no-state decoder first produced an apparent CE
    # gain while the actual T1 inference path collapsed to punctuation.
    if step < state_warmup_steps:
        return "state_transition"
    if step < state_warmup_steps + semantic_warmup_steps:
        return "semantic_anchor"
    if step < state_warmup_steps + semantic_warmup_steps + answer_bridge_steps:
        return "answer_bridge"
    return "recurrence"


def _semantic_execution_depth(
    task_depth: int,
    spec: UnifiedIntrinsicTrainingSpec,
) -> int:
    """Return the public execution depth at which the task is complete."""

    if type(task_depth) is not int or task_depth not in spec.train_depths:
        raise ValueError("semantic task depth is outside the trained recurrence horizon")
    return task_depth


def _phase_gradients(gradients: Any, phase: str) -> Any:
    """Keep the T1 semantic anchor fixed while training residual recurrence.

    Shared adapters learn the scoped T1 anchor.  Typed-state interpretation is
    owned by the continuous depth residuals, so a joint update cannot erase a
    useful shallow candidate or alter ordinary model inference.
    """

    if phase not in {
        "semantic_anchor",
        "answer_bridge",
        "state_transition",
        "recurrence",
    }:
        raise ValueError("unified optimization phase is invalid")
    masked = []
    for name, value in tree_flatten(gradients):
        shared_adapter = (
            name.startswith("model.")
            and "continuous_depth_" not in name
            and (name.endswith(".lora_a") or name.endswith(".lora_b"))
        )
        neural_answer_bridge = name.startswith("controller.answer_")
        if phase == "semantic_anchor":
            keep = shared_adapter
        elif phase == "answer_bridge":
            keep = neural_answer_bridge
        else:
            keep = not (shared_adapter or neural_answer_bridge)
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
    if name.startswith(
        (
            "controller.answer_query",
            "controller.answer_key",
            "controller.answer_value",
            "controller.answer_output",
            "controller.answer_gate_query",
            "controller.answer_gate_logit",
            "controller.answer_role_projection",
            "controller.answer_role_bias",
            "controller.answer_place_projection",
            "controller.answer_place_state_projection",
            "controller.answer_place_width_projection",
            "controller.answer_place_bias",
            "controller.answer_digit_gate_logit",
        )
    ):
        return "state_answer_bridge"
    if name.startswith(
        ("controller.action_value_embeddings", "controller.action_slot_embeddings")
    ):
        return "typed_action_codebook"
    if name.startswith(
        (
            "controller.action_query",
            "controller.action_key",
            "controller.action_value",
            "controller.action_output",
            "controller.action_depth",
            "controller.action_bias",
            "controller.action_literal_copy_logit",
            "controller.opcode_copy_logit",
            "controller.state_action_projection",
        )
    ):
        return "typed_action_transition"
    if name.startswith(
        (
            "controller.state_transition_",
            "controller.state_readout_",
            "controller.state_literal_copy_logit",
        )
    ):
        return "typed_state_transition"
    if name.startswith(
        (
            "controller.state_value_embeddings",
            "controller.state_slot_embeddings",
            "controller.literal_value_embeddings",
            "controller.literal_grounding_logit",
        )
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


def _apply_training_gradients(
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    gradients: Any,
    *,
    phase: str,
    max_norm: float,
    totals: dict[str, Any],
    loss: Any,
) -> None:
    """Apply one ownership-masked update and retain pre-clip diagnostics."""

    gradients = _phase_gradients(gradients, phase)
    gradients, gradient_norm, gradient_group_norms = _clip_gradient_groups(
        gradients,
        max_norm,
    )
    mx.eval(gradient_norm, *gradient_group_norms.values())
    totals["max_preclip_gradient_norm"] = max(
        float(totals["max_preclip_gradient_norm"]),
        float(gradient_norm.item()),
    )
    prior_group_norms = totals["max_preclip_gradient_norms"]
    for group, group_norm in gradient_group_norms.items():
        prior_group_norms[group] = max(
            float(prior_group_norms.get(group, 0.0)),
            float(group_norm.item()),
        )
    optimizer.update(bundle, gradients)
    mx.eval(bundle.parameters(), optimizer.state, loss)


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
    interchangeable_token_ids: frozenset[int] | None = None,
) -> tuple[Any, tuple[int, ...]]:
    """Use generated history without relabeling or corrupting its grammar."""

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
        if use_generated and interchangeable_token_ids is not None:
            # A generated digit may replace another digit because this exposes
            # the decoder to a wrong value while preserving the same grammar
            # state.  A digit may not replace syntax (or vice versa): doing so
            # shifts every later role/place target and trains against labels
            # that no longer describe the autoregressive prefix.
            use_generated = produced == target or (
                produced in interchangeable_token_ids
                and target in interchangeable_token_ids
            )
        effective.append(produced if use_generated else target)
        if use_generated:
            selected.append(position)
    return mx.array([effective], dtype=answer_tokens.dtype), tuple(selected)


def _record_student_rollin(
    totals: dict[str, Any],
    answer_tokens: Any,
    generated_tokens: Any,
    effective_tokens: Any,
    selected: tuple[int, ...],
    probability: float,
) -> None:
    """Record generated-history exposure without treating it as authority."""

    answer = [int(value) for value in answer_tokens.tolist()[0]]
    generated = [int(value) for value in generated_tokens.tolist()[0]]
    totals["examples"] += 1
    totals["answer_tokens"] += len(answer)
    totals["generated_positions"] += len(selected)
    totals["generated_matches"] += sum(
        generated[index] == answer[index] for index in selected
    )
    totals["last_generated_sha256"] = _sha256_tokens(generated_tokens)
    totals["last_effective_sha256"] = _sha256_tokens(effective_tokens)
    totals["last_probability"] = probability


def _answer_role_place_targets(
    family: str,
    answer_tokens: Any,
    contract: RecurrentAnswerEmissionContract,
) -> tuple[Any, Any]:
    """Label syntax versus terminal-register digit positions exactly."""

    syntax = dict(contract.syntax)
    layouts = {
        # Role zero means no pointer. Positive role N selects categorical
        # state slot N-1, so public result register 1 is role class 2.
        "khop": ((syntax["khop"], None), ((), 2), (syntax["close"], None)),
        "modular": (
            (syntax["modular"], None),
            ((), 2),
            (syntax["close"], None),
        ),
        "register_trace": (
            (syntax["register_head"], None),
            ((), 2),
            (syntax["register_mid_r1"], None),
            ((), 3),
            (syntax["register_mid_r2"], None),
            ((), 4),
            (syntax["close"], None),
        ),
    }
    if family not in layouts:
        raise ValueError("answer bridge family is outside the admitted grammar")
    values = tuple(int(value) for value in answer_tokens.tolist()[0])
    roles = [0] * len(values)
    places = [0] * len(values)
    cursor = 0
    digit_ids = set(contract.digit_token_ids)
    for fixed, role in layouts[family]:
        if role is None:
            stop = cursor + len(fixed)
            if values[cursor:stop] != fixed:
                raise ValueError("answer tokens differ from the canonical grammar")
            cursor = stop
            continue
        start = cursor
        while cursor < len(values) and values[cursor] in digit_ids:
            cursor += 1
        width = cursor - start
        if width not in {1, 2}:
            raise ValueError("answer value is outside the admitted two-digit grammar")
        for position in range(start, cursor):
            roles[position] = role
        if width == 1:
            places[start] = 2
        else:
            places[start] = 1
            places[start + 1] = 2
    if values[cursor:] != (contract.eos_token_id,):
        raise ValueError("answer tokens do not terminate with the bound EOS token")
    return (
        mx.array([roles], dtype=mx.int32),
        mx.array([places], dtype=mx.int32),
    )


def _answer_binding_loss(
    role_logits: Any,
    place_logits: Any,
    role_targets: Any,
    place_targets: Any,
) -> Any:
    """Supervise neural slot selection without exposing answer values."""

    if role_targets.shape != place_targets.shape or len(role_targets.shape) != 2:
        raise ValueError("answer binding targets differ")
    token_count = int(role_targets.shape[-1])
    if (
        len(role_logits.shape) != 3
        or len(place_logits.shape) != 3
        or role_logits.shape[:2] != place_logits.shape[:2]
        or int(role_logits.shape[1]) < token_count
        or int(place_logits.shape[1]) < token_count
    ):
        raise ValueError("answer binding logits differ from target positions")
    role_terms = nn.losses.cross_entropy(
        role_logits[:, :token_count, :].astype(mx.float32),
        role_targets,
        reduction="none",
    )
    place_terms = nn.losses.cross_entropy(
        place_logits[:, :token_count, :].astype(mx.float32),
        place_targets,
        reduction="none",
    )
    role_weights = mx.where(role_targets == 0, 0.25, 1.0)
    place_weights = mx.where(place_targets == 0, 0.25, 1.0)
    role_loss = mx.sum(role_terms * role_weights) / mx.sum(role_weights)
    place_loss = mx.sum(place_terms * place_weights) / mx.sum(place_weights)
    return 0.5 * (role_loss + place_loss)


def _answer_bridge_task(tasks: list[Any], bridge_index: int) -> Any:
    """Cover every family, then every family/depth cell, before repetition."""

    if type(bridge_index) is not int or bridge_index < 0 or not tasks:
        raise ValueError("answer bridge task schedule is invalid")
    cells = sorted({(str(task.family), int(task.depth)) for task in tasks})
    first_by_family = [
        min((cell for cell in cells if cell[0] == family), key=lambda cell: cell[1])
        for family in sorted({family for family, _depth in cells})
    ]
    ordered_cells = first_by_family + [
        cell for cell in cells if cell not in first_by_family
    ]
    cell = ordered_cells[bridge_index % len(ordered_cells)]
    cell_tasks = [
        task for task in tasks if (str(task.family), int(task.depth)) == cell
    ]
    cycle = bridge_index // len(ordered_cells)
    return cell_tasks[cycle % len(cell_tasks)]


def _cached_answer_binding_features(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer_tokens: Any,
    plan: Any,
) -> tuple[Any, Any, Any]:
    """Run the expensive tissue once and detach its causal binding features."""

    full = mx.concatenate([prompt, answer_tokens], axis=1)
    features: list[tuple[Any, Any, Any]] = []
    unified_recurrent_hidden_states(
        bundle.model,
        full,
        plan,
        bundle.controller,
        state_slot_start=int(prompt.shape[-1]),
        answer_binding_feature_trajectory=features,
    )
    if not features:
        raise RuntimeError("answer bridge emitted no reusable causal features")
    selected = tuple(mx.stop_gradient(value) for value in features[-1])
    mx.eval(*selected)
    return selected


def _cached_answer_binding_loss(
    bundle: UnifiedTrainingBundle,
    features: tuple[Any, Any, Any],
    targets: tuple[Any, Any],
) -> Any:
    role_logits, place_logits = bundle.controller.answer_binding_logits(*features)
    return _answer_binding_loss(role_logits, place_logits, *targets)


def _generate_student_rollin(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer_tokens: Any,
    plan: Any,
    *,
    eos_token_id: int | None,
    answer_emission_contract: RecurrentAnswerEmissionContract | None = None,
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
                answer_emission_contract=answer_emission_contract,
            )
            token = int(mx.argmax(logits[0, -1]).item())
            stopped = eos_token_id is not None and token == eos_token_id
        generated.append(token)
        tokens = mx.concatenate(
            [tokens, mx.array([[token]], dtype=tokens.dtype)],
            axis=1,
        )
    return mx.array([generated], dtype=answer_tokens.dtype)


def _evaluate_answer_bridge_admission(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    contract: RecurrentAnswerEmissionContract,
) -> dict[str, Any]:
    """Require exact emission on one unseen task from every family/depth cell."""

    cells = {(str(task.family), int(task.depth)) for task in tasks}
    selected = [
        _answer_bridge_task(tasks, index)
        for index in range(len(cells))
    ]
    selected_cells = {(str(task.family), int(task.depth)) for task in selected}
    if selected_cells != cells:
        raise RuntimeError("answer bridge admission did not cover every task cell")
    rows: list[dict[str, Any]] = []
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for task in selected:
            prompt, answer = encode_example(tokenizer, task, bridge)
            generated = _generate_student_rollin(
                bundle,
                prompt,
                answer,
                spec.plan_at(max(spec.train_depths)),
                eos_token_id=tokenizer.eos_token_id,
                answer_emission_contract=contract,
                state_slot_start=int(prompt.shape[-1]),
            )
            expected_values = tuple(int(value) for value in answer.tolist()[0])
            generated_values = tuple(int(value) for value in generated.tolist()[0])
            mismatches = [
                {
                    "position": position,
                    "expected_token_id": expected,
                    "generated_token_id": observed,
                }
                for position, (observed, expected) in enumerate(
                    zip(generated_values, expected_values, strict=True)
                )
                if observed != expected
            ]
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "task_depth": task.depth,
                    "exact": generated_values == expected_values,
                    "matching_tokens": sum(
                        observed == expected
                        for observed, expected in zip(
                            generated_values,
                            expected_values,
                            strict=True,
                        )
                    ),
                    "token_count": len(expected_values),
                    "mismatches": mismatches,
                    "expected_sha256": _sha256_tokens(answer),
                    "generated_sha256": _sha256_tokens(generated),
                }
            )
    exact = sum(row["exact"] for row in rows)
    matching = sum(row["matching_tokens"] for row in rows)
    token_count = sum(row["token_count"] for row in rows)
    body = {
        "schema": "aura.unified_intrinsic.answer_bridge_admission.v3",
        "depth": max(spec.train_depths),
        "cells": len(cells),
        "tasks": len(rows),
        "exact": exact,
        "exact_accuracy": exact / len(rows),
        "token_accuracy": matching / token_count,
        "admitted": exact == len(rows),
        "rows": rows,
    }
    return {**body, "admission_sha256": _canonical_sha256(body)}


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


def _invocation_stop_step(
    start_step: int,
    max_steps: int,
    max_invocation_steps: int | None,
) -> int:
    """Return an operational stop boundary without changing campaign identity."""

    if start_step < 0 or max_steps < 1 or start_step > max_steps:
        raise ValueError("unified recurrence invocation step range is invalid")
    if max_invocation_steps is None:
        return max_steps
    if max_invocation_steps < 1:
        raise ValueError("maximum invocation steps must be positive")
    return min(max_steps, start_step + max_invocation_steps)


def _training_halt_reason(
    *,
    step: int,
    max_steps: int,
    invocation_stop_step: int,
) -> str:
    if step >= max_steps:
        return "max_steps"
    if step >= invocation_stop_step:
        return "invocation_step_limit"
    return "wall_clock"


def _training_verdict(
    *,
    complete: bool,
    answer_bridge_admission: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> str:
    """Label only terminal evidence as a scientific training verdict."""

    if not complete:
        return "incomplete_checkpoint"
    if answer_bridge_admission is not None and not answer_bridge_admission["admitted"]:
        return "answer_bridge_not_admitted"
    if final and final["heldout_depth_helps"]:
        return "heldout_depth_gain"
    if final and final["trained_depth_helps"]:
        return "trained_depth_gain_only"
    return "no_heldout_depth_gain"


def _initial_rollin_totals() -> dict[str, Any]:
    return {
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
        "answer_bridge_inner_updates": 0,
    }


def _restore_rollin_totals(training_state: dict[str, Any]) -> dict[str, Any]:
    if not training_state:
        return _initial_rollin_totals()
    candidate = training_state.get("rollin_totals")
    expected = _initial_rollin_totals()
    if not isinstance(candidate, dict) or set(candidate) != set(expected):
        raise RuntimeError("unified recurrence roll-in checkpoint state differs")
    for key in (
        "examples",
        "answer_tokens",
        "generated_positions",
        "generated_matches",
    ):
        value = candidate[key]
        if type(value) is not int or value < 0:
            raise RuntimeError("unified recurrence roll-in counters differ")
    for key in ("last_generated_sha256", "last_effective_sha256"):
        value = candidate[key]
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError("unified recurrence roll-in digest differs")
    maximum = candidate["max_preclip_gradient_norm"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
    ):
        raise RuntimeError("unified recurrence roll-in gradient maximum differs")
    group_norms = candidate["max_preclip_gradient_norms"]
    if not isinstance(group_norms, dict) or any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for name, value in group_norms.items()
    ):
        raise RuntimeError("unified recurrence roll-in gradient groups differ")
    for key in ("last_probability", "last_state_teacher_forcing_probability"):
        value = candidate[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise RuntimeError("unified recurrence roll-in probability differs")
    return {
        **candidate,
        "max_preclip_gradient_norms": dict(group_norms),
    }


def _rollin_report(
    totals: dict[str, Any],
    *,
    initial_probability: float,
    final_probability: float,
) -> dict[str, Any]:
    snapshot = copy.deepcopy(totals)
    generated_positions = int(snapshot["generated_positions"])
    return {
        **snapshot,
        "initial_probability": initial_probability,
        "final_probability": final_probability,
        "generated_match_rate": (
            int(snapshot["generated_matches"]) / generated_positions
            if generated_positions
            else None
        ),
        "labels_from_generated_tokens": False,
    }


def _checkpoint_tensor_bytes(tensors: dict[str, Any], out_dir: Path) -> bytes:
    scratch = out_dir / f".checkpoint.{os.getpid()}.{uuid.uuid4().hex}.safetensors"
    try:
        mx.save_safetensors(str(scratch), tensors)
        return scratch.read_bytes()
    finally:
        durable_unlink(scratch, missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discard_checkpoint_stage(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    for name in ("bundle.safetensors", "complete.json"):
        durable_unlink(path / name, missing_ok=True)
    try:
        path.rmdir()
    except OSError:
        pass


def _publish_latest_checkpoint_generation(
    out_dir: Path,
    *,
    stem: str,
    payload: bytes,
    step: int,
    history: list[dict[str, Any]],
    identity: dict[str, Any],
    optimization_phase: str,
    training_state: dict[str, Any],
) -> dict[str, Any]:
    """Publish immutable bytes, then atomically advance the latest pointer."""

    generations = ensure_private_directory(out_dir / "checkpoint_generations")
    checkpoint_id = f"{stem}-step-{step:08d}-{uuid.uuid4().hex}"
    generation_dir = generations / checkpoint_id
    stage_dir = ensure_private_directory(
        generations / f".checkpoint-stage-{uuid.uuid4().hex}"
    )
    try:
        weights_path = stage_dir / "bundle.safetensors"
        atomic_write_bytes(weights_path, payload, mode=0o400)
        checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        body = {
            "schema": TRAINING_SCHEMA,
            "checkpoint_generation_schema": CHECKPOINT_GENERATION_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "stem": stem,
            "step": step,
            "optimization_phase": optimization_phase,
            "history": history,
            "training_state": training_state,
            "identity": identity,
            "checkpoint_file": weights_path.name,
            "checkpoint_size_bytes": len(payload),
            "checkpoint_sha256": checkpoint_sha256,
        }
        complete = {**body, "receipt_sha256": _canonical_sha256(body)}
        complete_bytes = (
            json.dumps(
                complete,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        atomic_write_bytes(stage_dir / "complete.json", complete_bytes, mode=0o400)
        os.chmod(stage_dir, 0o500)
        os.rename(stage_dir, generation_dir)
        _fsync_directory(generations)
    finally:
        _discard_checkpoint_stage(stage_dir)
    pointer = {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "checkpoint": f"checkpoint_generations/{checkpoint_id}",
        "complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "identity_sha256": identity["identity_sha256"],
        "step": step,
        "stem": stem,
    }
    atomic_write_text(
        out_dir / f"{stem}_pointer.json",
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        mode=0o600,
    )

    # Preserve the historical fixed paths for evaluators without duplicating
    # the tensor payload. The immutable generation remains resume authority.
    compatibility_weights = out_dir / f"{stem}.safetensors"
    atomic_write_bytes(compatibility_weights, payload, mode=0o400)
    legacy_body = {
        key: value
        for key, value in body.items()
        if key
        not in {
            "checkpoint_generation_schema",
            "checkpoint_id",
            "stem",
            "checkpoint_file",
            "checkpoint_size_bytes",
        }
    }
    _atomic_json(
        out_dir / f"{stem}.json",
        {**legacy_body, "receipt_sha256": _canonical_sha256(legacy_body)},
    )
    return complete


def _load_latest_checkpoint(
    out_dir: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], Path] | None:
    try:
        resolved = resolve_checkpoint_generation(
            out_dir,
            stem="checkpoint_latest",
            required=False,
        )
    except UnifiedCheckpointError as exc:
        raise RuntimeError(str(exc)) from exc
    if resolved is not None:
        return resolved.receipt, resolved.weights_path

    pointer_path = out_dir / "checkpoint_latest_pointer.json"
    legacy_receipt_path = out_dir / "checkpoint_latest.json"
    legacy_weights_path = out_dir / "checkpoint_latest.safetensors"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="ascii"))
        except (json.JSONDecodeError, OSError, UnicodeError) as exc:
            raise RuntimeError("unified recurrence checkpoint pointer is unreadable") from exc
        if not isinstance(pointer, dict) or set(pointer) != {
            "schema",
            "checkpoint",
            "complete_sha256",
            "identity_sha256",
            "step",
        } or pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA:
            raise RuntimeError("unified recurrence checkpoint pointer differs")
        relative = pointer.get("checkpoint")
        if not isinstance(relative, str) or not relative.startswith(
            "checkpoint_generations/"
        ):
            raise RuntimeError("unified recurrence checkpoint pointer path is invalid")
        try:
            generation_dir = (out_dir / relative).resolve(strict=True)
            generation_root = (out_dir / "checkpoint_generations").resolve(
                strict=True
            )
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                "unified recurrence checkpoint generation is unavailable"
            ) from exc
        if generation_dir.parent != generation_root or not generation_dir.is_dir():
            raise RuntimeError("unified recurrence checkpoint pointer escapes its root")
        complete_path = generation_dir / "complete.json"
        try:
            complete_bytes = complete_path.read_bytes()
            receipt = json.loads(complete_bytes.decode("ascii"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError) as exc:
            raise RuntimeError("unified recurrence checkpoint generation is unreadable") from exc
        identity = receipt.get("identity") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or hashlib.sha256(complete_bytes).hexdigest()
            != pointer.get("complete_sha256")
            or receipt.get("checkpoint_generation_schema")
            != CHECKPOINT_GENERATION_SCHEMA
            or receipt.get("checkpoint_id") != generation_dir.name
            or receipt.get("step") != pointer.get("step")
            or not isinstance(identity, dict)
            or identity.get("identity_sha256")
            != pointer.get("identity_sha256")
        ):
            raise RuntimeError("unified recurrence checkpoint generation differs")
        receipt_body = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _canonical_sha256(receipt_body):
            raise RuntimeError("unified recurrence checkpoint receipt differs")
        weights_name = receipt.get("checkpoint_file")
        if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
            raise RuntimeError("unified recurrence checkpoint weight path is invalid")
        weights_path = generation_dir / weights_name
        try:
            size = weights_path.stat().st_size
            digest = _file_sha256(weights_path)
        except OSError as exc:
            raise RuntimeError("unified recurrence checkpoint weights are unreadable") from exc
        if (
            size != receipt.get("checkpoint_size_bytes")
            or digest != receipt.get("checkpoint_sha256")
        ):
            raise RuntimeError("unified recurrence checkpoint weights differ")
        return receipt, weights_path

    legacy_present = (legacy_receipt_path.is_file(), legacy_weights_path.is_file())
    if any(legacy_present) and not all(legacy_present):
        raise RuntimeError("unified recurrence legacy checkpoint is incomplete")
    if not all(legacy_present):
        if required:
            raise RuntimeError("unified recurrence resume checkpoint is unavailable")
        return None
    try:
        receipt = json.loads(legacy_receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise RuntimeError("unified recurrence legacy checkpoint is unreadable") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("unified recurrence legacy checkpoint receipt differs")
    return receipt, legacy_weights_path


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
    training_state: dict[str, Any] | None = None,
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
    payload = _checkpoint_tensor_bytes(tensors, out_dir)
    with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
        _publish_latest_checkpoint_generation(
            out_dir,
            stem=stem,
            payload=payload,
            step=step,
            history=history,
            identity=identity,
            optimization_phase=optimization_phase,
            training_state=dict(training_state or {}),
        )


def _restore_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    identity: dict[str, Any],
    *,
    semantic_warmup_steps: int = 0,
    state_warmup_steps: int = 0,
    answer_bridge_steps: int = 0,
    required: bool = False,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
        loaded = _load_latest_checkpoint(out_dir, required=required)
    if loaded is None:
        return 0, [], {}
    receipt, weights_path = loaded
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
        answer_bridge_steps,
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
    training_state = receipt.get("training_state", {})
    if not isinstance(training_state, dict):
        raise RuntimeError("unified recurrence checkpoint training state differs")
    return (
        int(receipt["step"]),
        list(receipt.get("history", [])),
        dict(training_state),
    )


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
    initial_state_totals = {depth: 0.0 for depth in depths}
    state_value_totals = {depth: 0.0 for depth in depths}
    state_control_totals = {depth: 0.0 for depth in depths}
    initial_value_totals = {depth: 0.0 for depth in depths}
    initial_control_totals = {depth: 0.0 for depth in depths}
    state_value_exact_totals = {depth: 0.0 for depth in depths}
    initial_value_exact_totals = {depth: 0.0 for depth in depths}
    action_totals = {depth: 0.0 for depth in depths}
    action_exact_totals = {depth: 0.0 for depth in depths}
    with recurrence_adapter_scope(start=None, stop=None):
        for task in tasks:
            prompt, answer = encode_example(tokenizer, task, bridge)
            for depth in depths:
                initial_state_logits: list[Any] = []
                action_logits: list[Any] = []
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
                        initial_state_logit_trajectory=initial_state_logits,
                        action_logit_trajectory=action_logits,
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
                    if len(initial_state_logits) != 1:
                        raise RuntimeError(
                            "evaluation emitted no initial state decision"
                        )
                    _initial_loss, initial_accuracy = structured_initial_state_loss(
                        initial_state_logits[0],
                        targets,
                    )
                    state_breakdown = structured_state_accuracy_breakdown(
                        state_logits,
                        targets,
                    )
                    initial_breakdown = structured_initial_state_accuracy_breakdown(
                        initial_state_logits[0],
                        targets,
                    )
                    program = getattr(task, "transition_program", None)
                    if program is None:
                        raise RuntimeError("evaluation task has no exact action program")
                    action_targets = action_targets_from_program(program, depth)
                    _action_loss, action_accuracy, _action_steps = (
                        structured_action_loss(action_logits, action_targets)
                    )
                    action_breakdown = structured_action_accuracy_breakdown(
                        action_logits,
                        action_targets,
                    )
                    state_totals[depth] += state_accuracy
                    initial_state_totals[depth] += initial_accuracy
                    state_value_totals[depth] += float(
                        state_breakdown["value_accuracy"] or 0.0
                    )
                    state_control_totals[depth] += float(
                        state_breakdown["control_accuracy"] or 0.0
                    )
                    initial_value_totals[depth] += float(
                        initial_breakdown["value_accuracy"] or 0.0
                    )
                    initial_control_totals[depth] += float(
                        initial_breakdown["control_accuracy"] or 0.0
                    )
                    state_value_exact_totals[depth] += float(
                        state_breakdown["value_exact_accuracy"] or 0.0
                    )
                    initial_value_exact_totals[depth] += float(
                        initial_breakdown["value_exact_accuracy"] or 0.0
                    )
                    action_totals[depth] += action_accuracy
                    action_exact_totals[depth] += float(
                        action_breakdown["instruction_exact_accuracy"] or 0.0
                    )
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
    initial_state_accuracy = {
        f"T{depth}": (
            initial_state_totals[depth] / state_counts[depth]
            if state_counts[depth]
            else None
        )
        for depth in depths
    }
    action_accuracy = {
        f"T{depth}": (
            action_totals[depth] / state_counts[depth]
            if state_counts[depth]
            else None
        )
        for depth in depths
    }
    state_value_accuracy = {
        f"T{depth}": state_value_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    state_control_accuracy = {
        f"T{depth}": state_control_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_value_accuracy = {
        f"T{depth}": initial_value_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_control_accuracy = {
        f"T{depth}": initial_control_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    state_value_exact_accuracy = {
        f"T{depth}": state_value_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_value_exact_accuracy = {
        f"T{depth}": initial_value_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    action_instruction_exact_accuracy = {
        f"T{depth}": action_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
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
        "initial_state_accuracy": initial_state_accuracy,
        "state_value_accuracy": state_value_accuracy,
        "state_control_accuracy": state_control_accuracy,
        "initial_value_accuracy": initial_value_accuracy,
        "initial_control_accuracy": initial_control_accuracy,
        "action_accuracy": action_accuracy,
        "state_value_exact_accuracy": state_value_exact_accuracy,
        "initial_value_exact_accuracy": initial_value_exact_accuracy,
        "action_instruction_exact_accuracy": action_instruction_exact_accuracy,
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
    parser.add_argument("--expected-model-identity-sha256")
    parser.add_argument(
        "--exclusive-model-lane",
        action="store_true",
        help="require an atomically exclusive model-memory lease before loading",
    )
    parser.add_argument(
        "--campaign-binding-json",
        help="canonical immutable source/model/runtime/training identity for checkpoints",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="create-once canonical dataset generated before resident model load",
    )
    parser.add_argument(
        "--tokenized-dataset",
        type=Path,
        help="create-once tokenizer-bound dataset generated before resident model load",
    )
    parser.add_argument("--prelude-end", type=int)
    parser.add_argument("--coda-start", type=int)
    parser.add_argument("--prelude-fraction", type=float, default=0.25)
    parser.add_argument("--coda-fraction", type=float, default=0.25)
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
    parser.add_argument(
        "--window-tissue-mode",
        choices=("scoped_lora", "controller_only"),
        default="scoped_lora",
        help=(
            "train scoped transformer adapters or leave every base-model tensor "
            "frozen and train only the recurrent controller"
        ),
    )
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
        "--answer-bridge-steps",
        type=int,
        default=0,
        help="isolated state-to-token adaptation steps after semantic warmup",
    )
    parser.add_argument(
        "--answer-bridge-inner-steps",
        type=int,
        default=1,
        help=(
            "head-only optimizer updates per expensive bridge feature pass; "
            "values above one use detached causal features"
        ),
    )
    parser.add_argument(
        "--state-learning-rate",
        type=float,
        help="state-transition phase rate; defaults to recurrent learning rate",
    )
    parser.add_argument(
        "--answer-bridge-learning-rate",
        type=float,
        help="isolated state-to-token rate; defaults to --learning-rate",
    )
    parser.add_argument(
        "--answer-bridge-rollin-probability",
        type=float,
        default=0.25,
        help="initial generated-history fraction during answer-bridge adaptation",
    )
    parser.add_argument(
        "--answer-bridge-rollin-final-probability",
        type=float,
        default=1.0,
        help="final generated-history fraction during answer-bridge adaptation",
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
    parser.add_argument(
        "--max-invocation-steps",
        type=int,
        help=(
            "stop this process after N additional durable steps without changing "
            "the scientific campaign identity; resume continues the same run"
        ),
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-group", type=int, default=4)
    parser.add_argument(
        "--grounding-batch-size",
        type=int,
        default=32,
        help="equal-token-length batch size for frozen-prelude codebook grounding",
    )
    parser.add_argument("--seed", type=int, default=20260810198)
    parser.add_argument("--init-seed", type=int, default=20260810198)
    parser.add_argument("--bridge", default="assistant_answer")
    parser.add_argument("--memory-fraction", type=float, default=0.48)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float)
    parser.add_argument("--resource-stage-path", type=Path)
    parser.add_argument("--resource-startup-lethal-mb", type=float)
    parser.add_argument("--resource-steady-lethal-mb", type=float)
    parser.add_argument("--resource-guard-timeout-s", type=float, default=120.0)
    parser.add_argument("--preload-ready-path", type=Path)
    parser.add_argument("--preload-release-path", type=Path)
    parser.add_argument("--preload-key-path", type=Path)
    parser.add_argument("--preload-config-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-if-available",
        action="store_true",
        help=(
            "restore a valid durable checkpoint when present, otherwise begin "
            "at step zero; intended for one immutable supervised replay command"
        ),
    )
    args = parser.parse_args()
    campaign_binding = _parse_campaign_binding(args.campaign_binding_json)
    if args.resume and args.resume_if_available:
        raise ValueError("resume modes are mutually exclusive")
    if not (
        0 <= args.semantic_warmup_steps < args.max_steps
        and args.state_warmup_steps >= 0
        and args.answer_bridge_steps >= 0
        and args.semantic_warmup_steps
        + args.state_warmup_steps
        + args.answer_bridge_steps
        < args.max_steps
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
        0.0 <= args.answer_bridge_rollin_probability
        <= args.answer_bridge_rollin_final_probability
        <= 1.0
    ):
        raise ValueError("answer bridge roll-in probability must increase inside [0, 1]")
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
    if args.answer_bridge_inner_steps < 1:
        raise ValueError("answer bridge inner steps must be positive")
    if args.max_minutes <= 0.0:
        raise ValueError("maximum minutes must be positive")
    if any(
        value is not None and (not math.isfinite(value) or value <= 0.0)
        for value in (
            args.memory_limit_gb,
            args.cache_limit_gb,
            args.wired_limit_gb,
        )
    ):
        raise ValueError("explicit MLX memory limits must be finite and positive")
    if (
        args.memory_limit_gb is not None
        and args.cache_limit_gb is not None
        and args.cache_limit_gb >= args.memory_limit_gb
    ):
        raise ValueError("MLX cache limit must be below active memory limit")
    if (
        args.memory_limit_gb is not None
        and args.wired_limit_gb is not None
        and args.wired_limit_gb <= args.memory_limit_gb
    ):
        raise ValueError("MLX wired limit must exceed active memory limit")
    if args.max_invocation_steps is not None and args.max_invocation_steps < 1:
        raise ValueError("maximum invocation steps must be positive")
    if args.expected_model_identity_sha256 is not None and (
        len(args.expected_model_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_model_identity_sha256
        )
    ):
        raise ValueError("expected model identity SHA-256 is invalid")
    if args.grounding_batch_size < 1:
        raise ValueError("grounding batch size must be positive")
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
    answer_bridge_learning_rate = (
        args.learning_rate
        if args.answer_bridge_learning_rate is None
        else args.answer_bridge_learning_rate
    )
    if (
        args.learning_rate <= 0.0
        or recurrent_learning_rate <= 0.0
        or state_learning_rate <= 0.0
        or answer_bridge_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if args.state_weight <= 0.0 or args.stutter_weight < 0.0:
        raise ValueError("state weight must be positive and stutter weight non-negative")
    if args.window_tissue_mode == "controller_only" and args.semantic_warmup_steps:
        raise ValueError(
            "controller-only tissue cannot schedule transformer semantic warmup"
        )
    resource_guard_values = (
        args.resource_stage_path,
        args.resource_startup_lethal_mb,
        args.resource_steady_lethal_mb,
    )
    resource_guard_enabled = all(value is not None for value in resource_guard_values)
    if any(value is not None for value in resource_guard_values) != resource_guard_enabled:
        raise ValueError("resource guard arguments must be supplied together")
    if resource_guard_enabled and not (
        math.isfinite(float(args.resource_startup_lethal_mb))
        and math.isfinite(float(args.resource_steady_lethal_mb))
        and float(args.resource_startup_lethal_mb)
        > float(args.resource_steady_lethal_mb)
        > 0.0
        and math.isfinite(args.resource_guard_timeout_s)
        and args.resource_guard_timeout_s > 0.0
    ):
        raise ValueError("resource guard ceilings or timeout are invalid")
    preload_values = (
        args.preload_ready_path,
        args.preload_release_path,
        args.preload_key_path,
        args.preload_config_sha256,
    )
    preload_enabled = all(value is not None for value in preload_values)
    if any(value is not None for value in preload_values) != preload_enabled:
        raise ValueError("preload barrier arguments must be supplied together")
    if resource_guard_enabled and not preload_enabled:
        raise ValueError("external resource guard requires a signed preload barrier")
    preload_host_pressure: dict[str, Any] | None = None
    if resource_guard_enabled:
        preload_release = verify_release(
            args.preload_release_path.expanduser(),
            ready_path=args.preload_ready_path.expanduser(),
            key_path=args.preload_key_path.expanduser(),
            config_sha256=str(args.preload_config_sha256),
            require_live_evidence=True,
        )
        preload_host_pressure = dict(preload_release["host_pressure"])
    elif preload_enabled:
        raise ValueError("preload barrier requires the external resource guard")
    else:
        preload_host_pressure = host_pressure()
    if preload_enabled and campaign_binding is None:
        raise ValueError("resident preload requires a campaign checkpoint binding")
    if campaign_binding is not None and (
        campaign_binding["campaign_config_sha256"] != args.preload_config_sha256
    ):
        raise ValueError("campaign checkpoint and preload identities differ")
    if preload_enabled or resource_guard_enabled:
        if (
            preload_host_pressure.get("available") is not True
            or preload_host_pressure.get("under_pressure") is not False
        ):
            raise RuntimeError(
                "resident unified training refused unavailable or pressured host"
            )

    from mlx_lm import load

    from core.learning import recurrence_curriculum as curriculum
    from core.runtime.model_lane_control import standalone_model_lane

    train_depths = tuple(int(value) for value in args.train_depths.split(","))
    heldout_depths = tuple(
        int(value) for value in args.heldout_depths.split(",")
    )
    prelude_end, coda_start, window_geometry = _resolve_recurrent_window(
        args.model,
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        prelude_fraction=args.prelude_fraction,
        coda_fraction=args.coda_fraction,
    )
    spec = UnifiedIntrinsicTrainingSpec(
        prelude_end=prelude_end,
        coda_start=coda_start,
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
    out_dir = args.out_dir.expanduser().resolve()
    ensure_private_directory(out_dir)
    if args.tokenized_dataset is not None and args.dataset is None:
        raise RuntimeError("tokenized dataset requires a frozen source dataset")
    if args.dataset is not None:
        dataset_path = args.dataset.expanduser().resolve(strict=True)
        train_tasks, holdout = _load_frozen_dataset(dataset_path)
        if (
            {task.family for task in train_tasks + holdout} != set(families)
            or {task.depth for task in train_tasks + holdout} != set(task_depths)
            or len(train_tasks) != len(families) * len(task_depths) * args.per_cell
            or len(holdout)
            != len(families) * len(task_depths) * args.holdout_per_cell
        ):
            raise RuntimeError("unified recurrence frozen dataset differs from CLI")
    else:
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
    missing_programs = [
        task.task_id
        for task in train_tasks + holdout
        if task.transition_program is None
    ]
    if missing_programs:
        raise RuntimeError(
            "action-supervised curriculum contains tasks without exact programs: "
            + ",".join(missing_programs[:5])
        )

    dataset_identity = (
        _freeze_dataset(out_dir, train_tasks, holdout)
        if args.dataset is None
        else freeze_source_dataset(dataset_path, train_tasks, holdout)
    )
    source_sha256s = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in TRAINING_SOURCE_FILES
    }
    runtime_identity = _runtime_identity()
    model_identity = _model_identity(args.model)
    if (
        args.expected_model_identity_sha256 is not None
        and model_identity["identity_sha256"]
        != args.expected_model_identity_sha256
    ):
        raise RuntimeError("resident unified model identity differs from campaign")
    started = time.time()
    deadline = started + args.max_minutes * 60.0
    with standalone_model_lane(
        owner_id=f"train-unified-intrinsic:{out_dir.name}",
        model_path=args.model,
        purpose=_model_lane_purpose(args.window_tissue_mode),
        preemptible=False,
        require_exclusive=args.exclusive_model_lane,
        allow_owner_eviction=False,
        metadata={
            "tool": "train_unified_intrinsic_recurrence",
            "operator_launched": True,
            "window_tissue_mode": args.window_tissue_mode,
        },
    ), mlx_memory_envelope(
        fraction=args.memory_fraction,
        memory_gb=args.memory_limit_gb,
        cache_gb=args.cache_limit_gb,
        wired_gb=args.wired_limit_gb,
        restore_limits_on_exit=False,
    ) as envelope:
        mx.random.seed(args.init_seed)
        model, tokenizer = load(args.model)
        model.freeze()
        tokenizer_identity = resident_bootstrap_tokenizer_identity(
            Path(args.model),
            tokenizer,
        )
        if args.tokenized_dataset is not None:
            tokenized_path = args.tokenized_dataset.expanduser().resolve(strict=True)
            tokenized_dataset_identity = verify_tokenized_dataset(
                tokenized_path,
                tokenizer,
                train_tasks,
                holdout,
                bridge=bridge,
                dataset_identity=dataset_identity,
                tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
            )
        else:
            tokenized_dataset_identity = freeze_tokenized_dataset(
                out_dir / TOKENIZED_DATASET_FILENAME,
                tokenizer,
                train_tasks,
                holdout,
                bridge=bridge,
                dataset_identity=dataset_identity,
                tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
            )
        literal_digit_ids = tokenizer_digit_token_ids(tokenizer)
        literal_contract = LiteralObservationContract(literal_digit_ids)
        opcode_contract = tokenizer_opcode_contract(tokenizer)
        answer_emission_contract = tokenizer_answer_emission_contract(
            tokenizer,
            opcode_contract,
        )
        wiring = _configure_window_tissue(
            model,
            spec,
            mode=args.window_tissue_mode,
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
                literal_digit_token_ids=literal_digit_ids,
                opcode_token_patterns=opcode_contract.patterns,
                opcode_context_patterns=opcode_contract.contexts,
            )
        )
        state_codebook_grounding = _ground_state_value_embeddings(
            model,
            tokenizer,
            controller,
            prelude_end=spec.prelude_end,
            batch_size=args.grounding_batch_size,
        )
        initial_controller_sha256 = controller.parameter_sha256()
        bundle = UnifiedTrainingBundle(model, controller)
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
        identity = {
            "schema": TRAINING_SCHEMA,
            "model": model_identity,
            "runtime": runtime_identity,
            "dataset": dataset_identity,
            "tokenizer": tokenizer_identity,
            "tokenized_dataset": tokenized_dataset_identity,
            "spec": spec.to_dict(),
            "window_geometry": window_geometry,
            "families": list(families),
            "task_depths": list(task_depths),
            "per_cell": args.per_cell,
            "holdout_per_cell": args.holdout_per_cell,
            "seed": args.seed,
            "init_seed": args.init_seed,
            "semantic_warmup_steps": args.semantic_warmup_steps,
            "state_warmup_steps": args.state_warmup_steps,
            "answer_bridge_steps": args.answer_bridge_steps,
            "answer_bridge_inner_steps": args.answer_bridge_inner_steps,
            "max_steps": args.max_steps,
            "answer_bridge_rollin_probability": (
                args.answer_bridge_rollin_probability
            ),
            "answer_bridge_rollin_final_probability": (
                args.answer_bridge_rollin_final_probability
            ),
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
            "answer_bridge_learning_rate": answer_bridge_learning_rate,
            "recurrent_learning_rate": recurrent_learning_rate,
            "state_learning_rate": state_learning_rate,
            "bridge": args.bridge,
            "window_tissue_mode": args.window_tissue_mode,
            "lora_rank": args.lora_rank,
            "controller_rank": args.controller_rank,
            "state_weight": args.state_weight,
            "stutter_weight": args.stutter_weight,
            "state_codebook": (
                "frozen_prelude_state_action_and_tokenizer_literal_labels"
            ),
            "state_codebook_sha256": state_codebook_grounding["sha256"],
            "state_codebook_grounding": state_codebook_grounding,
            "initial_controller_sha256": initial_controller_sha256,
            "literal_observation_contract": {
                **literal_contract.to_dict(),
                "contract_sha256": literal_contract.contract_sha256,
            },
            "opcode_observation_contract": {
                **opcode_contract.to_dict(),
                "contract_sha256": opcode_contract.contract_sha256,
            },
            "answer_emission_contract": {
                **answer_emission_contract.to_dict(),
                "contract_sha256": answer_emission_contract.contract_sha256,
            },
            "depth_basis_size": args.depth_basis_size,
            "lora_targets": list(targets),
            "wiring": wiring,
            "readout_sha256": readout_sha256,
            "source_sha256s": source_sha256s,
            "campaign_binding": campaign_binding,
            "optimizer_contract": {
                "class": "mlx.optimizers.Adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "bias_correction": False,
                "phase_learning_rates": {
                    "semantic_anchor": args.learning_rate,
                    "answer_bridge": answer_bridge_learning_rate,
                    "state_transition": state_learning_rate,
                    "recurrence": recurrent_learning_rate,
                },
                "phase_transition_resets_optimizer_state": True,
            },
            "mlx_memory_envelope": envelope.to_receipt(),
        }
        identity["identity_sha256"] = _canonical_sha256(identity)
        def phase_learning_rate(phase: str) -> float:
            return {
                "semantic_anchor": args.learning_rate,
                "answer_bridge": answer_bridge_learning_rate,
                "state_transition": state_learning_rate,
                "recurrence": recurrent_learning_rate,
            }[phase]

        optimizer = _adam(
            phase_learning_rate(
                _optimization_phase(
                    0,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                )
            )
        )
        should_restore = args.resume or args.resume_if_available
        step, history, restored_training_state = (
            _restore_checkpoint(
                out_dir,
                bundle,
                optimizer,
                identity,
                semantic_warmup_steps=args.semantic_warmup_steps,
                state_warmup_steps=args.state_warmup_steps,
                answer_bridge_steps=args.answer_bridge_steps,
                required=args.resume,
            )
            if should_restore
            else (0, [], {})
        )
        rollin_totals = _restore_rollin_totals(restored_training_state)
        invocation_start_step = step
        invocation_stop_step = _invocation_stop_step(
            invocation_start_step,
            args.max_steps,
            args.max_invocation_steps,
        )
        optimizer.learning_rate = phase_learning_rate(
            _optimization_phase(
                step,
                args.semantic_warmup_steps,
                args.state_warmup_steps,
                args.answer_bridge_steps,
            )
        )
        mx.eval(optimizer.learning_rate)
        resource_guard_receipt: dict[str, Any] | None = None
        if resource_guard_enabled:
            resource_guard_receipt = _await_resource_guard(
                args.resource_stage_path.expanduser(),
                trainer_sha256=_file_sha256(Path(__file__).resolve(strict=True)),
                startup_lethal_mb=float(args.resource_startup_lethal_mb),
                steady_lethal_mb=float(args.resource_steady_lethal_mb),
                timeout_s=float(args.resource_guard_timeout_s),
            )
        print(
            f"[unified] step={step} trainable={sum(v.size for v in _trainable(bundle).values()):,} "
            f"readout={readout_sha256[:12]}",
            flush=True,
        )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        with checkpointed_window(model, group_size=args.checkpoint_group):
            while step < invocation_stop_step and time.time() < deadline:
                phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                )
                if phase == "answer_bridge":
                    bridge_start = (
                        args.state_warmup_steps + args.semantic_warmup_steps
                    )
                    task = _answer_bridge_task(train_tasks, step - bridge_start)
                else:
                    task = train_tasks[step % len(train_tasks)]
                prompt, answer = encode_example(tokenizer, task, bridge)
                with recurrence_adapter_scope(start=None, stop=None):
                    update_applied = False
                    if phase == "answer_bridge" and args.answer_bridge_inner_steps > 1:
                        semantic_depth = _semantic_execution_depth(task.depth, spec)
                        binding_targets = _answer_role_place_targets(
                            task.family,
                            answer,
                            answer_emission_contract,
                        )
                        features = _cached_answer_binding_features(
                            bundle,
                            prompt,
                            answer,
                            spec.plan_at(semantic_depth),
                        )
                        for _inner_step in range(args.answer_bridge_inner_steps):
                            loss, gradients = nn.value_and_grad(
                                bundle,
                                _cached_answer_binding_loss,
                            )(bundle, features, binding_targets)
                            _apply_training_gradients(
                                bundle,
                                optimizer,
                                gradients,
                                phase=phase,
                                max_norm=args.max_gradient_norm,
                                totals=rollin_totals,
                                loss=loss,
                            )
                        rollin_totals["answer_bridge_inner_updates"] += (
                            args.answer_bridge_inner_steps
                        )
                        update_applied = True
                    elif phase in {"semantic_anchor", "answer_bridge"}:
                        semantic_depth = _semantic_execution_depth(task.depth, spec)
                        effective = None
                        binding_targets = None
                        if phase == "answer_bridge":
                            binding_targets = _answer_role_place_targets(
                                task.family,
                                answer,
                                answer_emission_contract,
                            )
                            bridge_start = (
                                args.state_warmup_steps + args.semantic_warmup_steps
                            )
                            bridge_stop = bridge_start + args.answer_bridge_steps
                            rollin_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=bridge_start,
                                max_steps=bridge_stop,
                                initial=args.answer_bridge_rollin_probability,
                                final=args.answer_bridge_rollin_final_probability,
                            )
                            generated = _generate_student_rollin(
                                bundle,
                                prompt,
                                answer,
                                spec.plan_at(semantic_depth),
                                eos_token_id=tokenizer.eos_token_id,
                                answer_emission_contract=answer_emission_contract,
                                state_slot_start=int(prompt.shape[-1]),
                            )
                            effective, selected = _deterministic_student_mix(
                                answer,
                                generated,
                                probability=rollin_probability,
                                seed=args.seed * 1_000_003 + step,
                                interchangeable_token_ids=frozenset(
                                    answer_emission_contract.digit_token_ids
                                ),
                            )
                            _record_student_rollin(
                                rollin_totals,
                                answer,
                                generated,
                                effective,
                                selected,
                                rollin_probability,
                            )

                        def semantic_objective(
                            candidate: UnifiedTrainingBundle,
                            objective_prompt: Any,
                            objective_answer: Any,
                            objective_depth: int = semantic_depth,
                            objective_rollin: Any | None = effective,
                            objective_binding_targets: Any = binding_targets,
                        ):
                            role_logits: list[Any] = []
                            place_logits: list[Any] = []
                            _recurrent, _states, losses, _state_logits = (
                                unified_answer_and_recurrent_trajectory(
                                    candidate.model,
                                    objective_prompt,
                                    objective_answer,
                                    spec.plan_at(objective_depth),
                                    candidate.controller,
                                    decoder_input_tokens=objective_rollin,
                                    use_state_slots=True,
                                    answer_role_logit_trajectory=role_logits,
                                    answer_place_logit_trajectory=place_logits,
                                )
                            )
                            if objective_binding_targets is None:
                                return losses[-1]
                            if not role_logits or len(role_logits) != len(place_logits):
                                raise RuntimeError(
                                    "answer bridge emitted no binding trajectory"
                                )
                            role_targets, place_targets = objective_binding_targets
                            binding_loss = _answer_binding_loss(
                                role_logits[-1],
                                place_logits[-1],
                                role_targets,
                                place_targets,
                            )
                            return losses[-1] + binding_loss

                        loss, gradients = nn.value_and_grad(
                            bundle,
                            semantic_objective,
                        )(bundle, prompt, answer)
                    else:
                        if phase == "state_transition":
                            effective = answer
                            objective_spec = state_spec
                            state_teacher_probability = 1.0
                        else:
                            recurrent_start = (
                                args.semantic_warmup_steps
                                + args.state_warmup_steps
                                + args.answer_bridge_steps
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
                                answer_emission_contract=answer_emission_contract,
                                state_slot_start=int(prompt.shape[-1]),
                            )
                            effective, selected = _deterministic_student_mix(
                                answer,
                                generated,
                                probability=rollin_probability,
                                seed=args.seed * 1_000_003 + step,
                                interchangeable_token_ids=frozenset(
                                    answer_emission_contract.digit_token_ids
                                ),
                            )
                            _record_student_rollin(
                                rollin_totals,
                                answer,
                                generated,
                                effective,
                                selected,
                                rollin_probability,
                            )
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
                            transition_program: Any = task.transition_program,
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
                                transition_program=transition_program,
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
                    if not update_applied:
                        _apply_training_gradients(
                            bundle,
                            optimizer,
                            gradients,
                            phase=phase,
                            max_norm=args.max_gradient_norm,
                            totals=rollin_totals,
                            loss=loss,
                        )
                step += 1
                next_phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                )
                if next_phase != phase:
                    optimizer = _adam(phase_learning_rate(next_phase))
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
                    report["student_rollin"] = _rollin_report(
                        rollin_totals,
                        initial_probability=args.student_rollin_probability,
                        final_probability=rollin_final_probability,
                    )
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
                            training_state={"rollin_totals": rollin_totals},
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
                            training_state={"rollin_totals": rollin_totals},
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
                        training_state={"rollin_totals": rollin_totals},
                    )
                envelope.reclaim(force=True)

        if (
            not history
            or int(history[-1].get("step", -1)) != step
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
                args.answer_bridge_steps,
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
                    args.answer_bridge_steps,
                ),
                training_state={"rollin_totals": rollin_totals},
            )
        final_readout = readout_fingerprint(model, spec.coda_start)
        if final_readout != readout_sha256:
            raise RuntimeError("unified training changed the frozen readout")
        final = history[-1] if history else None
        answer_bridge_admission = (
            _evaluate_answer_bridge_admission(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                answer_emission_contract,
            )
            if args.answer_bridge_steps > 0 and step >= args.max_steps
            else None
        )
        if answer_bridge_admission is not None and answer_bridge_admission["admitted"]:
            _save_checkpoint(
                out_dir,
                bundle,
                optimizer,
                step=step,
                history=history,
                identity=identity,
                stem="checkpoint_answer_bridge_admitted",
                optimization_phase=_optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                ),
                training_state={"rollin_totals": rollin_totals},
            )
        halt_reason = _training_halt_reason(
            step=step,
            max_steps=args.max_steps,
            invocation_stop_step=invocation_stop_step,
        )
        with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
            latest_checkpoint = _load_latest_checkpoint(out_dir, required=True)
        if latest_checkpoint is None:
            raise RuntimeError("unified recurrence final checkpoint is unavailable")
        checkpoint_receipt, checkpoint_weights_path = latest_checkpoint
        body = {
            "schema": TRAINING_SCHEMA,
            "identity": identity,
            "steps": step,
            "history": history,
            "final": final,
            "readout_sha256_before": readout_sha256,
            "readout_sha256_after": final_readout,
            "readout_frozen": True,
            "complete": step >= args.max_steps,
            "halt_reason": halt_reason,
            "invocation": {
                "start_step": invocation_start_step,
                "end_step": step,
                "max_invocation_steps": args.max_invocation_steps,
                "planned_stop_step": invocation_stop_step,
                "max_minutes": args.max_minutes,
                "preload_host_pressure": preload_host_pressure,
                "resource_guard": resource_guard_receipt,
            },
            "latest_checkpoint": {
                "step": checkpoint_receipt["step"],
                "optimization_phase": checkpoint_receipt["optimization_phase"],
                "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
                "checkpoint_size_bytes": checkpoint_weights_path.stat().st_size,
                "receipt_sha256": checkpoint_receipt["receipt_sha256"],
            },
            "elapsed_minutes": round((time.time() - started) / 60.0, 3),
            "answer_bridge_admission": answer_bridge_admission,
            "verdict": _training_verdict(
                complete=step >= args.max_steps,
                answer_bridge_admission=answer_bridge_admission,
                final=final,
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        _atomic_canonical_json(out_dir / "training_receipt.json", receipt)
        print(f"[verdict] {receipt['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
