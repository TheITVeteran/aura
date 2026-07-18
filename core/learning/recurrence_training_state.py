"""Crash-consistent checkpoints for recurrence-native v2 training."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)

TRAINING_CHECKPOINT_SCHEMA = "aura.recurrence_native_checkpoint.v2"
LATEST_POINTER_SCHEMA = "aura.recurrence_native_checkpoint_pointer.v1"


class RecurrenceCheckpointError(RuntimeError):
    """Checkpoint bytes or identities are incomplete, stale, or inconsistent."""


@dataclass(frozen=True)
class LoadedRecurrenceCheckpoint:
    checkpoint_dir: Path
    state: dict[str, Any]
    adapter_tensors: dict[str, Any]
    optimizer_state: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_safetensors(path: Path, tensors: Mapping[str, Any]) -> bytes:
    import mlx.core as mx

    if not tensors:
        raise RecurrenceCheckpointError("checkpoint tensor mapping is empty")
    scratch = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        payload = scratch.read_bytes()
    finally:
        durable_unlink(scratch, missing_ok=True)
    atomic_write_bytes(path, payload, mode=0o600)
    return payload


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise RecurrenceCheckpointError(f"{role} is unreadable") from exc
    if not isinstance(payload, dict):
        raise RecurrenceCheckpointError(f"{role} must be a JSON object")
    return payload


def _contained_checkpoint(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.startswith("checkpoints/"):
        raise RecurrenceCheckpointError("latest checkpoint path is invalid")
    candidate = (root / relative).resolve(strict=True)
    checkpoint_root = (root / "checkpoints").resolve(strict=True)
    if candidate.parent != checkpoint_root or not candidate.is_dir():
        raise RecurrenceCheckpointError("latest checkpoint escapes checkpoint root")
    return candidate


def save_recurrence_checkpoint(
    out_dir: str | Path,
    *,
    adapter_tensors: Mapping[str, Any],
    optimizer_tensors: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Path:
    """Publish an immutable generation, then atomically advance ``latest``."""

    root = ensure_private_directory(Path(out_dir).expanduser())
    checkpoints = ensure_private_directory(root / "checkpoints")
    lock_path = root / ".checkpoint.lock"
    with interprocess_file_lock(lock_path):
        checkpoint_id = f"step-{int(state.get('step', -1)):08d}-{uuid.uuid4().hex}"
        checkpoint_dir = ensure_private_directory(checkpoints / checkpoint_id)
        adapter_path = checkpoint_dir / "adapter.safetensors"
        optimizer_path = checkpoint_dir / "optimizer.safetensors"
        adapter_bytes = _write_safetensors(adapter_path, adapter_tensors)
        optimizer_bytes = _write_safetensors(optimizer_path, optimizer_tensors)
        complete = dict(state)
        complete.update(
            {
                "schema": TRAINING_CHECKPOINT_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "adapter": {
                    "path": adapter_path.name,
                    "sha256": sha256_bytes(adapter_bytes),
                    "size_bytes": len(adapter_bytes),
                },
                "optimizer": {
                    "path": optimizer_path.name,
                    "sha256": sha256_bytes(optimizer_bytes),
                    "size_bytes": len(optimizer_bytes),
                },
            }
        )
        complete_bytes = canonical_json_bytes(complete)
        atomic_write_bytes(
            checkpoint_dir / "complete.json",
            complete_bytes,
            mode=0o600,
        )
        pointer = {
            "schema": LATEST_POINTER_SCHEMA,
            "checkpoint": f"checkpoints/{checkpoint_id}",
            "complete_sha256": sha256_bytes(complete_bytes),
        }
        atomic_write_text(
            root / "latest.json",
            canonical_json_bytes(pointer).decode("ascii"),
            encoding="ascii",
            mode=0o600,
        )
        return cast(Path, checkpoint_dir)


def load_recurrence_checkpoint(
    out_dir: str | Path,
    *,
    expected_config_sha256: str,
    expected_dataset_sha256: str,
    expected_execution_spec_sha256: str,
) -> LoadedRecurrenceCheckpoint:
    """Load only the fully published generation named by ``latest.json``."""

    import mlx.core as mx
    from mlx.utils import tree_unflatten

    root = Path(out_dir).expanduser().resolve(strict=True)
    with interprocess_file_lock(root / ".checkpoint.lock"):
        pointer = _read_json(root / "latest.json", role="latest checkpoint pointer")
        if set(pointer) != {"schema", "checkpoint", "complete_sha256"}:
            raise RecurrenceCheckpointError("latest checkpoint pointer schema differs")
        if pointer["schema"] != LATEST_POINTER_SCHEMA:
            raise RecurrenceCheckpointError("latest checkpoint pointer version differs")
        checkpoint_dir = _contained_checkpoint(root, pointer["checkpoint"])
        complete_path = checkpoint_dir / "complete.json"
        complete_bytes = complete_path.read_bytes()
        if sha256_bytes(complete_bytes) != pointer["complete_sha256"]:
            raise RecurrenceCheckpointError("checkpoint completion digest mismatch")
        state = _read_json(complete_path, role="checkpoint completion record")
        if state.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
            raise RecurrenceCheckpointError("checkpoint schema differs")
        expected = {
            "config_sha256": expected_config_sha256,
            "dataset_sha256": expected_dataset_sha256,
            "execution_spec_sha256": expected_execution_spec_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RecurrenceCheckpointError(f"checkpoint {key} mismatch")

        tensors: dict[str, dict[str, Any]] = {}
        for role in ("adapter", "optimizer"):
            binding = state.get(role)
            if not isinstance(binding, dict) or set(binding) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise RecurrenceCheckpointError(f"checkpoint {role} binding differs")
            path = checkpoint_dir / str(binding["path"])
            if path.parent.resolve(strict=True) != checkpoint_dir:
                raise RecurrenceCheckpointError(f"checkpoint {role} path escapes")
            payload = path.read_bytes()
            if len(payload) != binding["size_bytes"]:
                raise RecurrenceCheckpointError(f"checkpoint {role} size mismatch")
            if sha256_bytes(payload) != binding["sha256"]:
                raise RecurrenceCheckpointError(f"checkpoint {role} digest mismatch")
            loaded_tensors = mx.load(str(path))
            if not isinstance(loaded_tensors, dict):
                raise RecurrenceCheckpointError(
                    f"checkpoint {role} tensor container is invalid"
                )
            tensors[role] = dict(loaded_tensors)
        optimizer_state = tree_unflatten(tensors["optimizer"])
        if not isinstance(optimizer_state, dict):
            raise RecurrenceCheckpointError("optimizer checkpoint tree is invalid")
        return LoadedRecurrenceCheckpoint(
            checkpoint_dir=checkpoint_dir,
            state=state,
            adapter_tensors=tensors["adapter"],
            optimizer_state=optimizer_state,
        )


__all__ = [
    "LATEST_POINTER_SCHEMA",
    "TRAINING_CHECKPOINT_SCHEMA",
    "LoadedRecurrenceCheckpoint",
    "RecurrenceCheckpointError",
    "canonical_json_bytes",
    "load_recurrence_checkpoint",
    "save_recurrence_checkpoint",
    "sha256_bytes",
]
