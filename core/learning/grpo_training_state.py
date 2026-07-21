"""Crash-consistent, identity-bound checkpoints for GRPO training."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)

GRPO_CHECKPOINT_SCHEMA = "aura.grpo_checkpoint.v2"
GRPO_POINTER_SCHEMA = "aura.grpo_checkpoint_pointer.v1"
_HASH_KEYS = ("protocol_sha256", "dataset_sha256")
_REQUIRED_STATE_KEYS = {
    *_HASH_KEYS,
    "step",
    "curriculum",
    "telemetry",
    "history",
    "baseline_eval",
    "calibration",
    "elapsed_training_s",
    "invocation_count",
    "rng_strategy",
    "optimizer_updates",
    "last_step_kind",
    "last_step_committed",
}


class GRPOCheckpointError(RuntimeError):
    """Checkpoint bytes, identities, or exact resume state are invalid."""


@dataclass(frozen=True)
class LoadedGRPOCheckpoint:
    checkpoint_dir: Path
    state: dict[str, Any]
    adapter_tensors: dict[str, Any]
    optimizer_state: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeError) as exc:
        raise GRPOCheckpointError("checkpoint state is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_state(state: Mapping[str, Any]) -> None:
    missing = _REQUIRED_STATE_KEYS.difference(state)
    if missing:
        raise GRPOCheckpointError(
            "checkpoint exact state is incomplete: " + ", ".join(sorted(missing))
        )
    for key in _HASH_KEYS:
        if not _valid_sha256(state.get(key)):
            raise GRPOCheckpointError(f"checkpoint {key} is invalid")
    step = state.get("step")
    if type(step) is not int or step < 0:
        raise GRPOCheckpointError("checkpoint step is invalid")
    for key in ("curriculum", "telemetry"):
        if not isinstance(state.get(key), dict):
            raise GRPOCheckpointError(f"checkpoint {key} is invalid")
    history = state.get("history")
    if not isinstance(history, list) or any(
        not isinstance(entry, dict) for entry in history
    ):
        raise GRPOCheckpointError("checkpoint history is invalid")
    for key in ("baseline_eval", "calibration"):
        if state.get(key) is not None and not isinstance(state.get(key), dict):
            raise GRPOCheckpointError(f"checkpoint {key} is invalid")
    elapsed = state.get("elapsed_training_s")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise GRPOCheckpointError("checkpoint elapsed_training_s is invalid")
    invocation_count = state.get("invocation_count")
    if type(invocation_count) is not int or invocation_count < 1:
        raise GRPOCheckpointError("checkpoint invocation_count is invalid")
    optimizer_updates = state.get("optimizer_updates")
    if (
        type(optimizer_updates) is not int
        or optimizer_updates < 0
        or optimizer_updates > step
    ):
        raise GRPOCheckpointError("checkpoint optimizer update count is invalid")
    if state.get("rng_strategy") != "stateless_sha256_step_seeded_v1":
        raise GRPOCheckpointError("checkpoint rng strategy differs")
    last_step_kind = state.get("last_step_kind")
    allowed_step_kinds = {"initial", "optimizer_update", "degenerate_group"}
    if last_step_kind not in allowed_step_kinds:
        raise GRPOCheckpointError("checkpoint last step kind is invalid")
    if step == 0 and last_step_kind != "initial":
        raise GRPOCheckpointError("initial checkpoint has a non-initial step kind")
    if step > 0 and last_step_kind == "initial":
        raise GRPOCheckpointError("advanced checkpoint has an initial step kind")
    if type(state.get("last_step_committed")) is not bool:
        raise GRPOCheckpointError("checkpoint step commit flag is invalid")
    if step > 0 and state.get("last_step_committed") is not True:
        raise GRPOCheckpointError("checkpoint claims an incomplete training step")


def _write_safetensors(path: Path, tensors: Mapping[str, Any]) -> bytes:
    import mlx.core as mx

    if not tensors:
        raise GRPOCheckpointError("checkpoint tensor mapping is empty")
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
        value = json.loads(path.read_text(encoding="ascii"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise GRPOCheckpointError(f"{role} is unreadable") from exc
    if not isinstance(value, dict):
        raise GRPOCheckpointError(f"{role} must be a JSON object")
    return value


def _checkpoint_from_pointer(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative.startswith("checkpoints/"):
        raise GRPOCheckpointError("latest checkpoint path is invalid")
    checkpoint = (root / relative).resolve(strict=True)
    checkpoint_root = (root / "checkpoints").resolve(strict=True)
    if checkpoint.parent != checkpoint_root or not checkpoint.is_dir():
        raise GRPOCheckpointError("latest checkpoint escapes checkpoint root")
    return checkpoint


def _tensor_binding(
    checkpoint_dir: Path,
    state: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    import mlx.core as mx

    binding = state.get(role)
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise GRPOCheckpointError(f"checkpoint {role} binding differs")
    relative = binding.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise GRPOCheckpointError(f"checkpoint {role} path is invalid")
    path = (checkpoint_dir / relative).resolve(strict=True)
    if path.parent != checkpoint_dir:
        raise GRPOCheckpointError(f"checkpoint {role} path escapes")
    payload = path.read_bytes()
    if type(binding.get("size_bytes")) is not int or len(payload) != binding["size_bytes"]:
        raise GRPOCheckpointError(f"checkpoint {role} size mismatch")
    if not _valid_sha256(binding.get("sha256")) or sha256_bytes(payload) != binding["sha256"]:
        raise GRPOCheckpointError(f"checkpoint {role} digest mismatch")
    tensors = mx.load(str(path))
    if not isinstance(tensors, dict) or not tensors:
        raise GRPOCheckpointError(f"checkpoint {role} tensor container is invalid")
    return dict(tensors)


def _prune_generations(checkpoints: Path, *, keep: int, current: Path) -> None:
    generations = sorted(
        (path for path in checkpoints.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    retained = {path.resolve() for path in generations[:keep]}
    retained.add(current.resolve())
    for stale in generations:
        if stale.resolve() in retained:
            continue
        try:
            shutil.rmtree(stale)
        except OSError:
            # Pruning is capacity hygiene, not checkpoint publication. A
            # stale generation may remain, but latest never points to it.
            continue


def save_grpo_checkpoint(
    out_dir: str | Path,
    *,
    adapter_tensors: Mapping[str, Any],
    optimizer_tensors: Mapping[str, Any],
    state: Mapping[str, Any],
    keep: int = 3,
) -> Path:
    """Publish an immutable generation, then atomically advance latest."""
    if type(keep) is not int or keep < 1:
        raise GRPOCheckpointError("checkpoint keep must be a positive integer")
    _validate_state(state)
    root = ensure_private_directory(Path(out_dir).expanduser())
    checkpoints = ensure_private_directory(root / "checkpoints")
    with interprocess_file_lock(root / ".checkpoint.lock"):
        checkpoint_id = f"step-{int(state['step']):08d}-{uuid.uuid4().hex}"
        checkpoint_dir = ensure_private_directory(checkpoints / checkpoint_id)
        try:
            adapter_path = checkpoint_dir / "adapter.safetensors"
            optimizer_path = checkpoint_dir / "optimizer.safetensors"
            adapter_bytes = _write_safetensors(adapter_path, adapter_tensors)
            optimizer_bytes = _write_safetensors(optimizer_path, optimizer_tensors)
            complete = {
                **dict(state),
                "schema": GRPO_CHECKPOINT_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "created_unix": time.time(),
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
            complete_bytes = canonical_json_bytes(complete)
            atomic_write_bytes(checkpoint_dir / "complete.json", complete_bytes, mode=0o600)
            pointer = {
                "schema": GRPO_POINTER_SCHEMA,
                "checkpoint": f"checkpoints/{checkpoint_id}",
                "complete_sha256": sha256_bytes(complete_bytes),
            }
            atomic_write_text(
                root / "latest.json",
                canonical_json_bytes(pointer).decode("ascii"),
                encoding="ascii",
                mode=0o600,
            )
        except Exception:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            raise
        _prune_generations(checkpoints, keep=keep, current=checkpoint_dir)
        return checkpoint_dir


def load_grpo_checkpoint(
    out_dir: str | Path,
    *,
    expected_protocol_sha256: str,
    expected_dataset_sha256: str,
) -> LoadedGRPOCheckpoint:
    """Load only the complete generation bound to this protocol and data."""
    from mlx.utils import tree_unflatten

    root = Path(out_dir).expanduser().resolve(strict=True)
    with interprocess_file_lock(root / ".checkpoint.lock"):
        pointer = _read_json(root / "latest.json", role="latest checkpoint pointer")
        if set(pointer) != {"schema", "checkpoint", "complete_sha256"}:
            raise GRPOCheckpointError("latest checkpoint pointer schema differs")
        if pointer.get("schema") != GRPO_POINTER_SCHEMA:
            raise GRPOCheckpointError("latest checkpoint pointer version differs")
        checkpoint_dir = _checkpoint_from_pointer(root, pointer.get("checkpoint"))
        complete_bytes = (checkpoint_dir / "complete.json").read_bytes()
        if not _valid_sha256(pointer.get("complete_sha256")) or sha256_bytes(
            complete_bytes
        ) != pointer.get("complete_sha256"):
            raise GRPOCheckpointError("checkpoint completion digest mismatch")
        state = _read_json(
            checkpoint_dir / "complete.json", role="checkpoint completion record"
        )
        if state.get("schema") != GRPO_CHECKPOINT_SCHEMA:
            raise GRPOCheckpointError("checkpoint schema differs")
        if state.get("checkpoint_id") != checkpoint_dir.name:
            raise GRPOCheckpointError("checkpoint identity differs")
        _validate_state(state)
        expected = {
            "protocol_sha256": expected_protocol_sha256,
            "dataset_sha256": expected_dataset_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise GRPOCheckpointError(f"checkpoint {key} mismatch")
        adapters = _tensor_binding(checkpoint_dir, state, "adapter")
        optimizer_tensors = _tensor_binding(checkpoint_dir, state, "optimizer")
        optimizer_state = tree_unflatten(optimizer_tensors)
        if not isinstance(optimizer_state, dict):
            raise GRPOCheckpointError("optimizer checkpoint tree is invalid")
        return LoadedGRPOCheckpoint(
            checkpoint_dir=checkpoint_dir,
            state=state,
            adapter_tensors=adapters,
            optimizer_state=optimizer_state,
        )


__all__ = [
    "GRPO_CHECKPOINT_SCHEMA",
    "GRPO_POINTER_SCHEMA",
    "GRPOCheckpointError",
    "LoadedGRPOCheckpoint",
    "canonical_json_bytes",
    "load_grpo_checkpoint",
    "save_grpo_checkpoint",
    "sha256_bytes",
]
