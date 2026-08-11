"""Authoritative immutable checkpoint resolution for unified recurrence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256

TRAINING_SCHEMA: Final = "aura.unified_intrinsic_training.v1"
CHECKPOINT_GENERATION_SCHEMA: Final = "aura.unified_intrinsic_checkpoint.v3"
CHECKPOINT_POINTER_SCHEMA: Final = "aura.unified_intrinsic_checkpoint_pointer.v2"
_STEM: Final = re.compile(r"checkpoint_[a-z][a-z0-9_]{0,63}")
_CHECKPOINT_ID: Final = re.compile(
    r"checkpoint_[a-z][a-z0-9_]{0,63}-step-[0-9]{8}-[0-9a-f]{32}"
)
_STAGING_ID: Final = re.compile(r"\.checkpoint-stage-[0-9a-f]{32}")
MAX_GENERATION_ENTRIES: Final = 10_000


class UnifiedCheckpointError(RuntimeError):
    """An authoritative checkpoint generation is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class ResolvedUnifiedCheckpoint:
    receipt: dict[str, Any]
    weights_path: Path
    generation_dir: Path
    pointer: dict[str, Any]


def _stable_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint artifact is a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or not 0 < before.st_size <= max_bytes
            ):
                raise UnifiedCheckpointError(
                    "unified checkpoint artifact identity differs"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint artifact is unreadable"
        ) from exc
    payload = b"".join(chunks)
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise UnifiedCheckpointError("unified checkpoint artifact changed while read")
    return payload


def _canonical_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(path, max_bytes=max_bytes)
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedCheckpointError("unified checkpoint JSON is invalid") from exc
    if not isinstance(decoded, dict) or raw != canonical_bytes(decoded) + b"\n":
        raise UnifiedCheckpointError("unified checkpoint JSON is not canonical")
    return decoded, raw


def _stable_file_identity(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint artifact is a symlink")
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or not 0 < before.st_size <= max_bytes
            ):
                raise UnifiedCheckpointError(
                    "unified checkpoint artifact identity differs"
                )
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint artifact is unreadable"
        ) from exc
    if (
        remaining
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise UnifiedCheckpointError("unified checkpoint artifact changed while read")
    return {
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "mode": stat.S_IMODE(before.st_mode),
    }


def _validate_stem(stem: str) -> str:
    if not isinstance(stem, str) or _STEM.fullmatch(stem) is None:
        raise UnifiedCheckpointError("unified checkpoint stem is invalid")
    return stem


def _directory_identity(path: Path, *, modes: frozenset[int]) -> os.stat_result:
    if path.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint directory is a symlink")
    try:
        observed = path.stat()
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) not in modes
    ):
        raise UnifiedCheckpointError("unified checkpoint directory custody differs")
    return observed


def unpointed_checkpoint_inventory(output_dir: Path) -> dict[str, int]:
    """Validate but never promote generations left before a first pointer commit."""

    expanded = output_dir.expanduser()
    if expanded.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint output is a symlink")
    output = expanded.resolve(strict=True)
    _directory_identity(output, modes=frozenset({0o700}))
    pointer = output / "checkpoint_latest_pointer.json"
    if pointer.exists() or pointer.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint pointer already exists")
    generation_root = output / "checkpoint_generations"
    if not generation_root.exists() and not generation_root.is_symlink():
        return {"orphan_generations": 0, "staged_generations": 0}
    _directory_identity(generation_root, modes=frozenset({0o700}))
    orphan = 0
    staged = 0
    for count, candidate in enumerate(generation_root.iterdir(), start=1):
        if count > MAX_GENERATION_ENTRIES:
            raise UnifiedCheckpointError(
                "unified checkpoint generation inventory is unbounded"
            )
        if candidate.is_symlink():
            raise UnifiedCheckpointError("unified checkpoint generation is a symlink")
        if _CHECKPOINT_ID.fullmatch(candidate.name):
            _directory_identity(candidate, modes=frozenset({0o500}))
            orphan += 1
        elif _STAGING_ID.fullmatch(candidate.name):
            _directory_identity(candidate, modes=frozenset({0o500, 0o700}))
            staged += 1
        else:
            raise UnifiedCheckpointError("unified checkpoint generation name differs")
    return {"orphan_generations": orphan, "staged_generations": staged}


def resolve_checkpoint_generation(
    output_dir: Path,
    *,
    stem: str = "checkpoint_latest",
    required: bool = True,
) -> ResolvedUnifiedCheckpoint | None:
    """Resolve only the generation named by a validated atomic pointer."""

    stem = _validate_stem(stem)
    expanded_output = output_dir.expanduser()
    if expanded_output.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint output is a symlink")
    output_dir = expanded_output.resolve(strict=True)
    _directory_identity(output_dir, modes=frozenset({0o700}))
    pointer_path = output_dir / f"{stem}_pointer.json"
    if not pointer_path.exists():
        if required:
            raise UnifiedCheckpointError(
                f"unified recurrence {stem} checkpoint is unavailable"
            )
        return None
    pointer, pointer_raw = _canonical_json(pointer_path, max_bytes=64 * 1024)
    del pointer_raw
    if (
        set(pointer)
        != {
            "schema",
            "checkpoint",
            "complete_sha256",
            "identity_sha256",
            "step",
            "stem",
        }
        or pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA
        or pointer.get("stem") != stem
        or type(pointer.get("step")) is not int
        or int(pointer["step"]) < 0
    ):
        raise UnifiedCheckpointError("unified recurrence checkpoint pointer differs")
    relative = pointer.get("checkpoint")
    if not isinstance(relative, str):
        raise UnifiedCheckpointError("unified recurrence checkpoint pointer path is invalid")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.parts[:1] != ("checkpoint_generations",)
        or len(relative_path.parts) != 2
        or _CHECKPOINT_ID.fullmatch(relative_path.parts[1]) is None
    ):
        raise UnifiedCheckpointError("unified recurrence checkpoint pointer path is invalid")
    generation_root_path = output_dir / "checkpoint_generations"
    generation_path = output_dir / relative_path
    if generation_root_path.is_symlink() or generation_path.is_symlink():
        raise UnifiedCheckpointError(
            "unified recurrence checkpoint generation is a symlink"
        )
    generation_root = generation_root_path.resolve(strict=True)
    generation_dir = generation_path.resolve(strict=True)
    _directory_identity(generation_root, modes=frozenset({0o700}))
    _directory_identity(generation_dir, modes=frozenset({0o500}))
    if (
        generation_dir.parent != generation_root
        or generation_dir.name != relative_path.parts[1]
        or generation_dir.is_symlink()
        or not generation_dir.is_dir()
    ):
        raise UnifiedCheckpointError("unified recurrence checkpoint generation differs")
    receipt, receipt_raw = _canonical_json(
        generation_dir / "complete.json",
        max_bytes=256 * 1024 * 1024,
    )
    identity = receipt.get("identity") if isinstance(receipt, dict) else None
    if (
        hashlib.sha256(receipt_raw).hexdigest() != pointer.get("complete_sha256")
        or receipt.get("schema") != TRAINING_SCHEMA
        or receipt.get("checkpoint_generation_schema")
        != CHECKPOINT_GENERATION_SCHEMA
        or receipt.get("checkpoint_id") != generation_dir.name
        or receipt.get("step") != pointer.get("step")
        or receipt.get("stem") != stem
        or not isinstance(identity, dict)
        or identity.get("identity_sha256") != pointer.get("identity_sha256")
    ):
        raise UnifiedCheckpointError("unified recurrence checkpoint generation differs")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != canonical_sha256(receipt_body):
        raise UnifiedCheckpointError("unified recurrence checkpoint receipt differs")
    weights_name = receipt.get("checkpoint_file")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise UnifiedCheckpointError("unified recurrence checkpoint weight path is invalid")
    weights_path = generation_dir / weights_name
    weights = _stable_file_identity(
        weights_path,
        max_bytes=64 * 1024 * 1024 * 1024,
    )
    if (
        weights["size_bytes"] != receipt.get("checkpoint_size_bytes")
        or weights["sha256"] != receipt.get("checkpoint_sha256")
        or int(weights["mode"]) & 0o222
    ):
        raise UnifiedCheckpointError("unified recurrence checkpoint weights differ")
    return ResolvedUnifiedCheckpoint(
        receipt=receipt,
        weights_path=weights_path,
        generation_dir=generation_dir,
        pointer=pointer,
    )


__all__ = [
    "CHECKPOINT_GENERATION_SCHEMA",
    "CHECKPOINT_POINTER_SCHEMA",
    "TRAINING_SCHEMA",
    "ResolvedUnifiedCheckpoint",
    "UnifiedCheckpointError",
    "resolve_checkpoint_generation",
    "unpointed_checkpoint_inventory",
]
