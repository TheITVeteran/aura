"""Authoritative immutable checkpoint resolution for unified recurrence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import time
import uuid
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
SOURCE_MIGRATION_SCHEMA: Final = (
    "aura.unified_intrinsic.checkpoint_source_migration.v2"
)
CHECKPOINT_RETENTION_SCHEMA: Final = (
    "aura.unified_intrinsic.checkpoint_retention.v1"
)
CHECKPOINT_MIRROR_DEDUP_SCHEMA: Final = (
    "aura.unified_intrinsic.checkpoint_mirror_dedup.v1"
)
_POINTER_FILENAME: Final = re.compile(
    r"(checkpoint_[a-z][a-z0-9_]{0,63})_pointer\.json"
)


class UnifiedCheckpointError(RuntimeError):
    """An authoritative checkpoint generation is absent or inconsistent."""


_BOOTSTRAP_TOPOLOGY_FIELDS: Final = (
    "bridge",
    "window_tissue_mode",
    "lora_rank",
    "controller_rank",
    "state_codebook_sha256",
    "literal_observation_contract",
    "opcode_observation_contract",
    "answer_emission_contract",
    "depth_basis_size",
    "lora_targets",
    "readout_sha256",
)


def _model_tensor_identity(value: Any) -> dict[str, Any] | None:
    """Drop path aliases while retaining the immutable checkpoint identity."""

    if not isinstance(value, dict):
        return None
    weights = value.get("weights")
    if not isinstance(weights, list) or not weights:
        return None
    normalized: list[dict[str, Any]] = []
    for row in weights:
        if not isinstance(row, dict):
            return None
        size = row.get("size", row.get("size_bytes"))
        if (
            not isinstance(row.get("name"), str)
            or not isinstance(row.get("sha256"), str)
            or type(size) is not int
            or size < 1
        ):
            return None
        normalized.append(
            {
                "name": row["name"],
                "sha256": row["sha256"],
                "size": size,
            }
        )
    return {
        "config_sha256": value.get("config_sha256"),
        "weights": sorted(normalized, key=lambda row: row["name"]),
    }


def _recurrent_window_identity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "prelude_end": value.get("prelude_end"),
        "coda_start": value.get("coda_start"),
    }


def bootstrap_topology_mismatches(
    parent_identity: Any,
    child_identity: Any,
) -> tuple[str, ...]:
    """Compare tissue topology while permitting a deliberate new curriculum.

    Dataset, task family, objective weights, natural trace depths, canonical
    path aliases, and optimizer history belong to the new campaign. The model
    tensors, recurrent window, controller schema, tokenizer-grounded contracts,
    and frozen readout must remain identical.
    """

    if not isinstance(parent_identity, dict) or not isinstance(child_identity, dict):
        return ("identity",)
    mismatches = [
        field
        for field in _BOOTSTRAP_TOPOLOGY_FIELDS
        if canonical_sha256(parent_identity.get(field))
        != canonical_sha256(child_identity.get(field))
    ]
    if canonical_sha256(_model_tensor_identity(parent_identity.get("model"))) != (
        canonical_sha256(_model_tensor_identity(child_identity.get("model")))
    ):
        mismatches.append("model_tensor_identity")
    if canonical_sha256(_recurrent_window_identity(parent_identity.get("spec"))) != (
        canonical_sha256(_recurrent_window_identity(child_identity.get("spec")))
    ):
        mismatches.append("recurrent_window")
    return tuple(mismatches)


@dataclass(frozen=True, slots=True)
class ResolvedUnifiedCheckpoint:
    receipt: dict[str, Any]
    weights_path: Path
    generation_dir: Path
    pointer: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CheckpointRetentionCandidate:
    name: str
    stem: str
    step: int
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class CheckpointRetentionPlan:
    output_dir: Path
    generation_root: Path
    rollback_generations_per_stem: int
    protected_generations: tuple[str, ...]
    rollback_generations: tuple[str, ...]
    staged_generations: tuple[str, ...]
    candidates: tuple[CheckpointRetentionCandidate, ...]


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


@dataclass(frozen=True, slots=True)
class _RetentionGeneration:
    candidate: CheckpointRetentionCandidate
    receipt: dict[str, Any]
    receipt_raw: bytes


def _retention_generation(path: Path) -> _RetentionGeneration:
    """Validate enough immutable structure to decide whether deletion is safe.

    This deliberately does not hash the tensor payload. Hashing every obsolete
    gigabyte on every checkpoint publication would turn retention into the
    dominant training cost. Pointer targets still bind the canonical receipt;
    unpointed candidates must have an intact signed receipt, exact file shape,
    owner custody, a read-only tensor and the declared byte count.
    """

    metadata = _directory_identity(path, modes=frozenset({0o500}))
    try:
        entries = tuple(path.iterdir())
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint generation is unreadable"
        ) from exc
    if any(entry.is_symlink() for entry in entries):
        raise UnifiedCheckpointError("unified checkpoint generation is a symlink")

    receipt, receipt_raw = _canonical_json(
        path / "complete.json",
        max_bytes=256 * 1024 * 1024,
    )
    try:
        complete = (path / "complete.json").lstat()
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint generation receipt is unavailable"
        ) from exc
    stem = receipt.get("stem")
    step = receipt.get("step")
    identity = receipt.get("identity")
    weights_name = receipt.get("checkpoint_file")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("schema") != TRAINING_SCHEMA
        or receipt.get("checkpoint_generation_schema")
        != CHECKPOINT_GENERATION_SCHEMA
        or receipt.get("checkpoint_id") != path.name
        or not isinstance(stem, str)
        or _validate_stem(stem) != stem
        or type(step) is not int
        or step < 0
        or path.name
        != f"{stem}-step-{step:08d}-{path.name.rsplit('-', 1)[-1]}"
        or not isinstance(identity, dict)
        or not isinstance(identity.get("identity_sha256"), str)
        or len(identity["identity_sha256"]) != 64
        or not isinstance(weights_name, str)
        or Path(weights_name).name != weights_name
        or receipt.get("receipt_sha256") != canonical_sha256(receipt_body)
    ):
        raise UnifiedCheckpointError("unified checkpoint generation differs")

    weights_path = path / weights_name
    try:
        weights = weights_path.lstat()
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint generation weights are unavailable"
        ) from exc
    expected_entries = {"complete.json", weights_name}
    if (
        {entry.name for entry in entries} != expected_entries
        or not stat.S_ISREG(complete.st_mode)
        or complete.st_uid != os.geteuid()
        or stat.S_IMODE(complete.st_mode) != 0o400
        or not stat.S_ISREG(weights.st_mode)
        or weights.st_uid != os.geteuid()
        or stat.S_IMODE(weights.st_mode) != 0o400
        or type(receipt.get("checkpoint_size_bytes")) is not int
        or weights.st_size != receipt.get("checkpoint_size_bytes")
        or not isinstance(receipt.get("checkpoint_sha256"), str)
        or len(receipt["checkpoint_sha256"]) != 64
    ):
        raise UnifiedCheckpointError("unified checkpoint generation weights differ")

    return _RetentionGeneration(
        candidate=CheckpointRetentionCandidate(
            name=path.name,
            stem=stem,
            step=step,
            size_bytes=int(weights.st_size + len(receipt_raw)),
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            mtime_ns=int(metadata.st_mtime_ns),
        ),
        receipt=receipt,
        receipt_raw=receipt_raw,
    )


def checkpoint_retention_plan(
    output_dir: Path,
    *,
    rollback_generations_per_stem: int = 2,
) -> CheckpointRetentionPlan:
    """Plan deletion only from complete generations no pointer can reach.

    Every pointer target is protected, irrespective of stem. Two additional
    complete generations per stem are retained by default for rollback. A
    staged generation is left alone. A symlink, unknown name, malformed
    pointer, incomplete generation, or custody mismatch refuses the whole plan
    before a single path becomes eligible.
    """

    if (
        isinstance(rollback_generations_per_stem, bool)
        or not isinstance(rollback_generations_per_stem, int)
        or not 0 <= rollback_generations_per_stem <= 64
    ):
        raise ValueError("rollback generation retention must be between 0 and 64")
    expanded = output_dir.expanduser()
    if expanded.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint output is a symlink")
    output = expanded.resolve(strict=True)
    _directory_identity(output, modes=frozenset({0o700}))
    generation_root_path = output / "checkpoint_generations"
    if generation_root_path.is_symlink():
        raise UnifiedCheckpointError(
            "unified recurrence checkpoint generation is a symlink"
        )
    if not generation_root_path.exists():
        return CheckpointRetentionPlan(
            output_dir=output,
            generation_root=generation_root_path,
            rollback_generations_per_stem=rollback_generations_per_stem,
            protected_generations=(),
            rollback_generations=(),
            staged_generations=(),
            candidates=(),
        )
    generation_root = generation_root_path.resolve(strict=True)
    _directory_identity(generation_root, modes=frozenset({0o700}))

    generations: dict[str, _RetentionGeneration] = {}
    staged: list[str] = []
    try:
        generation_entries = sorted(generation_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint generation inventory is unreadable"
        ) from exc
    if len(generation_entries) > MAX_GENERATION_ENTRIES:
        raise UnifiedCheckpointError(
            "unified checkpoint generation inventory is unbounded"
        )
    for candidate in generation_entries:
        if candidate.is_symlink():
            raise UnifiedCheckpointError("unified checkpoint generation is a symlink")
        if _STAGING_ID.fullmatch(candidate.name):
            _directory_identity(candidate, modes=frozenset({0o500, 0o700}))
            staged.append(candidate.name)
            continue
        if _CHECKPOINT_ID.fullmatch(candidate.name) is None:
            raise UnifiedCheckpointError("unified checkpoint generation name differs")
        generations[candidate.name] = _retention_generation(candidate)

    protected: set[str] = set()
    pointer_count = 0
    for pointer_path in sorted(output.glob("checkpoint_*_pointer.json")):
        pointer_count += 1
        if pointer_count > 256:
            raise UnifiedCheckpointError("unified checkpoint pointer inventory is unbounded")
        match = _POINTER_FILENAME.fullmatch(pointer_path.name)
        if match is None or pointer_path.is_symlink():
            raise UnifiedCheckpointError("unified recurrence checkpoint pointer differs")
        stem = _validate_stem(match.group(1))
        pointer, _pointer_raw = _canonical_json(pointer_path, max_bytes=64 * 1024)
        relative = pointer.get("checkpoint")
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
            or not isinstance(relative, str)
        ):
            raise UnifiedCheckpointError("unified recurrence checkpoint pointer differs")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.parts[:1] != ("checkpoint_generations",)
            or len(relative_path.parts) != 2
            or _CHECKPOINT_ID.fullmatch(relative_path.parts[1]) is None
        ):
            raise UnifiedCheckpointError(
                "unified recurrence checkpoint pointer path is invalid"
            )
        target = generations.get(relative_path.parts[1])
        if target is None:
            raise UnifiedCheckpointError(
                "unified recurrence checkpoint generation is unavailable"
            )
        receipt = target.receipt
        if (
            hashlib.sha256(target.receipt_raw).hexdigest()
            != pointer.get("complete_sha256")
            or receipt.get("stem") != stem
            or receipt.get("step") != pointer.get("step")
            or receipt.get("identity", {}).get("identity_sha256")
            != pointer.get("identity_sha256")
        ):
            raise UnifiedCheckpointError(
                "unified recurrence checkpoint generation differs"
            )
        protected.add(target.candidate.name)

    unpointed_by_stem: dict[str, list[CheckpointRetentionCandidate]] = {}
    for generation in generations.values():
        candidate = generation.candidate
        if candidate.name not in protected:
            unpointed_by_stem.setdefault(candidate.stem, []).append(candidate)

    rollback: set[str] = set()
    removable: list[CheckpointRetentionCandidate] = []
    for candidates in unpointed_by_stem.values():
        ordered = sorted(
            candidates,
            key=lambda candidate: (candidate.step, candidate.name),
            reverse=True,
        )
        rollback.update(
            candidate.name
            for candidate in ordered[:rollback_generations_per_stem]
        )
        removable.extend(ordered[rollback_generations_per_stem:])
    removable.sort(key=lambda candidate: (candidate.stem, candidate.step, candidate.name))

    return CheckpointRetentionPlan(
        output_dir=output,
        generation_root=generation_root,
        rollback_generations_per_stem=rollback_generations_per_stem,
        protected_generations=tuple(sorted(protected)),
        rollback_generations=tuple(sorted(rollback)),
        staged_generations=tuple(sorted(staged)),
        candidates=tuple(removable),
    )


def _retention_plan_material(plan: CheckpointRetentionPlan) -> dict[str, Any]:
    return {
        "output_dir": str(plan.output_dir),
        "rollback_generations_per_stem": plan.rollback_generations_per_stem,
        "protected_generations": list(plan.protected_generations),
        "rollback_generations": list(plan.rollback_generations),
        "staged_generations": list(plan.staged_generations),
        "candidates": [
            {
                "name": candidate.name,
                "stem": candidate.stem,
                "step": candidate.step,
                "size_bytes": candidate.size_bytes,
                "device": candidate.device,
                "inode": candidate.inode,
                "mtime_ns": candidate.mtime_ns,
            }
            for candidate in plan.candidates
        ],
    }


def _write_retention_receipt(
    path: Path,
    *,
    receipt_id: str,
    plan: CheckpointRetentionPlan,
    state: str,
    deleted: tuple[str, ...],
    error: str = "",
) -> dict[str, Any]:
    material = _retention_plan_material(plan)
    body = {
        "schema": CHECKPOINT_RETENTION_SCHEMA,
        "receipt_id": receipt_id,
        "state": state,
        "created_unix_ns": time.time_ns(),
        "plan_sha256": canonical_sha256(material),
        **material,
        "candidate_count": len(plan.candidates),
        "candidate_bytes": sum(candidate.size_bytes for candidate in plan.candidates),
        "deleted_generations": list(deleted),
        "error": error,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    from core.runtime.file_write_gateway import get_file_write_gateway

    get_file_write_gateway().write_text(
        path,
        (canonical_bytes(receipt) + b"\n").decode("ascii"),
        encoding="ascii",
        source="training.unified_checkpoint_retention",
    )
    return receipt


def prune_checkpoint_generations(
    output_dir: Path,
    *,
    rollback_generations_per_stem: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove pointer-unreachable generations and return an auditable receipt."""

    plan = checkpoint_retention_plan(
        output_dir,
        rollback_generations_per_stem=rollback_generations_per_stem,
    )
    material = _retention_plan_material(plan)
    if dry_run or not plan.candidates:
        body = {
            "schema": CHECKPOINT_RETENTION_SCHEMA,
            "state": "dry_run" if dry_run else "noop",
            "plan_sha256": canonical_sha256(material),
            **material,
            "candidate_count": len(plan.candidates),
            "candidate_bytes": sum(
                candidate.size_bytes for candidate in plan.candidates
            ),
            "deleted_generations": [],
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}

    # Validate every inode immediately before the first deletion so a changed
    # tree refuses as a whole instead of leaving a half-applied retention set.
    for candidate in plan.candidates:
        path = plan.generation_root / candidate.name
        observed = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or (
                observed.st_dev,
                observed.st_ino,
                observed.st_mtime_ns,
            )
            != (candidate.device, candidate.inode, candidate.mtime_ns)
        ):
            raise UnifiedCheckpointError(
                "unified checkpoint generation changed before retention"
            )

    from core.runtime.atomic_writer import ensure_private_directory
    from core.runtime.file_write_gateway import get_file_write_gateway

    receipt_id = uuid.uuid4().hex
    receipt_directory = ensure_private_directory(
        plan.output_dir / "checkpoint_retention_receipts"
    )
    quarantine_directory = ensure_private_directory(
        plan.output_dir / "checkpoint_retention_quarantine"
    )
    receipt_path = receipt_directory / f"retention-{receipt_id}.json"
    _write_retention_receipt(
        receipt_path,
        receipt_id=receipt_id,
        plan=plan,
        state="planned",
        deleted=(),
    )

    deleted: list[str] = []
    try:
        gateway = get_file_write_gateway()
        for candidate in plan.candidates:
            removed = gateway.delete_owned_readonly_tree(
                plan.generation_root / candidate.name,
                source="training.unified_checkpoint_retention",
                expected_device=candidate.device,
                expected_inode=candidate.inode,
                expected_mtime_ns=candidate.mtime_ns,
                quarantine_directory=quarantine_directory,
            )
            if not removed:
                raise UnifiedCheckpointError(
                    "unified checkpoint retention candidate disappeared"
                )
            deleted.append(candidate.name)
    except (OSError, RuntimeError) as exc:
        _write_retention_receipt(
            receipt_path,
            receipt_id=receipt_id,
            plan=plan,
            state="failed",
            deleted=tuple(deleted),
            error=f"{type(exc).__name__}:{exc}",
        )
        raise
    receipt = _write_retention_receipt(
        receipt_path,
        receipt_id=receipt_id,
        plan=plan,
        state="complete",
        deleted=tuple(deleted),
    )
    return {**receipt, "receipt_path": str(receipt_path)}


def deduplicate_checkpoint_compatibility_mirrors(
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Point fixed-name compatibility files at authoritative generations.

    Every pointer and immutable generation is fully verified before any mirror
    changes. Fixed-name files are explicitly non-authoritative; replacing a
    separate copy with a hard link preserves legacy readers while reclaiming
    the duplicate allocation. No generation, pointer, receipt, or rollback is
    deleted.
    """

    expanded = output_dir.expanduser()
    if expanded.is_symlink():
        raise UnifiedCheckpointError("unified checkpoint output is a symlink")
    output = expanded.resolve(strict=True)
    _directory_identity(output, modes=frozenset({0o700}))

    candidates: list[dict[str, Any]] = []
    for pointer_path in sorted(output.glob("checkpoint_*_pointer.json")):
        match = _POINTER_FILENAME.fullmatch(pointer_path.name)
        if match is None or pointer_path.is_symlink():
            raise UnifiedCheckpointError(
                "unified recurrence checkpoint pointer differs"
            )
        stem = _validate_stem(match.group(1))
        resolved = resolve_checkpoint_generation(output, stem=stem, required=True)
        if resolved is None:  # pragma: no cover - required=True is exhaustive
            raise UnifiedCheckpointError(
                "unified recurrence checkpoint generation is unavailable"
            )
        source = resolved.weights_path
        source_identity = source.lstat()
        target = output / f"{stem}.safetensors"
        target_identity: os.stat_result | None = None
        already_linked = False
        if target.is_symlink():
            raise UnifiedCheckpointError(
                "unified checkpoint compatibility mirror is a symlink"
            )
        if target.exists():
            target_identity = target.lstat()
            if (
                not stat.S_ISREG(target_identity.st_mode)
                or target_identity.st_uid != os.geteuid()
            ):
                raise UnifiedCheckpointError(
                    "unified checkpoint compatibility mirror differs"
                )
            already_linked = (
                target_identity.st_dev == source_identity.st_dev
                and target_identity.st_ino == source_identity.st_ino
            )
        candidates.append(
            {
                "stem": stem,
                "source": source,
                "source_device": int(source_identity.st_dev),
                "source_inode": int(source_identity.st_ino),
                "target": target,
                "target_device": (
                    int(target_identity.st_dev) if target_identity is not None else None
                ),
                "target_inode": (
                    int(target_identity.st_ino) if target_identity is not None else None
                ),
                "target_mtime_ns": (
                    int(target_identity.st_mtime_ns)
                    if target_identity is not None
                    else None
                ),
                "reclaimable_bytes": (
                    int(target_identity.st_blocks * 512)
                    if target_identity is not None and not already_linked
                    else 0
                ),
                "already_linked": already_linked,
            }
        )

    material = {
        "schema": CHECKPOINT_MIRROR_DEDUP_SCHEMA,
        "output_dir": str(output),
        "mirrors": [
            {
                "stem": row["stem"],
                "source": str(Path(row["source"]).relative_to(output)),
                "target": str(Path(row["target"]).relative_to(output)),
                "reclaimable_bytes": row["reclaimable_bytes"],
                "already_linked": row["already_linked"],
            }
            for row in candidates
        ],
    }
    plan_sha256 = canonical_sha256(material)
    if dry_run:
        body = {
            **material,
            "state": "dry_run",
            "plan_sha256": plan_sha256,
            "replaced": [],
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}

    replaced: list[str] = []
    receipt_id = uuid.uuid4().hex
    receipt_directory = output / "checkpoint_storage_receipts"
    from core.runtime.atomic_writer import (
        atomic_hardlink_replace,
        atomic_write_text,
        ensure_private_directory,
    )

    ensure_private_directory(receipt_directory)
    receipt_path = receipt_directory / f"mirror-dedup-{receipt_id}.json"
    state = "failed"
    error = "deduplication_interrupted"
    try:
        for row in candidates:
            if row["already_linked"]:
                continue
            source = Path(row["source"])
            target = Path(row["target"])
            source_now = source.lstat()
            if (
                source_now.st_dev != row["source_device"]
                or source_now.st_ino != row["source_inode"]
            ):
                raise UnifiedCheckpointError(
                    "unified checkpoint generation changed before mirror deduplication"
                )
            if row["target_inode"] is not None:
                target_now = target.lstat()
                if (
                    target_now.st_dev != row["target_device"]
                    or target_now.st_ino != row["target_inode"]
                    or target_now.st_mtime_ns != row["target_mtime_ns"]
                ):
                    raise UnifiedCheckpointError(
                        "unified checkpoint mirror changed before deduplication"
                    )
            elif target.exists() or target.is_symlink():
                raise UnifiedCheckpointError(
                    "unified checkpoint mirror appeared before deduplication"
                )
            atomic_hardlink_replace(source, target)
            replaced.append(str(row["stem"]))
        state = "complete"
        error = ""
    except (OSError, RuntimeError) as exc:
        state = "failed"
        error = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        body = {
            **material,
            "state": state,
            "plan_sha256": plan_sha256,
            "replaced": replaced,
            "error": error,
        }
        receipt = {**body, "receipt_sha256": canonical_sha256(body)}
        atomic_write_text(
            receipt_path,
            (canonical_bytes(receipt) + b"\n").decode("ascii"),
            encoding="ascii",
            mode=0o600,
        )
    return {**receipt, "receipt_path": str(receipt_path)}


def adopt_source_migration_identity(
    output_dir: Path,
    computed_identity: dict[str, Any],
) -> dict[str, Any]:
    """Verify and adopt an exact source-only resume identity when present.

    A migrated run retains the original experiment initialization while its
    operational bootstrap is the current resume tissue. The normal bootstrap
    path cannot represent both facts, so this verifier adopts the stored
    migration identity only after independently checking both checkpoints and
    every non-transport identity field.
    """

    output = output_dir.expanduser().resolve(strict=True)
    migration_path = output.parent / "checkpoint-source-migration.json"
    if not migration_path.exists() and not migration_path.is_symlink():
        return computed_identity
    migration, _raw = _canonical_json(migration_path, max_bytes=4 * 1024 * 1024)
    material = {
        key: value for key, value in migration.items() if key != "migration_sha256"
    }
    source = migration.get("source")
    destination = migration.get("destination")
    if (
        migration.get("schema") != SOURCE_MIGRATION_SCHEMA
        or migration.get("state") != "complete"
        or migration.get("migration_sha256") != canonical_sha256(material)
        or not isinstance(source, dict)
        or not isinstance(destination, dict)
        or migration.get("payload_byte_identical") is not True
        or migration.get("optimizer_and_bundle_bytes_preserved") is not True
        or migration.get("history_preserved") is not True
        or migration.get("training_state_preserved") is not True
        or migration.get("scientific_initialization_preserved") is not True
        or migration.get("training_profile_preserved") is not True
    ):
        raise UnifiedCheckpointError("unified checkpoint source migration differs")

    target = resolve_checkpoint_generation(output, required=True)
    if target is None:  # pragma: no cover - required=True is authoritative
        raise UnifiedCheckpointError("unified checkpoint source migration target missing")
    stored_identity = target.receipt.get("identity")
    campaign_binding = computed_identity.get("campaign_binding")
    if (
        not isinstance(stored_identity, dict)
        or not isinstance(campaign_binding, dict)
        or destination.get("campaign_id") != campaign_binding.get("campaign_id")
        or destination.get("config_sha256")
        != campaign_binding.get("campaign_config_sha256")
        or destination.get("generation") != target.generation_dir.name
        or destination.get("step") != target.receipt.get("step")
        or destination.get("checkpoint_sha256")
        != target.receipt.get("checkpoint_sha256")
        or destination.get("receipt_sha256") != target.receipt.get("receipt_sha256")
        or destination.get("identity_sha256")
        != stored_identity.get("identity_sha256")
    ):
        raise UnifiedCheckpointError("unified checkpoint source migration target differs")

    try:
        source_checkpoint = resolve_checkpoint_generation(
            Path(str(source["output"])), required=True
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise UnifiedCheckpointError(
            "unified checkpoint source migration source invalid"
        ) from exc
    if (
        source_checkpoint is None
        or source.get("generation") != source_checkpoint.generation_dir.name
        or source.get("step") != source_checkpoint.receipt.get("step")
        or source.get("checkpoint_sha256")
        != source_checkpoint.receipt.get("checkpoint_sha256")
        or source.get("receipt_sha256")
        != source_checkpoint.receipt.get("receipt_sha256")
        or source.get("identity_sha256")
        != source_checkpoint.receipt.get("identity", {}).get("identity_sha256")
        or source.get("checkpoint_sha256") != destination.get("checkpoint_sha256")
        or source.get("step") != destination.get("step")
    ):
        raise UnifiedCheckpointError("unified checkpoint source migration source differs")

    implementation = Path(__file__).with_name("migrate_unified_intrinsic_checkpoint.py")
    if _stable_file_identity(implementation, max_bytes=4 * 1024 * 1024)["sha256"] != (
        migration.get("migration_tool_sha256")
    ):
        raise UnifiedCheckpointError(
            "unified checkpoint source migration implementation differs"
        )

    expected = copy.deepcopy(computed_identity)
    expected.pop("identity_sha256", None)
    current_controller = expected.get("initial_controller_sha256")
    original_controller = stored_identity.get("initial_controller_sha256")
    original_bootstrap = stored_identity.get("bootstrap")
    if (
        not isinstance(current_controller, str)
        or len(current_controller) != 64
        or not isinstance(original_controller, str)
        or len(original_controller) != 64
        or (
            original_bootstrap is not None
            and not isinstance(original_bootstrap, dict)
        )
    ):
        raise UnifiedCheckpointError("unified checkpoint source migration origin differs")
    expected["initial_controller_sha256"] = original_controller
    expected["bootstrap"] = copy.deepcopy(original_bootstrap)
    expected["source_migration_controller_sha256"] = current_controller
    current_schedule = expected.get("phase_schedule")
    original_schedule = stored_identity.get("phase_schedule")
    if canonical_bytes(current_schedule) != canonical_bytes(original_schedule):
        current_schedule_body = copy.deepcopy(current_schedule)
        original_schedule_body = copy.deepcopy(original_schedule)
        if not isinstance(current_schedule_body, dict) or not isinstance(
            original_schedule_body, dict
        ):
            raise UnifiedCheckpointError(
                "unified checkpoint source migration phase schedule differs"
            )
        current_mode = current_schedule_body.pop("mode", None)
        original_mode = original_schedule_body.pop("mode", None)
        current_bootstrap = current_schedule_body.pop("bootstrap_required", None)
        original_bootstrap_required = original_schedule_body.pop(
            "bootstrap_required", None
        )
        if not (
            current_mode == "bootstrap_process_acquisition_only"
            and original_mode == "process_acquisition_only"
            and current_bootstrap is True
            and original_bootstrap_required is False
            and canonical_bytes(current_schedule_body)
            == canonical_bytes(original_schedule_body)
        ):
            raise UnifiedCheckpointError(
                "unified checkpoint source migration phase schedule differs"
            )
        # The imported checkpoint is an operational resume mechanism, not a new
        # scientific bootstrap. Preserve the phase schedule under which the
        # experiment actually began while requiring every phase boundary and
        # optimization field to remain byte-equivalent after canonicalization.
        expected["phase_schedule"] = copy.deepcopy(original_schedule)
    expected["identity_sha256"] = canonical_sha256(expected)
    if canonical_bytes(expected) != canonical_bytes(stored_identity):
        raise UnifiedCheckpointError("unified checkpoint source migration identity differs")
    return copy.deepcopy(stored_identity)


__all__ = [
    "CHECKPOINT_GENERATION_SCHEMA",
    "CHECKPOINT_MIRROR_DEDUP_SCHEMA",
    "CHECKPOINT_POINTER_SCHEMA",
    "CHECKPOINT_RETENTION_SCHEMA",
    "SOURCE_MIGRATION_SCHEMA",
    "TRAINING_SCHEMA",
    "ResolvedUnifiedCheckpoint",
    "UnifiedCheckpointError",
    "adopt_source_migration_identity",
    "checkpoint_retention_plan",
    "deduplicate_checkpoint_compatibility_mirrors",
    "prune_checkpoint_generations",
    "resolve_checkpoint_generation",
    "unpointed_checkpoint_inventory",
]
