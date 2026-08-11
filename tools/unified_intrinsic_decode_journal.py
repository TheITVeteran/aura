"""Crash-safe progress custody for resident decoded evaluations.

Each candidate is committed once under an immutable experiment identity.  A
later invocation may reuse only byte-equivalent, hash-valid records for the
same task and arm; partial, reordered, or cross-experiment state fails closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.canonical_json import canonical_json_bytes
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)

MANIFEST_SCHEMA = "aura.unified_intrinsic.decode_progress_manifest.v1"
RECORD_SCHEMA = "aura.unified_intrinsic.decode_progress_record.v1"


class DecodeProgressError(RuntimeError):
    """A progress artifact is malformed or belongs to another experiment."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _document(value: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    body = dict(value)
    if digest_key in body:
        raise DecodeProgressError(f"{digest_key} is reserved")
    return {**body, digest_key: _sha256(body)}


def _read_exact(path: Path, *, digest_key: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecodeProgressError(f"decode progress artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DecodeProgressError(f"decode progress artifact is not an object: {path}")
    if raw != canonical_json_bytes(value) + b"\n":
        raise DecodeProgressError(f"decode progress artifact is not canonical: {path}")
    body = {key: item for key, item in value.items() if key != digest_key}
    if value.get(digest_key) != _sha256(body):
        raise DecodeProgressError(f"decode progress artifact hash differs: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any], *, digest_key: str) -> None:
    payload = canonical_json_bytes(dict(value)) + b"\n"
    if atomic_write_bytes_if_absent(path, payload, mode=0o400):
        return
    if _read_exact(path, digest_key=digest_key) != dict(value):
        raise DecodeProgressError(f"decode progress artifact already differs: {path}")


class DecodeProgressJournal:
    """Write-once candidate journal bound to one decoded experiment."""

    def __init__(self, root: Path, experiment: Mapping[str, Any]) -> None:
        self.root = ensure_private_directory(root.expanduser().absolute())
        experiment_body = dict(experiment)
        self.experiment_sha256 = _sha256(experiment_body)
        self.manifest = _document(
            {
                "schema": MANIFEST_SCHEMA,
                "experiment": experiment_body,
                "experiment_sha256": self.experiment_sha256,
            },
            "manifest_sha256",
        )
        _write_once(
            self.root / "manifest.json",
            self.manifest,
            digest_key="manifest_sha256",
        )

    def _record_path(self, sequence: int) -> Path:
        if type(sequence) is not int or sequence < 0:
            raise DecodeProgressError("decode progress sequence must be nonnegative")
        return self.root / f"candidate-{sequence:08d}.json"

    def load(
        self,
        sequence: int,
        *,
        task_id: str,
        arm: str,
        prompt_sha256: str,
    ) -> dict[str, Any] | None:
        path = self._record_path(sequence)
        if not path.exists():
            return None
        record = _read_exact(path, digest_key="record_sha256")
        candidate = record.get("candidate")
        if (
            record.get("schema") != RECORD_SCHEMA
            or record.get("experiment_sha256") != self.experiment_sha256
            or record.get("sequence") != sequence
            or not isinstance(candidate, dict)
            or candidate.get("task_id") != task_id
            or candidate.get("arm") != arm
            or candidate.get("prompt_sha256") != prompt_sha256
        ):
            raise DecodeProgressError(f"decode progress record binding differs: {path}")
        return dict(candidate)

    def commit(self, sequence: int, candidate: Mapping[str, Any]) -> dict[str, Any]:
        task_id = candidate.get("task_id")
        arm = candidate.get("arm")
        prompt_sha256 = candidate.get("prompt_sha256")
        if not all(isinstance(value, str) and value for value in (task_id, arm)):
            raise DecodeProgressError("decode candidate identity is incomplete")
        if not (
            isinstance(prompt_sha256, str)
            and len(prompt_sha256) == 64
            and all(character in "0123456789abcdef" for character in prompt_sha256)
        ):
            raise DecodeProgressError("decode candidate prompt identity is invalid")
        record = _document(
            {
                "schema": RECORD_SCHEMA,
                "experiment_sha256": self.experiment_sha256,
                "sequence": sequence,
                "candidate": dict(candidate),
            },
            "record_sha256",
        )
        _write_once(
            self._record_path(sequence),
            record,
            digest_key="record_sha256",
        )
        return record


__all__ = ["DecodeProgressError", "DecodeProgressJournal"]
