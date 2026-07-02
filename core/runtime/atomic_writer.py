"""Canonical AtomicWriter — single durable-write gateway.

Postgres-grade durability requires that every persistent write is:

    write to a temp file in the same directory
    flush + fsync the file
    fsync the parent directory
    atomic rename over the target

Crash points (between any of those steps) must leave the target either
unchanged (old committed state) or fully replaced (new committed state).

This module exposes:

- atomic_write_bytes(path, payload)
- atomic_write_text(path, text)
- atomic_append_text(path, text)
- atomic_write_json(path, obj, schema_version)

with explicit schema-version envelopes so loaders can detect ancient
records and refuse rather than silently misread.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.AtomicWriter")

PathLike = str | Path

DEFAULT_TEMP_PREFIX = ".aura_atomic_"
_append_locks: dict[Path, threading.Lock] = {}
_append_locks_guard = threading.Lock()


class AtomicWriteError(RuntimeError):
    """Raised when an atomic write cannot complete."""


def _fsync_file(fd: int) -> None:
    try:
        os.fsync(fd)
    except (AttributeError, OSError):
        # Best-effort on platforms where fsync is unavailable.
        pass  # no-op: intentional


def _fsync_dir(directory: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except (FileNotFoundError, PermissionError, OSError):
        return
    try:
        _fsync_file(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: PathLike, payload: bytes, *, durable: bool = True) -> None:
    """Atomically replace `path` with `payload`.

    ``durable=False`` keeps the write atomic (readers never observe a torn
    file) but skips both fsyncs. Probe files, caches, and other content that
    is worthless after a crash must use it: under memory-pressure thrash a
    single fsync has blocked the live event loop for 20 minutes.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix=DEFAULT_TEMP_PREFIX, dir=str(parent))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            if durable:
                _fsync_file(fh.fileno())
        os.replace(tmp_path, target)
        if durable:
            _fsync_dir(parent)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass  # no-op: intentional
        raise


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8", durable: bool = True) -> None:
    atomic_write_bytes(path, text.encode(encoding), durable=durable)


def atomic_append_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Durably append text without reading or rewriting the existing target."""
    target = Path(path)
    with _append_locks_guard:
        lock = _append_locks.setdefault(target.resolve(), threading.Lock())
    with lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(str(target), flags, 0o600)
        except OSError as exc:
            raise AtomicWriteError(f"cannot open append target: {target}") from exc
        try:
            payload = text.encode(encoding)
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            _fsync_file(fd)
        except OSError as exc:
            raise AtomicWriteError(f"cannot append to target: {target}") from exc
        finally:
            os.close(fd)
        _fsync_dir(target.parent)


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    schema_version: int,
    schema_name: str | None = None,
    indent: int | None = 2,
) -> None:
    """Atomically write a JSON envelope `{schema, version, payload}`."""
    if not isinstance(schema_version, int) or schema_version < 1:
        raise AtomicWriteError("schema_version must be a positive int")
    target = Path(path)
    inferred_schema = schema_name or target.stem or target.name or "atomic_json"
    envelope = {
        "schema": inferred_schema,
        "schema_name": inferred_schema,
        "schema_version": schema_version,
        "payload": obj,
    }
    text = json.dumps(envelope, indent=indent, sort_keys=True, default=str)
    atomic_write_text(path, text)


def read_json_envelope(path: PathLike) -> dict[str, Any]:
    target = Path(path)
    raw = target.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "schema_version" not in data:
        raise AtomicWriteError(
            f"file at {target} is not a versioned envelope (missing schema_version)"
        )
    return data


def cleanup_partial_writes(directory: PathLike) -> int:
    """Remove leftover temp files from interrupted writes. Returns count."""
    parent = Path(directory)
    if not parent.exists():
        return 0
    removed = 0
    for child in parent.iterdir():
        if child.name.startswith(DEFAULT_TEMP_PREFIX):
            try:
                child.unlink()
                removed += 1
            except OSError:
                continue
    return removed
