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
- durable_replace(source, target)
- durable_unlink(path)

with explicit schema-version envelopes so loaders can detect ancient
records and refuse rather than silently misread.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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


def ensure_private_directory(path: PathLike) -> Path:
    """Create a durability directory and restrict it to the current user."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    _fsync_dir(directory.parent)
    return directory


async def async_ensure_private_directory(path: PathLike) -> Path:
    import asyncio

    return await asyncio.to_thread(ensure_private_directory, path)


@contextmanager
def interprocess_file_lock(path: PathLike) -> Iterator[None]:
    """Serialize a multi-file transaction across threads and processes."""

    target = Path(path)
    ensure_private_directory(target.parent)
    fd = os.open(str(target), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


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


async def async_atomic_write_bytes(path: PathLike, payload: bytes, *, durable: bool = True) -> None:
    """Event-loop-safe atomic write: the fsync happens on a worker thread.

    Under memory-pressure thrash a single on-loop fsync has frozen the live
    event loop for ~20 minutes (12 recorded crashes). Async callers must use
    this lane; the sync functions are for threads and sync bootstrap only.
    """
    import asyncio

    await asyncio.to_thread(atomic_write_bytes, path, payload, durable=durable)


async def async_atomic_write_text(
    path: PathLike, text: str, *, encoding: str = "utf-8", durable: bool = True
) -> None:
    await async_atomic_write_bytes(path, text.encode(encoding), durable=durable)


async def async_atomic_append_text(
    path: PathLike, text: str, *, encoding: str = "utf-8"
) -> None:
    import asyncio

    await asyncio.to_thread(atomic_append_text, path, text, encoding=encoding)


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
            fcntl.flock(fd, fcntl.LOCK_EX)
            payload = text.encode(encoding)
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            _fsync_file(fd)
        except OSError as exc:
            raise AtomicWriteError(f"cannot append to target: {target}") from exc
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
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


async def async_atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    schema_version: int,
    schema_name: str | None = None,
    indent: int | None = 2,
) -> None:
    """Event-loop-safe atomic_write_json: serialization + fsync on a worker thread."""
    import asyncio

    await asyncio.to_thread(
        atomic_write_json,
        path,
        obj,
        schema_version=schema_version,
        schema_name=schema_name,
        indent=indent,
    )


def durable_replace(source: PathLike, target: PathLike) -> None:
    """Atomically move ``source`` over ``target`` and fsync both directories."""

    source_path = Path(source)
    target_path = Path(target)
    if not os.path.lexists(source_path):
        raise FileNotFoundError(f"durable replace source does not exist: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source_path, target_path)
    _fsync_dir(source_path.parent)
    if target_path.parent != source_path.parent:
        _fsync_dir(target_path.parent)


async def async_durable_replace(source: PathLike, target: PathLike) -> None:
    import asyncio

    await asyncio.to_thread(durable_replace, source, target)


def durable_unlink(path: PathLike, *, missing_ok: bool = False) -> bool:
    """Delete a file/symlink and fsync its parent directory."""

    target = Path(path)
    if not os.path.lexists(target):
        if missing_ok:
            return False
        raise FileNotFoundError(target)
    if target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(target)
    os.unlink(target)
    _fsync_dir(target.parent)
    return True


async def async_durable_unlink(path: PathLike, *, missing_ok: bool = False) -> bool:
    import asyncio

    return await asyncio.to_thread(durable_unlink, path, missing_ok=missing_ok)


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
