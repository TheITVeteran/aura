"""core/runtime/file_write_gateway.py — Canonical File Write Gateway.

All file writing operations should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import hashlib
import logging
import os
import stat
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.atomic_writer import (
    PathLike,
    atomic_append_text,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    durable_replace,
    durable_unlink,
    interprocess_file_lock,
)

logger = logging.getLogger("Aura.FileWriteGateway")
_FILE_WRITE_DOMAINS = (
    "file_write",
    "memory_write",
    "state_mutation",
    "self_modification",
    "tool_execution",
)


class FileWriteTransactionError(RuntimeError):
    """A multi-file gateway commit failed or could not be rolled back."""


@dataclass(frozen=True, slots=True)
class FileWriteBatchEntry:
    path: PathLike
    payload: bytes
    mode: int = 0o600


@dataclass(frozen=True, slots=True)
class FileWriteBatchReceipt:
    transaction_id: str
    paths: tuple[str, ...]
    sha256: tuple[tuple[str, str], ...]


def _validated_permissions(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("permissions must be an integer")
    if mode < 0 or mode & ~0o777:
        raise ValueError("permissions must contain only rwx permission bits")
    return mode


class FileWriteGateway:
    """Single canonical owner for filesystem write operations."""

    def __init__(self) -> None:
        self._allowed_domains = _FILE_WRITE_DOMAINS

    def ensure_directory(self, path: PathLike, *, source: str = "unknown") -> str:
        """Create a private directory through the governed filesystem lane."""

        directory = Path(path).expanduser()
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.ensure_directory:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import ensure_private_directory

        return str(ensure_private_directory(directory))

    async def ensure_directory_async(self, path: PathLike, *, source: str = "unknown") -> str:
        """Create a private directory off the event loop after inline governance."""

        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.ensure_directory:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_ensure_private_directory

        return str(await async_ensure_private_directory(path))

    def write_bytes(self, path: PathLike, payload: bytes, *, source: str = "unknown") -> None:
        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_write_bytes(target, bytes(payload))

    def write_bytes_batch(
        self,
        entries: Sequence[FileWriteBatchEntry],
        *,
        source: str = "unknown",
    ) -> FileWriteBatchReceipt:
        """Commit a same-directory file set with exception rollback.

        Each target replacement is independently crash-atomic and durable. If
        an ordinary write fails, already replaced targets are restored before
        the error escapes. A process/power loss can still occur between file
        replacements, so consumers of a coupled set must validate their mutual
        consistency on read before using it.
        """

        batch = tuple(entries)
        if not batch:
            raise ValueError("file batch must contain at least one entry")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes_batch:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        normalized: list[tuple[Path, bytes, int]] = []
        seen: set[Path] = set()
        parent: Path | None = None
        for entry in batch:
            target = _coerce_target(entry.path)
            if target.is_symlink():
                raise FileWriteTransactionError(
                    f"refusing batch replacement through symlink: {target}"
                )
            if not isinstance(entry.payload, (bytes, bytearray, memoryview)):
                raise TypeError("batch payloads must be bytes-like")
            resolved_parent = target.parent.resolve()
            if parent is None:
                parent = resolved_parent
            elif parent != resolved_parent:
                raise ValueError("all batch targets must share one directory")
            absolute_target = resolved_parent / target.name
            if absolute_target in seen:
                raise ValueError(f"duplicate batch target: {target}")
            seen.add(absolute_target)
            normalized.append(
                (
                    absolute_target,
                    bytes(entry.payload),
                    _validated_permissions(entry.mode),
                )
            )

        assert parent is not None
        from core.runtime.atomic_writer import ensure_private_directory

        ensure_private_directory(parent)
        lock_path = parent / ".aura_file_write_batch.lock"
        transaction_id = uuid.uuid4().hex
        with interprocess_file_lock(lock_path):
            originals: dict[Path, tuple[bytes, int] | None] = {}
            for target, _payload, _mode in normalized:
                if target.exists():
                    if not target.is_file():
                        raise FileWriteTransactionError(
                            f"batch target is not a regular file: {target}"
                        )
                    originals[target] = (
                        target.read_bytes(),
                        stat.S_IMODE(target.stat().st_mode),
                    )
                else:
                    originals[target] = None

            committed: list[Path] = []
            try:
                for target, payload, mode in normalized:
                    committed.append(target)
                    atomic_write_bytes(target, payload, mode=mode)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                rollback_failures: list[str] = []
                for target in reversed(committed):
                    try:
                        original = originals[target]
                        if original is None:
                            durable_unlink(target, missing_ok=True)
                        else:
                            atomic_write_bytes(
                                target,
                                original[0],
                                mode=original[1],
                            )
                    except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                        rollback_failures.append(
                            f"{target}:{type(rollback_exc).__name__}:{rollback_exc}"
                        )
                detail = (
                    f"; rollback failures={rollback_failures}"
                    if rollback_failures
                    else "; prior targets restored"
                )
                raise FileWriteTransactionError(
                    f"file batch {transaction_id} did not commit{detail}"
                ) from exc

        paths = tuple(str(target) for target, _payload, _mode in normalized)
        hashes = tuple(
            (str(target), hashlib.sha256(payload).hexdigest())
            for target, payload, _mode in normalized
        )
        return FileWriteBatchReceipt(
            transaction_id=transaction_id,
            paths=paths,
            sha256=hashes,
        )

    def write_text(
        self,
        path: PathLike,
        text: str,
        *,
        encoding: str = "utf-8",
        source: str = "unknown",
        durable: bool = True,
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_write_text(target, text, encoding=encoding, durable=durable)

    def append_text(self, path: PathLike, text: str, *, encoding: str = "utf-8", source: str = "unknown") -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_append_text(target, text, encoding=encoding)

    # ── Event-loop-safe lane ─────────────────────────────────────────
    # Governance is checked inline (fail fast, caller's context); only the
    # blocking disk write is offloaded. Async callers must use these — an
    # on-loop fsync froze the live event loop for ~20 minutes under thrash.

    async def write_text_async(
        self,
        path: PathLike,
        text: str,
        *,
        encoding: str = "utf-8",
        source: str = "unknown",
        durable: bool = True,
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_text

        await async_atomic_write_text(target, text, encoding=encoding, durable=durable)

    async def write_bytes_async(
        self, path: PathLike, payload: bytes, *, source: str = "unknown", durable: bool = True
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_bytes

        await async_atomic_write_bytes(target, bytes(payload), durable=durable)

    def open_owned_binary(
        self,
        path: PathLike,
        *,
        mode: str,
        permissions: int = 0o600,
        source: str = "unknown",
    ) -> BinaryIO:
        """Open a process-owned mutable binary file through governance.

        This narrow primitive exists for `mmap` rings and process-lifetime lock
        files that cannot use replace-on-write persistence. Symlinks and
        arbitrary mode strings are rejected.
        """

        target = _coerce_target(path)
        if target.is_symlink():
            raise OSError(f"refusing owned binary open through symlink: {target}")
        flag_by_mode = {
            "a+b": os.O_RDWR | os.O_CREAT | os.O_APPEND,
            "w+b": os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            "r+b": os.O_RDWR,
        }
        if mode not in flag_by_mode:
            raise ValueError(f"unsupported owned binary mode: {mode!r}")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.open_owned_binary:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import ensure_private_directory

        ensure_private_directory(target.parent)
        flags = flag_by_mode[mode]
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(target), flags, _validated_permissions(permissions))
        try:
            os.fchmod(fd, _validated_permissions(permissions))
            return cast("BinaryIO", os.fdopen(fd, mode))
        except (OSError, ValueError):
            os.close(fd)
            raise

    def replace_file(
        self,
        path: PathLike,
        destination: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Durably replace a file through the governed synchronous lane."""

        src = _coerce_target(path)
        dst = _coerce_target(destination)
        if src.is_symlink() or dst.is_symlink():
            raise OSError("refusing durable replacement involving a symlink")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_file:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        durable_replace(src, dst)
        return str(dst)

    async def append_text_async(
        self, path: PathLike, text: str, *, encoding: str = "utf-8", source: str = "unknown"
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_append_text

        await async_atomic_append_text(target, text, encoding=encoding)

    @staticmethod
    def _replace_symlink_unchecked(link: Path, target: Path) -> str:
        from core.runtime.atomic_writer import ensure_private_directory

        if not target.exists():
            raise FileNotFoundError(f"symlink target does not exist: {target}")
        if link.exists() and link.is_dir() and not link.is_symlink():
            raise IsADirectoryError(f"refusing to replace directory with symlink: {link}")
        ensure_private_directory(link.parent)
        temporary = link.with_name(
            f".{link.name}.{os.getpid()}.{time.time_ns()}.symlink.tmp"
        )
        try:
            temporary.symlink_to(
                target.resolve(),
                target_is_directory=target.is_dir(),
            )
            os.replace(temporary, link)
        finally:
            if os.path.lexists(temporary):
                temporary.unlink()
        return str(target.resolve())

    def replace_symlink(
        self,
        path: PathLike,
        target_path: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Atomically create or replace a symlink through the file-write lane."""
        link = _coerce_path_allow_dir(path)
        target = _coerce_path_allow_dir(target_path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_symlink:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        return self._replace_symlink_unchecked(link, target)

    async def replace_symlink_async(
        self,
        path: PathLike,
        target_path: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Atomically replace a symlink off the event loop after governance."""
        link = _coerce_path_allow_dir(path)
        target = _coerce_path_allow_dir(target_path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_symlink:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        import asyncio

        return await asyncio.to_thread(
            self._replace_symlink_unchecked,
            link,
            target,
        )

    def delete_file(self, path: PathLike, *, source: str = "unknown") -> bool:
        """Delete a single file through the same governance lane as writes."""
        target = _coerce_target(path)
        if target.exists() and target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(f"target path is a directory: {target}")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_file:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        try:
            return durable_unlink(target, missing_ok=True)
        except FileNotFoundError:
            return False

    async def delete_path_async(
        self, path: PathLike, *, recursive: bool = False, source: str = "unknown"
    ) -> bool:
        """Delete a file or directory tree under governance, off the event loop.

        Directories require ``recursive=True`` — refusing an implicit tree
        delete is the difference between "remove this file" and "remove
        everything under here", and callers must state which they mean.

        Returns True if something was deleted, False if the path was absent.
        """
        target = _coerce_path_allow_dir(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _delete() -> bool:
            if not os.path.lexists(target):
                return False
            if target.is_symlink():
                target.unlink()
                return True
            if target.is_dir():
                if not recursive:
                    raise IsADirectoryError(
                        f"refusing to delete directory without recursive=True: {target}"
                    )
                import shutil

                shutil.rmtree(target)
                return True
            target.unlink()
            return True

        import asyncio

        return await asyncio.to_thread(_delete)

    async def move_path_async(
        self, path: PathLike, destination: PathLike, *, source: str = "unknown"
    ) -> str:
        """Move a file or directory under governance, off the event loop.

        Returns the final destination path as a string.
        """
        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.move_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _move() -> str:
            if not os.path.lexists(src):
                raise FileNotFoundError(f"move source does not exist: {src}")
            import shutil

            return str(shutil.move(str(src), str(dst)))

        import asyncio

        return await asyncio.to_thread(_move)

    async def copy_path_async(
        self, path: PathLike, destination: PathLike, *, source: str = "unknown"
    ) -> str:
        """Copy a file or directory tree under governance, off the event loop.

        Returns the final destination path as a string.
        """
        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.copy_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _copy() -> str:
            if not os.path.lexists(src):
                raise FileNotFoundError(f"copy source does not exist: {src}")
            import shutil

            if src.is_dir():
                return str(shutil.copytree(str(src), str(dst), symlinks=True))
            return str(shutil.copy2(str(src), str(dst), follow_symlinks=False))

        import asyncio

        return await asyncio.to_thread(_copy)

    def drain_text(self, path: PathLike, *, encoding: str = "utf-8", source: str = "unknown") -> str:
        """Atomically drain a text queue file and return its previous contents.

        The target is first moved aside, then read and deleted. Writers that
        append during the drain create a fresh target file, so entries are not
        lost by a read-then-clear race.
        """
        target = _coerce_target(path)
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.drain_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        if not target.exists():
            return ""
        drain_path = target.with_name(
            f".aura_drain_{target.name}_{os.getpid()}_{time.time_ns()}"
        )
        try:
            target.replace(drain_path)
        except FileNotFoundError:
            return ""
        try:
            return drain_path.read_text(encoding=encoding)
        finally:
            try:
                drain_path.unlink()
            except FileNotFoundError:
                pass

    def write_json(
        self,
        path: PathLike,
        obj: Any,
        *,
        schema_version: int,
        schema_name: str | None = None,
        indent: int | None = 2,
        source: str = "unknown",
    ) -> None:
        target = _coerce_target(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_json:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_write_json(
            target,
            obj,
            schema_version=schema_version,
            schema_name=schema_name,
            indent=indent,
        )

    async def write_json_async(
        self,
        path: PathLike,
        obj: Any,
        *,
        schema_version: int,
        schema_name: str | None = None,
        indent: int | None = 2,
        source: str = "unknown",
    ) -> None:
        target = _coerce_target(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_json:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_json

        await async_atomic_write_json(
            target,
            obj,
            schema_version=schema_version,
            schema_name=schema_name,
            indent=indent,
        )


def _coerce_target(path: PathLike) -> Path:
    if path is None:
        raise ValueError("target path is required")
    target = Path(path).expanduser()
    if target.exists() and target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(f"target path is a directory: {target}")
    return target


def _coerce_path_allow_dir(path: PathLike) -> Path:
    """Coerce a path argument for operations that legitimately act on directories."""
    if path is None:
        raise ValueError("target path is required")
    return Path(path).expanduser()


_gateway: FileWriteGateway | None = None


def get_file_write_gateway() -> FileWriteGateway:
    global _gateway
    if _gateway is None:
        _gateway = FileWriteGateway()
    return _gateway


def _registry_write_bytes(path: PathLike, payload: bytes, source: str) -> None:
    get_file_write_gateway().write_bytes(path, payload, source=source)


def _registry_write_text(path: PathLike, text: str, encoding: str, source: str) -> None:
    get_file_write_gateway().write_text(path, text, encoding=encoding, source=source)


try:
    from core.runtime.service_registry import install_file_write_sinks

    install_file_write_sinks(
        write_bytes=_registry_write_bytes,
        write_text=_registry_write_text,
    )
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
    logger.debug("Runtime file-write bridge unavailable: %s", exc)


__all__ = [
    "FileWriteBatchEntry",
    "FileWriteBatchReceipt",
    "FileWriteGateway",
    "FileWriteTransactionError",
    "get_file_write_gateway",
]
