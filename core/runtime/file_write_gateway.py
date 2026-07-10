"""core/runtime/file_write_gateway.py — Canonical File Write Gateway.

All file writing operations should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

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
)

logger = logging.getLogger("Aura.FileWriteGateway")
_FILE_WRITE_DOMAINS = (
    "file_write",
    "memory_write",
    "state_mutation",
    "self_modification",
    "tool_execution",
)


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

    def delete_file(self, path: PathLike, *, source: str = "unknown") -> bool:
        """Delete a single file through the same governance lane as writes."""
        target = _coerce_target(path)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"target path is a directory: {target}")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_file:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        try:
            target.unlink()
            return True
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
    if target.exists() and target.is_dir():
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


__all__ = ["FileWriteGateway", "get_file_write_gateway"]
