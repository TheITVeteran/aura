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


def _coerce_target(path: PathLike) -> Path:
    if path is None:
        raise ValueError("target path is required")
    target = Path(path).expanduser()
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"target path is a directory: {target}")
    return target


_gateway: FileWriteGateway | None = None


def get_file_write_gateway() -> FileWriteGateway:
    global _gateway
    if _gateway is None:
        _gateway = FileWriteGateway()
    return _gateway


__all__ = ["FileWriteGateway", "get_file_write_gateway"]
