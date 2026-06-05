"""core/runtime/file_write_gateway.py — Canonical File Write Gateway.

All file writing operations should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.atomic_writer import (
    atomic_append_text,
    atomic_write_bytes,
    atomic_write_text,
    atomic_write_json,
    PathLike,
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

    def write_text(self, path: PathLike, text: str, *, encoding: str = "utf-8", source: str = "unknown") -> None:
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
        atomic_write_text(target, text, encoding=encoding)

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

    def write_json(
        self,
        path: PathLike,
        obj: Any,
        *,
        schema_version: int,
        schema_name: Optional[str] = None,
        indent: Optional[int] = 2,
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


_gateway: Optional[FileWriteGateway] = None


def get_file_write_gateway() -> FileWriteGateway:
    global _gateway
    if _gateway is None:
        _gateway = FileWriteGateway()
    return _gateway


__all__ = ["FileWriteGateway", "get_file_write_gateway"]
