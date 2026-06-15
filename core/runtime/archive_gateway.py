"""Canonical archive gateway.

Backups and updates need tarball creation/extraction. Those operations are
consequential filesystem effects, so production code should not call
``tarfile.open(..., "w")`` directly outside this gateway.
"""
from __future__ import annotations

import logging
import os
import tarfile
from collections.abc import Iterable
from pathlib import Path

from core.governance_context import governance_runtime_active, require_governance

logger = logging.getLogger("Aura.ArchiveGateway")

_ARCHIVE_DOMAINS = (
    "file_write",
    "state_mutation",
    "self_modification",
    "tool_execution",
)


class ArchiveGateway:
    """Single owner for archive creation/extraction."""

    def create_tar_gz(
        self,
        archive: str | Path,
        source: str | Path,
        *,
        arcname: str | None = None,
        source_label: str = "unknown",
    ) -> Path:
        if governance_runtime_active():
            require_governance(
                f"archive_gateway.create_tar_gz:{source_label}",
                strict=True,
                allowed_domains=_ARCHIVE_DOMAINS,
            )
        archive_path = Path(archive).expanduser()
        source_path = Path(source).expanduser()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(source_path, arcname=arcname or source_path.name)
        return archive_path

    def create_tar_gz_from_sources(
        self,
        archive: str | Path,
        sources: Iterable[str | Path],
        *,
        source_label: str = "unknown",
    ) -> Path:
        if governance_runtime_active():
            require_governance(
                f"archive_gateway.create_tar_gz_from_sources:{source_label}",
                strict=True,
                allowed_domains=_ARCHIVE_DOMAINS,
            )
        archive_path = Path(archive).expanduser()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tf:
            for source in sources:
                source_path = Path(source).expanduser()
                if source_path.exists():
                    tf.add(source_path, arcname=source_path.name)
        return archive_path

    def extract_tar_gz(
        self,
        archive: str | Path,
        target: str | Path,
        *,
        source_label: str = "unknown",
    ) -> Path:
        if governance_runtime_active():
            require_governance(
                f"archive_gateway.extract_tar_gz:{source_label}",
                strict=True,
                allowed_domains=_ARCHIVE_DOMAINS,
            )
        archive_path = Path(archive).expanduser()
        target_path = Path(target).expanduser()
        target_path.mkdir(parents=True, exist_ok=True)
        target_root = target_path.resolve()
        with tarfile.open(archive_path, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                destination = (target_root / member.name).resolve()
                if os.path.commonpath([str(target_root), str(destination)]) != str(target_root):
                    raise ValueError(f"archive member escapes target directory: {member.name}")
            # ``data`` rejects device files and unsafe links in addition to
            # the explicit destination traversal check above.
            tf.extractall(target_path, members=members, filter="data")
        return target_path


_gateway: ArchiveGateway | None = None


def get_archive_gateway() -> ArchiveGateway:
    global _gateway
    if _gateway is None:
        _gateway = ArchiveGateway()
    return _gateway


__all__ = ["ArchiveGateway", "get_archive_gateway"]
