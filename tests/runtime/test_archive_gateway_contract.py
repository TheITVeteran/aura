from __future__ import annotations

import io
import tarfile

import pytest

from core.runtime import archive_gateway


def test_archive_gateway_creates_multi_source_tar_gz(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(archive_gateway, "governance_runtime_active", lambda: False)
    source_a = tmp_path / "memory"
    source_b = tmp_path / "wallet"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "trace.txt").write_text("memory-trace", encoding="utf-8")
    (source_b / "ledger.txt").write_text("wallet-ledger", encoding="utf-8")
    archive_path = tmp_path / "migration.tar.gz"

    result = archive_gateway.ArchiveGateway().create_tar_gz_from_sources(
        archive_path,
        [source_a, source_b, tmp_path / "missing"],
        source_label="test.archive_gateway.multi_source",
    )

    assert result == archive_path
    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())
        assert "memory/trace.txt" in names
        assert "wallet/ledger.txt" in names


def test_archive_gateway_rejects_link_outside_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(archive_gateway, "governance_runtime_active", lambda: False)
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        payload = b"safe"
        regular = tarfile.TarInfo("safe.txt")
        regular.size = len(payload)
        tar.addfile(regular, io.BytesIO(payload))
        link = tarfile.TarInfo("escape-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tar.addfile(link)

    with pytest.raises(tarfile.TarError):
        archive_gateway.ArchiveGateway().extract_tar_gz(
            archive_path,
            tmp_path / "restore",
            source_label="test.archive_gateway.unsafe_link",
        )
