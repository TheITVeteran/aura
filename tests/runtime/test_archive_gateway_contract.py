from __future__ import annotations

import tarfile

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
