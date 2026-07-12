"""Contract tests for tools/state_backup.py.

The claims that matter: hot WAL databases snapshot consistently, excludes
hold, the manifest is honest (hash mismatch = verify failure), the ring is
bounded, and corruption is detected on verify — because an unverified
backup is a hope, not a backup.
"""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from tools.state_backup import (
    create_backup,
    prune_ring,
    verify_backup,
)


@pytest.fixture()
def state_root(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "storage").mkdir()
    (root / "data" / "training").mkdir()
    (root / "data" / "error_logs").mkdir()

    db = root / "data" / "aura_state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE facts (k TEXT, v TEXT)")
        conn.execute("INSERT INTO facts VALUES ('alpha', '1')")

    (root / "data" / "notes.txt").write_text("plain state")
    (root / "data" / "training" / "huge.bin").write_text("excluded")
    (root / "data" / "error_logs" / "stall_1.txt").write_text("excluded")
    (root / "storage" / "blob.json").write_text("{}")
    return root


@pytest.fixture()
def out_dir(tmp_path):
    return tmp_path / "backups"


class TestCreate:
    def test_archive_layout_and_excludes(self, state_root, out_dir):
        archive = create_backup(state_root, out_dir)
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert "data/aura_state.db" in names
        assert "data/notes.txt" in names
        assert "storage/blob.json" in names
        assert not any("training" in n for n in names)
        assert not any("error_logs" in n for n in names)

    def test_manifest_written_with_hash(self, state_root, out_dir):
        archive = create_backup(state_root, out_dir)
        lines = (out_dir / "manifest.jsonl").read_text().splitlines()
        entry = json.loads(lines[-1])
        assert entry["archive"] == archive.name
        assert len(entry["sha256"]) == 64
        assert entry["sqlite_api_copies"] == 1
        assert entry["staged_files"] >= 3

    def test_hot_wal_database_snapshots_consistently(self, state_root, out_dir):
        """An open writer with an uncheckpointed WAL must not corrupt or
        lose committed rows in the snapshot — this is why the backup API
        exists and why raw tar was wrong."""
        db = state_root / "data" / "aura_state.db"
        writer = sqlite3.connect(db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO facts VALUES ('beta', '2')")
        writer.commit()  # committed but likely un-checkpointed (lives in -wal)
        try:
            assert (db.parent / "aura_state.db-wal").exists(), "test premise: WAL present"
            archive = create_backup(state_root, out_dir)
        finally:
            writer.close()
        with tarfile.open(archive) as tar:
            extract_dir = out_dir / "x"
            tar.extractall(extract_dir, filter="data")
        with sqlite3.connect(extract_dir / "data" / "aura_state.db") as conn:
            rows = dict(conn.execute("SELECT k, v FROM facts").fetchall())
        assert rows == {"alpha": "1", "beta": "2"}

    def test_wal_shm_siblings_not_shipped_raw(self, state_root, out_dir):
        writer = sqlite3.connect(state_root / "data" / "aura_state.db")
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO facts VALUES ('c', '3')")
        writer.commit()
        try:
            archive = create_backup(state_root, out_dir)
        finally:
            writer.close()
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert not any(n.endswith(("-wal", "-shm")) for n in names)

    def test_empty_root_fails_loudly(self, tmp_path, out_dir):
        with pytest.raises(SystemExit):
            create_backup(tmp_path / "nothing", out_dir)


class TestRing:
    def test_ring_bounded_and_legacy_untouched(self, state_root, out_dir):
        out_dir.mkdir(parents=True)
        legacy = out_dir / "aura_backup_20260402_160752.zip"
        legacy.write_text("bryan's old backup")
        archives = [create_backup(state_root, out_dir, keep=3) for _ in range(5)]
        ring = sorted(out_dir.glob("aura_state_*.tar.gz"))
        assert len(ring) == 3
        assert archives[-1] in ring  # newest survives
        assert legacy.exists()

    def test_prune_keeps_newest(self, out_dir):
        out_dir.mkdir(parents=True)
        import time as _time
        paths = []
        for i in range(4):
            p = out_dir / f"aura_state_2026010{i}_000000-1.tar.gz"
            p.write_text(str(i))
            _time.sleep(0.01)
            paths.append(p)
        assert prune_ring(out_dir, keep=2) == 2
        assert paths[3].exists() and paths[2].exists()
        assert not paths[0].exists() and not paths[1].exists()


class TestVerify:
    def test_verify_ok_on_fresh_backup(self, state_root, out_dir, capsys):
        create_backup(state_root, out_dir)
        assert verify_backup(out_dir) == 0
        assert "quick_check" in capsys.readouterr().out

    def test_verify_detects_archive_corruption(self, state_root, out_dir, capsys):
        archive = create_backup(state_root, out_dir)
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 0xFF
        archive.write_bytes(bytes(data))
        assert verify_backup(out_dir, archive) == 1
        assert "sha256 mismatch" in capsys.readouterr().out

    def test_verify_fails_when_no_ring(self, out_dir):
        out_dir.mkdir(parents=True)
        assert verify_backup(out_dir) == 1

    def test_verify_without_manifest_still_checks_structure(
        self, state_root, out_dir, capsys
    ):
        archive = create_backup(state_root, out_dir)
        (out_dir / "manifest.jsonl").unlink()
        assert verify_backup(out_dir, archive) == 0
        assert "provenance unavailable" in capsys.readouterr().out
