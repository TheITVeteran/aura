"""Contract tests for the SystemIntegrityMonitor SQLite sweep.

Pins the three properties the rewrite exists for: (1) every real store is
discovered — nested, *.sqlite3, sibling roots — not just top-level *.db;
(2) the periodic path is quick_check on a slow cadence with boot coverage,
not full integrity_check page scans every 5 minutes; (3) a corrupt store
stays visible on skip-cycles (state, not event).
"""

from __future__ import annotations

import sqlite3

import pytest

from core.resilience.integrity_monitor import SystemIntegrityMonitor


def _make_db(path, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])


def _corrupt_db(path) -> None:
    """A valid header with mangled btree pages — quick_check must catch it.

    Page 1 free space can absorb garbage silently; real detection requires
    mangling the table btree pages (page 2 onward at the 4096 page size).
    """
    _make_db(path, rows=500)
    data = bytearray(path.read_bytes())
    assert len(data) > 8192, "test premise: table btree spans page 2+"
    for i in range(4096, min(len(data), 6144)):
        data[i] = 0xFF
    path.write_bytes(bytes(data))


@pytest.fixture()
def root(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return tmp_path


class TestDiscovery:
    def test_finds_nested_and_sqlite3_and_sibling_roots(self, root):
        _make_db(root / "data" / "top.db")
        _make_db(root / "data" / "traces" / "nested.sqlite3")
        _make_db(root / "storage" / "vault.db")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        found = {p.name for p in mon._discover_sqlite_stores()}
        assert found == {"top.db", "nested.sqlite3", "vault.db"}

    def test_skips_excluded_trees_wal_and_impostors(self, root):
        _make_db(root / "data" / "real.db")
        _make_db(root / "data" / "training" / "excluded.db")
        _make_db(root / "data" / "error_logs" / "excluded2.sqlite3")
        (root / "data" / "real.db-wal").write_bytes(b"wal bytes")
        (root / "data" / "fake.db").write_text("not a sqlite file")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        found = {p.name for p in mon._discover_sqlite_stores()}
        assert found == {"real.db"}

    def test_discovery_bounded(self, root):
        for i in range(12):
            _make_db(root / "data" / f"s{i:02d}.db", rows=1)
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        assert len(mon._discover_sqlite_stores(max_stores=5)) == 5


class TestSweep:
    def test_healthy_stores_pass(self, root):
        _make_db(root / "data" / "ok.db")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        report = __import__("asyncio").get_event_loop_policy().new_event_loop().run_until_complete(
            mon.run_check(include_databases=True)
        )
        assert report.db_checks.get("data/ok.db") == "ok"
        assert not [e for e in report.errors if "DB" in e]

    def test_corrupt_store_reported_with_runbook_pointer(self, root):
        _corrupt_db(root / "data" / "broken.db")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        loop = __import__("asyncio").new_event_loop()
        try:
            report = loop.run_until_complete(mon.run_check(include_databases=True))
        finally:
            loop.close()
        assert any("broken.db" in e for e in report.errors)
        assert any("memory-corruption" in e for e in report.errors)
        assert report.db_checks.get("data/broken.db") not in (None, "ok")

    def test_hot_wal_writer_is_not_a_false_positive(self, root):
        db = root / "data" / "hot.db"
        _make_db(db)
        writer = sqlite3.connect(db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO t VALUES (99)")
        writer.commit()
        try:
            mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
            loop = __import__("asyncio").new_event_loop()
            try:
                report = loop.run_until_complete(mon.run_check(include_databases=True))
            finally:
                loop.close()
            assert report.db_checks.get("data/hot.db") == "ok"
        finally:
            writer.close()


class TestCadence:
    def test_first_cycle_sweeps_then_respects_every_n(self, root, monkeypatch):
        monkeypatch.setenv("AURA_INTEGRITY_DB_SWEEP_EVERY_N", "6")
        _make_db(root / "data" / "a.db")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            first = loop.run_until_complete(mon.run_check())
            assert first.db_checks, "boot cycle must sweep"
            mon._check_count = 1  # what the monitor loop does after a cycle
            second = loop.run_until_complete(mon.run_check())
        finally:
            loop.close()
        # skip-cycle carries the last verdict forward instead of re-scanning
        assert second.db_checks == first.db_checks

    def test_known_corruption_stays_visible_on_skip_cycles(self, root, monkeypatch):
        monkeypatch.setenv("AURA_INTEGRITY_DB_SWEEP_EVERY_N", "6")
        _corrupt_db(root / "data" / "bad.db")
        mon = SystemIntegrityMonitor(data_dir=str(root / "data"))
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            first = loop.run_until_complete(mon.run_check())
            assert any("bad.db" in e for e in first.errors)
            mon._check_count = 1
            second = loop.run_until_complete(mon.run_check())
        finally:
            loop.close()
        assert any("bad.db" in e for e in second.errors), (
            "corruption is state, not an event — skip-cycles must not go green"
        )
