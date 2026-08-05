"""The "single-process lock" was never the constraint. This measures what is.

The review listed as a limitation: "Transactional state commits via SQLite WAL
enforce single-process execution locks, limiting distributed multi-node
scaling."

The first half is false, and the test below is the measurement rather than
another assertion. Four separate OS processes write to one WAL database
concurrently; all of them succeed.

What WAL genuinely requires is a shared-memory index (``-shm``) that every
participant mmaps, which is a same-machine mechanism. Across hosts on a network
filesystem it is not coherent, and the failure mode is not an error — it is
silent database corruption. Nothing was checking for that, which is the gap
this closes: the constraint is SINGLE HOST, now enforced, rather than single
process, which was never true.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
import subprocess
from pathlib import Path

import pytest

from core.runtime.store_locality import (
    LOCAL_FILESYSTEMS,
    NETWORK_FILESYSTEMS,
    StoreLocality,
    assert_wal_safe,
    describe_store,
)


def _write_rows(db_path: str, tag: str, count: int, results) -> None:
    try:
        connection = sqlite3.connect(db_path, timeout=15.0)
        connection.execute("PRAGMA journal_mode=WAL")
        written = 0
        for index in range(count):
            connection.execute("INSERT INTO rows(who, i) VALUES (?,?)", (tag, index))
            connection.commit()
            written += 1
        results.put((tag, written, ""))
    except sqlite3.Error as exc:  # pragma: no cover - the thing under test
        results.put((tag, -1, str(exc)))


def test_four_processes_write_to_one_wal_database(tmp_path):
    """The measurement that contradicts the "single-process lock" claim."""
    db_path = str(tmp_path / "concurrent.db")
    setup = sqlite3.connect(db_path)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE rows(who TEXT, i INT)")
    setup.commit()
    setup.close()

    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_write_rows, args=(db_path, f"p{n}", 100, results))
        for n in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            pytest.fail(f"WAL writer {worker.pid} exceeded its 30s deadline")
        assert worker.exitcode == 0, f"WAL writer {worker.pid} exited {worker.exitcode}"

    try:
        reported = [results.get(timeout=5) for _ in range(4)]
    except queue.Empty:
        pytest.fail("a WAL writer exited without publishing its result")
    for tag, written, error in reported:
        assert error == "", f"{tag} failed: {error}"
        assert written == 100, f"{tag} wrote {written}"

    check = sqlite3.connect(db_path)
    total = check.execute("SELECT count(*) FROM rows").fetchone()[0]
    per_writer = dict(check.execute("SELECT who, count(*) FROM rows GROUP BY who").fetchall())
    check.close()

    assert total == 400
    assert per_writer == {"p0": 100, "p1": 100, "p2": 100, "p3": 100}


class TestTheRealBoundaryIsTheFilesystem:
    def test_darwin_probe_does_not_spawn_when_native_statfs_succeeds(
        self, tmp_path, monkeypatch
    ):
        import core.runtime.store_locality as module

        monkeypatch.setattr(module, "_fstype_darwin_native", lambda _path: "apfs")

        class SubprocessMustNotRun:
            def run(self, *args, **kwargs):
                pytest.fail("mount subprocess must not run")

        monkeypatch.setattr(
            module,
            "get_subprocess_gateway",
            lambda: SubprocessMustNotRun(),
        )

        assert module._fstype_darwin(tmp_path) == "apfs"

    def test_darwin_probe_falls_back_when_native_statfs_is_unavailable(
        self, tmp_path, monkeypatch
    ):
        import core.runtime.store_locality as module

        monkeypatch.setattr(module, "_fstype_darwin_native", lambda _path: "")

        class MountProbe:
            def run(self, *args, **kwargs):
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="/dev/disk3s5 on / (apfs, local, journaled)\n",
                    stderr="",
                )

        monkeypatch.setattr(
            module,
            "get_subprocess_gateway",
            lambda: MountProbe(),
        )

        assert module._fstype_darwin(tmp_path) == "apfs"

    def test_a_local_store_is_recognised_and_allowed(self, tmp_path):
        locality = describe_store(tmp_path / "ledger.db")
        assert locality.is_network is False
        assert locality.wal_is_safe is True
        assert locality.fstype in LOCAL_FILESYSTEMS or not locality.is_known

    def test_a_relative_path_is_measured_where_it_actually_lives(self):
        """A relative path compared against "/" reports the wrong filesystem."""
        locality = describe_store("data/somewhere.db")
        assert Path(locality.path).is_absolute()

    def test_a_store_that_does_not_exist_yet_is_still_measurable(self, tmp_path):
        locality = describe_store(tmp_path / "not" / "created" / "yet.db")
        assert locality.fstype
        assert locality.wal_is_safe is True

    @pytest.mark.parametrize("fstype", sorted(NETWORK_FILESYSTEMS)[:6])
    def test_wal_is_refused_on_a_network_filesystem(self, fstype, monkeypatch):
        """Silent corruption is worse than an unavailable ledger."""
        import core.runtime.store_locality as module

        monkeypatch.setattr(
            module,
            "describe_store",
            lambda _path: StoreLocality("/mnt/share/x.db", fstype, True, True),
        )
        with pytest.raises(RuntimeError, match="refusing a WAL store"):
            module.assert_wal_safe("/mnt/share/x.db", subsystem="test")

    def test_an_unknown_filesystem_is_recorded_not_refused(self, monkeypatch):
        """Refusing every mount this module has not heard of takes hosts down."""
        import core.runtime.store_locality as module

        recorded = []
        monkeypatch.setattr(
            module,
            "describe_store",
            lambda _path: StoreLocality("/odd/x.db", "someexoticfs", False, False),
        )
        monkeypatch.setattr(
            module, "record_degradation", lambda *a, **k: recorded.append((a, k))
        )
        locality = module.assert_wal_safe("/odd/x.db", subsystem="test")
        assert locality.wal_is_safe is True
        assert len(recorded) == 1


def test_the_receipts_ledger_checks_before_it_opens(tmp_path):
    """The guard is wired where a real store is opened, not just available."""
    import inspect

    from core.runtime import receipts

    source = inspect.getsource(receipts.ReceiptStore._ledger_connection_locked)
    assert "assert_wal_safe" in source


def test_the_real_data_directory_is_wal_safe():
    """The host this actually runs on."""
    locality = describe_store("data/aura_state.db")
    assert locality.wal_is_safe is True
    assert_wal_safe("data/aura_state.db", subsystem="test")
