"""The reaper must drain crash-artifact backlogs, bounded and off-loop.

Live evidence: 18,226 stall dumps (558MB) accumulated because pruning only
ran when NEW stalls happened — a healthy runtime never drained the backlog.
"""
from __future__ import annotations

import asyncio
import threading

from core.ops.lymphatic_reaper import LymphaticReaper


def _make_reaper(tmp_path, monkeypatch) -> LymphaticReaper:
    reaper = LymphaticReaper(interval_s=300.0, data_dir=tmp_path / "home-data")
    monkeypatch.setattr(reaper, "_crash_artifact_root", lambda: tmp_path / "data")
    return reaper


def _write_stalls(tmp_path, count: int, *, start: int = 1_000_000) -> None:
    stalls = tmp_path / "data" / "error_logs" / "stalls"
    stalls.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (stalls / f"stall_{start + i}.txt").write_text(f"stall dump {i}\n" * 5)


class TestCrashArtifactSweep:
    def test_backlog_beyond_retention_is_drained_oldest_first(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        _write_stalls(tmp_path, reaper.CRASH_ARTIFACT_POLICIES[0][2] + 40)

        reclaimed = reaper._crash_artifact_sweep()

        stalls = sorted((tmp_path / "data" / "error_logs" / "stalls").iterdir())
        keep = reaper.CRASH_ARTIFACT_POLICIES[0][2]
        assert len(stalls) == keep
        assert reclaimed > 0
        # Oldest (lowest epoch) deleted; newest retained.
        assert stalls[0].name == f"stall_{1_000_000 + 40}.txt"

    def test_deletions_are_batch_bounded_per_sweep(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        keep = reaper.CRASH_ARTIFACT_POLICIES[0][2]
        backlog = keep + reaper.CRASH_SWEEP_DELETE_BATCH + 300
        _write_stalls(tmp_path, backlog)

        reaper._crash_artifact_sweep()

        remaining = len(list((tmp_path / "data" / "error_logs" / "stalls").iterdir()))
        assert remaining == backlog - reaper.CRASH_SWEEP_DELETE_BATCH, (
            "one sweep must delete at most CRASH_SWEEP_DELETE_BATCH files"
        )
        # A second sweep continues draining toward retention.
        reaper._crash_artifact_sweep()
        remaining_after_second = len(
            list((tmp_path / "data" / "error_logs" / "stalls").iterdir())
        )
        assert remaining_after_second == keep

    def test_within_retention_nothing_is_deleted(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        _write_stalls(tmp_path, 10)
        assert reaper._crash_artifact_sweep() == 0
        assert len(list((tmp_path / "data" / "error_logs" / "stalls").iterdir())) == 10

    def test_memory_tombstones_have_their_own_retention(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        mem = tmp_path / "data" / "error_logs" / "memory"
        mem.mkdir(parents=True)
        for i in range(30):
            (mem / f"death_syslog_{1_000_000 + i}.log").write_text("syslog capture")
        # Unrelated files in the same directory are untouched.
        (mem / "sentinel.log").write_text("keep me")

        reaper._crash_artifact_sweep()

        remaining = sorted(p.name for p in mem.iterdir())
        assert "sentinel.log" in remaining
        assert len([n for n in remaining if n.startswith("death_syslog_")]) == 20

    def test_missing_directories_are_fine(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        assert reaper._crash_artifact_sweep() == 0


class TestSweepRunsOffLoop:
    def test_steps_execute_on_worker_threads(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)
        step_threads: dict[str, threading.Thread] = {}

        def tracking(name):
            def _fn():
                step_threads[name] = threading.current_thread()
                return 0

            return _fn

        monkeypatch.setattr(reaper, "_hunt_orphans", tracking("hunt_orphans"))
        monkeypatch.setattr(reaper, "_filesystem_sweep", tracking("filesystem_sweep"))
        monkeypatch.setattr(reaper, "_crash_artifact_sweep", tracking("crash_artifact_sweep"))
        monkeypatch.setattr(reaper, "_defragment_memory", tracking("defragment_memory"))

        async def scenario():
            loop_thread = threading.current_thread()
            status = await reaper.sweep()
            return loop_thread, status

        loop_thread, status = asyncio.run(scenario())
        assert set(step_threads) == {
            "hunt_orphans", "filesystem_sweep", "crash_artifact_sweep", "defragment_memory",
        }
        assert all(t is not loop_thread for t in step_threads.values()), (
            "reaper steps are blocking I/O and must not run on the event loop"
        )
        assert status["step_errors"] == {}

    def test_step_failure_is_contained(self, tmp_path, monkeypatch):
        reaper = _make_reaper(tmp_path, monkeypatch)

        def boom():
            raise OSError("disk unavailable")

        monkeypatch.setattr(reaper, "_hunt_orphans", boom)

        status = asyncio.run(reaper.sweep())
        assert "hunt_orphans" in status["step_errors"]
        assert status["processes_reaped"] == 0
