from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core.resilience.memory_governor import MemoryGovernor


class _Process:
    def __init__(self, pid: int, cmdline: list[str] | None = None, rss_mb: float = 64.0):
        self.pid = pid
        self._cmdline = cmdline or [
            "python",
            "-c",
            "from multiprocessing.spawn import spawn_main",
        ]
        self._rss = int(rss_mb * 1024 * 1024)

    def is_alive(self) -> bool:
        return True

    def as_dict(self, attrs=None):
        return {
            "pid": self.pid,
            "name": "python",
            "cmdline": self._cmdline,
            "memory_info": SimpleNamespace(rss=self._rss),
        }


def _client(process: _Process, state: str = "ready", initialized: bool = True):
    return SimpleNamespace(
        _process=process,
        _lane_state=state,
        _init_done=initialized,
        _warmup_in_flight=state == "warming",
    )


def test_registered_spawn_main_worker_is_counted_exactly_once(monkeypatch):
    governor = MemoryGovernor(SimpleNamespace())
    worker = _Process(111)
    unrelated = _Process(222)
    governor._proc = SimpleNamespace(children=lambda recursive=True: [worker, worker, unrelated])
    fake_module = SimpleNamespace(_CLIENTS={"/models/32b": _client(worker)})
    monkeypatch.setitem(sys.modules, "core.brain.llm.mlx_client", fake_module)

    managed = list(governor._iter_managed_runtime_processes())

    assert [process.info["pid"] for process in managed] == [111]


def test_canonical_process_tree_includes_root_exactly_once(monkeypatch):
    governor = MemoryGovernor(SimpleNamespace())
    governor._proc = SimpleNamespace(pid=42)
    monkeypatch.setattr(
        "core.resilience.memory_governor.process_memory_bytes",
        lambda _pid: 2 * 1024**3,
    )
    monkeypatch.setattr(
        "core.resilience.memory_governor.get_memory_pressure_snapshot",
        lambda force_refresh=True: SimpleNamespace(process_rss_gb=22.0),
    )

    core_mb, runtime_mb = governor._sample_rss_sync()

    assert core_mb == 2048.0
    assert runtime_mb == 20 * 1024.0
    assert core_mb + runtime_mb == 22 * 1024.0


def test_model_lifecycle_resets_and_settles_before_new_trend_epoch(monkeypatch):
    governor = MemoryGovernor(SimpleNamespace())
    governor._model_settling_s = 120.0
    snapshots = iter(
        [
            {
                "state": "model_loading",
                "allocation_identity": (("/models/32b", 111, "warming", False),),
            },
            {
                "state": "steady",
                "allocation_identity": (("/models/32b", 111, "ready", True),),
            },
            {
                "state": "steady",
                "allocation_identity": (("/models/32b", 111, "ready", True),),
            },
        ]
    )
    monkeypatch.setattr(governor, "_model_lifecycle_snapshot", lambda: next(snapshots))
    governor._runaway.observe(10_000.0, now=1.0)

    assert governor._update_runaway_lifecycle(10.0) == ("model_loading", True)
    assert governor._runaway.assess(now=10.0).samples == 0
    assert governor._update_runaway_lifecycle(20.0) == ("settling", True)
    assert governor._runaway.assess(now=20.0).samples == 0
    assert governor._update_runaway_lifecycle(141.0) == ("steady", False)


@pytest.mark.asyncio
async def test_absolute_critical_cleanup_remains_active_during_model_load(monkeypatch):
    governor = MemoryGovernor(SimpleNamespace())
    governor._sample_rss_sync = lambda: (1000.0, governor.threshold_critical + 1.0)
    governor._model_lifecycle_snapshot = lambda: {
        "state": "model_loading",
        "allocation_identity": (("/models/32b", 111, "warming", False),),
    }
    governor._iter_managed_runtime_processes = lambda: iter(())
    calls: list[str] = []

    async def _cleanup(reason="memory_pressure"):
        calls.append(reason)

    governor._critical_cleanup = _cleanup
    monkeypatch.setattr(
        "core.utils.memory_monitor.AppleSiliconMemoryMonitor._get_pressure_sysctl",
        lambda _self: 50.0,
    )

    await governor._enforce_policy()

    assert calls
    assert governor._last_policy_sample["trend_provisional"] is True
    assert governor._last_policy_sample["runaway"]["state"] == "provisional"
