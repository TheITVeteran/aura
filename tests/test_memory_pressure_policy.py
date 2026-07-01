from __future__ import annotations

from types import SimpleNamespace

import pytest


def _vm(*, total_gb: float, available_gb: float, percent: float) -> SimpleNamespace:
    gib = 1024**3
    return SimpleNamespace(
        total=int(total_gb * gib),
        available=int(available_gb * gib),
        percent=percent,
    )


def _patch_process_rss(
    monkeypatch,
    memory_monitor,
    rss_gb: float,
    *,
    child_rss_gb: tuple[float, ...] = (),
) -> None:
    gib = 1024**3

    class _Process:
        def __init__(self, *_args, _rss: float | None = None, **_kwargs):
            self._rss = rss_gb if _rss is None else _rss
            self.pid = 12345

        def memory_info(self):
            return SimpleNamespace(rss=int(self._rss * gib))

        def children(self, recursive=True):
            return [_Process(_rss=child_rss) for child_rss in child_rss_gb]

    monkeypatch.setattr(memory_monitor.psutil, "Process", _Process)
    monkeypatch.setattr(memory_monitor, "_darwin_phys_footprint_bytes", lambda _pid: 0)


def test_memory_pressure_snapshot_classifies_64gb_emergency(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=2.5, percent=96.0),
    )
    _patch_process_rss(monkeypatch, memory_monitor, 2.0)

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "emergency"
    assert snapshot.emergency is True
    assert snapshot.refuse_heavy_local_generation is True
    assert snapshot.max_token_cap == 32
    assert "memory_pressure:96.0%" in snapshot.reason


def test_memory_pressure_snapshot_caps_but_does_not_refuse_high_pressure(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=9.5, percent=86.0),
    )
    _patch_process_rss(monkeypatch, memory_monitor, 2.0)

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "high"
    assert snapshot.refuse_heavy_local_generation is False
    assert snapshot.max_token_cap == 192


def test_memory_pressure_snapshot_refuses_when_aura_process_exceeds_rss_limit(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=24.0, percent=62.0),
    )
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "12")
    _patch_process_rss(monkeypatch, memory_monitor, 12.5)

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "critical"
    assert snapshot.critical is True
    assert snapshot.refuse_heavy_local_generation is True
    assert snapshot.process_rss_gb == pytest.approx(12.5)
    assert snapshot.process_rss_limit_gb == pytest.approx(12.0)
    assert "process_tree_rss:12.5GB/12.0GB" in snapshot.reason


def test_memory_pressure_snapshot_counts_child_inference_workers(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=24.0, percent=62.0),
    )
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "18")
    _patch_process_rss(monkeypatch, memory_monitor, 3.0, child_rss_gb=(15.5, 2.0))

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "emergency"
    assert snapshot.refuse_heavy_local_generation is True
    assert snapshot.process_rss_gb == pytest.approx(20.5)
    assert "process_tree_rss:20.5GB/18.0GB" in snapshot.reason


def test_memory_pressure_snapshot_uses_darwin_footprint_when_larger_than_rss(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=24.0, percent=62.0),
    )
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "18")
    _patch_process_rss(monkeypatch, memory_monitor, 3.0)
    monkeypatch.setattr(
        memory_monitor,
        "_darwin_phys_footprint_bytes",
        lambda _pid: int(21.0 * 1024**3),
    )

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "emergency"
    assert snapshot.refuse_heavy_local_generation is True
    assert snapshot.process_rss_gb == pytest.approx(21.0)
    assert "process_tree_rss:21.0GB/18.0GB" in snapshot.reason


def test_darwin_footprint_uses_current_value_not_lifetime_peak():
    import core.utils.memory_monitor as memory_monitor

    usage = memory_monitor._DarwinRUsageInfoV4()
    usage.ri_resident_size = 2 * 1024**3
    usage.ri_phys_footprint = 6 * 1024**3
    usage.ri_lifetime_max_phys_footprint = 105 * 1024**3

    assert memory_monitor._current_darwin_footprint_bytes(usage) == 6 * 1024**3


def test_darwin_footprint_falls_back_to_current_resident_size():
    import core.utils.memory_monitor as memory_monitor

    usage = memory_monitor._DarwinRUsageInfoV4()
    usage.ri_resident_size = 3 * 1024**3
    usage.ri_phys_footprint = 0
    usage.ri_lifetime_max_phys_footprint = 105 * 1024**3

    assert memory_monitor._current_darwin_footprint_bytes(usage) == 3 * 1024**3


def test_background_policy_blocks_on_process_tree_rss_even_when_system_memory_is_low(
    monkeypatch,
):
    import core.runtime.background_policy as background_policy
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=28.0, percent=56.0),
    )
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "18")
    _patch_process_rss(monkeypatch, memory_monitor, 20.0)
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )

    reason = background_policy.background_activity_reason(allow_no_user_anchor=True)

    assert reason.startswith("process_tree_rss:20.0GB/18.0GB")


def test_constitutive_compute_budget_throttles_on_process_tree_rss_pressure(monkeypatch):
    import core.runtime.background_policy as background_policy
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=28.0, percent=56.0),
    )
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "18")
    _patch_process_rss(monkeypatch, memory_monitor, 20.0)
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )

    budget = background_policy.constitutive_compute_budget(
        "liquid_substrate",
        60.0,
        min_hz=0.5,
        memory_critical_hz=0.5,
    )

    assert budget.effective_hz == pytest.approx(0.5)
    assert budget.reason.startswith("process_tree_rss:20.0GB/18.0GB")


def test_background_policy_defers_optional_work_under_cpu_pressure(monkeypatch):
    import core.runtime.background_policy as background_policy

    monkeypatch.delenv("AURA_PROOF_RUN", raising=False)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_HEAT_GUARD", "1")
    monkeypatch.setenv("AURA_BACKGROUND_MAX_CPU_PERCENT", "75")
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(
            pressure_pct=20.0,
            reason="memory_pressure_20.0",
        ),
    )
    monkeypatch.setattr(
        background_policy,
        "get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )
    monkeypatch.setattr(background_policy.psutil, "cpu_percent", lambda interval=None: 93.4)
    monkeypatch.setattr(background_policy.psutil, "cpu_count", lambda: 18)
    monkeypatch.setattr(background_policy.psutil, "sensors_temperatures", lambda: {}, raising=False)

    reason = background_policy.background_activity_reason(allow_no_user_anchor=True)

    assert reason == "cpu_pressure_93.4"


def test_constitutive_compute_budget_throttles_under_cpu_pressure(monkeypatch):
    import core.runtime.background_policy as background_policy

    monkeypatch.delenv("AURA_PROOF_RUN", raising=False)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_HEAT_GUARD", "1")
    monkeypatch.setenv("AURA_BACKGROUND_MAX_CPU_PERCENT", "75")
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(
            pressure_pct=20.0,
            reason="memory_pressure_20.0",
        ),
    )
    monkeypatch.setattr(
        background_policy,
        "get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )
    monkeypatch.setattr(background_policy.psutil, "cpu_percent", lambda interval=None: 91.2)
    monkeypatch.setattr(background_policy.psutil, "cpu_count", lambda: 18)
    monkeypatch.setattr(background_policy.psutil, "sensors_temperatures", lambda: {}, raising=False)

    budget = background_policy.constitutive_compute_budget(
        "liquid_substrate",
        60.0,
        min_hz=0.5,
        compute_pressure_hz=1.0,
    )

    assert budget.effective_hz == pytest.approx(1.0)
    assert budget.reason == "cpu_pressure_91.2"


@pytest.mark.asyncio
async def test_mlx_client_refuses_heavy_generation_under_emergency_memory(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from core.brain.llm.mlx_client import MLXLocalClient

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=2.0, percent=96.0),
    )
    _patch_process_rss(monkeypatch, memory_monitor, 2.0)

    client = MLXLocalClient("/models/Aura-Cortex-32B-MLX")
    request_lock_calls = 0

    async def record_unexpected_request_lock(*_args, **_kwargs):
        nonlocal request_lock_calls
        request_lock_calls += 1
        return False

    monkeypatch.setattr(client, "_acquire_request_lock", record_unexpected_request_lock)

    result = await client.generate("hello", foreground_request=True, origin="desktop")

    assert result is None
    assert request_lock_calls == 0
    assert client._lane_state == "cold"
