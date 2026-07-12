from types import SimpleNamespace

from core.utils.memory_monitor import AppleSiliconMemoryMonitor


def test_memory_monitor_clamps_psutil_percent(monkeypatch):
    from core.runtime.resource_observation import HostResourceObserver

    monitor = AppleSiliconMemoryMonitor(observer=HostResourceObserver())
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=132.4),
    )

    assert monitor._get_pressure_sysctl() == 100


def test_memory_monitor_uses_available_total_when_percent_missing(monkeypatch):
    from core.runtime.resource_observation import HostResourceObserver

    monitor = AppleSiliconMemoryMonitor(observer=HostResourceObserver())
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=None, total=1_000, available=250),
    )

    assert monitor._get_pressure_sysctl() == 75


def test_memory_monitor_fails_closed_on_sampling_error():
    observer = SimpleNamespace(
        memory=lambda: SimpleNamespace(
            available=False,
            error="sample failed",
        )
    )
    monitor = AppleSiliconMemoryMonitor(observer=observer)

    assert monitor._get_pressure_sysctl() == 100
