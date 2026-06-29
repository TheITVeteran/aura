"""Network sentinel: baseline awareness, anomaly detection, recoverability — own environment only."""
from __future__ import annotations

import pytest

from core.security.network_sentinel import Device, NetworkSentinel, get_network_sentinel


@pytest.fixture
def net():
    # settle period 0 so post-baseline anomalies are detected immediately in tests
    return NetworkSentinel(settle_period_s=0.0)


def test_known_device_is_normal(net):
    net.learn_baseline([Device("aa:bb:cc", name="home-router", kind="router")])
    v = net.observe(Device("aa:bb:cc", name="home-router"), now=10_000.0)
    assert v.known and not v.anomalous and v.action == "normal"


def test_new_device_after_baseline_is_anomalous(net):
    net.learn_baseline([Device("aa:bb:cc", name="router")])
    v = net.observe(Device("de:ad:be:ef", name="mystery-laptop"), now=10_000.0)
    assert not v.known and v.anomalous
    assert v.action == "investigate"
    assert v.threat > 0


def test_settle_in_learns_devices_as_baseline():
    net = NetworkSentinel(settle_period_s=3600.0)  # still settling
    v = net.observe(Device("new:01", name="phone"))
    assert not v.anomalous          # learned, not flagged
    assert v.action == "observe"


def test_anomaly_feeds_immune_system(net, monkeypatch):
    fed = []
    import core.security.immune_system as im

    class _Stub:
        def assess(self, *a, **k):
            fed.append(k.get("threat_class"))
    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())

    net.learn_baseline([Device("aa:bb:cc")])
    net.observe(Device("intruder:99"), now=10_000.0)
    assert fed, "anomalous device did not notify the immune system"


def test_enumerate_uses_registered_scanner(net):
    net.register_scanner(lambda: [Device("x:1", name="a"), Device("x:2", name="b")])
    devices = net.enumerate()
    assert {d.fingerprint for d in devices} == {"x:1", "x:2"}


def test_sweep_assesses_each_device(net):
    net.learn_baseline([Device("known:1")])
    net.register_scanner(lambda: [Device("known:1"), Device("unknown:2")])
    verdicts = net.sweep()
    by_fp = {v.fingerprint: v for v in verdicts}
    assert by_fp["known:1"].known
    assert by_fp["unknown:2"].anomalous


def test_enumerate_failopen_on_broken_scanner(net):
    attempts = []

    def _failing_scanner():
        attempts.append("called")
        raise RuntimeError("scan failed")
    net.register_scanner(_failing_scanner)
    assert net.enumerate() == []   # never raises
    assert attempts == ["called"]


def test_recovery_plan_reports_restore_points(net):
    plan = net.recovery_plan()
    assert "restore_points" in plan
    assert isinstance(plan["recoverable"], bool)


def test_singleton_stable():
    assert get_network_sentinel() is get_network_sentinel()
