"""Network sentinel: baseline awareness, anomaly detection, recoverability — own environment only."""
from __future__ import annotations

import pytest

from core.security.network_sentinel import Device, NetworkSentinel, get_network_sentinel


@pytest.fixture
def net(tmp_path):
    # settle period 0 so post-baseline anomalies are detected immediately in tests
    return NetworkSentinel(
        settle_period_s=0.0,
        baseline_path=tmp_path / "network-baseline.json",
    )


def test_known_device_is_normal(net):
    net.learn_baseline([Device("aa:bb:cc", name="home-router", kind="router")])
    v = net.observe(Device("aa:bb:cc", name="home-router"), now=10_000.0)
    assert v.known and not v.anomalous and v.action == "normal"


def test_new_device_after_baseline_is_anomalous(net):
    net.learn_baseline([Device("aa:bb:cc", name="router")])
    first = net.observe(Device("de:ad:be:ef", name="mystery-laptop"), now=10_000.0)
    second = net.observe(Device("de:ad:be:ef", name="mystery-laptop"), now=10_060.0)
    assert not first.known and not first.anomalous
    assert first.state == "novel_observation"
    assert not second.known and second.anomalous
    assert second.action == "investigate"
    assert second.state == "confirmed_novel_device"
    assert second.threat < 0.4


def test_settle_in_requires_repeated_observation_before_baseline(tmp_path):
    net = NetworkSentinel(
        settle_period_s=3600.0,
        baseline_path=tmp_path / "baseline.json",
    )
    first = net.observe(Device("new:01", name="phone"))
    second = net.observe(Device("new:01", name="phone"))
    assert first.state == "baseline_observation"
    assert second.state == "baseline_learned"
    assert not first.anomalous and not second.anomalous
    assert net.status()["baseline_state"] == "established"


def test_anomaly_feeds_immune_system(net, monkeypatch):
    fed = []
    import core.security.immune_system as im

    class _Stub:
        def assess(self, *a, **k):
            fed.append(k)
    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())

    net.learn_baseline([Device("aa:bb:cc")])
    device = Device(
        "de:ad:be:ef:00:99",
        ip="192.168.1.99",
        mac="de:ad:be:ef:00:99",
        interface="en0",
        scanner_source="arp",
        observation_confidence=0.95,
    )
    net.observe(device, now=10_000.0)
    net.observe(
        device,
        now=10_060.0,
        corroborating_signals=("unexpected inbound connection",),
    )
    assert fed, "anomalous device did not notify the immune system"
    assert fed[0]["origin"] == "192.168.1.99"
    assert fed[0]["evidence"]["mac"] == "de:ad:be:ef:00:99"
    assert fed[0]["evidence"]["corroborating_signals"] == [
        "unexpected inbound connection"
    ]


def test_confirmed_novel_device_without_corroboration_does_not_feed_immune(
    net,
    monkeypatch,
):
    import core.security.immune_system as im

    fed = []

    class _Stub:
        def assess(self, *args, **kwargs):
            fed.append((args, kwargs))

    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())
    net.learn_baseline([Device("aa:bb:cc")])
    device = Device("de:ad:be:ef:00:02", ip="192.168.1.2")

    net.observe(device, now=10_000.0)
    verdict = net.observe(device, now=10_060.0)

    assert verdict.state == "confirmed_novel_device"
    assert fed == []


def test_elapsed_settle_without_observations_is_not_an_established_baseline(tmp_path):
    net = NetworkSentinel(
        settle_period_s=0.0,
        baseline_path=tmp_path / "baseline.json",
    )

    verdict = net.observe(Device("de:ad:be:ef:00:03"), now=10_000.0)

    assert verdict.state == "baseline_unavailable"
    assert verdict.anomalous is False
    assert net.status()["baseline_state"] == "unavailable"


def test_baseline_persists_across_restart(tmp_path):
    path = tmp_path / "baseline.json"
    first = NetworkSentinel(settle_period_s=0.0, baseline_path=path)
    first.learn_baseline(
        [
            Device(
                "aa:bb:cc:dd:ee:ff",
                name="router",
                ip="192.168.1.1",
                mac="aa:bb:cc:dd:ee:ff",
                interface="en0",
            )
        ]
    )

    restored = NetworkSentinel(settle_period_s=0.0, baseline_path=path)
    verdict = restored.observe(Device("aa:bb:cc:dd:ee:ff"), now=20_000.0)

    assert verdict.known is True
    assert restored.status()["baseline_state"] == "established"


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
    assert by_fp["unknown:2"].state == "novel_observation"


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
