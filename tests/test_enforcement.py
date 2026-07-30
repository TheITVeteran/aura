"""Enforcement backends: real defensive actions behind the immune seams."""
from __future__ import annotations

from core.security.enforcement import (
    AppLayerFirewall,
    Quarantine,
    ResourceMonitor,
    arp_scan,
    install_default_enforcement,
)

# ── app-layer firewall ──────────────────────────────────────────────────────

def test_firewall_blocks_and_unblocks():
    fw = AppLayerFirewall()
    fw.block("1.2.3.4")
    assert fw.is_blocked("1.2.3.4")
    assert "1.2.3.4" in fw.blocked()
    fw.unblock("1.2.3.4")
    assert not fw.is_blocked("1.2.3.4")


def test_firewall_ignores_local_and_unknown():
    fw = AppLayerFirewall()
    fw.block("local")
    fw.block("unknown")
    fw.block("")
    assert fw.blocked() == []


# ── quarantine ───────────────────────────────────────────────────────────────

def test_quarantine_isolates_a_file(tmp_path):
    q = Quarantine(quarantine_dir=tmp_path / "q")
    suspect = tmp_path / "malware.bin"
    suspect.write_bytes(b"evil")
    dest = q.isolate(str(suspect))
    assert dest is not None
    assert not suspect.exists()        # moved out of harm's way
    from pathlib import Path
    assert Path(dest).exists()


def test_quarantine_missing_file_is_safe(tmp_path):
    q = Quarantine(quarantine_dir=tmp_path / "q")
    assert q.isolate(str(tmp_path / "nope")) is None


# ── resource monitor → immune ────────────────────────────────────────────────

def test_resource_monitor_reports_strain(monkeypatch):
    mon = ResourceMonitor(cpu_high=80.0, mem_high=80.0, disk_high=95.0)
    monkeypatch.setattr(mon, "sample", lambda: {"cpu": 99.0, "mem": 40.0, "disk": 10.0, "procs": 300.0})

    fed = []
    import core.security.immune_system as im

    class _Stub:
        def assess(self, *a, **k):
            fed.append(k.get("threat_class"))
    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())

    report = mon.check_and_report()
    assert report is not None
    assert "cpu" in report["breaches"]
    assert fed, "resource strain was not reported to the immune system"


def test_resource_monitor_quiet_when_healthy(monkeypatch):
    mon = ResourceMonitor()
    monkeypatch.setattr(mon, "sample", lambda: {"cpu": 10.0, "mem": 30.0, "disk": 40.0, "procs": 200.0})
    assert mon.check_and_report() is None


def test_resource_monitor_failopen_on_no_sample(monkeypatch):
    mon = ResourceMonitor()
    monkeypatch.setattr(mon, "sample", lambda: {})
    assert mon.check_and_report() is None


# ── arp scanner (own network) ────────────────────────────────────────────────

def test_arp_scan_parses_devices(monkeypatch):
    from types import SimpleNamespace
    sample = (
        "router (192.168.1.1) at dc:eb:69:85:68:17 on en0 ifscope [ethernet]\n"
        "? (192.168.1.42) at 3a:87:3c:de:5c:7c on en0 ifscope [ethernet]\n"
        "? (192.168.1.99) at (incomplete) on en0 ifscope [ethernet]\n"
    )
    import core.runtime.subprocess_gateway as sg
    import core.security.enforcement as enforcement
    monkeypatch.setattr(
        sg, "get_subprocess_gateway",
        lambda: SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=sample, returncode=0)),
    )
    monkeypatch.setattr(enforcement, "_local_interface_macs", lambda: set())
    devices = arp_scan()
    by_mac = {d.fingerprint: d for d in devices}
    macs = set(by_mac)
    assert "dc:eb:69:85:68:17" in macs
    assert "3a:87:3c:de:5c:7c" in macs
    assert len(devices) == 2   # the incomplete entry is skipped
    assert by_mac["3a:87:3c:de:5c:7c"].name == "192.168.1.42"
    assert by_mac["3a:87:3c:de:5c:7c"].ip == "192.168.1.42"
    assert by_mac["3a:87:3c:de:5c:7c"].interface == "en0"
    assert by_mac["3a:87:3c:de:5c:7c"].scanner_source == "arp"


def test_arp_scan_filters_local_broadcast_multicast_and_malformed(monkeypatch):
    from types import SimpleNamespace

    sample = (
        "? (192.168.1.2) at 02:00:00:00:00:01 on en0 ifscope [ethernet]\n"
        "? (192.168.1.3) at ff:ff:ff:ff:ff:ff on en0 ifscope [ethernet]\n"
        "? (224.0.0.1) at 02:00:00:00:00:03 on en0 ifscope [ethernet]\n"
        "? (192.168.1.4) at 01:00:5e:00:00:01 on en0 ifscope [ethernet]\n"
        "? (192.168.1.5) at 02:00:00:00:00:05 on en0 ifscope [ethernet]\n"
    )
    import core.runtime.subprocess_gateway as sg
    import core.security.enforcement as enforcement

    monkeypatch.setattr(
        sg,
        "get_subprocess_gateway",
        lambda: SimpleNamespace(
            run=lambda *args, **kwargs: SimpleNamespace(
                stdout=sample,
                returncode=0,
            )
        ),
    )
    monkeypatch.setattr(
        enforcement,
        "_local_interface_macs",
        lambda: {"02:00:00:00:00:01"},
    )

    devices = arp_scan()

    assert [device.ip for device in devices] == ["192.168.1.5"]


def test_arp_scan_failopen(monkeypatch):
    import core.runtime.subprocess_gateway as sg

    monkeypatch.setattr(
        sg,
        "get_subprocess_gateway",
        lambda: (_ for _ in ()).throw(RuntimeError("no arp")),
    )
    assert arp_scan() == []


# ── installation wires the seams ─────────────────────────────────────────────

def test_install_registers_mitigations_and_scanner():
    result = install_default_enforcement()
    assert result["installed"] is True
    from core.security.immune_system import get_immune_system
    handlers = get_immune_system()._handlers
    assert "isolate" in handlers and "quarantine" in handlers and "rate_limit" in handlers
    # a blocked origin flows through the registered isolate handler
    from core.security.enforcement import get_firewall
    get_firewall().unblock("9.9.9.9")
    get_immune_system().assess_and_respond(
        "test", "exploit", severity=0.9, origin="9.9.9.9", targeted_vuln="v",
    )
    assert get_firewall().is_blocked("9.9.9.9")
