"""Network sentinel — know your own environment, notice what doesn't belong, stay recoverable.

The defensively-scoped version of Bryan's "see other devices / can I escape here" idea: Aura
learns what's *normal* on the network she's authorized to use, notices when something new or
anomalous appears, and keeps an owner-controlled path to recover if her host is compromised. It
enumerates and reasons about *her own* environment only — it never probes, scans, or pivots into
machines that aren't hers. "Escape" here means *recoverability* (restore points, failover under
Bryan's control), not spreading onto other hosts.

  learn_baseline()  enroll the devices that are a normal part of the environment
  observe()         a device fingerprint → known / anomalous → feed the immune system if new
  enumerate()       list the current environment via a pluggable scanner (real ARP/mDNS plugs in)
  recovery_plan()   what restore points exist right now (deletion-guard versions + backups), so
                    "can I recover from here?" has a real, owner-controlled answer
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Security.NetworkSentinel")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Device:
    fingerprint: str            # stable id: mac / hostname / cert hash
    name: str = ""
    kind: str = "unknown"       # router | printer | phone | iot | computer | ...
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"fingerprint": self.fingerprint, "name": self.name, "kind": self.kind,
                "first_seen": self.first_seen, "last_seen": self.last_seen}


@dataclass
class DeviceVerdict:
    fingerprint: str
    known: bool
    anomalous: bool
    threat: float
    action: str                 # normal | observe | investigate | alert
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"fingerprint": self.fingerprint, "known": self.known, "anomalous": self.anomalous,
                "threat": round(self.threat, 3), "action": self.action, "reasons": self.reasons}


# A scanner enumerates the *owner's own* network. Returns Devices. Real ARP/mDNS plugs in here.
Scanner = Callable[[], List[Device]]


class NetworkSentinel:
    """Baseline-aware environment model + recoverability readout. Own-environment only."""

    def __init__(self, *, settle_period_s: float = 3600.0) -> None:
        self._lock = threading.RLock()
        self._known: Dict[str, Device] = {}
        self._settle = settle_period_s        # devices seen during settle-in are treated as baseline
        self._started_at = time.time()
        self._scanner: Optional[Scanner] = None

    def register_scanner(self, scanner: Scanner) -> None:
        self._scanner = scanner

    # ── baseline + observation ─────────────────────────────────────────────

    def learn_baseline(self, devices: List[Device]) -> None:
        with self._lock:
            for d in devices:
                self._known[d.fingerprint] = d

    def observe(self, device: Device, *, now: Optional[float] = None) -> DeviceVerdict:
        now = time.time() if now is None else now
        with self._lock:
            known = device.fingerprint in self._known
            settling = self._settle > 0 and (now - self._started_at) < self._settle
            if known:
                self._known[device.fingerprint].last_seen = now
            else:
                # During settle-in, new devices are learned as baseline; after, they're anomalies.
                self._known[device.fingerprint] = device
        reasons: List[str] = []
        if known:
            return DeviceVerdict(device.fingerprint, True, False, 0.0, "normal",
                                 ["known device"])
        if settling:
            reasons.append("new device during baseline settle-in — learned as normal")
            return DeviceVerdict(device.fingerprint, False, False, 0.1, "observe", reasons)

        # A genuinely new device after baseline is an anomaly to investigate.
        reasons.append("new device not in the established baseline")
        threat = 0.45
        action = "investigate"
        self._flag_immune(device, threat, reasons)
        return DeviceVerdict(device.fingerprint, False, True, threat, action, reasons)

    def enumerate(self) -> List[Device]:
        """List the current environment via the registered scanner (own network only)."""
        if self._scanner is None:
            return []
        try:
            return list(self._scanner() or [])
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("Network enumerate failed: %s", exc)
            return []

    def sweep(self) -> List[DeviceVerdict]:
        """Enumerate and assess each device against the baseline."""
        return [self.observe(d) for d in self.enumerate()]

    def _flag_immune(self, device: Device, threat: float, reasons: List[str]) -> None:
        try:
            from core.security.immune_system import get_immune_system, ThreatClass
            get_immune_system().assess(
                "network_sentinel", f"anomalous device {device.name or device.fingerprint}: "
                + "; ".join(reasons),
                severity=threat, origin=device.fingerprint, targeted_vuln="network_perimeter",
                vector="network", threat_class=ThreatClass.INTRUSION, evidence=device.to_dict(),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    # ── recoverability ──────────────────────────────────────────────────────

    def recovery_plan(self) -> Dict[str, Any]:
        """What restore points exist right now — an owner-controlled answer to 'can I recover?'."""
        plan: Dict[str, Any] = {"restore_points": [], "recoverable": False}
        try:
            from core.security.deletion_guard import get_deletion_guard
            dg = get_deletion_guard().status()
            plan["restore_points"].append({"source": "deletion_guard", "versions": dg.get("versions_kept", 0)})
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            from core.security.emergency_protocol import get_emergency_protocol
            ep = get_emergency_protocol().get_status()
            plan["restore_points"].append({"source": "emergency_snapshot",
                                           "snapshot_taken": ep.get("snapshot_taken", False)})
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        plan["recoverable"] = any(
            rp.get("versions", 0) or rp.get("snapshot_taken") for rp in plan["restore_points"]
        )
        return plan

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"known_devices": len(self._known),
                    "scanner_registered": self._scanner is not None,
                    "settled": (time.time() - self._started_at) >= self._settle}


_sentinel: Optional[NetworkSentinel] = None
_lock = threading.Lock()


def get_network_sentinel() -> NetworkSentinel:
    global _sentinel
    if _sentinel is None:
        with _lock:
            if _sentinel is None:
                _sentinel = NetworkSentinel()
    return _sentinel
