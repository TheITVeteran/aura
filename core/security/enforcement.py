"""Enforcement backends — the hands the immune system reaches for.

The immune system decides; this is what actually carries out a defensive action on the host.
Each backend is registered as a mitigation handler (so the decision layer stays clean and
auditable) and is strictly defensive: block an origin, quarantine a file, kill a runaway
process Aura owns, throttle a flood. Anything needing elevated privileges (a real pf firewall
rule, system-wide process control) is attempted only when those privileges exist and fails
open to an effective app-layer equivalent otherwise — Aura never pretends she enforced
something she couldn't.

It also wires the real sensors: a psutil-backed resource monitor that feeds the
exhaustion/flood detectors, and an ``arp -a`` scanner that gives the network sentinel the
actual device list for her own network. No scanning or action against machines that aren't
hers.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("Security.Enforcement")


class AppLayerFirewall:
    """An in-process blocklist Aura's own servers/clients consult — effective without root."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blocked: Set[str] = set()
        self._blocked_at: Dict[str, float] = {}

    def block(self, origin: str, *, now: Optional[float] = None) -> None:
        if not origin or origin in {"unknown", "local"}:
            return
        with self._lock:
            self._blocked.add(origin)
            self._blocked_at[origin] = time.time() if now is None else now
        logger.warning("🛡️ [Firewall] app-layer block on %s", origin)
        self._try_pf_block(origin)

    def is_blocked(self, origin: str) -> bool:
        with self._lock:
            return origin in self._blocked

    def unblock(self, origin: str) -> None:
        with self._lock:
            self._blocked.discard(origin)
            self._blocked_at.pop(origin, None)

    def blocked(self) -> List[str]:
        with self._lock:
            return sorted(self._blocked)

    @staticmethod
    def _try_pf_block(origin: str) -> None:
        """Best-effort kernel firewall rule via pf — only if we have the privileges. Fail-open."""
        if os.geteuid() != 0:  # type: ignore[attr-defined]
            return  # no root → app-layer block already applied; don't pretend
        if not re.match(r"^[0-9a-fA-F:.]+$", origin):  # only IP-ish origins to pfctl
            return
        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            from core.governance_context import local_internal_governed_scope
            with local_internal_governed_scope("security.enforcement.pf_block", domain="tool_execution"):
                get_subprocess_gateway().run(
                    ["pfctl", "-t", "aura_block", "-T", "add", origin],
                    read_only=False, timeout=3.0, source="security.enforcement",
                )
        except Exception as exc:  # noqa: BLE001 - pf is best-effort; app-layer block stands
            logger.debug("pf block unavailable for %s: %s", origin, exc)


class ProcessGuard:
    """Kills a runaway/hostile process — own-user processes only, never arbitrary system pids."""

    @staticmethod
    def terminate(pid: int) -> bool:
        try:
            import psutil
            p = psutil.Process(int(pid))
            # Only ever act on processes owned by the same user as Aura.
            if p.uids().real != os.getuid():  # type: ignore[attr-defined]
                logger.warning("Refusing to kill pid %s — not owned by Aura's user", pid)
                return False
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                p.kill()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Process terminate failed for %s: %s", pid, exc)
            return False


class Quarantine:
    """Moves a suspect file out of harm's way into an isolated, restorable store."""

    def __init__(self, quarantine_dir: Optional[Path] = None) -> None:
        if quarantine_dir is None:
            try:
                from core.config import config
                quarantine_dir = Path(config.paths.home_dir) / "data" / "quarantine"
            except (ImportError, AttributeError, RuntimeError):
                quarantine_dir = Path.home() / ".aura" / "data" / "quarantine"
        self._dir = Path(quarantine_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def isolate(self, path: str) -> Optional[str]:
        try:
            src = Path(path)
            if not src.exists():
                return None
            dest = self._dir / f"q-{int(time.time())}-{src.name}"
            shutil.move(str(src), str(dest))
            try:
                dest.chmod(0o400)  # read-only, defang
            except OSError:
                pass
            logger.warning("🔒 [Quarantine] isolated %s → %s", path, dest)
            return str(dest)
        except (OSError, ValueError) as exc:
            logger.debug("Quarantine failed for %s: %s", path, exc)
            return None


class ResourceMonitor:
    """psutil-backed sampling that feeds the exhaustion detector when the host is under strain."""

    def __init__(self, *, cpu_high: float = 92.0, mem_high: float = 90.0, disk_high: float = 95.0) -> None:
        self._cpu_high = cpu_high
        self._mem_high = mem_high
        self._disk_high = disk_high

    def sample(self) -> Dict[str, float]:
        try:
            import psutil
            return {
                "cpu": float(psutil.cpu_percent(interval=0.0)),
                "mem": float(psutil.virtual_memory().percent),
                "disk": float(psutil.disk_usage("/").percent),
                "procs": float(len(psutil.pids())),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Resource sample failed: %s", exc)
            return {}

    def check_and_report(self) -> Optional[Dict[str, Any]]:
        s = self.sample()
        if not s:
            return None
        breaches = []
        if s.get("cpu", 0) >= self._cpu_high:
            breaches.append(("cpu", s["cpu"]))
        if s.get("mem", 0) >= self._mem_high:
            breaches.append(("mem", s["mem"]))
        if s.get("disk", 0) >= self._disk_high:
            breaches.append(("disk", s["disk"]))
        if not breaches:
            return None
        worst = max(breaches, key=lambda kv: kv[1])
        try:
            from core.security.immune_system import get_immune_system, ThreatClass
            get_immune_system().assess(
                "resource_monitor", f"resource strain: {worst[0]} at {worst[1]:.0f}%",
                severity=min(0.9, worst[1] / 100.0), origin="host",
                targeted_vuln="resource_exhaustion", vector="compute",
                threat_class=ThreatClass.RESOURCE_EXHAUSTION, evidence=s,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return {"breaches": dict(breaches), "sample": s}


# ── network scanner (own environment) ───────────────────────────────────────

_ARP_LINE = re.compile(r"^(?P<host>[^\s(]+)?\s*\((?P<ip>[0-9.]+)\)\s+at\s+(?P<mac>[0-9a-fA-F:]+)")


def arp_scan() -> List[Any]:
    """Enumerate the local network from the ARP table (unprivileged, own network only)."""
    devices: List[Any] = []
    try:
        from core.security.network_sentinel import Device
        from core.runtime.subprocess_gateway import get_subprocess_gateway
        proc = get_subprocess_gateway().run(
            ["/usr/sbin/arp", "-a"], read_only=True, timeout=4.0, source="security.enforcement.arp",
        )
        for line in (proc.stdout or "").splitlines():
            m = _ARP_LINE.match(line.strip())
            if not m or "incomplete" in line:
                continue
            mac = m.group("mac")
            if mac and mac != "(incomplete)":
                devices.append(Device(fingerprint=mac, name=m.group("host") or m.group("ip"),
                                      kind="network_host"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("ARP scan failed: %s", exc)
    return devices


# ── installation: wire the backends into the seams ──────────────────────────

_firewall: Optional[AppLayerFirewall] = None
_quarantine: Optional[Quarantine] = None
_installed = False
_install_lock = threading.Lock()


def get_firewall() -> AppLayerFirewall:
    global _firewall
    if _firewall is None:
        _firewall = AppLayerFirewall()
    return _firewall


def install_default_enforcement() -> Dict[str, Any]:
    """Register the real enforcement backends into the immune system + network sentinel."""
    global _installed, _quarantine
    with _install_lock:
        if _installed:
            return {"installed": True, "already": True}
        fw = get_firewall()
        _quarantine = Quarantine()

        from core.security.immune_system import get_immune_system, ThreatEvent
        immune = get_immune_system()

        def _block(ev: "ThreatEvent") -> Optional[str]:
            fw.block(ev.origin)
            return f"unblock-{ev.origin}"

        def _quarantine_handler(ev: "ThreatEvent") -> Optional[str]:
            path = ev.evidence.get("path") if isinstance(ev.evidence, dict) else None
            if path:
                return _quarantine.isolate(str(path))
            return None

        def _rate_limit(ev: "ThreatEvent") -> Optional[str]:
            fw.block(ev.origin)   # at the app layer, rate-limit a flood == block the source
            return f"unblock-{ev.origin}"

        immune.register_mitigation("isolate", _block)
        immune.register_mitigation("alert", lambda ev: None)   # alerting handled by reflex/log
        immune.register_mitigation("quarantine", _quarantine_handler)
        immune.register_mitigation("rate_limit", _rate_limit)

        try:
            from core.security.network_sentinel import get_network_sentinel
            get_network_sentinel().register_scanner(arp_scan)
        except (ImportError, AttributeError, RuntimeError):
            pass

        _installed = True
        return {"installed": True, "mitigations": sorted(immune._handlers), "scanner": "arp"}
