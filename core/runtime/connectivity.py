"""Bounded external connectivity probe for live capability planning.

This service does not perform browsing or content fetches. It only answers the
question: "does this machine currently appear to have usable internet?" so
conversation and tool routing can degrade honestly when web access is absent.
"""
from __future__ import annotations

import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation


@dataclass(frozen=True)
class ConnectivityStatus:
    checked_at: float
    online: bool
    mode: str
    target: str
    latency_ms: float | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConnectivityProbe:
    """Low-cost TCP connect probe with cache and explicit offline mode."""

    def __init__(
        self,
        *,
        target: str | None = None,
        port: int | None = None,
        timeout_s: float | None = None,
        ttl_s: float | None = None,
    ) -> None:
        self.target = target or os.environ.get("AURA_CONNECTIVITY_TARGET", "1.1.1.1")
        self.port = int(port or os.environ.get("AURA_CONNECTIVITY_PORT", "53"))
        self.timeout_s = float(timeout_s or os.environ.get("AURA_CONNECTIVITY_TIMEOUT_S", "0.6"))
        self.ttl_s = float(ttl_s or os.environ.get("AURA_CONNECTIVITY_TTL_S", "20"))
        self._last: ConnectivityStatus | None = None

    def status(self, *, force: bool = False) -> ConnectivityStatus:
        offline_override = os.environ.get("AURA_FORCE_OFFLINE", "").strip().lower() in {"1", "true", "on", "yes"}
        now = time.time()
        if offline_override:
            self._last = ConnectivityStatus(
                checked_at=now,
                online=False,
                mode="forced_offline",
                target=f"{self.target}:{self.port}",
                reason="AURA_FORCE_OFFLINE is set",
            )
            return self._last
        if not force and self._last and now - self._last.checked_at < self.ttl_s:
            return self._last
        started = time.perf_counter()
        try:
            with socket.create_connection((self.target, self.port), timeout=self.timeout_s):
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._last = ConnectivityStatus(
                checked_at=now,
                online=True,
                mode="tcp_probe",
                target=f"{self.target}:{self.port}",
                latency_ms=latency_ms,
            )
        except OSError as exc:
            record_degradation(
                "connectivity_probe",
                exc,
                severity="debug",
                action="marked external internet unavailable for this planning window",
            )
            self._last = ConnectivityStatus(
                checked_at=now,
                online=False,
                mode="tcp_probe",
                target=f"{self.target}:{self.port}",
                reason=str(exc)[:180],
            )
        return self._last


_PROBE: ConnectivityProbe | None = None


def get_connectivity_probe() -> ConnectivityProbe:
    global _PROBE
    if _PROBE is None:
        _PROBE = ConnectivityProbe()
    return _PROBE


def get_connectivity_status(*, force: bool = False) -> ConnectivityStatus:
    return get_connectivity_probe().status(force=force)


def render_connectivity_prompt_block(status: ConnectivityStatus | dict[str, Any] | None) -> str:
    if status is None:
        return ""
    data = status.to_dict() if isinstance(status, ConnectivityStatus) else dict(status)
    online = bool(data.get("online"))
    mode = str(data.get("mode") or "unknown")
    target = str(data.get("target") or "unknown")
    if online:
        latency = data.get("latency_ms")
        return (
            "## CONNECTIVITY\n"
            f"- External internet appears available via {mode} ({target}, latency_ms={latency}).\n"
            "- Web/browser actions still require normal governance and tool receipts."
        )
    reason = str(data.get("reason") or "probe failed")
    return (
        "## CONNECTIVITY\n"
        f"- External internet is not currently verified ({mode}, {target}): {reason}\n"
        "- Answer from local knowledge when possible. For tasks requiring web access, explain the limitation in Aura's normal voice and do not fabricate live sources."
    )


__all__ = [
    "ConnectivityProbe",
    "ConnectivityStatus",
    "get_connectivity_probe",
    "get_connectivity_status",
    "render_connectivity_prompt_block",
]
