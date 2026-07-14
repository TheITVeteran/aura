"""core/consciousness/whole_system_phi_service.py — live whole-system Φ.
==========================================================================
The runtime host for core/consciousness/integrated_information.py:
harvests real channels each cognitive cycle, keeps a sliding window, and
periodically produces a provenance-complete PhiEstimate off the event
loop.  Registered as ``whole_system_phi``; the report is consumed by the
phi phase (state metadata + logs), persisted through the governed write
gateway, and exposed via status() for health/observability.

This SUPERSEDES the epistemics of the hand-picked 16-node φ: channels are
whatever the runtime actually exposes, the complex is discovered by grain
search, the MIP is exact, and every number ships with its null, CI, and
bounded claim.  The legacy phi scalar keeps its scale for downstream
gates; this service adds the honest measurement alongside, plus the
interventional rows from the perturbational probe.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.service_access import optional_service

logger = logging.getLogger("Aura.WholeSystemPhi")

_MIN_SAMPLES_FLAG = declare(
    "AURA_WSPHI_MIN_SAMPLES", kind=FlagKind.INT, default=240,
    description="Observations required before the first whole-system Φ estimate",
    owner="core.consciousness.whole_system_phi_service",
)
_WINDOW_FLAG = declare(
    "AURA_WSPHI_WINDOW", kind=FlagKind.INT, default=1200,
    description="Sliding-window length (observations) for whole-system Φ",
    owner="core.consciousness.whole_system_phi_service",
)
_EVERY_FLAG = declare(
    "AURA_WSPHI_ESTIMATE_EVERY", kind=FlagKind.INT, default=180,
    description="Recompute whole-system Φ every N new observations",
    owner="core.consciousness.whole_system_phi_service",
)
_PHI_DIR_FLAG = declare(
    "AURA_PHI_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for persisted whole-system Φ reports",
    owner="core.consciousness.whole_system_phi_service",
)


def _maybe(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


class WholeSystemPhiService:
    """Sliding-window channel collector + periodic estimator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._window: deque[dict[str, float]] = deque(maxlen=int(_WINDOW_FLAG.value()))
        self._since_estimate = 0
        self._latest: Any = None            # PhiEstimate
        self._latest_probe: dict[str, Any] = {}
        self._interventional: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._estimating = False
        self._estimates_done = 0
        self._last_error = ""

    # ── channel harvest ──────────────────────────────────────────────────
    def sample_runtime_channels(self, state: Any = None) -> dict[str, float]:
        """One reading of every cheap live scalar the runtime exposes.
        Each source is optional and guarded; a missing organ just means a
        narrower channel set — reported, never fabricated."""
        channels: dict[str, float] = {}

        try:
            affect = optional_service("affect_engine", "affect_facade", default=None)
            if affect is not None and hasattr(affect, "get_state_sync"):
                st = affect.get_state_sync()
                if isinstance(st, dict):
                    for key in ("valence", "arousal", "dominance"):
                        v = _maybe(st.get(key))
                        if v is not None:
                            channels[f"affect.{key}"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="affect channels skipped")

        try:
            stakes = optional_service("existential_stakes", default=None)
            if stakes is not None and hasattr(stakes, "get_existential_threat"):
                v = _maybe(stakes.get_existential_threat())
                if v is not None:
                    channels["survival.threat"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="stakes channel skipped")

        try:
            unity = optional_service("unity_state", default=None)
            if unity is not None:
                for attr, name in (("unity_score", "unity.score"),
                                   ("fragmentation_score", "unity.fragmentation")):
                    v = _maybe(getattr(unity, attr, None))
                    if v is not None:
                        channels[name] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="unity channels skipped")

        try:
            will = optional_service("unified_will", default=None)
            wstate = getattr(will, "_state", None)
            if wstate is not None:
                for attr in ("confidence", "assertiveness", "identity_coherence"):
                    v = _maybe(getattr(wstate, attr, None))
                    if v is not None:
                        channels[f"will.{attr}"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="will channels skipped")

        try:
            covenant = optional_service("ulysses_covenant", default=None)
            if covenant is not None and hasattr(covenant, "integrity_score"):
                v = _maybe(covenant.integrity_score())
                if v is not None:
                    channels["covenant.integrity"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="covenant channel skipped")

        if state is not None:
            for path, name in (
                (("consciousness", "phi"), "state.phi"),
                (("consciousness", "arousal"), "state.arousal"),
                (("cognition", "cognitive_load"), "state.cognitive_load"),
                (("phenomenal", "intensity"), "state.phenomenal_intensity"),
            ):
                obj = state
                for part in path:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                v = _maybe(obj)
                if v is not None:
                    channels[name] = v

        try:
            import psutil

            proc = psutil.Process()
            channels["body.rss_gb"] = proc.memory_info().rss / 1e9
            cpu = _maybe(proc.cpu_percent(interval=None))
            if cpu is not None:
                channels["body.cpu_pct"] = cpu
        except (ImportError, OSError, RuntimeError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="body channels skipped")

        return channels

    def observe(self, channels: dict[str, float]) -> None:
        clean = {str(k): v for k, v in (channels or {}).items()
                 if _maybe(v) is not None}
        if len(clean) < 2:
            return
        with self._lock:
            self._window.append(clean)
            self._since_estimate += 1

    def observe_runtime(self, state: Any = None) -> None:
        self.observe(self.sample_runtime_channels(state))

    def add_interventional_transitions(
        self, transitions: list[tuple[tuple[int, ...], tuple[int, ...]]],
        probe_report: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._interventional.extend(transitions or [])
            self._interventional = self._interventional[-200:]
            if probe_report:
                self._latest_probe = dict(probe_report)

    # ── estimation ───────────────────────────────────────────────────────
    def _matrix(self) -> tuple[Any, tuple[str, ...]]:
        import numpy as np

        with self._lock:
            rows = list(self._window)
        # channels present in ≥90% of the window (late-boot organs join later)
        counts: dict[str, int] = {}
        for r in rows:
            for k in r:
                counts[k] = counts.get(k, 0) + 1
        names = tuple(sorted(k for k, c in counts.items()
                             if c >= 0.9 * len(rows)))
        X = np.asarray(
            [[row.get(k, 0.0) for k in names] for row in rows], dtype=float
        )
        return X, names

    def ready(self) -> bool:
        with self._lock:
            return (len(self._window) >= int(_MIN_SAMPLES_FLAG.value())
                    and self._since_estimate >= int(_EVERY_FLAG.value()))

    def maybe_estimate(self) -> Any:
        """Heavy — run off-loop (asyncio.to_thread).  Returns the fresh
        PhiEstimate when due, else None."""
        with self._lock:
            if self._estimating or not self.ready():
                return None
            self._estimating = True
        try:
            from core.consciousness.integrated_information import (
                estimate_whole_system_phi,
            )

            X, names = self._matrix()
            if X.shape[1] < 4:
                return None
            with self._lock:
                extra = list(self._interventional)
            est = estimate_whole_system_phi(
                X, channel_names=names,
                seed=int(time.time()) % 100_000,
                extra_transitions=extra,
            )
            with self._lock:
                self._latest = est
                self._since_estimate = 0
                self._estimates_done += 1
                self._last_error = ""
            logger.info("WholeSystemPhi: %s", est.claim)
            return est
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            record_degradation("whole_system_phi", exc, severity="warning",
                               action="whole-system Φ estimate skipped")
            return None
        finally:
            with self._lock:
                self._estimating = False

    async def persist_latest(self) -> str:
        """Write the latest report through the governed async write lane."""
        with self._lock:
            est = self._latest
            probe = dict(self._latest_probe)
        if est is None:
            return ""
        from pathlib import Path

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        root = Path(str(_PHI_DIR_FLAG.value() or "")
                    or (Path.home() / ".aura" / "data" / "phi"))
        target = root / "whole_system_latest.json"
        payload = {"estimate": est.to_dict(), "probe": probe}
        with local_internal_governed_scope("whole_system_phi",
                                           domain="state_mutation"):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(root, source="whole_system_phi")
            await gateway.write_json_async(
                target, payload, schema_version=1,
                schema_name="whole_system_phi_report",
                source="whole_system_phi",
            )
        return str(target)

    # ── observability ────────────────────────────────────────────────────
    def latest(self) -> Any:
        with self._lock:
            return self._latest

    def status(self) -> dict[str, Any]:
        with self._lock:
            est = self._latest
            return {
                "window": len(self._window),
                "estimates_done": self._estimates_done,
                "since_estimate": self._since_estimate,
                "interventional_rows": len(self._interventional),
                "last_error": self._last_error,
                "latest": est.to_dict() if est is not None else None,
                "latest_probe": dict(self._latest_probe),
            }

    def is_alive(self) -> bool:
        return True


_service: WholeSystemPhiService | None = None
_service_lock = threading.Lock()


def get_whole_system_phi() -> WholeSystemPhiService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = WholeSystemPhiService()
    return _service


def boot_whole_system_phi() -> WholeSystemPhiService:
    service = get_whole_system_phi()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("whole_system_phi", service,
                                           required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("whole_system_phi", exc, severity="warning",
                           action="service built but not registered")
    return service
