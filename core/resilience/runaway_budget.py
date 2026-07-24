"""core/resilience/runaway_budget.py — turn slow runaways into hard failures.

The problem this exists for, stated plainly: Aura's degradation system is too
forgiving. Thousands of broad excepts became receipts, and a receipt is not a
failure — it is a note. So the system never sees a hard failure. What it sees is
a slow 110GB runaway, recorded faithfully, one polite warning at a time, until
the host freezes. ``autonomy_latitude.py`` exists because there was a runaway.
It will happen again.

The existing MemoryGovernor is level-triggered: prune at 28GB, unload at 34GB,
emergency at 40GB. Level triggers are necessary and insufficient. Two ways they
miss:

  1. Slow drift. The 4h soak measured ~242MB/h linear growth. At that rate the
     28GB trigger is days away — the trend is unmistakable long before any level
     fires, and nothing looks at the trend.

  2. Ineffective mitigation. When pruning fires and RSS keeps climbing anyway,
     the governor prunes again. And again. Each cycle emits a receipt saying it
     handled things. A mitigation loop that never converges is indistinguishable
     from one that works, because the only signal is "I ran".

This module adds the missing judgement: *is the thing I am doing about it
working?* It watches a scalar over time, fits its trend, counts how often
mitigation has fired, and escalates to a RUNAWAY verdict — a hard, loud,
fail-closed condition — when growth persists despite repeated intervention.

A runaway is not a warning. It is a refusal: new consequential work stops until
the trend reverses. That is the point. A system that cannot fail hard cannot
stop hurting itself.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("Aura.Resilience.RunawayBudget")


class RunawayState(StrEnum):
    """What the trend says, independent of the current level."""

    NOMINAL = "nominal"    # flat or falling
    DRIFT = "drift"        # rising steadily, mitigation not yet proven useless
    RUNAWAY = "runaway"    # rising despite repeated mitigation — fail closed


@dataclass(frozen=True)
class RunawayVerdict:
    state: RunawayState
    slope_per_hour: float
    samples: int
    window_s: float
    net_change: float
    mitigations_in_window: int
    projected_breach_s: float | None   # seconds until the ceiling at this slope
    reason: str

    def is_runaway(self) -> bool:
        return self.state is RunawayState.RUNAWAY

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "slope_per_hour": round(self.slope_per_hour, 3),
            "samples": self.samples,
            "window_s": round(self.window_s, 1),
            "net_change": round(self.net_change, 3),
            "mitigations_in_window": self.mitigations_in_window,
            "projected_breach_s": (
                round(self.projected_breach_s, 1)
                if self.projected_breach_s is not None
                else None
            ),
            "reason": self.reason,
        }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError, OverflowError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass
class RunawayPolicy:
    """When drift becomes a refusal.

    Defaults are tuned to the observed failure: ~242MB/h sustained growth is
    well above ``min_slope``, so the soak's leak would have been called a drift
    within the hour and a runaway once pruning had visibly failed to hold it.
    """

    # Below this slope the trend is noise, not growth.
    min_slope_per_hour: float = 50.0
    # Need this much history before trusting a trend at all.
    min_samples: int = 8
    min_window_s: float = 300.0
    # Mitigation has fired at least this often in the window and growth
    # continued ⇒ the mitigation does not work on this problem.
    ineffective_after_mitigations: int = 3
    # A hard ceiling for projection. None disables projection escalation.
    ceiling: float | None = None
    # If the projection says we breach the ceiling within this horizon, that is
    # a runaway even if mitigation has not yet had its three chances — because
    # by then it will be too late to matter.
    projection_horizon_s: float = 3600.0

    @classmethod
    def for_memory_mb(cls) -> "RunawayPolicy":
        try:
            from core.runtime import resource_psutil as psutil

            total_mb = psutil.virtual_memory().total / (1024 * 1024)
        except (ImportError, AttributeError, OSError, RuntimeError):
            total_mb = 65536.0
        return cls(
            min_slope_per_hour=_env_float("AURA_RUNAWAY_MIN_SLOPE_MB_H", 200.0),
            min_samples=_env_int("AURA_RUNAWAY_MIN_SAMPLES", 8),
            min_window_s=_env_float("AURA_RUNAWAY_MIN_WINDOW_S", 300.0),
            ineffective_after_mitigations=_env_int("AURA_RUNAWAY_MITIGATIONS", 3),
            ceiling=_env_float("AURA_RUNAWAY_CEILING_MB", total_mb * 0.75),
            projection_horizon_s=_env_float("AURA_RUNAWAY_HORIZON_S", 3600.0),
        )


class RunawayDetector:
    """Trend-aware runaway detection for one scalar (RSS, handles, queue depth…).

    Deliberately not a threshold. Thresholds answer "how bad is it now"; this
    answers "is it getting worse, and is anything I do about it helping".
    """

    def __init__(
        self,
        name: str,
        policy: RunawayPolicy | None = None,
        *,
        max_samples: int = 240,
    ):
        self.name = name
        self.policy = policy or RunawayPolicy()
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._mitigations: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.RLock()
        self._last_state = RunawayState.NOMINAL

    # -- input ------------------------------------------------------------
    def observe(self, value: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._samples.append((now, float(value)))

    def record_mitigation(self, now: float | None = None) -> None:
        """Note that something was done about it (a prune, an unload, a GC).

        This is the input that makes the detector able to say "and it did not
        help" — the judgement the receipt-only system structurally cannot make.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._mitigations.append(now)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._mitigations.clear()
            self._last_state = RunawayState.NOMINAL

    # -- judgement --------------------------------------------------------
    def assess(self, now: float | None = None) -> RunawayVerdict:
        now = time.time() if now is None else now
        with self._lock:
            samples = list(self._samples)
            mitigations = list(self._mitigations)

        if len(samples) < self.policy.min_samples:
            return self._verdict(
                RunawayState.NOMINAL, 0.0, len(samples), 0.0, 0.0, 0, None,
                f"insufficient history ({len(samples)}/{self.policy.min_samples})",
            )

        t0, t1 = samples[0][0], samples[-1][0]
        window = t1 - t0
        if window < self.policy.min_window_s:
            return self._verdict(
                RunawayState.NOMINAL, 0.0, len(samples), window, 0.0, 0, None,
                f"window too short ({window:.0f}s/{self.policy.min_window_s:.0f}s)",
            )

        slope_per_hour = _linear_slope(samples) * 3600.0
        net_change = samples[-1][1] - samples[0][1]
        mitigations_in_window = sum(1 for m in mitigations if m >= t0)

        if slope_per_hour < self.policy.min_slope_per_hour:
            return self._verdict(
                RunawayState.NOMINAL, slope_per_hour, len(samples), window,
                net_change, mitigations_in_window, None,
                f"slope {slope_per_hour:.1f}/h below {self.policy.min_slope_per_hour:.1f}/h",
            )

        # Growth is real. Two independent ways it becomes a refusal.
        projected = _projected_breach_s(
            samples[-1][1], slope_per_hour, self.policy.ceiling
        )

        # 1. Mitigation has had its chances and growth continued.
        if (
            mitigations_in_window >= self.policy.ineffective_after_mitigations
            and net_change > 0
        ):
            return self._verdict(
                RunawayState.RUNAWAY, slope_per_hour, len(samples), window,
                net_change, mitigations_in_window, projected,
                f"growing {slope_per_hour:.1f}/h despite {mitigations_in_window} "
                f"mitigations in {window / 60:.0f}min (net +{net_change:.1f}) — "
                "the mitigation does not work on this problem",
            )

        # 2. The projection says we run out of room before mitigation could
        #    plausibly get its chances.
        if projected is not None and projected <= self.policy.projection_horizon_s:
            return self._verdict(
                RunawayState.RUNAWAY, slope_per_hour, len(samples), window,
                net_change, mitigations_in_window, projected,
                f"growing {slope_per_hour:.1f}/h — projected to breach ceiling "
                f"{self.policy.ceiling:.0f} in {projected / 60:.0f}min",
            )

        return self._verdict(
            RunawayState.DRIFT, slope_per_hour, len(samples), window,
            net_change, mitigations_in_window, projected,
            f"rising {slope_per_hour:.1f}/h — watching whether mitigation holds it",
        )

    def _verdict(
        self, state, slope, n, window, net, mitigations, projected, reason
    ) -> RunawayVerdict:
        verdict = RunawayVerdict(
            state=state,
            slope_per_hour=slope,
            samples=n,
            window_s=window,
            net_change=net,
            mitigations_in_window=mitigations,
            projected_breach_s=projected,
            reason=reason,
        )
        if state is not self._last_state:
            self._announce(verdict)
            self._last_state = state
        return verdict

    def _announce(self, verdict: RunawayVerdict) -> None:
        if verdict.state is RunawayState.RUNAWAY:
            # CRITICAL, not warning. This is the hard failure the receipt-only
            # path could never produce.
            logger.critical(
                "🚨 RUNAWAY [%s]: %s", self.name, verdict.reason,
            )
            try:
                from core.runtime.errors import record_degradation

                # severity is the Literal STRING "critical" — errors.Severity
                # is a typing alias, not an enum. `Severity.CRITICAL` raised
                # AttributeError("CRITICAL") here, so every live RUNAWAY
                # (e.g. 2026-07-21 22:32 and 23:13, 39-54GB/h projections)
                # logged "Could not record runaway degradation: CRITICAL"
                # and the fail-closed record was never written.
                record_degradation(
                    "runaway_budget",
                    RuntimeError(f"runaway: {self.name}: {verdict.reason}"),
                    action="failing closed: refusing new consequential work",
                    severity="critical",
                    enforce_failure_policy=False,
                    extra=verdict.to_dict(),
                )
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.error("Could not record runaway degradation: %s", exc)
        elif verdict.state is RunawayState.DRIFT:
            logger.warning("📈 DRIFT [%s]: %s", self.name, verdict.reason)
        else:
            logger.info("✅ [%s] trend back to nominal: %s", self.name, verdict.reason)


def _linear_slope(samples: list[tuple[float, float]]) -> float:
    """Least-squares slope in units per second. Robust to uneven sampling."""
    n = len(samples)
    if n < 2:
        return 0.0
    t0 = samples[0][0]
    xs = [t - t0 for t, _ in samples]
    ys = [v for _, v in samples]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def _projected_breach_s(
    current: float, slope_per_hour: float, ceiling: float | None
) -> float | None:
    if ceiling is None or slope_per_hour <= 0:
        return None
    if current >= ceiling:
        return 0.0
    return (ceiling - current) / slope_per_hour * 3600.0


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


class RunawayBudget:
    """Process-wide registry of runaway detectors and the resulting refusal.

    ``is_failing_closed()`` is the causal surface: when any tracked signal is in
    runaway, consequential work is refused. Without that this module would be
    one more thing that writes a receipt about the fire.
    """

    def __init__(self):
        self._detectors: dict[str, RunawayDetector] = {}
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, RunawayVerdict], None]] = []

    def detector(
        self, name: str, policy: RunawayPolicy | None = None
    ) -> RunawayDetector:
        with self._lock:
            if name not in self._detectors:
                self._detectors[name] = RunawayDetector(name, policy)
            return self._detectors[name]

    def on_runaway(self, fn: Callable[[str, RunawayVerdict], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def assess_all(self, now: float | None = None) -> dict[str, RunawayVerdict]:
        with self._lock:
            detectors = dict(self._detectors)
        verdicts = {}
        for name, det in detectors.items():
            verdict = det.assess(now)
            verdicts[name] = verdict
            if verdict.is_runaway():
                self._notify(name, verdict)
        return verdicts

    def _notify(self, name: str, verdict: RunawayVerdict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(name, verdict)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.error("Runaway listener failed: %s", exc)

    def is_failing_closed(self, now: float | None = None) -> bool:
        """True when any tracked signal is in runaway.

        Consequential sinks consult this. A runaway must cost something, or it
        is just a louder log line.
        """
        return any(v.is_runaway() for v in self.assess_all(now).values())

    def runaway_reasons(self, now: float | None = None) -> list[str]:
        return [
            f"{name}: {v.reason}"
            for name, v in self.assess_all(now).items()
            if v.is_runaway()
        ]

    def status(self) -> dict[str, Any]:
        return {
            name: verdict.to_dict() for name, verdict in self.assess_all().items()
        }

    def reset(self) -> None:
        with self._lock:
            for det in self._detectors.values():
                det.reset()
            self._detectors.clear()
            self._listeners.clear()


_budget: RunawayBudget | None = None
_budget_lock = threading.RLock()


def get_runaway_budget() -> RunawayBudget:
    global _budget
    with _budget_lock:
        if _budget is None:
            _budget = RunawayBudget()
        return _budget


def reset_runaway_budget() -> None:
    global _budget
    with _budget_lock:
        if _budget is not None:
            _budget.reset()
        _budget = None


__all__ = [
    "RunawayBudget",
    "RunawayDetector",
    "RunawayPolicy",
    "RunawayState",
    "RunawayVerdict",
    "get_runaway_budget",
    "reset_runaway_budget",
]
