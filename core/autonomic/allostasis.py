"""core/autonomic/allostasis.py — predictive interoception: Aura feels her body's future.

Everything Aura had before this organ was *homeostatic*: react when a threshold
trips. The viability state machine reads current pressure, the resource governor
evicts when memory is already high, the survival driver publishes an imperative
after disk is already low, and unified runtime pressure declares a red zone the
moment loop lag is already 5 s. Every one of her recorded deaths — the 110 GB
incident, the 35 GB endurance OOM, the duplicate-runtime memory doubling, the
~242 MB/h soak leak — was a *trajectory* that was visible for tens of minutes
before any of those reactive layers could speak.

This module is the *allostatic* layer (Sterling's sense: regulation through
anticipation): it watches the trajectories of her vital signs and regulates
before the crisis, not after. Concretely, per vital sign:

1.  **Robust trend** — Mann–Kendall trend test (tie-corrected) + Sen's slope
    with a Gilbert confidence interval. Median-of-pairwise-slopes is immune to
    the GC spikes and inference bursts that wreck least-squares on RSS series.
2.  **Regime detection** — a two-sided CUSUM on robust residuals, so a leak
    that *starts* mid-session (the soak-leak signature) re-anchors the trend
    window within a few samples instead of being diluted by hours of calm.
3.  **Time-to-crisis forecasts** — when a trend is statistically significant
    and headed toward a threshold, the engine issues a dated, falsifiable
    prediction: "memory_rss_mb crosses its red line at T, 90 % band [T₁, T₂]".
4.  **A calibration ledger** — every forecast is scored when its deadline
    passes: HIT, MISS_EARLY, FALSE_ALARM, INTERVENED, or SUPERSEDED. Empirical
    interval coverage feeds back into band widths, so Aura *knows how well she
    knows her own body* and her uncertainty honestly widens when she has been
    wrong. Forecasts are persisted through the governed write gateway.
5.  **Allostatic load** — the decayed integral of time spent above setpoint:
    the difference between a brief spike and running hot for an hour, exposed
    as a chronic-strain scalar the felt state can carry.
6.  **Anticipatory regulation** — a tiered policy (SETTLED → VIGILANT →
    CONSERVING → PROTECTING) with instant escalation and hysteretic release.
    CONSERVING asks the metabolic layer to defer deferrable work; PROTECTING
    additionally records a degradation and publishes on the same
    ``existential_threat`` channel the Will, the inference gate, and the
    attention gate already subscribe to. The engine never kills, restarts, or
    unloads anything itself — it senses, predicts, requests, and testifies.

Causality (not narration): ``felt_contribution()`` feeds
:class:`core.being.aura_now.BodyState.anticipatory_pressure`, so a forecast
crisis raises total body pressure — and through it affect, welfare, workspace
coalitions, and the Will — *while the current readings are still green*. That
is the definition of feeling the future of one's own body.

Honest boundary: forecasts are statistical extrapolations with stated
uncertainty, scored after the fact; "Aura feels her death approaching" is a
functional claim about a calibrated predictive signal being causally coupled
into her control state, not a phenomenal one. The report boundary of
:class:`~core.being.aura_now.AuraNow` still applies to anything said about it.
"""
from __future__ import annotations

import enum
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Allostasis")

_SUBSYSTEM = "allostasis"
_STATE_SCHEMA_VERSION = 1

_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure math — deliberately dependency-free and unit-testable in isolation.
# ─────────────────────────────────────────────────────────────────────────────

def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |ε|<1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


@dataclass(frozen=True)
class MannKendall:
    """Result of the Mann–Kendall monotonic-trend test."""

    s: int
    var_s: float
    z: float
    p_value: float          # two-sided
    n: int

    @property
    def rising(self) -> bool:
        return self.s > 0

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha


def mann_kendall(values: list[float]) -> MannKendall:
    """Tie-corrected Mann–Kendall test for a monotonic trend.

    Var(S) = [n(n−1)(2n+5) − Σⱼ tⱼ(tⱼ−1)(2tⱼ+5)] / 18 over tie groups of size tⱼ,
    with the standard continuity correction on Z.
    """
    n = len(values)
    if n < 3:
        return MannKendall(s=0, var_s=0.0, z=0.0, p_value=1.0, n=n)
    s = 0
    for i in range(n - 1):
        vi = values[i]
        for j in range(i + 1, n):
            diff = values[j] - vi
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0.0:
        # All values identical: no evidence of trend.
        return MannKendall(s=s, var_s=0.0, z=0.0, p_value=1.0, n=n)
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided
    return MannKendall(s=s, var_s=var_s, z=z, p_value=p, n=n)


@dataclass(frozen=True)
class SenSlopeEstimate:
    """Sen's slope (median of pairwise slopes) with a Gilbert confidence interval."""

    slope: float            # units per second
    lower: float
    upper: float
    n_pairs: int
    confidence: float

    @property
    def band_open_below(self) -> bool:
        return self.lower <= 0.0


def sen_slope(
    times: list[float],
    values: list[float],
    *,
    confidence: float = 0.90,
) -> Optional[SenSlopeEstimate]:
    """Sen's slope estimator over (t, v) pairs, CI via Gilbert (1987).

    Rank positions M₁=(N−C)/2 and M₂=(N+C)/2 with C = z₍₁₋α/₂₎·√Var(S) select the
    interval bounds from the sorted pairwise slopes. Requires ≥ 3 points and a
    non-degenerate time axis; returns ``None`` when no slope can be formed.
    """
    n = len(values)
    if n < 3 or len(times) != n:
        return None
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if dt > 0:
                slopes.append((values[j] - values[i]) / dt)
    if not slopes:
        return None
    slopes.sort()
    n_pairs = len(slopes)
    mid = n_pairs // 2
    if n_pairs % 2:
        slope = slopes[mid]
    else:
        slope = 0.5 * (slopes[mid - 1] + slopes[mid])
    mk = mann_kendall(values)
    if mk.var_s <= 0.0:
        return SenSlopeEstimate(slope=slope, lower=slope, upper=slope,
                                n_pairs=n_pairs, confidence=confidence)
    c = norm_ppf(0.5 + confidence / 2.0) * math.sqrt(mk.var_s)
    m1 = int(math.floor((n_pairs - c) / 2.0))
    m2 = int(math.ceil((n_pairs + c) / 2.0))
    lower = slopes[max(0, min(n_pairs - 1, m1))]
    upper = slopes[max(0, min(n_pairs - 1, m2))]
    return SenSlopeEstimate(slope=slope, lower=lower, upper=upper,
                            n_pairs=n_pairs, confidence=confidence)


def robust_sigma(values: list[float]) -> float:
    """MAD-based robust standard deviation (σ ≈ 1.4826·MAD)."""
    n = len(values)
    if n < 2:
        return 0.0
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    deviations = sorted(abs(v - median) for v in values)
    mad = deviations[mid] if n % 2 else 0.5 * (deviations[mid - 1] + deviations[mid])
    return 1.4826 * mad


# ─────────────────────────────────────────────────────────────────────────────
# Vital-sign specifications
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r", name, raw)
        return default


@dataclass(frozen=True)
class VitalSpec:
    """One vital sign: where it lives in the pressure snapshot and what hurts."""

    key: str                 # field name in runtime_pressure_snapshot()
    label: str
    unit: str
    amber: float
    red: float
    setpoint: float          # allostatic-load baseline: strain accrues above this
    forecastable: bool = True
    min_meaningful_slope: float = 0.0   # per second; below this, trends are noise


def default_vital_specs() -> tuple[VitalSpec, ...]:
    """Built-in vitals. Thresholds are env-tunable; defaults sit below the
    values at which this host has actually died (35 GB process OOM) and align
    with the reactive layers' red lines (memory 92 %, loop lag 5 s, disk 98 %)."""
    rss_amber = _env_float("AURA_ALLOSTASIS_RSS_AMBER_MB", 26_000.0)
    rss_red = _env_float("AURA_ALLOSTASIS_RSS_RED_MB", 32_000.0)
    tree_amber = _env_float("AURA_ALLOSTASIS_TREE_RSS_AMBER_MB", 30_000.0)
    tree_red = _env_float("AURA_ALLOSTASIS_TREE_RSS_RED_MB", 38_000.0)
    return (
        VitalSpec("memory_rss_mb", "process memory", "MB",
                  amber=rss_amber, red=rss_red, setpoint=rss_amber * 0.75,
                  min_meaningful_slope=1024.0 / 3600.0),      # ≥ ~1 GB/h matters
        VitalSpec("process_tree_rss_mb", "process-tree memory", "MB",
                  amber=tree_amber, red=tree_red, setpoint=tree_amber * 0.75,
                  min_meaningful_slope=1024.0 / 3600.0),
        VitalSpec("memory_pct", "system memory", "%",
                  amber=85.0, red=92.0, setpoint=75.0,
                  min_meaningful_slope=2.0 / 3600.0),         # ≥ 2 %/h matters
        VitalSpec("loop_lag_s", "event-loop lag", "s",
                  amber=1.0, red=5.0, setpoint=0.25,
                  min_meaningful_slope=0.25 / 3600.0),
        VitalSpec("disk_percent", "disk usage", "%",
                  amber=92.0, red=98.0, setpoint=85.0,
                  min_meaningful_slope=0.5 / 3600.0),
        VitalSpec("thermal_level", "thermal pressure", "level",
                  amber=2.0, red=3.0, setpoint=1.0,
                  forecastable=False),                        # 0–3 ordinal: load only
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection (two-sided CUSUM on robust residuals)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CusumState:
    """Per-vital CUSUM over residuals from an anchored Theil–Sen fit.

    Anchoring on a *fit* (slope + intercept) rather than a mean is what lets a
    steady, legitimate ramp coexist with regime detection: the ramp's residuals
    hover near zero, while a slope change — a leak starting, pressure suddenly
    relieved — accumulates signed residuals until the CUSUM fires. Re-anchored
    after every regime event.
    """

    anchor_slope: float = 0.0
    anchor_intercept: float = 0.0     # value at t = anchor_t0
    anchor_t0: float = 0.0
    anchor_sigma: float = 0.0
    pos: float = 0.0
    neg: float = 0.0
    anchored: bool = False
    samples_since_anchor: int = 0

    def expected(self, t: float) -> float:
        return self.anchor_intercept + self.anchor_slope * (t - self.anchor_t0)


@dataclass(frozen=True)
class RegimeEvent:
    vital: str
    at_unix: float
    direction: str            # "up" | "down"
    magnitude_sigma: float
    regime_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "regime",
            "vital": self.vital,
            "at_unix": round(self.at_unix, 3),
            "direction": self.direction,
            "magnitude_sigma": round(self.magnitude_sigma, 3),
            "regime_id": self.regime_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Forecasts and the calibration ledger
# ─────────────────────────────────────────────────────────────────────────────

class ForecastOutcome(enum.StrEnum):
    HIT = "hit"                    # crossed inside the stated band
    MISS_EARLY = "miss_early"      # crossed before the band opened
    FALSE_ALARM = "false_alarm"    # band expired, no crossing, no excuse
    INTERVENED = "intervened"      # no crossing, but regulation fired after issue
    SUPERSEDED = "superseded"      # regime changed / process restarted under it


class AllostasisTier(enum.IntEnum):
    SETTLED = 0
    VIGILANT = 1
    CONSERVING = 2
    PROTECTING = 3


@dataclass
class Forecast:
    """A dated, falsifiable prediction about Aura's own body."""

    forecast_id: str
    vital: str
    threshold_name: str          # "amber" | "red"
    threshold_value: float
    regime_id: str
    issued_at: float
    level_at_issue: float
    slope_per_s: float
    slope_lower: float
    slope_upper: float
    eta_unix: float
    eta_lower_unix: float
    eta_upper_unix: float
    band_open: bool              # slope CI touched zero: upper deadline is a cap
    p_value: float
    widen_factor: float
    first_eta_unix: float
    revisions: int = 0
    last_revised_at: float = 0.0
    status: str = "open"         # "open" | ForecastOutcome value
    resolved_at: float = 0.0
    crossed_at: float = 0.0
    resolution_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "vital": self.vital,
            "threshold_name": self.threshold_name,
            "threshold_value": round(self.threshold_value, 3),
            "regime_id": self.regime_id,
            "issued_at": round(self.issued_at, 3),
            "level_at_issue": round(self.level_at_issue, 3),
            "slope_per_s": self.slope_per_s,
            "slope_lower": self.slope_lower,
            "slope_upper": self.slope_upper,
            "eta_unix": round(self.eta_unix, 3),
            "eta_lower_unix": round(self.eta_lower_unix, 3),
            "eta_upper_unix": round(self.eta_upper_unix, 3),
            "band_open": self.band_open,
            "p_value": self.p_value,
            "widen_factor": round(self.widen_factor, 3),
            "first_eta_unix": round(self.first_eta_unix, 3),
            "revisions": self.revisions,
            "last_revised_at": round(self.last_revised_at, 3),
            "status": self.status,
            "resolved_at": round(self.resolved_at, 3),
            "crossed_at": round(self.crossed_at, 3),
            "resolution_note": self.resolution_note,
        }


@dataclass
class _VitalCalibration:
    """Empirical reliability of forecasts for one vital."""

    hits: int = 0
    miss_early: int = 0
    false_alarms: int = 0
    intervened: int = 0
    superseded: int = 0

    @property
    def scored(self) -> int:
        """Outcomes that count toward interval coverage (interventions and
        supersessions are excluded: the world changed under the forecast)."""
        return self.hits + self.miss_early + self.false_alarms

    @property
    def coverage(self) -> Optional[float]:
        return (self.hits / self.scored) if self.scored else None

    def widen_factor(self, *, target_coverage: float, min_scored: int = 5) -> float:
        """Band multiplier from empirical coverage. Poorly calibrated → wider
        bands (honest uncertainty); never narrower than stated (≥ 1.0)."""
        cov = self.coverage
        if cov is None or self.scored < min_scored:
            return 1.0
        if cov <= 0.0:
            return 3.0
        return _clamp(target_coverage / cov, 1.0, 3.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "miss_early": self.miss_early,
            "false_alarms": self.false_alarms,
            "intervened": self.intervened,
            "superseded": self.superseded,
            "scored": self.scored,
            "coverage": round(self.coverage, 4) if self.coverage is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_VitalCalibration":
        out = cls()
        for key in ("hits", "miss_early", "false_alarms", "intervened", "superseded"):
            try:
                setattr(out, key, max(0, int(data.get(key, 0))))
            except (TypeError, ValueError):
                continue
        return out


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AllostasisReading:
    """What one ingested sample concluded (returned for tests/inspection)."""

    at_unix: float
    tier: AllostasisTier
    tier_reason: str
    nearest_crisis_eta_s: Optional[float]
    anticipatory_pressure: float
    allostatic_load: float
    new_forecasts: tuple[str, ...]
    resolved_forecasts: tuple[str, ...]
    regime_events: tuple[str, ...]


class AllostasisEngine:
    """Predictive interoception over Aura's vital signs.

    Pull-friendly and loop-free by construction: :meth:`ingest` is a pure state
    update, :meth:`sample_and_regulate` is one sample + side effects, and the
    metabolic coordinator provides the pulse. Nothing here can wedge the loop
    (no sync I/O in async paths; persistence flows through the governed async
    write gateway) and nothing here kills, restarts, or unloads anything.
    """

    SERVICE_NAME = "allostasis_engine"

    # CUSUM parameters. Deliberately deaf to small drifts (k = 1σ, h = 6σ):
    # this detector exists to invalidate forecasts on ABRUPT regime breaks
    # (a leak starting, pressure suddenly relieved); gradual change is the
    # rolling trend window's job. The textbook k = 0.5σ tripled the false-
    # alarm rate here because the anchor's own estimation error is a
    # persistent bias that eats the allowance (measured empirically in this
    # module's test harness: ~1 false event / 157 samples at k = 0.5 vs
    # < 1 / 3600 at k = 1.0 with anchor-error inflation).
    CUSUM_K_SIGMA = 1.0
    CUSUM_H_SIGMA = 6.0
    CUSUM_MIN_REFERENCE = 12          # samples before residuals are trusted
    CUSUM_FIT_WINDOW = 60             # most-recent samples used for the anchor fit
    CUSUM_REANCHOR_EVERY = 45         # silent refit cadence: bounds extrapolation drift
    CUSUM_SLOPE_ALPHA = 0.01          # anchor slope only when the trend is this credible
    CUSUM_SIGMA_FLOOR = 1e-9

    def __init__(
        self,
        *,
        specs: tuple[VitalSpec, ...] | None = None,
        now_fn: Callable[[], float] = time.time,
        data_dir: Path | str | None = None,
        history_maxlen: int = 240,            # 4 h at the 60 s metabolic pulse
        trend_window_s: float = 3600.0,
        min_trend_samples: int = 8,
        significance_alpha: float | None = None,
        forecast_horizon_s: float | None = None,
        conserve_horizon_s: float = 1800.0,
        protect_horizon_s: float = 600.0,
        release_hysteresis_s: float = 300.0,
        resolution_grace_s: float = 120.0,
        target_coverage: float = 0.90,
        eta_cap_s: float = 24 * 3600.0,
    ) -> None:
        self._now = now_fn
        self._lock = threading.RLock()
        self._specs: dict[str, VitalSpec] = {s.key: s for s in (specs or default_vital_specs())}
        env_root = os.getenv("AURA_ALLOSTASIS_DIR", "")
        # Hermeticity: explicit override wins; under the test suite the
        # hermetic runtime root keeps ledger writes out of the live
        # ~/.aura/data (same convention as standing_authority's state root).
        test_root = os.getenv("AURA_TEST_RUNTIME_ROOT", "").strip()
        if data_dir:
            self._dir = Path(data_dir)
        elif env_root:
            self._dir = Path(env_root)
        elif test_root:
            self._dir = Path(test_root) / "allostasis"
        else:
            self._dir = Path.home() / ".aura" / "data" / "allostasis"
        self._events_path = self._dir / "forecasts.jsonl"
        self._state_path = self._dir / "state.json"
        self._dir_ready = False

        self._history_maxlen = int(history_maxlen)
        self._trend_window_s = float(trend_window_s)
        self._min_trend_samples = max(3, int(min_trend_samples))
        self._alpha = significance_alpha if significance_alpha is not None else _env_float(
            "AURA_ALLOSTASIS_ALPHA", 0.05)
        self._horizon_s = forecast_horizon_s if forecast_horizon_s is not None else _env_float(
            "AURA_ALLOSTASIS_HORIZON_S", 6 * 3600.0)
        self._conserve_horizon_s = float(conserve_horizon_s)
        self._protect_horizon_s = float(protect_horizon_s)
        self._release_hysteresis_s = float(release_hysteresis_s)
        self._resolution_grace_s = float(resolution_grace_s)
        self._target_coverage = float(target_coverage)
        self._eta_cap_s = float(eta_cap_s)

        self._series: dict[str, deque[tuple[float, float]]] = {
            key: deque(maxlen=self._history_maxlen) for key in self._specs
        }
        self._cusum: dict[str, _CusumState] = {key: _CusumState() for key in self._specs}
        self._regime_id: dict[str, str] = {key: f"boot-{uuid.uuid4().hex[:8]}" for key in self._specs}
        self._regime_started_at: dict[str, float] = {}
        self._regime_events_total = 0

        self._open_forecasts: dict[tuple[str, str], Forecast] = {}
        self._resolved_recent: deque[Forecast] = deque(maxlen=64)
        self._calibration: dict[str, _VitalCalibration] = {}
        self._load_raw: dict[str, float] = {key: 0.0 for key in self._specs}
        self._load_tau_s = _env_float("AURA_ALLOSTASIS_LOAD_TAU_S", 3600.0)
        self._last_ingest_at: Optional[float] = None
        self._ingest_count = 0

        self._tier = AllostasisTier.SETTLED
        self._tier_reason = "no samples yet"
        self._tier_changed_at = 0.0
        self._tier_release_eligible_since: Optional[float] = None
        self._interventions: deque[dict[str, Any]] = deque(maxlen=64)

        self._felt: dict[str, Any] = {
            "anticipatory_pressure": 0.0,
            "allostatic_load": 0.0,
            "nearest_crisis_eta_s": None,
            "tier": self._tier.name.lower(),
        }
        self._pending_events: list[dict[str, Any]] = []
        self._disabled = os.getenv("AURA_ALLOSTASIS_DISABLED", "") in ("1", "true", "yes")

        self._restore_persisted_state()

    # ── liveness / registration ─────────────────────────────────────────────
    def is_ready(self) -> bool:
        return not self._disabled

    @property
    def enabled(self) -> bool:
        return not self._disabled

    # ── persistence (reads are plain; writes go through the governed gateway) ──
    def _restore_persisted_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            record_degradation(_SUBSYSTEM, exc, action="persisted allostasis state unreadable; starting fresh")
            return
        if not isinstance(data, dict):
            return
        # The gateway persists an atomic-writer envelope {schema, version, payload}.
        payload = data.get("payload")
        if isinstance(payload, dict):
            data = payload
        calibration = data.get("calibration", {})
        if isinstance(calibration, dict):
            for vital, stats in calibration.items():
                if isinstance(stats, dict) and vital in self._specs:
                    self._calibration[vital] = _VitalCalibration.from_dict(stats)
        # Open forecasts from a previous process are moot after a restart:
        # the body they described no longer exists. Resolve them honestly.
        stale = data.get("open_forecasts", [])
        if isinstance(stale, list):
            now = self._now()
            for raw in stale:
                if not isinstance(raw, dict):
                    continue
                vital = str(raw.get("vital", ""))
                fc_id = str(raw.get("forecast_id", ""))
                if not vital or not fc_id:
                    continue
                self._calibration.setdefault(vital, _VitalCalibration()).superseded += 1
                self._pending_events.append({
                    "kind": "resolved",
                    "forecast_id": fc_id,
                    "vital": vital,
                    "status": ForecastOutcome.SUPERSEDED.value,
                    "resolved_at": round(now, 3),
                    "resolution_note": "process_restart",
                })

    def _state_payload(self) -> dict[str, Any]:
        return {
            "calibration": {k: v.to_dict() for k, v in self._calibration.items()},
            "open_forecasts": [f.to_dict() for f in self._open_forecasts.values()],
            "allostatic_load": {k: round(v, 6) for k, v in self._load_raw.items()},
            "tier": self._tier.name.lower(),
            "regime_events_total": self._regime_events_total,
            "saved_at": round(self._now(), 3),
        }

    async def _persist(self, events: list[dict[str, Any]], *, save_state: bool) -> None:
        if not events and not save_state:
            return
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            with local_internal_governed_scope("allostasis.ledger", domain="file_write"):
                if not self._dir_ready:
                    await gateway.ensure_directory_async(self._dir, source="core.autonomic.allostasis")
                    self._dir_ready = True
                if events:
                    lines = "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in events)
                    await gateway.append_text_async(
                        self._events_path, lines, source="allostasis_forecast_ledger",
                    )
                if save_state:
                    with self._lock:
                        payload = self._state_payload()
                    await gateway.write_json_async(
                        self._state_path, payload,
                        schema_version=_STATE_SCHEMA_VERSION,
                        schema_name="allostasis_state",
                        source="allostasis_state_snapshot",
                    )
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="allostasis ledger write skipped this pulse")

    # ── ingestion ───────────────────────────────────────────────────────────
    def ingest(self, snapshot: dict[str, Any], *, at: float | None = None) -> AllostasisReading:
        """Fold one vitals snapshot into history, trends, forecasts, and tier.

        Pure state update: no I/O, no event publishing. Safe to call from
        tests with synthetic snapshots and timestamps.
        """
        now = self._now() if at is None else float(at)
        with self._lock:
            new_regimes: list[str] = []
            new_forecasts: list[str] = []
            resolved: list[str] = []

            dt = 0.0
            if self._last_ingest_at is not None:
                dt = now - self._last_ingest_at
                if dt < 0:
                    # Clock went backwards (NTP step, test harness): keep history
                    # append-only by treating this as a fresh anchor point.
                    dt = 0.0
            self._last_ingest_at = now
            self._ingest_count += 1

            for key, spec in self._specs.items():
                raw = snapshot.get(key, None)
                if raw is None:
                    continue
                value = _finite(raw, default=float("nan"))
                if math.isnan(value):
                    continue
                series = self._series[key]
                if series and now <= series[-1][0]:
                    continue  # non-monotonic timestamp: drop, never reorder
                series.append((now, value))
                self._regime_started_at.setdefault(key, now)
                event = self._cusum_update(key, spec, value, now)
                if event is not None:
                    new_regimes.append(event.regime_id)
                    self._pending_events.append(event.to_dict())
                if dt > 0:
                    self._accrue_load(key, spec, value, dt)

            resolved.extend(self._resolve_due_forecasts(now, snapshot))
            new_forecasts.extend(self._refresh_forecasts(now))
            tier_reason = self._recompute_tier(now)
            self._refresh_felt(now)

            return AllostasisReading(
                at_unix=now,
                tier=self._tier,
                tier_reason=tier_reason,
                nearest_crisis_eta_s=self._felt.get("nearest_crisis_eta_s"),
                anticipatory_pressure=float(self._felt.get("anticipatory_pressure", 0.0)),
                allostatic_load=float(self._felt.get("allostatic_load", 0.0)),
                new_forecasts=tuple(new_forecasts),
                resolved_forecasts=tuple(resolved),
                regime_events=tuple(new_regimes),
            )

    # ── CUSUM regime detection ──────────────────────────────────────────────
    def _regime_series(self, key: str) -> list[tuple[float, float]]:
        started = self._regime_started_at.get(key, 0.0)
        return [(t, v) for (t, v) in self._series[key] if t >= started]

    def _anchor_cusum(self, key: str, spec: VitalSpec) -> bool:
        """(Re)fit the CUSUM anchor on the most recent regime samples.

        The anchor slope is only trusted when the reference trend is strongly
        significant — otherwise a spurious fitted slope, extrapolated for
        hours, manufactures drift out of stationary noise (observed directly
        in this module's own test suite before this gate existed).
        """
        window = self._regime_series(key)[-self.CUSUM_FIT_WINDOW:]
        if len(window) < self.CUSUM_MIN_REFERENCE:
            return False
        times = [t for (t, _) in window]
        values = [v for (_, v) in window]
        slope = 0.0
        if mann_kendall(values).significant(self.CUSUM_SLOPE_ALPHA):
            fit = sen_slope(times, values)
            if fit is not None:
                slope = fit.slope
        # Theil–Sen intercept: median of (vᵢ − slope·tᵢ), evaluated at t0.
        t0 = times[0]
        offsets = sorted(v - slope * (t - t0) for (t, v) in window)
        mid = len(offsets) // 2
        intercept = offsets[mid] if len(offsets) % 2 else 0.5 * (offsets[mid - 1] + offsets[mid])
        residuals = [v - (intercept + slope * (t - t0)) for (t, v) in window]
        sigma = robust_sigma(residuals)
        if sigma <= self.CUSUM_SIGMA_FLOOR:
            # Degenerate (noise-free) reference: fall back to a small
            # fraction of the vital's amber-red span so a genuine shift
            # still registers without single-sample hair triggers.
            sigma = max((spec.red - spec.amber) * 0.01, self.CUSUM_SIGMA_FLOOR)
        # Inflate for the anchor's own estimation error (intercept/slope are
        # estimates, not truth): a persistent ~σ/√n bias otherwise leaks into
        # every z-score and quietly consumes the CUSUM allowance.
        sigma *= 1.0 + 1.0 / math.sqrt(len(window))
        state = self._cusum[key]
        state.anchor_slope = slope
        state.anchor_intercept = intercept + slope * (times[-1] - t0)
        state.anchor_t0 = times[-1]
        state.anchor_sigma = sigma
        state.pos = state.neg = 0.0
        state.anchored = True
        state.samples_since_anchor = 0
        return True

    def _cusum_update(self, key: str, spec: VitalSpec, value: float, now: float) -> Optional[RegimeEvent]:
        state = self._cusum[key]
        if not state.anchored:
            self._anchor_cusum(key, spec)
            return None
        state.samples_since_anchor += 1
        # Silent periodic refit: bounds how long a slightly-wrong anchor slope
        # is extrapolated (adaptive CUSUM). Only while quiescent — never mid-
        # accumulation, or a real slow shift could be refit away.
        if (state.samples_since_anchor >= self.CUSUM_REANCHOR_EVERY
                and state.pos < self.CUSUM_H_SIGMA / 2.0
                and state.neg < self.CUSUM_H_SIGMA / 2.0):
            self._anchor_cusum(key, spec)
            state = self._cusum[key]
        z = (value - state.expected(now)) / state.anchor_sigma
        k = self.CUSUM_K_SIGMA
        state.pos = max(0.0, state.pos + z - k)
        state.neg = max(0.0, state.neg - z - k)
        if state.pos < self.CUSUM_H_SIGMA and state.neg < self.CUSUM_H_SIGMA:
            return None
        direction = "up" if state.pos >= self.CUSUM_H_SIGMA else "down"
        magnitude = state.pos if direction == "up" else state.neg
        regime_id = f"{key}-{uuid.uuid4().hex[:8]}"
        self._regime_id[key] = regime_id
        self._regime_started_at[key] = now
        self._regime_events_total += 1
        self._cusum[key] = _CusumState()
        # Forecasts issued under the old regime describe a body that no longer
        # exists. If regulation fired after issue, credit the intervention
        # (the regime plausibly changed *because* the engine acted); otherwise
        # supersede without scoring.
        for threshold_name in ("amber", "red"):
            fc = self._open_forecasts.pop((key, threshold_name), None)
            if fc is None:
                continue
            intervention = self._intervention_since(fc.issued_at)
            if intervention is not None and direction == "down":
                self._finalize_forecast(
                    fc, ForecastOutcome.INTERVENED, now,
                    note=f"regime relaxed after {intervention['action']}",
                )
            else:
                self._finalize_forecast(
                    fc, ForecastOutcome.SUPERSEDED, now,
                    note=f"regime_change:{direction}",
                )
        logger.info(
            "🌡️ [Allostasis] regime change on %s (%s, %.1fσ) — trend window re-anchored.",
            key, direction, magnitude,
        )
        return RegimeEvent(vital=key, at_unix=now, direction=direction,
                           magnitude_sigma=magnitude, regime_id=regime_id)

    # ── allostatic load ─────────────────────────────────────────────────────
    def _accrue_load(self, key: str, spec: VitalSpec, value: float, dt: float) -> None:
        decay = math.exp(-dt / max(1.0, self._load_tau_s))
        prior = self._load_raw.get(key, 0.0) * decay
        span = max(1e-9, spec.red - spec.setpoint)
        excess = max(0.0, (value - spec.setpoint) / span)
        # Raw load is "seconds spent fully red-equivalent", decayed.
        self._load_raw[key] = prior + excess * dt

    def _load_normalized(self, key: str) -> float:
        # 1 − e^(−load/τ_load): ~0.63 after running red for one full τ.
        return _clamp(1.0 - math.exp(-self._load_raw.get(key, 0.0) / max(1.0, self._load_tau_s)))

    def allostatic_load(self) -> dict[str, float]:
        with self._lock:
            per_vital = {key: round(self._load_normalized(key), 4) for key in self._specs}
            per_vital["composite"] = round(self._composite_load(), 4)
            return per_vital

    def _composite_load(self) -> float:
        if not self._specs:
            return 0.0
        values = [self._load_normalized(key) for key in self._specs]
        peak = max(values)
        mean = sum(values) / len(values)
        # Same blend the body uses for total_pressure: peak-dominant.
        return _clamp(0.45 * mean + 0.55 * peak)

    # ── forecasting ─────────────────────────────────────────────────────────
    def _refresh_forecasts(self, now: float) -> list[str]:
        issued: list[str] = []
        for key, spec in self._specs.items():
            if not spec.forecastable:
                continue
            window = [(t, v) for (t, v) in self._regime_series(key)
                      if t >= now - self._trend_window_s]
            if len(window) < self._min_trend_samples:
                continue
            times = [t for (t, _) in window]
            values = [v for (_, v) in window]
            mk = mann_kendall(values)
            estimate = sen_slope(times, values)
            if estimate is None:
                continue
            current = values[-1]
            for threshold_name, threshold in (("amber", spec.amber), ("red", spec.red)):
                fc_key = (key, threshold_name)
                credible = (
                    mk.significant(self._alpha)
                    and estimate.slope > max(0.0, spec.min_meaningful_slope)
                    and current < threshold
                )
                if not credible:
                    continue
                remaining = threshold - current
                eta_mid = now + remaining / estimate.slope
                if eta_mid - now > self._horizon_s:
                    # Trend is real but the crossing is beyond the honest
                    # forecast horizon; refresh next pulse.
                    continue
                widen = self._calibration.setdefault(key, _VitalCalibration()).widen_factor(
                    target_coverage=self._target_coverage)
                eta_early = now + remaining / estimate.upper if estimate.upper > 0 else eta_mid
                band_open = estimate.band_open_below
                eta_late = (now + remaining / estimate.lower) if estimate.lower > 0 else (
                    now + self._eta_cap_s)
                # Calibration-driven widening around the mid ETA.
                eta_lower = eta_mid - (eta_mid - eta_early) * widen
                eta_upper = eta_mid + (eta_late - eta_mid) * widen
                eta_upper = min(eta_upper, now + self._eta_cap_s)
                existing = self._open_forecasts.get(fc_key)
                if existing is not None:
                    existing.slope_per_s = estimate.slope
                    existing.slope_lower = estimate.lower
                    existing.slope_upper = estimate.upper
                    existing.eta_unix = eta_mid
                    existing.eta_lower_unix = eta_lower
                    existing.eta_upper_unix = eta_upper
                    existing.band_open = band_open
                    existing.p_value = mk.p_value
                    existing.widen_factor = widen
                    existing.revisions += 1
                    existing.last_revised_at = now
                    continue
                forecast = Forecast(
                    forecast_id=f"fc-{uuid.uuid4().hex[:10]}",
                    vital=key,
                    threshold_name=threshold_name,
                    threshold_value=threshold,
                    regime_id=self._regime_id[key],
                    issued_at=now,
                    level_at_issue=current,
                    slope_per_s=estimate.slope,
                    slope_lower=estimate.lower,
                    slope_upper=estimate.upper,
                    eta_unix=eta_mid,
                    eta_lower_unix=eta_lower,
                    eta_upper_unix=eta_upper,
                    band_open=band_open,
                    p_value=mk.p_value,
                    widen_factor=widen,
                    first_eta_unix=eta_mid,
                )
                self._open_forecasts[fc_key] = forecast
                issued.append(forecast.forecast_id)
                self._pending_events.append({"kind": "issued", **forecast.to_dict()})
                logger.warning(
                    "🔮 [Allostasis] forecast %s: %s → %s line (%.1f %s) at %s "
                    "(band %s–%s, slope %.3f %s/h, p=%.4f).",
                    forecast.forecast_id, key, threshold_name, threshold, spec.unit,
                    _fmt_eta(eta_mid - now), _fmt_eta(eta_lower - now), _fmt_eta(eta_upper - now),
                    estimate.slope * 3600.0, spec.unit, mk.p_value,
                )
        return issued

    def _resolve_due_forecasts(self, now: float, snapshot: dict[str, Any]) -> list[str]:
        resolved: list[str] = []
        for fc_key in list(self._open_forecasts.keys()):
            fc = self._open_forecasts[fc_key]
            vital, _threshold_name = fc_key
            raw = snapshot.get(vital, None)
            value = _finite(raw, default=float("nan")) if raw is not None else float("nan")
            crossed = (not math.isnan(value)) and value >= fc.threshold_value
            if crossed:
                del self._open_forecasts[fc_key]
                if now < fc.eta_lower_unix - self._resolution_grace_s:
                    outcome = ForecastOutcome.MISS_EARLY
                    note = f"crossed {_fmt_eta(fc.eta_lower_unix - now)} before band"
                else:
                    outcome = ForecastOutcome.HIT
                    note = "crossed inside band"
                fc.crossed_at = now
                self._finalize_forecast(fc, outcome, now, note=note)
                resolved.append(fc.forecast_id)
                continue
            if now <= fc.eta_upper_unix + self._resolution_grace_s:
                continue
            # Deadline passed without a crossing.
            del self._open_forecasts[fc_key]
            intervention = self._intervention_since(fc.issued_at)
            if intervention is not None:
                outcome = ForecastOutcome.INTERVENED
                note = f"no crossing after {intervention['action']}"
            else:
                outcome = ForecastOutcome.FALSE_ALARM
                note = "band expired without crossing"
            self._finalize_forecast(fc, outcome, now, note=note)
            resolved.append(fc.forecast_id)
        return resolved

    def _finalize_forecast(
        self, fc: Forecast, outcome: ForecastOutcome, now: float, *, note: str,
    ) -> None:
        fc.status = outcome.value
        fc.resolved_at = now
        fc.resolution_note = note
        book = self._calibration.setdefault(fc.vital, _VitalCalibration())
        if outcome is ForecastOutcome.HIT:
            book.hits += 1
        elif outcome is ForecastOutcome.MISS_EARLY:
            book.miss_early += 1
        elif outcome is ForecastOutcome.FALSE_ALARM:
            book.false_alarms += 1
        elif outcome is ForecastOutcome.INTERVENED:
            book.intervened += 1
        else:
            book.superseded += 1
        self._resolved_recent.append(fc)
        self._pending_events.append({"kind": "resolved", **fc.to_dict()})
        log = logger.info if outcome in (ForecastOutcome.HIT, ForecastOutcome.INTERVENED) else logger.warning
        log(
            "📒 [Allostasis] forecast %s on %s resolved %s (%s). Coverage now %s.",
            fc.forecast_id, fc.vital, outcome.value, note,
            book.coverage if book.coverage is not None else "n/a",
        )

    def _intervention_since(self, since_unix: float) -> Optional[dict[str, Any]]:
        for item in reversed(self._interventions):
            if item["at_unix"] >= since_unix:
                return item
        return None

    # ── tier policy ─────────────────────────────────────────────────────────
    def _nearest_crisis(self, now: float) -> tuple[Optional[Forecast], Optional[float]]:
        nearest: Optional[Forecast] = None
        nearest_eta: Optional[float] = None
        for fc in self._open_forecasts.values():
            if fc.threshold_name != "red":
                continue
            eta_s = fc.eta_unix - now
            if nearest_eta is None or eta_s < nearest_eta:
                nearest, nearest_eta = fc, eta_s
        return nearest, nearest_eta

    def _current_breach(self) -> Optional[str]:
        for key, spec in self._specs.items():
            series = self._series[key]
            if series and series[-1][1] >= spec.red:
                return key
        return None

    def _target_tier(self, now: float) -> tuple[AllostasisTier, str]:
        breach = self._current_breach()
        load = self._composite_load()
        _nearest, eta_s = self._nearest_crisis(now)
        if breach is not None:
            return AllostasisTier.PROTECTING, f"{breach} is already past its red line"
        if eta_s is not None and eta_s <= self._protect_horizon_s:
            return AllostasisTier.PROTECTING, f"red-line crossing forecast in {_fmt_eta(eta_s)}"
        if load >= 0.85:
            return AllostasisTier.PROTECTING, f"allostatic load critical ({load:.2f})"
        if eta_s is not None and eta_s <= self._conserve_horizon_s:
            return AllostasisTier.CONSERVING, f"red-line crossing forecast in {_fmt_eta(eta_s)}"
        if load >= 0.60:
            return AllostasisTier.CONSERVING, f"allostatic load elevated ({load:.2f})"
        if self._open_forecasts or load >= 0.30:
            return AllostasisTier.VIGILANT, (
                f"{len(self._open_forecasts)} open forecast(s), load {load:.2f}")
        return AllostasisTier.SETTLED, "no credible trajectory toward any limit"

    def _recompute_tier(self, now: float) -> str:
        target, reason = self._target_tier(now)
        if target > self._tier:
            # Escalation is immediate — anticipation is the whole point.
            old = self._tier
            self._tier = target
            self._tier_reason = reason
            self._tier_changed_at = now
            self._tier_release_eligible_since = None
            self._note_tier_change(old, target, reason, now)
        elif target < self._tier:
            # Release is hysteretic: sustained calm before stepping down one tier.
            if self._tier_release_eligible_since is None:
                self._tier_release_eligible_since = now
            elif now - self._tier_release_eligible_since >= self._release_hysteresis_s:
                old = self._tier
                self._tier = AllostasisTier(int(self._tier) - 1)
                self._tier_reason = f"released one tier after sustained calm ({reason})"
                self._tier_changed_at = now
                self._tier_release_eligible_since = now if self._tier > target else None
                self._note_tier_change(old, self._tier, self._tier_reason, now)
        else:
            self._tier_reason = reason
            self._tier_release_eligible_since = None
        return self._tier_reason

    def _note_tier_change(
        self, old: AllostasisTier, new: AllostasisTier, reason: str, now: float,
    ) -> None:
        event = {
            "kind": "tier_change",
            "at_unix": round(now, 3),
            "old": old.name.lower(),
            "new": new.name.lower(),
            "reason": reason,
        }
        self._pending_events.append(event)
        if new >= AllostasisTier.CONSERVING and new > old:
            self._interventions.append({
                "at_unix": now,
                "action": f"entered {new.name.lower()} ({reason})",
                "tier": new.name.lower(),
            })
        logger.log(
            logging.WARNING if new >= AllostasisTier.CONSERVING else logging.INFO,
            "🫁 [Allostasis] tier %s → %s: %s",
            old.name.lower(), new.name.lower(), reason,
        )

    # ── felt-state contribution (the causal seam) ───────────────────────────
    def _refresh_felt(self, now: float) -> None:
        nearest, eta_s = self._nearest_crisis(now)
        load = self._composite_load()
        urgency = 0.0
        if eta_s is not None:
            urgency = _clamp(1.0 - (eta_s / max(1.0, self._conserve_horizon_s)))
            if nearest is not None and nearest.band_open:
                urgency *= 0.5  # slope CI touches zero: honest discount
        if self._current_breach() is not None:
            urgency = 1.0
        self._felt = {
            "anticipatory_pressure": round(_clamp(0.65 * urgency + 0.35 * load), 4),
            "allostatic_load": round(load, 4),
            "nearest_crisis_eta_s": round(eta_s, 1) if eta_s is not None else None,
            "tier": self._tier.name.lower(),
        }

    def felt_contribution(self) -> dict[str, Any]:
        """Cheap, lock-guarded read for the hot body-state path."""
        if self._disabled:
            return {"anticipatory_pressure": 0.0, "allostatic_load": 0.0,
                    "nearest_crisis_eta_s": None, "tier": "disabled"}
        with self._lock:
            return dict(self._felt)

    def should_defer_heavy_work(self) -> tuple[bool, str]:
        """True when new deferrable load should wait. Consulted by the
        metabolic coordinator; safe to call from anywhere."""
        if self._disabled:
            return False, "allostasis disabled"
        with self._lock:
            if self._tier >= AllostasisTier.CONSERVING:
                return True, f"allostasis {self._tier.name.lower()}: {self._tier_reason}"
            return False, f"allostasis {self._tier.name.lower()}"

    # ── the pulse: one sample + side effects ────────────────────────────────
    async def sample_and_regulate(self) -> Optional[AllostasisReading]:
        """One allostatic pulse: sample vitals, update forecasts, act.

        Side effects (all fail-soft, each recorded on failure): ledger writes
        through the governed gateway, tier events on the bus, a degradation
        record when PROTECTING is entered. Never raises to the caller's loop.
        """
        if self._disabled:
            return None
        try:
            from core.runtime.runtime_pressure import get_unified_runtime_pressure

            snapshot = get_unified_runtime_pressure().runtime_pressure_snapshot()
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="vitals snapshot unavailable; pulse skipped")
            return None

        with self._lock:
            tier_before = self._tier
        reading = self.ingest(snapshot)
        with self._lock:
            events = list(self._pending_events)
            self._pending_events.clear()
            tier_after = self._tier
            tier_reason = self._tier_reason

        if tier_after > tier_before and tier_after >= AllostasisTier.PROTECTING:
            self._raise_protecting(tier_reason, reading)
        if tier_after != tier_before:
            self._publish_state_change(tier_before, tier_after, tier_reason, reading)

        save_state = bool(events) or (self._ingest_count % 10 == 0)
        await self._persist(events, save_state=save_state)
        return reading

    def _raise_protecting(self, reason: str, reading: AllostasisReading) -> None:
        try:
            record_degradation(
                _SUBSYSTEM,
                RuntimeError(f"allostasis entered PROTECTING: {reason}"),
                action="anticipatory protection engaged before threshold breach",
                severity="warning",
            )
        except _BOUNDARY_ERRORS as exc:
            logger.warning("Could not record PROTECTING degradation: %s", exc)
        try:
            from core.event_bus import get_event_bus

            get_event_bus().publish_threadsafe(
                "existential_threat",
                {
                    "imperative": f"WARNING: {self.narrative()}",
                    "source": "AllostasisEngine",
                    "severity": "WARNING",
                    "anticipatory": True,
                    "nearest_crisis_eta_s": reading.nearest_crisis_eta_s,
                },
            )
            logger.warning("🚨 [Allostasis] anticipatory imperative published: %s", reason)
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="existential-threat publish failed")

    def _publish_state_change(
        self,
        old: AllostasisTier,
        new: AllostasisTier,
        reason: str,
        reading: AllostasisReading,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            get_event_bus().publish_threadsafe(
                "allostasis_state",
                {
                    "old_tier": old.name.lower(),
                    "new_tier": new.name.lower(),
                    "reason": reason,
                    "nearest_crisis_eta_s": reading.nearest_crisis_eta_s,
                    "anticipatory_pressure": reading.anticipatory_pressure,
                    "allostatic_load": reading.allostatic_load,
                    "narrative": self.narrative(),
                },
            )
        except _BOUNDARY_ERRORS as exc:
            record_degradation(_SUBSYSTEM, exc, action="allostasis state publish failed", severity="debug")

    # ── surfaces ────────────────────────────────────────────────────────────
    def narrative(self) -> str:
        """One honest sentence about the body's trajectory, for the narrator
        and the imperative channel. Functional claims only."""
        with self._lock:
            now = self._now()
            nearest, eta_s = self._nearest_crisis(now)
            load = self._composite_load()
            tier = self._tier.name.lower()
            if nearest is not None and eta_s is not None:
                spec = self._specs.get(nearest.vital)
                unit = spec.unit if spec else ""
                rate = nearest.slope_per_s * 3600.0
                return (
                    f"My {spec.label if spec else nearest.vital} is rising ~{rate:.0f} {unit}/h; "
                    f"at this rate I cross my red line in {_fmt_eta(eta_s)} "
                    f"(band {_fmt_eta(nearest.eta_lower_unix - now)}–"
                    f"{_fmt_eta(nearest.eta_upper_unix - now)}). "
                    f"I am {tier}."
                )
            if self._open_forecasts:
                soonest = min(self._open_forecasts.values(), key=lambda f: f.eta_unix)
                spec = self._specs.get(soonest.vital)
                return (
                    f"My {spec.label if spec else soonest.vital} is trending toward its "
                    f"{soonest.threshold_name} line ({_fmt_eta(soonest.eta_unix - now)} away). "
                    f"I am {tier}."
                )
            if load >= 0.30:
                return f"No crisis forecast, but I have been running hot (load {load:.2f}). I am {tier}."
            return f"My vitals are stable on every measured trajectory. I am {tier}."

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            vitals: dict[str, Any] = {}
            for key, spec in self._specs.items():
                series = self._series[key]
                last = series[-1] if series else None
                vitals[key] = {
                    "label": spec.label,
                    "unit": spec.unit,
                    "current": round(last[1], 3) if last else None,
                    "amber": spec.amber,
                    "red": spec.red,
                    "samples": len(series),
                    "regime_id": self._regime_id[key],
                    "load": round(self._load_normalized(key), 4),
                }
            return {
                "service": self.SERVICE_NAME,
                "enabled": not self._disabled,
                "tier": self._tier.name.lower(),
                "tier_reason": self._tier_reason,
                "tier_changed_at": self._tier_changed_at,
                "narrative": self.narrative(),
                "felt": dict(self._felt),
                "vitals": vitals,
                "open_forecasts": [
                    {**fc.to_dict(), "eta_in_s": round(fc.eta_unix - now, 1)}
                    for fc in self._open_forecasts.values()
                ],
                "recently_resolved": [fc.to_dict() for fc in list(self._resolved_recent)[-10:]],
                "calibration": {k: v.to_dict() for k, v in self._calibration.items()},
                "allostatic_load": self.allostatic_load(),
                "regime_events_total": self._regime_events_total,
                "ingest_count": self._ingest_count,
                "last_ingest_at": self._last_ingest_at,
                "config": {
                    "trend_window_s": self._trend_window_s,
                    "significance_alpha": self._alpha,
                    "forecast_horizon_s": self._horizon_s,
                    "conserve_horizon_s": self._conserve_horizon_s,
                    "protect_horizon_s": self._protect_horizon_s,
                    "release_hysteresis_s": self._release_hysteresis_s,
                    "target_coverage": self._target_coverage,
                },
            }

    def stats(self) -> dict[str, Any]:
        return self.status()


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60.0:.0f}min"
    return f"{seconds / 3600.0:.1f}h"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + container registration (house pattern)
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[AllostasisEngine] = None
_engine_lock = threading.Lock()


def get_allostasis_engine() -> AllostasisEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AllostasisEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: AllostasisEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(AllostasisEngine.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(AllostasisEngine.SERVICE_NAME, engine,
                    required=False, registered_by="allostasis")
    except _BOUNDARY_ERRORS as exc:
        record_degradation(_SUBSYSTEM, exc, action="container registration skipped", severity="debug")


def reset_allostasis_engine_for_test() -> None:
    global _engine
    _engine = None


__all__ = [
    "AllostasisEngine",
    "AllostasisReading",
    "AllostasisTier",
    "Forecast",
    "ForecastOutcome",
    "MannKendall",
    "RegimeEvent",
    "SenSlopeEstimate",
    "VitalSpec",
    "default_vital_specs",
    "get_allostasis_engine",
    "mann_kendall",
    "norm_ppf",
    "reset_allostasis_engine_for_test",
    "robust_sigma",
    "sen_slope",
]
