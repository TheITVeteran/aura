"""Tests for core/autonomic/allostasis.py — the predictive-interoception organ.

Covers, in order:
  * the pure math (inverse normal CDF, Mann–Kendall with tie correction,
    Sen's slope + CI robustness, robust sigma);
  * time-to-crisis forecasting on synthetic ramps (accuracy, significance
    gating, direction gating, horizon gating);
  * CUSUM regime detection on residuals (steady ramps do NOT fire; slope
    changes DO; forecasts under a dead regime are superseded/credited);
  * the falsifiable ledger: every resolution outcome and the
    coverage-driven band widening;
  * allostatic load accrual and decay;
  * the anticipatory policy: tier escalation, hysteretic release, the
    heavy-work gate, and the felt-state contribution;
  * robustness to malformed snapshots and cold starts;
  * governed persistence (ledger + state) and restart supersession.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from core.autonomic.allostasis import (
    AllostasisEngine,
    AllostasisTier,
    Forecast,
    ForecastOutcome,
    VitalSpec,
    _VitalCalibration,
    default_vital_specs,
    mann_kendall,
    norm_ppf,
    robust_sigma,
    sen_slope,
)

MEM = "memory_rss_mb"


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def simple_spec(**overrides) -> VitalSpec:
    base = dict(
        key=MEM, label="process memory", unit="MB",
        amber=1000.0, red=1200.0, setpoint=800.0,
        forecastable=True, min_meaningful_slope=0.0,
    )
    base.update(overrides)
    return VitalSpec(**base)


def make_engine(tmp_path, clock: FakeClock, **overrides) -> AllostasisEngine:
    kwargs = dict(
        specs=(simple_spec(),),
        now_fn=clock.now,
        data_dir=tmp_path / "allostasis",
        min_trend_samples=8,
        trend_window_s=3600.0,
        forecast_horizon_s=6 * 3600.0,
        conserve_horizon_s=1800.0,
        protect_horizon_s=600.0,
        release_hysteresis_s=300.0,
        resolution_grace_s=120.0,
    )
    kwargs.update(overrides)
    return AllostasisEngine(**kwargs)


def feed(engine: AllostasisEngine, clock: FakeClock, values, dt: float = 60.0):
    readings = []
    for v in values:
        readings.append(engine.ingest({MEM: v}, at=clock.now()))
        clock.advance(dt)
    return readings


def ramp(start: float, step: float, n: int) -> list[float]:
    return [start + step * i for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Pure math
# ─────────────────────────────────────────────────────────────────────────────

class TestNormPpf:
    def test_known_quantiles(self):
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
        assert norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-4)
        assert norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-4)

    def test_symmetry(self):
        for p in (0.01, 0.1, 0.3, 0.45):
            assert norm_ppf(p) == pytest.approx(-norm_ppf(1.0 - p), abs=1e-8)

    def test_tail_regions(self):
        # Below/above the central branch switch (0.02425).
        assert norm_ppf(0.001) == pytest.approx(-3.090232, abs=1e-3)
        assert norm_ppf(0.999) == pytest.approx(3.090232, abs=1e-3)

    def test_domain_errors(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                norm_ppf(bad)


class TestMannKendall:
    def test_monotone_increasing_no_ties(self):
        mk = mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mk.s == 10
        assert mk.var_s == pytest.approx(5 * 4 * 15 / 18.0)
        assert mk.rising
        # z = (10-1)/sqrt(16.667) = 2.2045; two-sided p ≈ 0.0275
        assert mk.z == pytest.approx(2.2045, abs=1e-3)
        assert mk.significant(0.05)

    def test_tie_correction(self):
        # [1,1,2,2,3]: two tie groups of size 2 → correction 2·(2·1·9) = 36
        mk = mann_kendall([1.0, 1.0, 2.0, 2.0, 3.0])
        assert mk.s == 8
        assert mk.var_s == pytest.approx((300 - 36) / 18.0)

    def test_decreasing_is_negative_s(self):
        mk = mann_kendall([5.0, 4.0, 3.0, 2.0, 1.0])
        assert mk.s == -10
        assert not mk.rising

    def test_constant_series_no_trend(self):
        mk = mann_kendall([3.0] * 10)
        assert mk.s == 0
        assert mk.p_value == 1.0
        assert not mk.significant()

    def test_too_short(self):
        assert mann_kendall([1.0, 2.0]).p_value == 1.0

    def test_noise_false_positive_rate_bounded(self):
        rng = random.Random(1234)
        false_positives = 0
        trials = 200
        for _ in range(trials):
            series = [rng.gauss(0.0, 1.0) for _ in range(30)]
            if mann_kendall(series).significant(0.05):
                false_positives += 1
        # Nominal rate is 5%; allow generous slack for a finite sample.
        assert false_positives / trials <= 0.12


class TestSenSlope:
    def test_exact_linear(self):
        times = [60.0 * i for i in range(10)]
        values = [5.0 + 0.5 * t for t in times]
        est = sen_slope(times, values)
        assert est is not None
        assert est.slope == pytest.approx(0.5, abs=1e-12)
        # Noise-free: every pairwise slope identical → degenerate CI.
        assert est.lower == pytest.approx(0.5, abs=1e-12)
        assert est.upper == pytest.approx(0.5, abs=1e-12)

    def test_robust_to_outliers(self):
        rng = random.Random(7)
        times = [60.0 * i for i in range(24)]
        values = [100.0 + 1.0 * t + rng.gauss(0, 3) for t in times]
        values[5] += 500.0   # a GC spike
        values[17] -= 400.0  # a burst release
        est = sen_slope(times, values)
        assert est is not None
        assert est.slope == pytest.approx(1.0, rel=0.15)

    def test_ci_brackets_true_slope(self):
        rng = random.Random(99)
        hits = 0
        trials = 100
        for _ in range(trials):
            times = [60.0 * i for i in range(30)]
            values = [10.0 + 0.2 * t + rng.gauss(0, 4) for t in times]
            est = sen_slope(times, values, confidence=0.90)
            assert est is not None
            if est.lower <= 0.2 <= est.upper:
                hits += 1
        assert hits / trials >= 0.80  # nominal 90%, slack for finite sample

    def test_degenerate_inputs(self):
        assert sen_slope([0.0, 1.0], [1.0, 2.0]) is None
        assert sen_slope([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None


class TestRobustSigma:
    def test_known_mad(self):
        # median 3, |dev| = [2,1,0,1,97] → MAD = 1 → σ ≈ 1.4826
        assert robust_sigma([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(1.4826)

    def test_outlier_immunity(self):
        rng = random.Random(3)
        clean = [rng.gauss(0, 2.0) for _ in range(200)]
        spiked = list(clean)
        spiked[10] = 1e6
        assert robust_sigma(spiked) == pytest.approx(robust_sigma(clean), rel=0.05)

    def test_short_series(self):
        assert robust_sigma([1.0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting on synthetic ramps
# ─────────────────────────────────────────────────────────────────────────────

class TestForecasting:
    def test_clean_ramp_issues_accurate_red_forecast(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        t0 = clock.now()
        # 500 MB rising 5 MB per 60 s sample → crosses red (1200) at sample 140.
        feed(engine, clock, ramp(500.0, 5.0, 20))
        status = engine.status()
        open_fcs = {f["threshold_name"]: f for f in status["open_forecasts"]}
        assert "red" in open_fcs and "amber" in open_fcs
        true_red_crossing = t0 + 140 * 60.0
        assert open_fcs["red"]["eta_unix"] == pytest.approx(true_red_crossing, abs=120.0)
        true_amber_crossing = t0 + 100 * 60.0
        assert open_fcs["amber"]["eta_unix"] == pytest.approx(true_amber_crossing, abs=120.0)

    def test_flat_noise_issues_nothing(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        rng = random.Random(42)
        feed(engine, clock, [500.0 + rng.gauss(0, 2.0) for _ in range(40)])
        assert engine.status()["open_forecasts"] == []
        assert engine.status()["tier"] == "settled"

    def test_falling_trend_issues_nothing(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(900.0, -5.0, 20))
        assert engine.status()["open_forecasts"] == []

    def test_slow_trend_beyond_horizon_not_issued(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock, forecast_horizon_s=3600.0)
        # +0.5 MB/min from 500: red is ~1400 min away ≫ 1 h horizon.
        feed(engine, clock, ramp(500.0, 0.5, 20))
        assert engine.status()["open_forecasts"] == []

    def test_min_meaningful_slope_gates_noise_level_trends(self, tmp_path):
        clock = FakeClock()
        spec = simple_spec(min_meaningful_slope=10.0 / 60.0)  # require ≥ 10 MB/min
        engine = make_engine(tmp_path, clock, specs=(spec,))
        feed(engine, clock, ramp(500.0, 5.0, 20))  # only 5 MB/min
        assert engine.status()["open_forecasts"] == []

    def test_forecast_revised_not_duplicated(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 5.0, 30))
        status = engine.status()
        red = [f for f in status["open_forecasts"] if f["threshold_name"] == "red"]
        assert len(red) == 1
        assert red[0]["revisions"] >= 1
        assert red[0]["first_eta_unix"] > 0

    def test_ramp_to_crossing_resolves_hit(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 5.0, 141))  # sample 140 = 1200 = red
        cal = engine.status()["calibration"][MEM]
        assert cal["hits"] >= 1
        assert cal["coverage"] == pytest.approx(1.0)
        assert not any(
            f["threshold_name"] == "red" for f in engine.status()["open_forecasts"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeDetection:
    def test_steady_ramp_is_one_regime(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 5.0, 60))
        assert engine.status()["regime_events_total"] == 0

    def test_step_change_fires_regime(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        rng = random.Random(5)
        flat = [500.0 + rng.gauss(0, 2.0) for _ in range(30)]
        shifted = [560.0 + rng.gauss(0, 2.0) for _ in range(10)]
        readings = feed(engine, clock, flat + shifted)
        assert engine.status()["regime_events_total"] >= 1
        # Detected within the shifted block, not during the flat reference.
        first_regime_idx = next(
            i for i, r in enumerate(readings) if r.regime_events
        )
        assert first_regime_idx >= 30

    def test_stationary_noise_false_alarm_rate_bounded(self, tmp_path):
        # CUSUM has a finite average run length to false alarm; the tuned
        # detector measures ~1 event per 1000 samples (~17 h at the live
        # 60 s pulse). Deterministic seed: this exact series stays silent,
        # and any regression that cheapens the allowance will fire on it.
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        rng = random.Random(11)
        feed(engine, clock, [500.0 + rng.gauss(0, 3.0) for _ in range(120)])
        assert engine.status()["regime_events_total"] == 0

    def test_plateau_after_ramp_supersedes_forecast(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        # Gentle ramp: forecast issued but crisis stays > conserve horizon,
        # so no intervention is on record when the regime relaxes.
        feed(engine, clock, ramp(500.0, 2.0, 20))
        assert engine.status()["open_forecasts"]
        feed(engine, clock, [538.0] * 15)
        status = engine.status()
        assert status["regime_events_total"] >= 1
        outcomes = {f["status"] for f in status["recently_resolved"]}
        assert ForecastOutcome.SUPERSEDED.value in outcomes

    def test_plateau_after_intervention_credits_intervened(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        # Aggressive ramp: red ETA drops inside the conserve horizon, which
        # records an intervention (tier escalation). The subsequent plateau
        # is then credited to regulation, not written off as noise.
        feed(engine, clock, ramp(1000.0, 6.0, 15))
        assert engine.status()["tier"] in ("conserving", "protecting")
        defer, _ = engine.should_defer_heavy_work()
        assert defer
        plateau_value = 1000.0 + 6.0 * 14
        feed(engine, clock, [plateau_value] * 15)
        outcomes = {f["status"] for f in engine.status()["recently_resolved"]}
        assert ForecastOutcome.INTERVENED.value in outcomes
        assert engine.status()["calibration"][MEM]["intervened"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Ledger resolution paths (white-box for the branches synthetic ramps
# cannot reach deterministically)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_open_forecast(engine, clock, *, eta_offset: float, band: float,
                          threshold: float = 1200.0) -> Forecast:
    now = clock.now()
    fc = Forecast(
        forecast_id="fc-test",
        vital=MEM,
        threshold_name="red",
        threshold_value=threshold,
        regime_id="r-test",
        issued_at=now,
        level_at_issue=1000.0,
        slope_per_s=0.1,
        slope_lower=0.05,
        slope_upper=0.2,
        eta_unix=now + eta_offset,
        eta_lower_unix=now + eta_offset - band,
        eta_upper_unix=now + eta_offset + band,
        band_open=False,
        p_value=0.01,
        widen_factor=1.0,
        first_eta_unix=now + eta_offset,
    )
    engine._open_forecasts[(MEM, "red")] = fc
    return fc


class TestLedgerOutcomes:
    def test_false_alarm_on_expiry_without_crossing(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        _inject_open_forecast(engine, clock, eta_offset=300.0, band=60.0)
        clock.advance(600.0)  # past eta_upper + grace (120)
        engine.ingest({MEM: 900.0}, at=clock.now())
        cal = engine.status()["calibration"][MEM]
        assert cal["false_alarms"] == 1
        assert engine.status()["recently_resolved"][-1]["status"] == "false_alarm"

    def test_miss_early_when_crossing_before_band(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        _inject_open_forecast(engine, clock, eta_offset=3000.0, band=100.0)
        clock.advance(60.0)  # far before eta_lower − grace
        engine.ingest({MEM: 1250.0}, at=clock.now())
        cal = engine.status()["calibration"][MEM]
        assert cal["miss_early"] == 1

    def test_hit_inside_band(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        _inject_open_forecast(engine, clock, eta_offset=300.0, band=120.0)
        clock.advance(300.0)
        engine.ingest({MEM: 1250.0}, at=clock.now())
        assert engine.status()["calibration"][MEM]["hits"] == 1

    def test_expiry_with_intervention_is_intervened(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        fc = _inject_open_forecast(engine, clock, eta_offset=300.0, band=60.0)
        engine._interventions.append({
            "at_unix": fc.issued_at + 10.0,
            "action": "entered conserving (test)",
            "tier": "conserving",
        })
        clock.advance(600.0)
        engine.ingest({MEM: 900.0}, at=clock.now())
        assert engine.status()["calibration"][MEM]["intervened"] == 1


class TestCalibration:
    def test_widen_factor_needs_minimum_sample(self):
        book = _VitalCalibration(hits=1, false_alarms=1)
        assert book.widen_factor(target_coverage=0.9) == 1.0

    def test_widen_factor_from_poor_coverage(self):
        book = _VitalCalibration(hits=2, false_alarms=4)
        assert book.widen_factor(target_coverage=0.9) == pytest.approx(2.7)

    def test_widen_factor_capped(self):
        book = _VitalCalibration(hits=0, false_alarms=6)
        assert book.widen_factor(target_coverage=0.9) == 3.0

    def test_good_coverage_never_narrows(self):
        book = _VitalCalibration(hits=9, false_alarms=1)
        assert book.widen_factor(target_coverage=0.9) == pytest.approx(1.0)

    def test_interventions_do_not_count_against_coverage(self):
        book = _VitalCalibration(hits=3, intervened=50, superseded=20)
        assert book.scored == 3
        assert book.coverage == pytest.approx(1.0)

    def test_poor_calibration_widens_issued_bands(self, tmp_path):
        clock = FakeClock()
        tight_engine = make_engine(tmp_path / "a", clock)
        feed(tight_engine, clock, ramp(500.0, 5.0, 20))
        tight = [f for f in tight_engine.status()["open_forecasts"]
                 if f["threshold_name"] == "red"][0]

        clock2 = FakeClock()
        wide_engine = make_engine(tmp_path / "b", clock2)
        wide_engine._calibration[MEM] = _VitalCalibration(hits=2, false_alarms=4)
        feed(wide_engine, clock2, ramp(500.0, 5.0, 20))
        wide = [f for f in wide_engine.status()["open_forecasts"]
                if f["threshold_name"] == "red"][0]

        assert wide["widen_factor"] == pytest.approx(2.7)
        tight_band = tight["eta_upper_unix"] - tight["eta_lower_unix"]
        wide_band = wide["eta_upper_unix"] - wide["eta_lower_unix"]
        assert wide_band >= tight_band


# ─────────────────────────────────────────────────────────────────────────────
# Allostatic load
# ─────────────────────────────────────────────────────────────────────────────

class TestAllostaticLoad:
    def test_load_accrues_above_setpoint_and_decays_below(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, [1200.0] * 30)  # fully red for 30 min
        hot = engine.allostatic_load()[MEM]
        assert hot > 0.2
        feed(engine, clock, [500.0] * 60)   # calm for an hour
        cool = engine.allostatic_load()[MEM]
        assert cool < hot * 0.5

    def test_no_load_below_setpoint(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, [700.0] * 30)
        assert engine.allostatic_load()[MEM] == pytest.approx(0.0, abs=1e-6)

    def test_composite_is_peak_weighted(self, tmp_path):
        clock = FakeClock()
        specs = (simple_spec(), simple_spec(key="loop_lag_s", label="lag", unit="s",
                                            amber=1.0, red=5.0, setpoint=0.25))
        engine = make_engine(tmp_path, clock, specs=specs)
        feed_values = [{MEM: 1200.0, "loop_lag_s": 0.0} for _ in range(30)]
        for snap in feed_values:
            engine.ingest(snap, at=clock.now())
            clock.advance(60.0)
        load = engine.allostatic_load()
        assert load["composite"] > load[MEM] * 0.5  # peak dominates the blend


# ─────────────────────────────────────────────────────────────────────────────
# Tier policy, gate, felt contribution
# ─────────────────────────────────────────────────────────────────────────────

class TestTierPolicy:
    def test_escalates_through_tiers_as_crisis_approaches(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        seen = []
        for v in ramp(500.0, 10.0, 70):
            reading = engine.ingest({MEM: v}, at=clock.now())
            seen.append(reading.tier)
            clock.advance(60.0)
        assert AllostasisTier.VIGILANT in seen
        assert AllostasisTier.CONSERVING in seen
        assert AllostasisTier.PROTECTING in seen
        # Escalation order is monotone until the peak.
        peak = max(seen)
        assert seen.index(peak) > seen.index(AllostasisTier.VIGILANT)

    def test_current_breach_is_protecting(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, [1250.0] * 3)
        assert engine.status()["tier"] == "protecting"

    def test_release_is_hysteretic_one_tier_at_a_time(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock, release_hysteresis_s=300.0)
        feed(engine, clock, [1250.0] * 3)          # breach → protecting
        assert engine.status()["tier"] == "protecting"
        # Calm values, but load and hysteresis hold the tier up briefly.
        engine.ingest({MEM: 500.0}, at=clock.now())
        assert engine.status()["tier"] == "protecting"
        tiers = []
        for _ in range(40):
            clock.advance(60.0)
            reading = engine.ingest({MEM: 500.0}, at=clock.now())
            tiers.append(reading.tier)
        assert tiers[-1] == AllostasisTier.SETTLED
        # Never skips a tier on the way down.
        for earlier, later in zip(tiers, tiers[1:]):
            assert int(earlier) - int(later) <= 1

    def test_should_defer_heavy_work_by_tier(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        assert engine.should_defer_heavy_work()[0] is False
        feed(engine, clock, [1250.0] * 3)
        defer, reason = engine.should_defer_heavy_work()
        assert defer is True
        assert "protecting" in reason

    def test_chronic_load_alone_escalates(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        # Hold just below red so no breach and (flat) no forecast: chronic
        # strain is the only escalation path available.
        feed(engine, clock, [1190.0] * 200)
        status = engine.status()
        assert status["open_forecasts"] == []
        assert status["tier"] in ("conserving", "protecting")


class TestFeltContribution:
    def test_settled_contributes_nothing(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, [500.0] * 10)
        felt = engine.felt_contribution()
        assert felt["anticipatory_pressure"] == pytest.approx(0.0, abs=0.05)
        assert felt["nearest_crisis_eta_s"] is None

    def test_approaching_crisis_raises_anticipation_before_breach(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 10.0, 55))  # near red but NOT crossed
        felt = engine.felt_contribution()
        assert engine.status()["vitals"][MEM]["current"] < 1200.0
        assert felt["anticipatory_pressure"] > 0.3
        assert felt["nearest_crisis_eta_s"] is not None

    def test_anticipation_monotone_in_proximity(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 10.0, 30))
        early = engine.felt_contribution()["anticipatory_pressure"]
        feed(engine, clock, ramp(800.0, 10.0, 25))
        late = engine.felt_contribution()["anticipatory_pressure"]
        assert late > early

    def test_breach_saturates_urgency(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, [1250.0] * 3)
        assert engine.felt_contribution()["anticipatory_pressure"] >= 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_cold_status_and_narrative(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        status = engine.status()
        assert status["tier"] == "settled"
        assert isinstance(engine.narrative(), str) and engine.narrative()
        assert engine.is_ready()

    def test_malformed_snapshots_never_crash(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        for snap in (
            {},
            {MEM: None},
            {MEM: "not-a-number"},
            {MEM: float("nan")},
            {MEM: float("inf")},
            {"unknown_vital": 5.0},
        ):
            engine.ingest(snap, at=clock.now())
            clock.advance(60.0)
        assert engine.status()["vitals"][MEM]["samples"] == 0

    def test_non_monotonic_timestamps_dropped(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        engine.ingest({MEM: 500.0}, at=1_000_000.0)
        engine.ingest({MEM: 600.0}, at=999_000.0)   # in the past: dropped
        engine.ingest({MEM: 501.0}, at=1_000_060.0)
        assert engine.status()["vitals"][MEM]["samples"] == 2

    def test_disabled_engine_is_inert(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_ALLOSTASIS_DISABLED", "1")
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        assert engine.is_ready() is False
        assert engine.should_defer_heavy_work() == (False, "allostasis disabled")
        assert engine.felt_contribution()["anticipatory_pressure"] == 0.0

    def test_default_specs_are_coherent(self):
        for spec in default_vital_specs():
            assert spec.amber < spec.red
            assert spec.setpoint < spec.red

    def test_narrative_mentions_trajectory_when_forecasting(self, tmp_path):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        feed(engine, clock, ramp(500.0, 10.0, 40))
        text = engine.narrative()
        assert "red line" in text
        assert "band" in text


# ─────────────────────────────────────────────────────────────────────────────
# Persistence (governed writes) and restart behavior
# ─────────────────────────────────────────────────────────────────────────────

class _StubPressure:
    def __init__(self, values, clock):
        self._values = list(values)
        self._clock = clock

    def runtime_pressure_snapshot(self):
        value = self._values.pop(0) if self._values else 500.0
        return {MEM: value, "at_unix": self._clock.now()}


class TestPersistence:
    async def test_sample_and_regulate_writes_governed_ledger(self, tmp_path, monkeypatch):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        stub = _StubPressure(ramp(500.0, 5.0, 30), clock)
        import core.runtime.runtime_pressure as pressure_mod
        monkeypatch.setattr(pressure_mod, "get_unified_runtime_pressure", lambda: stub)

        for _ in range(30):
            reading = await engine.sample_and_regulate()
            assert reading is not None
            clock.advance(60.0)

        events_path = tmp_path / "allostasis" / "forecasts.jsonl"
        state_path = tmp_path / "allostasis" / "state.json"
        assert events_path.exists(), "forecast ledger was never persisted"
        assert state_path.exists(), "state snapshot was never persisted"
        kinds = [json.loads(line)["kind"] for line in
                 events_path.read_text().strip().splitlines()]
        assert "issued" in kinds
        envelope = json.loads(state_path.read_text())
        assert envelope["schema_name"] == "allostasis_state"
        assert envelope["payload"]["open_forecasts"], (
            "open forecasts missing from state snapshot"
        )

    async def test_restart_supersedes_stale_forecasts(self, tmp_path, monkeypatch):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        stub = _StubPressure(ramp(500.0, 5.0, 30), clock)
        import core.runtime.runtime_pressure as pressure_mod
        monkeypatch.setattr(pressure_mod, "get_unified_runtime_pressure", lambda: stub)
        for _ in range(30):
            await engine.sample_and_regulate()
            clock.advance(60.0)
        assert engine.status()["open_forecasts"]

        reborn = make_engine(tmp_path, FakeClock(clock.now() + 3600.0))
        assert reborn._calibration[MEM].superseded >= 1
        pending_kinds = {e.get("kind") for e in reborn._pending_events}
        assert "resolved" in pending_kinds

    async def test_pulse_survives_broken_snapshot_source(self, tmp_path, monkeypatch):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)

        def _broken():
            raise RuntimeError("observer down")

        import core.runtime.runtime_pressure as pressure_mod
        monkeypatch.setattr(pressure_mod, "get_unified_runtime_pressure", _broken)
        assert await engine.sample_and_regulate() is None  # no raise

    async def test_disabled_pulse_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_ALLOSTASIS_DISABLED", "1")
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        assert await engine.sample_and_regulate() is None


# ─────────────────────────────────────────────────────────────────────────────
# Event publication on escalation
# ─────────────────────────────────────────────────────────────────────────────

class _StubBus:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish_threadsafe(self, topic, data, priority=None):
        self.published.append((topic, data))


class TestEscalationSideEffects:
    async def test_protecting_publishes_existential_threat(self, tmp_path, monkeypatch):
        clock = FakeClock()
        engine = make_engine(tmp_path, clock)
        bus = _StubBus()
        import core.event_bus as bus_mod
        monkeypatch.setattr(bus_mod, "get_event_bus", lambda: bus)

        stub = _StubPressure(ramp(500.0, 12.0, 80), clock)
        import core.runtime.runtime_pressure as pressure_mod
        monkeypatch.setattr(pressure_mod, "get_unified_runtime_pressure", lambda: stub)

        for _ in range(80):
            await engine.sample_and_regulate()
            clock.advance(60.0)
            if engine.status()["tier"] == "protecting":
                break
        assert engine.status()["tier"] == "protecting"
        topics = [t for (t, _) in bus.published]
        assert "existential_threat" in topics
        assert "allostasis_state" in topics
        threat = next(d for (t, d) in bus.published if t == "existential_threat")
        assert threat["source"] == "AllostasisEngine"
        assert threat["anticipatory"] is True
