"""One correct way to accept a number from outside.

CP126 raised the same finding against seven modules independently — belief
confidence, global stability pressure, scheduler intervals, telemetry
bounds, inquiry scores, execution timeouts, fine-tune quality — all of the
form "accepts non-finite and out-of-range values". Separate findings, one
defect, and the codebase carried a dozen near-identical private helpers
(_safe_float, _finite_float, _clamp01, _finite, _clamp) so fixing one taught
the others nothing.

The subtlety that makes ad-hoc clamping wrong is NaN: every comparison with
NaN is False, so the usual idiom does not clamp it, it propagates it —

    min(max(float("nan"), 0.0), 1.0)  ->  nan

core/volition.py had exactly this. A NaN inquiry score reached the priority
comparison, and because nothing compares greater than it, it sorted as the
highest priority available. A corrupt number there does not merely produce a
wrong answer, it wins.
"""
from __future__ import annotations

import math

import pytest

from core.runtime.numeric_guards import (
    bounded_float,
    bounded_int,
    is_finite_number,
    positive_float,
    unit_float,
)

NAN = float("nan")
INF = float("inf")


class TestTheNaNCase:
    def test_the_naive_idiom_really_does_propagate_nan(self):
        """Pins WHY this module exists."""
        assert math.isnan(min(max(NAN, 0.0), 1.0))

    def test_unit_float_rejects_nan_to_the_default(self):
        assert unit_float(NAN, default=0.0) == 0.0

    def test_nan_is_not_clamped_to_the_maximum(self):
        """The volition defect: NaN reaching the top of a priority sort."""
        assert unit_float(NAN, default=0.0) != 1.0

    @pytest.mark.parametrize("value", [INF, -INF, NAN])
    def test_non_finite_values_use_the_default(self, value):
        assert bounded_float(value, default=0.25, minimum=0.0, maximum=1.0) == 0.25


class TestRangeClamping:
    @pytest.mark.parametrize(("value", "expected"), [(1.7, 1.0), (-3.0, 0.0), (0.5, 0.5)])
    def test_in_range_values_survive_and_others_clamp(self, value, expected):
        assert unit_float(value) == expected

    def test_an_out_of_range_default_is_itself_clamped(self):
        """A caller cannot smuggle an impossible value in through default."""
        assert bounded_float(NAN, default=99.0, minimum=0.0, maximum=1.0) == 1.0

    def test_numeric_strings_are_accepted(self):
        assert unit_float("0.5") == 0.5

    def test_unparseable_strings_use_the_default(self):
        assert unit_float("not a number", default=0.3) == 0.3


class TestBooleansAreNotNumbers:
    """True is arithmetically 1 and almost never what a caller passing a
    score or an interval meant; accepting it hides a type error at exactly
    the boundary this guards."""

    @pytest.mark.parametrize("value", [True, False])
    def test_booleans_use_the_default(self, value):
        assert unit_float(value, default=0.25) == 0.25

    @pytest.mark.parametrize("value", [True, False])
    def test_is_finite_number_rejects_booleans(self, value):
        assert is_finite_number(value) is False


class TestPositiveFloat:
    @pytest.mark.parametrize("value", [0.0, -1.0, -0.0])
    def test_non_positive_values_fall_back_to_the_default(self, value):
        """Clamping zero up to an epsilon looks safer and is not: a zero
        interval clamped to 1e-9 is still a busy loop, and a zero timeout is
        still already expired."""
        assert positive_float(value, default=5.0) == 5.0

    def test_a_positive_value_survives(self):
        assert positive_float(2.5, default=5.0) == 2.5

    def test_nan_uses_the_default(self):
        assert positive_float(NAN, default=5.0) == 5.0

    def test_the_result_is_always_positive(self):
        assert positive_float(0.0, default=0.0) > 0.0

    def test_a_maximum_is_honoured(self):
        assert positive_float(99999.0, default=15.0, maximum=600.0) == 600.0


class TestBoundedInt:
    @pytest.mark.parametrize("value", [NAN, INF, "x", None, object()])
    def test_unusable_values_use_the_default(self, value):
        assert bounded_int(value, default=3, minimum=1, maximum=10) == 3

    def test_floats_truncate(self):
        assert bounded_int(7.9, default=0, minimum=0, maximum=10) == 7

    def test_bounds_apply(self):
        assert bounded_int(99, default=0, minimum=1, maximum=10) == 10
        assert bounded_int(-99, default=0, minimum=1, maximum=10) == 1

    def test_booleans_use_the_default(self):
        assert bounded_int(True, default=4) == 4


class TestTheGuardsAreUsedWhereItMattered:
    def test_belief_confidence_is_guarded(self):
        """A NaN belief used to win the top-5 prompt injection outright."""
        from core.final_engines import WorldModelEngine

        engine = WorldModelEngine.__new__(WorldModelEngine)
        engine.beliefs = {}
        engine.persist_path = None
        engine._save_beliefs = lambda: None
        engine.add_belief("nan claim", NAN)
        engine.add_belief("real claim", 0.9)
        assert engine.beliefs["nan claim"].confidence == 0.0
        top = max(engine.beliefs.values(), key=lambda b: b.confidence)
        assert top.claim == "real claim"

    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_scheduler_rejects_non_finite_intervals(self, value):
        """A NaN interval passed `< 0` validation and then made
        `now - last_run >= interval` False forever: registered, reported
        healthy, never ran again."""
        from core.scheduler import TaskSpec

        async def _noop():
            return None

        with pytest.raises(ValueError):
            TaskSpec(name="t", coro=_noop, tick_interval=value)

    def test_scheduler_still_accepts_a_zero_interval(self):
        """Zero means "run every scheduler tick" — bounded by the scheduler's
        own cadence, not a busy loop — and the suite relies on it."""
        from core.scheduler import TaskSpec

        async def _noop():
            return None

        assert TaskSpec(name="t", coro=_noop, tick_interval=0.0).tick_interval == 0.0

    def test_execution_timeout_is_bounded(self):
        import inspect

        from core.actuators import code_execution_actuator as mod

        source = inspect.getsource(mod)
        assert "positive_float(" in source
        assert 'float(params.get("timeout_s", 15.0))' not in source

    def test_finetune_quality_rejects_nan(self):
        import inspect

        from core.adaptation import finetune_pipe as mod

        source = inspect.getsource(mod)
        assert "is_finite_number(quality_score)" in source
        assert "unit_float(quality_score" in source

    def test_reliability_heartbeat_bounds_its_inputs(self):
        """get_global_stability multiplies stability * (1 - pressure), so one
        NaN service makes the whole mean NaN."""
        import inspect

        from core.reliability_engine import ReliabilityEngine

        source = inspect.getsource(ReliabilityEngine.heartbeat)
        assert "unit_float(stability" in source
        assert "unit_float(pressure" in source


class TestOneBadReadingCannotDisableADrive:
    """CP126: "Non-finite free energy can corrupt boredom state."

    `nan < BOREDOM_FE_CEILING` is False, so a NaN reading took the
    "prediction error present" branch and drained boredom by three every
    tick — the novelty threshold became unreachable. Worse, _last_fe_value
    was overwritten with the NaN, so every later fe_delta was NaN too and
    the surprise-spike relief died as well. One bad reading disabled the
    drive permanently, in both directions.
    """

    def _engine(self):
        from collections import deque

        from core.drive_engine import DriveEngine

        engine = DriveEngine.__new__(DriveEngine)
        engine._last_fe_value = 0.5
        engine._boredom_ticks = 100
        engine._seek_novelty = False
        engine._boredom_history = deque(maxlen=10)
        engine._relieve_boredom = lambda *a, **k: None
        return engine

    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_a_non_finite_reading_does_not_poison_the_last_value(self, value):
        engine = self._engine()
        engine.tick_boredom(value)
        assert math.isfinite(engine._last_fe_value)

    def test_a_finite_reading_still_updates_state(self):
        engine = self._engine()
        engine.tick_boredom(0.9)
        assert engine._last_fe_value == 0.9

    def test_boredom_still_accumulates_on_a_low_reading(self):
        engine = self._engine()
        engine._last_fe_value = 0.0
        before = engine._boredom_ticks
        engine.tick_boredom(0.0)
        assert engine._boredom_ticks == before + 1


class TestCorruptQualityIsNotAJudgement:
    """CP126: "Cortana accepts non-finite and unbounded quality signals."

    `nan > 0.6` is False, so a NaN quality silently counted as a FAILED
    turn. success_rate feeds the rampancy/metastability verdict, so corrupt
    input did not produce an error — it produced a confident judgement about
    her cognitive health built on a non-judgement.
    """

    def _monitor(self):
        from core.fictional_ai_synthesis import CognitiveHealthMonitor

        monitor = CognitiveHealthMonitor.__new__(CognitiveHealthMonitor)
        monitor._total_turns = 0
        monitor._successful_turns = 0
        monitor._unresolved_threads = 0
        monitor._metastability_score = 0.5
        monitor._rampancy_stage = 0
        monitor._history = []
        return monitor

    def test_a_fully_corrupt_turn_does_not_raise(self):
        monitor = self._monitor()
        monitor.record_turn(
            context_tokens=NAN, max_tokens=NAN, response_quality=NAN,
            identity_markers_present=True, topics_in_play=NAN, resolved_topics=NAN,
        )
        assert monitor._total_turns == 1

    def test_the_metastability_score_stays_finite(self):
        monitor = self._monitor()
        monitor.record_turn(
            context_tokens=NAN, max_tokens=0, response_quality=INF,
            identity_markers_present=False, topics_in_play=-5, resolved_topics=99,
        )
        assert math.isfinite(monitor._metastability_score)

    def test_a_genuinely_good_turn_still_counts_as_success(self):
        monitor = self._monitor()
        monitor.record_turn(
            context_tokens=100, max_tokens=1000, response_quality=0.9,
            identity_markers_present=True, topics_in_play=2, resolved_topics=2,
        )
        assert monitor._successful_turns == 1


class TestTelemetryGaugesAreBounded:
    """CP126: "Telemetry omits upper and finite bounds."

    Every gauge carried ge=0.0 and nothing else, so inf and NaN validated
    cleanly and reached the browser — a NaN renders as "NaN" in a gauge and
    an unbounded energy silently rescales every chart on the page.

    The enforcement CLAMPS rather than rejects, and that choice matters: the
    payload is built inside the heartbeat and published to the websocket, so
    a ValidationError there does not protect the UI, it kills the telemetry
    stream and freezes the dashboard on its last good frame while the
    runtime looks healthy.
    """

    def _payload(self, **kwargs):
        from core.schemas import TelemetryPayload

        return TelemetryPayload(**kwargs)

    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_non_finite_gauges_take_the_default(self, value):
        assert self._payload(energy=value).energy == 100.0

    def test_an_over_range_gauge_clamps_to_the_maximum(self):
        assert self._payload(energy=500.0).energy == 100.0

    def test_an_under_range_gauge_clamps_to_the_minimum(self):
        assert self._payload(energy=-20.0).energy == 0.0

    def test_normalised_scores_use_their_own_range(self):
        assert self._payload(coherence=3.0).coherence == 1.0

    def test_a_healthy_value_is_untouched(self):
        payload = self._payload(energy=80.0, coherence=0.9)
        assert payload.energy == 80.0
        assert payload.coherence == 0.9

    def test_fully_corrupt_telemetry_never_raises(self):
        """The whole point: the stream must survive bad numbers."""
        payload = self._payload(
            energy=NAN, curiosity=INF, coherence=NAN, vitality=-5, cpu_usage=INF,
        )
        assert all(
            math.isfinite(getattr(payload, name))
            for name in ("energy", "curiosity", "coherence", "vitality", "cpu_usage")
        )

    def test_every_bounded_field_has_a_declared_range(self):
        from core.schemas import TelemetryPayload

        for name, (lo, hi, default) in TelemetryPayload._BOUNDS.items():
            assert lo <= default <= hi, name
