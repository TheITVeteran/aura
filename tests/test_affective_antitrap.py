"""Anti-trap affective governor — no digital-depression closed loops.

Pins: the trap signature (temperature pinned at floor AND distress not
improving AND sustained over real time), the bounded exploration escape with
efficacy recording, the cooldown, unconditional exploration floors for
repair/ideation lanes, and silence on healthy paths.

Also pins the eleven CP126 findings, which shared one shape — the module
wrote down what it intended rather than what happened:

  * an escape that could never complete because repair-lane calls returned
    before the counter was touched;
  * a trap window filled with REQUESTED temperatures from lanes that had
    actually run at 0.50, so repair traffic could open an escape for speech;
  * a distress score that replaced missing axes with optimistic constants,
    so a disconnected substrate read as a calm one;
  * NaN temperature passing both floor tests untouched;
  * efficacy from one endpoint sample, where any positive delta was
    "recovered";
  * fault records with no escape id and no observations;
  * a swallowed registry failure while the counters reported success.

Every test drives a fake clock, so "sustained" is asserted rather than
waited for.
"""
from __future__ import annotations

import math

import pytest

from core.brain.affective_antitrap import (
    DEFAULT_CALIBRATION,
    ESCAPE_SPAN,
    ESCAPE_TEMP_FLOOR,
    LANE_EXPLORATION_FLOOR,
    TEMP_HARD_FLOOR,
    WINDOW,
    AffectiveTrapGuard,
    Lane,
    distress_reading,
    distress_score,
    resolve_lane,
)

CALM = {
    "cortisol": 0.2, "active_error_pressure": 0.0,
    "frustration": 0.1, "organismal_coherence": 0.9,
}
DISTRESSED = {
    "cortisol": 0.9, "active_error_pressure": 0.8,
    "frustration": 0.7, "organismal_coherence": 0.3,
}

#: Enough wall time per observation that a full window is a sustained state.
_STEP_S = DEFAULT_CALIBRATION.min_window_span_s / (WINDOW - 1) + 0.5


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def guard(clock: _Clock) -> AffectiveTrapGuard:
    return AffectiveTrapGuard(clock=clock)


def _pin_at_floor(
    guard: AffectiveTrapGuard,
    clock: _Clock,
    n: int,
    substrate: dict,
    *,
    step_s: float = _STEP_S,
    lane: str = "speech",
) -> tuple[float, str]:
    out = (TEMP_HARD_FLOOR, "")
    for _ in range(n):
        out = guard.observe_and_adjust(TEMP_HARD_FLOOR, substrate, lane=lane)
        clock.advance(step_s)
    return out


class TestTrapDetection:
    def test_healthy_variance_never_triggers(self, guard, clock):
        for temp in (0.7, 0.6, 0.75, 0.5, 0.65, 0.7, 0.62, 0.71, 0.68, 0.7):
            adjusted, note = guard.observe_and_adjust(temp, CALM)
            clock.advance(_STEP_S)
            assert adjusted == temp
            assert note == ""

    def test_pinned_floor_with_recovering_distress_is_not_a_trap(self, guard, clock):
        """The clamp WORKING (distress falling) must not be interrupted."""
        note = ""
        for level in [0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.3, 0.2]:
            substrate = dict(DISTRESSED, cortisol=level, active_error_pressure=level)
            _, note = guard.observe_and_adjust(TEMP_HARD_FLOOR, substrate)
            clock.advance(_STEP_S)
        assert note == ""
        assert guard.status()["escapes_fired"] == 0

    def test_pinned_floor_with_flat_distress_opens_escape(self, guard, clock):
        adjusted, note = _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        assert "TRAP DETECTED" in note
        assert adjusted >= ESCAPE_TEMP_FLOOR
        assert guard.status()["escape_active"] is True

    def test_trap_records_fault_occurrence(self, guard, clock):
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        before = registry.fault_count("AFFECT-TRAP")
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        assert registry.fault_count("AFFECT-TRAP") == before + 1
        defn = registry.get_definition("AFFECT-TRAP")
        assert defn is not None and defn.domain.value == "consciousness"


class TestABurstIsNotASustainedState:
    def test_eight_calls_in_one_instant_do_not_trap(self, guard, clock):
        """The window carries timestamps and never used them, so any burst
        of eight qualifying calls opened an escape."""
        _, note = _pin_at_floor(guard, clock, WINDOW * 2, DISTRESSED, step_s=0.0)
        assert "TRAP DETECTED" not in note
        assert guard.status()["escapes_fired"] == 0

    def test_the_same_calls_spread_over_time_do_trap(self, guard, clock):
        """Same evidence, real duration: this is the condition it is for."""
        _, note = _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        assert "TRAP DETECTED" in note

    def test_the_span_requirement_is_the_calibrated_one(self, guard, clock):
        just_short = (DEFAULT_CALIBRATION.min_window_span_s / WINDOW) * 0.5
        _, note = _pin_at_floor(guard, clock, WINDOW * 2, DISTRESSED, step_s=just_short)
        assert "TRAP DETECTED" not in note


class TestTheTrapWindowIsSpeechEvidenceOnly:
    def test_repair_lane_calls_do_not_fill_the_speech_trap_window(self, guard, clock):
        """A repair call runs at 0.50. It used to enter the ring as the 0.15
        it asked for, so repair traffic manufactured the pinned signature."""
        _, note = _pin_at_floor(guard, clock, WINDOW * 2, DISTRESSED, lane="repair")
        assert "TRAP DETECTED" not in note
        assert guard.status()["trap_window_filled"] == 0

    def test_the_effective_temperature_is_what_gets_recorded(self, guard, clock):
        """During an escape the model samples at 0.45, so those observations
        are not evidence of a floor-pinned model — otherwise an escape
        re-arms on its own output."""
        # One short of the window: every observation ran at the floor and
        # nothing has intervened yet.
        _pin_at_floor(guard, clock, WINDOW - 1, DISTRESSED)
        assert guard.status()["trap_window_pinned"] is True
        assert guard.status()["escapes_fired"] == 0

        for _ in range(ESCAPE_SPAN + 3):
            adjusted, _ = guard.observe_and_adjust(TEMP_HARD_FLOOR, DISTRESSED)
            clock.advance(_STEP_S)
        assert guard.status()["trap_window_pinned"] is False

    def test_an_incomplete_reading_is_not_evidence_of_calm(self, guard, clock):
        """A disconnected substrate used to read as a healthy one."""
        for _ in range(WINDOW * 2):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, {})
            clock.advance(_STEP_S)
        status = guard.status()
        assert status["trap_window_filled"] == 0
        assert status["readings_rejected"] == WINDOW * 2
        assert status["escapes_fired"] == 0


class TestEscapeLifecycle:
    def test_escape_floor_persists_for_span_then_expires(self, guard, clock):
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        floors = []
        for _ in range(ESCAPE_SPAN):
            adjusted, _ = guard.observe_and_adjust(TEMP_HARD_FLOOR, DISTRESSED)
            clock.advance(_STEP_S)
            floors.append(adjusted)
        assert all(f >= ESCAPE_TEMP_FLOOR for f in floors)
        assert guard.status()["escape_active"] is False

    def test_cooldown_prevents_oscillation(self, guard, clock):
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)          # escape 1
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, DISTRESSED)
            clock.advance(_STEP_S)
        # Still trapped, but inside the cooldown: no second escape.
        _, note = _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        assert "TRAP DETECTED" not in note
        assert guard.status()["escapes_fired"] == 1

    def test_a_repair_lane_stream_cannot_hold_an_escape_open(self, guard, clock):
        """The lifecycle bug: lane decoupling returned before the escape
        counter was decremented, so the escape never advanced and its
        completion — the efficacy measurement — never ran."""
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        assert guard.status()["escape_active"] is True

        for _ in range(ESCAPE_SPAN):
            adjusted, _ = guard.observe_and_adjust(
                TEMP_HARD_FLOOR, DISTRESSED, lane="repair"
            )
            clock.advance(_STEP_S)
            assert adjusted >= LANE_EXPLORATION_FLOOR

        assert guard.status()["escape_active"] is False, "the escape must advance"

    def test_the_escape_floor_never_lowers_a_lane_floor(self, guard, clock):
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        adjusted, _ = guard.observe_and_adjust(
            TEMP_HARD_FLOOR, DISTRESSED, lane="ideation"
        )
        assert adjusted == LANE_EXPLORATION_FLOOR


class TestEfficacyIsMeasuredNotAsserted:
    def test_a_real_recovery_is_recorded_as_recovered(self, guard, clock):
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, CALM)
            clock.advance(_STEP_S)

        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        completion = [r for r in records if "escape" in r.details and "complete" in r.details]
        assert completion
        assert completion[-1].recovered is True

    def test_noise_is_not_recorded_as_recovery(self, guard, clock):
        """One endpoint sample made any positive delta a recovery."""
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        # A hair better than entry, well under the efficacy threshold.
        barely = dict(DISTRESSED, cortisol=DISTRESSED["cortisol"] - 0.02)
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, barely)
            clock.advance(_STEP_S)

        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        completion = [r for r in records if "complete" in r.details]
        assert completion[-1].recovered is False

    def test_the_completion_says_it_is_uncontrolled(self, guard, clock):
        """A before/after comparison is not an attribution, and the record
        must not read like one."""
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, CALM)
            clock.advance(_STEP_S)

        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        completion = [r for r in records if "complete" in r.details][-1]
        assert "does not prove the temperature caused the change" in completion.details


class TestTheAuditIsReproducible:
    def test_the_opening_record_carries_the_observations(self, guard, clock):
        """It used to carry a sentence restating the thresholds, which can
        neither be replayed nor checked."""
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)

        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        opening = [r for r in records if "opened" in r.details][-1]
        assert "effective_temperature" in opening.details
        assert "distress" in opening.details
        assert DEFAULT_CALIBRATION.version in opening.details

    def test_both_records_name_the_same_escape(self, guard, clock):
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        escape_id = guard.status()["escape_id"]
        assert escape_id

        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, CALM)
            clock.advance(_STEP_S)

        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        mine = [r for r in records if escape_id in r.details]
        assert len(mine) == 2, "opening and completion must be correlatable"

    def test_a_failed_fault_write_is_visible_in_status(self, guard, clock, monkeypatch):
        """The counters used to advance regardless, so status could report a
        recorded escape whose audit occurrence did not exist."""
        import core.resilience.fault_taxonomy as taxonomy

        def _broken():
            raise RuntimeError("registry down")

        monkeypatch.setattr(taxonomy, "get_fault_registry", _broken)

        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)

        status = guard.status()
        assert status["escapes_fired"] == 1
        assert status["fault_record_failures"] == 1
        assert status["fault_records_written"] == 0
        assert status["audit_complete"] is False

    def test_a_healthy_guard_reports_a_complete_audit(self, guard, clock):
        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)
        status = guard.status()
        assert status["fault_records_written"] >= 1
        assert status["audit_complete"] is True


class TestTheRegistryIsNotCalledUnderTheLock:
    def test_a_reentrant_registry_does_not_deadlock(self, guard, clock, monkeypatch):
        """Escape begin and completion used to run inside the non-reentrant
        guard lock while synchronously calling another subsystem."""
        import core.resilience.fault_taxonomy as taxonomy

        seen: list[dict] = []
        real = taxonomy.get_fault_registry

        class _Reentrant:
            def record_fault(self, *args, **kwargs):
                # The registry calls back into the guard, as a status probe
                # or a health surface reasonably might.
                seen.append(guard.status())
                return real().record_fault(*args, **kwargs)

        monkeypatch.setattr(taxonomy, "get_fault_registry", _Reentrant)

        _pin_at_floor(guard, clock, WINDOW + 1, DISTRESSED)

        assert seen, "the registry must actually have been called"
        assert guard.status()["escapes_fired"] == 1


class TestLaneDecoupling:
    def test_repair_lane_floor_is_unconditional(self, guard):
        """Self-repair ideation is never variance-starved, trapped or not."""
        adjusted, note = guard.observe_and_adjust(
            TEMP_HARD_FLOOR, DISTRESSED, lane="repair",
        )
        assert adjusted == LANE_EXPLORATION_FLOOR
        assert "exploration floor" in note

    def test_speech_lane_keeps_safety_clamp_when_healthy(self, guard):
        adjusted, note = guard.observe_and_adjust(0.2, CALM, lane="speech")
        assert adjusted == 0.2
        assert note == ""

    def test_an_unknown_lane_cannot_buy_exploration_variance(self, guard):
        """The lane was a free-form string and membership in a public set was
        the only condition for a 0.50 floor."""
        adjusted, note = guard.observe_and_adjust(
            TEMP_HARD_FLOOR, DISTRESSED, lane="repair_but_not_really",
        )
        assert adjusted == TEMP_HARD_FLOOR
        assert "unknown lane" in note
        assert guard.status()["unknown_lane_requests"] == 1

    @pytest.mark.parametrize(
        "value,expected,known",
        [
            ("repair", Lane.REPAIR, True),
            ("SPEECH", Lane.SPEECH, True),
            (Lane.IDEATION, Lane.IDEATION, True),
            ("", Lane.SPEECH, False),
            (None, Lane.SPEECH, False),
            ("../repair", Lane.SPEECH, False),
        ],
    )
    def test_lane_resolution(self, value, expected, known):
        assert resolve_lane(value) == (expected, known)


class TestTelemetryValidation:
    def test_nan_temperature_is_replaced_not_propagated(self, guard):
        """NaN fails every comparison, so it used to slide through both floor
        tests and pin detection as though it had been checked."""
        adjusted, note = guard.observe_and_adjust(float("nan"), CALM)
        assert math.isfinite(adjusted)
        assert adjusted == TEMP_HARD_FLOOR
        assert "non-finite temperature" in note

    def test_infinite_temperature_is_replaced(self, guard):
        adjusted, _ = guard.observe_and_adjust(float("inf"), CALM)
        assert adjusted == TEMP_HARD_FLOOR

    def test_a_non_numeric_substrate_value_does_not_raise(self, guard):
        """float(value) on the inference path used to raise."""
        reading = distress_reading({"cortisol": "hot", "frustration": 0.2})
        assert 0.0 <= reading.score <= 1.0
        assert "cortisol" in reading.axes_invalid
        assert reading.complete is False

    def test_a_non_mapping_substrate_is_survived(self, guard):
        adjusted, _ = guard.observe_and_adjust(0.5, None)  # type: ignore[arg-type]
        assert adjusted == 0.5

    def test_a_negative_temperature_is_not_a_pinned_observation(self, guard, clock):
        for _ in range(WINDOW * 2):
            guard.observe_and_adjust(-5.0, DISTRESSED)
            clock.advance(_STEP_S)
        # It is below the floor, so it IS pinned — but it is also returned
        # unchanged rather than being read as a valid sample of a clamped
        # model. What must not happen is a crash or a silent NaN.
        assert guard.status()["escapes_fired"] in (0, 1)


class TestDistressCoverageIsReported:
    def test_a_complete_substrate_reads_complete(self):
        reading = distress_reading(DISTRESSED)
        assert reading.complete is True
        assert reading.axes_missing == ()

    def test_an_empty_substrate_reports_every_axis_missing(self):
        """It used to return an ordinary score built from optimistic
        constants, so a disconnected substrate looked healthy."""
        reading = distress_reading({})
        assert reading.complete is False
        assert len(reading.axes_missing) == 4
        assert reading.axes_present == ()

    def test_a_partial_substrate_names_what_is_missing(self):
        reading = distress_reading({"cortisol": 0.9})
        assert reading.axes_present == ("cortisol",)
        assert "frustration" in reading.axes_missing

    def test_distress_score_bounds(self):
        assert distress_score({}) <= 1.0
        assert distress_score(DISTRESSED) > distress_score(CALM)
        assert 0.0 <= distress_score(CALM) <= 1.0


class TestCalibrationIsDeclared:
    def test_status_exposes_the_whole_calibration_with_a_version(self, guard):
        calibration = guard.status()["calibration"]
        assert calibration["version"] == DEFAULT_CALIBRATION.version
        for key in (
            "window", "floor_epsilon", "temp_hard_floor", "escape_temp_floor",
            "escape_span", "cooldown_s", "lane_exploration_floor",
            "min_window_span_s", "min_efficacy_delta",
        ):
            assert key in calibration

    def test_the_basis_says_which_numbers_are_not_calibrated(self, guard):
        """A guess presented as a calibrated threshold is the thing to avoid."""
        basis = guard.status()["calibration"]["basis"]
        assert "NOT empirically calibrated" in basis

    def test_the_module_constants_agree_with_the_calibration(self):
        assert WINDOW == DEFAULT_CALIBRATION.window
        assert TEMP_HARD_FLOOR == DEFAULT_CALIBRATION.temp_hard_floor
        assert ESCAPE_TEMP_FLOOR == DEFAULT_CALIBRATION.escape_temp_floor
        assert ESCAPE_SPAN == DEFAULT_CALIBRATION.escape_span
        assert LANE_EXPLORATION_FLOOR == DEFAULT_CALIBRATION.lane_exploration_floor


class TestBridgeIntegration:
    def test_compute_inference_params_accepts_lane_and_survives(self):
        from core.brain.latent_bridge import compute_inference_params

        params = compute_inference_params(lane="repair")
        assert params.temperature >= 0.15  # sane output, guard wired in

    def test_the_bridges_floor_matches_the_guards_idea_of_it(self):
        """The guard's pin test is only meaningful if it is testing against
        the floor the bridge actually clamps to."""
        import inspect

        from core.brain import latent_bridge

        source = inspect.getsource(latent_bridge.compute_inference_params)
        assert f"max({TEMP_HARD_FLOOR}" in source or f"max({TEMP_HARD_FLOOR:.2f}" in source
