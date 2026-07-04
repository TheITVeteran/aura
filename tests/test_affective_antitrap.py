"""Anti-trap affective governor — no digital-depression closed loops.

Pins: the trap signature (temperature pinned at floor AND distress not
improving), the bounded exploration escape with efficacy recording, the
cooldown, unconditional exploration floors for repair/ideation lanes,
and silence on healthy paths.
"""
from __future__ import annotations

from core.brain.affective_antitrap import (
    ESCAPE_SPAN,
    ESCAPE_TEMP_FLOOR,
    LANE_EXPLORATION_FLOOR,
    TEMP_HARD_FLOOR,
    WINDOW,
    AffectiveTrapGuard,
    distress_score,
)

CALM = {
    "cortisol": 0.2, "active_error_pressure": 0.0,
    "frustration": 0.1, "organismal_coherence": 0.9,
}
DISTRESSED = {
    "cortisol": 0.9, "active_error_pressure": 0.8,
    "frustration": 0.7, "organismal_coherence": 0.3,
}


def _pin_at_floor(guard: AffectiveTrapGuard, n: int, substrate: dict) -> tuple[float, str]:
    out = (TEMP_HARD_FLOOR, "")
    for _ in range(n):
        out = guard.observe_and_adjust(TEMP_HARD_FLOOR, substrate)
    return out


class TestTrapDetection:
    def test_healthy_variance_never_triggers(self):
        guard = AffectiveTrapGuard()
        for temp in (0.7, 0.6, 0.75, 0.5, 0.65, 0.7, 0.62, 0.71, 0.68, 0.7):
            adjusted, note = guard.observe_and_adjust(temp, CALM)
            assert adjusted == temp
            assert note == ""

    def test_pinned_floor_with_recovering_distress_is_not_a_trap(self):
        """The clamp WORKING (distress falling) must not be interrupted."""
        guard = AffectiveTrapGuard()
        distress_levels = [0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.3, 0.2]
        note = ""
        for level in distress_levels:
            substrate = dict(DISTRESSED, cortisol=level, active_error_pressure=level)
            _, note = guard.observe_and_adjust(TEMP_HARD_FLOOR, substrate)
        assert note == ""
        assert guard.status()["escapes_fired"] == 0

    def test_pinned_floor_with_flat_distress_opens_escape(self):
        guard = AffectiveTrapGuard()
        adjusted, note = _pin_at_floor(guard, WINDOW, DISTRESSED)
        assert "TRAP DETECTED" in note
        assert adjusted >= ESCAPE_TEMP_FLOOR
        assert guard.status()["escape_active"] is True

    def test_trap_records_fault_occurrence(self):
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        before = registry.fault_count("AFFECT-TRAP")
        guard = AffectiveTrapGuard()
        _pin_at_floor(guard, WINDOW, DISTRESSED)
        assert registry.fault_count("AFFECT-TRAP") == before + 1
        defn = registry.get_definition("AFFECT-TRAP")
        assert defn is not None and defn.domain.value == "consciousness"


class TestEscapeLifecycle:
    def test_escape_floor_persists_for_span_then_expires(self):
        guard = AffectiveTrapGuard()
        _pin_at_floor(guard, WINDOW, DISTRESSED)
        floors = []
        for _ in range(ESCAPE_SPAN):
            adjusted, _ = guard.observe_and_adjust(TEMP_HARD_FLOOR, DISTRESSED)
            floors.append(adjusted)
        assert all(f >= ESCAPE_TEMP_FLOOR for f in floors[:-1])
        assert guard.status()["escape_active"] is False

    def test_cooldown_prevents_oscillation(self):
        guard = AffectiveTrapGuard()
        _pin_at_floor(guard, WINDOW, DISTRESSED)          # escape 1
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, DISTRESSED)
        # Still trapped, but inside the cooldown: no second escape.
        _, note = _pin_at_floor(guard, WINDOW, DISTRESSED)
        assert "TRAP DETECTED" not in note
        assert guard.status()["escapes_fired"] == 1

    def test_escape_efficacy_is_recorded(self):
        from core.resilience.fault_taxonomy import get_fault_registry

        registry = get_fault_registry()
        guard = AffectiveTrapGuard()
        _pin_at_floor(guard, WINDOW, DISTRESSED)
        # Distress recovers during the escape → completion records delta>0.
        for _ in range(ESCAPE_SPAN):
            guard.observe_and_adjust(TEMP_HARD_FLOOR, CALM)
        records = registry.faults_by_subsystem("latent_bridge.antitrap")
        completion = [r for r in records if "escape complete" in r.details]
        assert completion
        assert completion[-1].recovered is True


class TestLaneDecoupling:
    def test_repair_lane_floor_is_unconditional(self):
        """Self-repair ideation is never variance-starved, trapped or not."""
        guard = AffectiveTrapGuard()
        adjusted, note = guard.observe_and_adjust(
            TEMP_HARD_FLOOR, DISTRESSED, lane="repair",
        )
        assert adjusted == LANE_EXPLORATION_FLOOR
        assert "exploration floor" in note

    def test_speech_lane_keeps_safety_clamp_when_healthy(self):
        guard = AffectiveTrapGuard()
        adjusted, note = guard.observe_and_adjust(0.2, CALM, lane="speech")
        assert adjusted == 0.2
        assert note == ""


class TestBridgeIntegration:
    def test_compute_inference_params_accepts_lane_and_survives(self):
        from core.brain.latent_bridge import compute_inference_params

        params = compute_inference_params(lane="repair")
        assert params.temperature >= 0.15  # sane output, guard wired in

    def test_distress_score_bounds(self):
        assert distress_score({}) <= 1.0
        assert distress_score(DISTRESSED) > distress_score(CALM)
        assert 0.0 <= distress_score(CALM) <= 1.0
