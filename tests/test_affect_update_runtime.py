from types import SimpleNamespace

from core.health.degraded_events import clear_degraded_events, record_degraded_event
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState


def test_affect_feedback_rewards_successful_self_disclosure_turn():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.conversation_energy = 0.72
    state.response_modifiers["response_contract"] = {"requires_aura_stance": True}
    state.response_modifiers["dialogue_validation"] = {"ok": True, "violations": []}

    trust_before = state.affect.emotions["trust"]
    anticipation_before = state.affect.emotions["anticipation"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["trust"] > trust_before
    assert state.affect.emotions["anticipation"] > anticipation_before
    assert state.affect.social_hunger < hunger_before


def test_low_energy_message_is_not_treated_as_social_withdrawal():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.conversation_energy = 0.12
    state.cognition.user_emotional_trend = "neutral"
    sadness_before = state.affect.emotions["sadness"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["sadness"] == sadness_before
    assert state.affect.social_hunger == hunger_before


def test_low_energy_cooling_off_still_registers_social_friction():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.conversation_energy = 0.12
    state.cognition.user_emotional_trend = "cooling_off"
    sadness_before = state.affect.emotions["sadness"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["sadness"] > sadness_before
    assert state.affect.social_hunger > hunger_before


def test_affect_feedback_registers_prompt_fishing_as_social_friction():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.response_modifiers["response_contract"] = {"requires_aura_stance": True}
    state.response_modifiers["dialogue_validation"] = {
        "ok": False,
        "violations": ["prompt_fishing_closer", "missing_first_person_stance"],
    }

    sadness_before = state.affect.emotions["sadness"]
    anger_before = state.affect.emotions["anger"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["sadness"] > sadness_before
    assert state.affect.emotions["anger"] > anger_before
    assert state.affect.social_hunger > hunger_before


def test_foreground_turn_releases_stale_fear_without_active_failure():
    phase = AffectUpdatePhase(SimpleNamespace(organs={}))
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.conversation_energy = 0.7
    state.affect.valence = -1.0
    state.affect.emotions["fear"] = 1.0
    state.affect.emotions["sadness"] = 0.8
    state.affect.emotions["anger"] = 0.4

    phase._ensure_affect_schema(state.affect)
    phase._regulate_stale_negative_affect(
        state.affect,
        state,
        "What tools can you use externally?",
        [{"type": "positive_interaction", "intensity": 0.2}],
    )
    phase._derive_metrics(state.affect)

    assert state.affect.emotions["fear"] < 1.0
    assert state.affect.emotions["sadness"] < 0.8
    assert state.affect.valence > -0.95
    assert "stale_negative_affect_regulated" in state.affect.markers


def test_foreground_stale_fear_regulation_does_not_mask_real_failures():
    phase = AffectUpdatePhase(SimpleNamespace(organs={}))
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.modifiers["system_failure_state"] = {
        "pressure": 0.6,
        "critical": 1,
    }
    state.affect.valence = -1.0
    state.affect.emotions["fear"] = 1.0
    state.affect.emotions["sadness"] = 0.8

    phase._ensure_affect_schema(state.affect)
    phase._regulate_stale_negative_affect(
        state.affect,
        state,
        "What tools can you use externally?",
        [{"type": "positive_interaction", "intensity": 0.2}],
    )

    assert state.affect.emotions["fear"] == 1.0
    assert state.affect.emotions["sadness"] == 0.8
    assert "stale_negative_affect_regulated" not in state.affect.markers


def test_affect_system_pressures_register_unified_failure_load():
    clear_degraded_events()
    record_degraded_event("router", "down", severity="critical", classification="foreground_blocking")
    record_degraded_event("memory", "stall", severity="error", classification="background_degraded")

    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    fear_before = state.affect.emotions["fear"]
    sadness_before = state.affect.emotions["sadness"]

    phase._apply_system_pressures(state.affect, state)

    assert state.cognition.modifiers["system_failure_state"]["pressure"] > 0.0
    assert state.affect.emotions["fear"] > fear_before
    assert state.affect.emotions["sadness"] > sadness_before


def test_affect_system_pressures_register_continuity_reentry_burden():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.modifiers["continuity_obligations"] = {
        "continuity_pressure": 0.78,
        "continuity_reentry_required": True,
        "continuity_scar": "time_gap, abrupt_shutdown",
    }
    fear_before = state.affect.emotions["fear"]
    anticipation_before = state.affect.emotions["anticipation"]
    hunger_before = state.affect.social_hunger

    phase._apply_system_pressures(state.affect, state)

    assert state.affect.emotions["fear"] > fear_before
    assert state.affect.emotions["anticipation"] > anticipation_before
    assert state.affect.social_hunger > hunger_before
