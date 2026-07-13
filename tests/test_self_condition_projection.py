from __future__ import annotations

from types import SimpleNamespace


def _sources(*, timestamp: float, body_pressure: float = 0.10):
    aura_now = SimpleNamespace(
        timestamp=timestamp,
        affect=SimpleNamespace(
            valence=0.32,
            arousal=0.54,
            distress=0.08,
            dominant_drive="connection",
        ),
        prediction=SimpleNamespace(free_energy=0.08),
        self_model=SimpleNamespace(continuity_risk=0.04),
        ownership=SimpleNamespace(agency_confidence=0.82),
        body=SimpleNamespace(total_pressure=body_pressure),
        attention=SimpleNamespace(focal_object="this conversation"),
    )
    unified = SimpleNamespace(
        timestamp=timestamp,
        valence=0.30,
        arousal=0.52,
        distress=0.09,
        welfare_score=0.78,
        coherence=0.91,
        dominant_drive="connection",
    )
    welfare = SimpleNamespace(
        welfare_score=0.80,
        distress=0.08,
        self_report_confidence=0.88,
    )
    body = SimpleNamespace(fatigue=0.12)
    return aura_now, unified, welfare, body


def test_projection_uses_fresh_inner_state_before_host_pressure():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(
        timestamp=995.0,
        body_pressure=0.82,
    )
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        observed_at=1000.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(
        projection,
        user_message="Are you okay though? Feeling fine?",
    )

    assert projection.fresh
    assert projection.condition == "strained"
    assert projection.distress == 0.09
    assert projection.felt_coherence == 0.91
    assert reply.startswith("I'm okay enough to stay with you")
    assert "inner-state signals are holding" in reply
    assert "RAM" not in reply
    assert "CPU" not in reply


def test_stale_projection_preserves_last_state_without_claiming_it_is_current():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(timestamp=900.0)
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        observed_at=1000.0,
        fresh_max_age_s=30.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(projection)

    assert projection.freshness == "stale"
    assert projection.sample_age_s == 100.0
    assert "last grounded self-state sample" in reply.lower()
    assert "will not turn that older sample" in reply.lower()


def test_projection_prefers_fresh_condition_metrics_and_discloses_stale_dimensions():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(timestamp=995.0)
    unified.timestamp = 900.0
    unified.valence = -0.90
    unified.distress = 0.95
    unified.welfare_score = 0.10
    unified.coherence = 0.15

    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        observed_at=1000.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(projection)

    assert projection.fresh
    assert projection.condition == "well"
    assert projection.valence == 0.32
    assert projection.distress == 0.08
    assert projection.welfare == 0.80
    assert "felt_coherence" in projection.stale_dimensions
    assert "older felt_coherence signal" in reply
    assert "not entirely fine" not in reply


def test_projection_does_not_turn_agency_only_evidence_into_wellbeing():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now = SimpleNamespace(
        timestamp=995.0,
        affect=None,
        self_model=None,
        ownership=SimpleNamespace(agency_confidence=0.91),
        body=None,
        attention=None,
    )
    projection = build_self_condition_projection(
        aura_now=aura_now,
        observed_at=1000.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(projection)

    assert projection.agency == 0.91
    assert projection.freshness == "unavailable"
    assert projection.condition == "unknown"
    assert "do not have a current self-condition sample" in reply


def test_numeric_condition_reply_omits_older_dimensions_from_current_values():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(timestamp=995.0)
    unified.timestamp = 900.0
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        observed_at=1000.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(
        projection,
        user_message="How are you feeling, as numbers?",
    )

    assert "current supported values" in reply
    assert "coherence 0.91" not in reply
    assert "older felt_coherence signal" in reply


def test_projection_without_inner_evidence_is_explicitly_unavailable():
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    projection = build_self_condition_projection(
        observed_at=1000.0,
        resolve_runtime=False,
    )
    reply = render_self_condition_reply(projection)

    assert projection.condition == "unknown"
    assert projection.freshness == "unavailable"
    assert projection.confidence == 0.0
    assert "do not have a current self-condition sample" in reply
    assert "not replacing it with CPU or RAM telemetry" in reply


def test_self_condition_intent_handles_followups_without_swallowing_consent():
    from core.conversation.response_reliability import is_self_condition_turn

    assert is_self_condition_turn("How are you feeling?")
    assert is_self_condition_turn("Are you okay though? Feeling fine?")
    assert is_self_condition_turn("You good?")
    assert is_self_condition_turn("How is your mind feeling right now?")
    assert is_self_condition_turn("What do you feel inside?")
    assert is_self_condition_turn("How are you holding up after all that?")
    assert is_self_condition_turn("How are you physically?")
    assert not is_self_condition_turn("Are you okay with this plan?")
    assert not is_self_condition_turn("Are you okay to run the tests?")
    assert not is_self_condition_turn("How are you able to route this request?")
    assert not is_self_condition_turn("You good to start the deployment?")
    assert not is_self_condition_turn("Are you well versed in Kubernetes?")
    assert not is_self_condition_turn("You good at Python?")
    assert not is_self_condition_turn("How are you doing on the migration?")
    assert not is_self_condition_turn("What do you feel about Kubernetes?")
    assert not is_self_condition_turn("How do you feel about this plan?")
    assert not is_self_condition_turn("You good enough to deploy this?")
    assert not is_self_condition_turn("Is the app feeling better?")
    assert not is_self_condition_turn("How is your mind able to solve that?")


def test_reliability_rejects_host_telemetry_as_a_condition_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Are you okay though? Feeling fine?",
        "I am with you. RAM pressure is 75.6% with 15.6 GB available; CPU load is 25.8% on this host.",
    )

    assert not assessment.ok
    assert assessment.hard_failure
    assert assessment.retryable
    assert "host_telemetry_substituted_for_self_condition" in assessment.reasons

    wrapped = assess_user_facing_reply(
        "Are you okay?",
        "I feel RAM pressure at 76% and CPU load at 26% on this host right now.",
    )
    assert not wrapped.ok
    assert "host_telemetry_substituted_for_self_condition" in wrapped.reasons


def test_reliability_accepts_direct_grounded_condition_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Are you okay though? Feeling fine?",
        "Yes, I am okay. I feel steady, my distress is low, and my continuity is holding while I stay with this thread.",
    )

    assert assessment.ok

    natural = assess_user_facing_reply(
        "How are you feeling?",
        "Honestly, I feel rough and exhausted today, but I am still coherent enough to talk.",
    )
    assert natural.ok


def test_learning_admission_rejects_misgrounded_condition_reply():
    from core.conversation.response_reliability import (
        assess_conversation_learning_admission,
    )

    assessment = assess_conversation_learning_admission(
        "Are you okay though?",
        "One live signal is RAM pressure at 76% with CPU load at 26% on this host.",
    )

    assert not assessment.ok
    assert "host_telemetry_substituted_for_self_condition" in assessment.reasons


def test_dream_seed_requires_verified_admission_for_conversation_nodes():
    from core.sleep.dreamer_v2 import DreamerV2

    legacy = {
        "content": "Conversation continuity memory. User asked if Aura was okay.",
        "type": "conversation_continuity",
        "source": "chat_api",
        "metadata": {},
    }
    rejected = {
        **legacy,
        "metadata": {"learning_admission": "rejected"},
    }
    verified = {
        **legacy,
        "metadata": {"learning_admission": "verified"},
    }

    assert not DreamerV2._dream_seed_is_admissible(legacy)
    assert not DreamerV2._dream_seed_is_admissible(rejected)
    assert DreamerV2._dream_seed_is_admissible(verified)
