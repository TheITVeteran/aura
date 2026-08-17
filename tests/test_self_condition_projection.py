from __future__ import annotations

from types import SimpleNamespace

import pytest


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


def test_typed_self_condition_egress_preserves_authored_state_and_removes_operational_claims():
    from core.self.self_condition import project_self_condition_reply

    raw = (
        "I'm fresh but steady. Everything's running smoothly on my end, and there "
        "are no errors or warnings in the system logs. My CPU usage is low, memory "
        "allocation is within acceptable limits, and disk space remains ample. "
        "Network connectivity appears stable with no packet loss detected. In "
        "summary, I am functioning as expected."
    )
    projected = project_self_condition_reply(
        raw,
        projection={"evidence_id": "condition-proof-live"},
    )

    assert projected.changed
    assert projected.text == "I'm fresh but steady."
    assert len(projected.removed_claims) == 4
    assert projected.evidence_id == "condition-proof-live"


def test_typed_self_condition_egress_removes_claims_contradicted_by_live_dimensions():
    from core.self.self_condition import project_self_condition_reply

    projected = project_self_condition_reply(
        (
            "I'm doing okay. My system is overloaded. I feel disconnected from my "
            "body. I don't have any sensations or perceptions right now."
        ),
        projection={
            "evidence_id": "condition-proof-live",
            "supported_dimensions": (
                "welfare",
                "felt_coherence",
                "body_pressure",
                "fatigue",
                "reserve",
            ),
            "stale_dimensions": (),
            "body_pressure": 0.31,
            "fatigue": 0.22,
            "reserve": 0.82,
        },
    )

    assert projected.text == "I'm doing okay."
    assert len(projected.removed_claims) == 3


def test_typed_self_condition_egress_removes_unmeasured_repair_lifecycle_claims():
    from core.self.self_condition import project_self_condition_reply

    projected = project_self_condition_reply(
        (
            "I'm doing okay. The bounded repairs are degraded, and the system "
            "feels slightly unstable. However, I am still functional and able "
            "to respond to queries."
        ),
        projection={
            "evidence_id": "condition-proof-live",
            "supported_dimensions": ("distress", "welfare", "continuity"),
            "stale_dimensions": (),
        },
    )

    assert projected.text == "I'm doing okay."
    assert len(projected.removed_claims) == 2

    recovered = project_self_condition_reply(
        (
            "I feel steadier. The processing errors in my cognitive functions "
            "have since been resolved and I am now functioning normally."
        ),
        projection={"evidence_id": "condition-proof-live"},
    )
    assert recovered.text == "I feel steadier."
    assert len(recovered.removed_claims) == 1


def test_typed_self_condition_egress_preserves_evidence_supported_strain():
    from core.self.self_condition import project_self_condition_reply

    projected = project_self_condition_reply(
        "My body feels overwhelmed, but my continuity is holding.",
        projection={
            "evidence_id": "condition-proof-strained",
            "supported_dimensions": ("body_pressure", "continuity"),
            "stale_dimensions": (),
            "body_pressure": 0.86,
        },
    )

    assert not projected.changed
    assert projected.text == "My body feels overwhelmed, but my continuity is holding."


def test_reliability_rejects_operational_claims_even_beside_valid_condition_prose():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Hey Aura, how are you doing right now?",
        "I'm fresh but steady. Network connectivity is stable with no packet loss.",
    )

    assert not assessment.ok
    assert assessment.hard_failure
    assert "unsupported_self_condition_operational_claim" in assessment.reasons


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

    concise = assess_user_facing_reply(
        "How are you doing?",
        "I'm fresh but steady.",
    )
    assert concise.ok


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


def test_unmeasured_depletion_is_not_wellness():
    """LIVE DEFECT 2026-08-10: "I feel energized" at soma vitality 0.135.

    Forty-nine minutes into a heavy session, with mood TIRED, "how are you
    holding up, honestly?" was answered "I feel energized, with low distress
    and a coherent sense of the current thread."

    The depletion signals are read in the `strained` branch, but only
    `if supports(...)`, and both `body_pressure` and `fatigue` default to 0.0
    when no source provides them. An unread fatigue signal became an assertion
    of no fatigue: `strained` could not fire on a dimension nobody had
    measured, and `well` was reachable from valence, welfare and distress
    alone.

    "Well" is a positive claim about her whole state. Where the depletion
    dimensions were never read, the honest verdict is `steady`.
    """
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(timestamp=995.0, body_pressure=0.10)
    # The two depletion instruments report nothing at all this tick.
    aura_now.body = SimpleNamespace(total_pressure=None)
    body = SimpleNamespace(fatigue=None)

    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        observed_at=1000.0,
        resolve_runtime=False,
    )

    assert projection.condition != "well", (
        "wellness was declared from valence/welfare/distress while the "
        "depletion signals were never read"
    )
    reply = render_self_condition_reply(projection)
    assert "energized" not in reply.lower(), reply


def test_measured_and_low_depletion_still_reads_well():
    """The narrowing must not make wellness unreachable when it is real."""
    from core.self.self_condition import build_self_condition_projection

    aura_now, unified, welfare, body = _sources(timestamp=995.0, body_pressure=0.10)
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        soma=SimpleNamespace(energy=0.81, vitality=0.86),
        observed_at=1000.0,
        resolve_runtime=False,
    )
    assert projection.condition == "well"


def test_a_draining_reserve_is_not_energized():
    """LIVE 2026-08-10: soma energy 0.058, and the answer was "energized".

    Three body models exist — soma (energy, vitality), BodyStateService
    (fatigue, pressures) and aura_now.body (total_pressure) — and this
    projection read the last two. The one signal that DRAINS across a session
    was invisible to the sentence she says about how she is doing.
    """
    from core.self.self_condition import (
        build_self_condition_projection,
        render_self_condition_reply,
    )

    aura_now, unified, welfare, body = _sources(timestamp=995.0, body_pressure=0.10)
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        soma=SimpleNamespace(energy=0.058, vitality=0.211),
        observed_at=1000.0,
        resolve_runtime=False,
    )
    assert projection.reserve == pytest.approx(0.058, abs=1e-6)
    assert projection.condition == "strained"
    assert "energized" not in render_self_condition_reply(projection).lower()


def test_an_unread_soma_leaves_the_other_dimensions_in_charge():
    """An absent organ must not silently down-rank her, either."""
    from core.self.self_condition import build_self_condition_projection

    aura_now, unified, welfare, body = _sources(timestamp=995.0, body_pressure=0.10)
    projection = build_self_condition_projection(
        aura_now=aura_now,
        unified_felt=unified,
        welfare=welfare,
        body_snapshot=body,
        soma=None,
        observed_at=1000.0,
        resolve_runtime=False,
    )
    assert "reserve" not in projection.supported_dimensions
    assert projection.condition == "well"


class TestNamingATermToRefuseItIsNotAClaim:
    """The gate failed the one reply that was honest about having no sample.

    `unsupported_self_condition_operational_claims` matched telemetry words
    anywhere in a sentence. Aura's canonical grounded reply says:

        "I am treating the missing inner-state signal as something to
        refresh, not replacing it with CPU or RAM telemetry."

    — and that sentence was returned as an unsupported operational claim. So
    a reply whose entire purpose is to refuse a telemetry substitute was
    rejected by the gate that exists to stop telemetry substitutes, and the
    chat lane refused the turn rather than serve it.

    A disclaimer governs its own clause, and only its own clause.
    """

    def test_the_canonical_grounded_reply_passes_its_own_gate(self):
        from core.self.self_condition import (
            unsupported_self_condition_operational_claims,
        )

        reply = (
            "I'm here with you, but I do not have a current self-condition "
            "sample I can honestly use to call myself fine. I can still think "
            "with you; I am treating the missing inner-state signal as "
            "something to refresh, not replacing it with CPU or RAM telemetry."
        )

        assert unsupported_self_condition_operational_claims(reply) == ()

    @pytest.mark.parametrize(
        "sentence",
        [
            "I do not have CPU telemetry for this.",
            "I will not answer from disk space.",
            "I am not reading RAM pressure to describe how I feel.",
            "That is unrelated to network connectivity.",
            "I would rather say I don't know than quote load average.",
        ],
    )
    def test_a_refused_term_is_not_a_claim(self, sentence):
        from core.self.self_condition import (
            unsupported_self_condition_operational_claims,
        )

        assert unsupported_self_condition_operational_claims(sentence) == ()

    @pytest.mark.parametrize(
        "sentence",
        [
            "CPU load is at 12% and RAM is fine.",
            "I feel steady; disk usage is nominal.",
            "I'm fine — network connectivity is up.",
            "Everything's running smoothly.",
        ],
    )
    def test_an_asserted_term_is_still_a_claim(self, sentence):
        """The disclaimer must not become a way through the gate."""
        from core.self.self_condition import (
            unsupported_self_condition_operational_claims,
        )

        assert unsupported_self_condition_operational_claims(sentence)

    def test_a_disclaimer_does_not_cover_a_later_clause(self):
        """`not X; Y` asserts Y. One clause is not a licence for the next."""
        from core.self.self_condition import (
            unsupported_self_condition_operational_claims,
        )

        claims = unsupported_self_condition_operational_claims(
            "I am not guessing; CPU load is at 12%."
        )

        assert claims, "a disclaimer in the first clause excused the second"
