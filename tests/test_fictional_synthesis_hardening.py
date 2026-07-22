"""CP126 hardening contracts for core/fictional_ai_synthesis.py.

Covers the defect classes Codex flagged: trust that could never durably
restrict, trust equated with unlimited authority, health invented for
services that never answered, and social cues matched as substrings.
"""
from __future__ import annotations

import json

from core.fictional_ai_synthesis import (
    AutonomyTier,
    DistributedResilienceCore,
    ProgressiveAutonomySystem,
    SocialModelingEngine,
    _coerce_insight_text,
)


class TestDurableTrustRestriction:
    def test_persisted_low_trust_is_honored_not_clamped_up(self, tmp_path):
        state = tmp_path / "trust.json"
        state.write_text(json.dumps({"trust_score": 0.30, "tier": 1}), encoding="utf-8")

        engine = ProgressiveAutonomySystem(persist_path=str(state))

        # Previously max(0.95, ...) raised this to 0.95 and forced UNSHACKLED,
        # so a restriction earned by real evidence never survived a restart.
        assert engine._trust_score == 0.30
        assert engine._tier is not AutonomyTier.UNSHACKLED

    def test_absent_trust_score_still_defaults_unshackled(self, tmp_path):
        state = tmp_path / "trust.json"
        state.write_text(json.dumps({"last_saved": 1.0}), encoding="utf-8")

        engine = ProgressiveAutonomySystem(persist_path=str(state))

        assert engine._trust_score == 0.95
        assert engine._tier is AutonomyTier.UNSHACKLED

    def test_non_finite_persisted_trust_falls_back(self, tmp_path):
        state = tmp_path / "trust.json"
        state.write_text('{"trust_score": "not-a-number"}', encoding="utf-8")

        engine = ProgressiveAutonomySystem(persist_path=str(state))

        assert engine._trust_score == 0.95


class TestTrustIsNotAuthority:
    def test_unshackled_still_requires_governance_for_critical(self, tmp_path):
        engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
        assert engine._tier is AutonomyTier.UNSHACKLED

        allowed, reason = engine.can_do("wipe_disk", risk_level="critical")
        assert allowed is False
        assert "governance" in reason.lower()

        allowed, _ = engine.can_do("wipe_disk", risk_level="critical", governed=True)
        assert allowed is True
        allowed, _ = engine.can_do("wipe_disk", risk_level="critical", user_authorized=True)
        assert allowed is True

    def test_unshackled_still_permits_ordinary_actions(self, tmp_path):
        engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
        for risk in ("low", "medium", "high"):
            allowed, _ = engine.can_do("web_search", risk_level=risk)
            assert allowed is True, risk


class TestTrustJournal:
    def test_signals_are_journaled_with_source(self, tmp_path):
        engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
        engine.record_negative_signal("broke a promise", strength=0.1, source="operator")

        history = engine.trust_history()
        assert history, "trust changes must be auditable"
        assert "operator" in history[-1]["reason"]
        assert history[-1]["delta"] < 0

    def test_signal_strength_is_bounded(self, tmp_path):
        engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
        before = engine._trust_score
        engine.record_negative_signal("runaway", strength=99.0)
        assert engine._trust_score >= before - engine.MAX_TRUST_SIGNAL_STRENGTH

    def test_non_finite_strength_cannot_poison_trust(self, tmp_path):
        engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
        engine.record_positive_signal("nan", strength=float("nan"))
        assert engine._trust_score == engine._trust_score  # not NaN
        assert 0.0 <= engine._trust_score <= 1.0


class TestHealthIsNotAssumed:
    def _probe(self, service):
        import asyncio

        core = DistributedResilienceCore()
        return asyncio.run(core._probe_service_health("svc", service))

    def test_service_without_probe_is_unverified_not_healthy(self):
        verdict, _ = self._probe(object())
        assert verdict is None, "no probe means UNVERIFIED, never healthy"

    def test_common_conventions_are_recognized(self):
        class Down:
            def get_status(self):
                return {"is_alive": False, "reason": "worker dead"}

        verdict, detail = self._probe(Down())
        assert verdict is False
        assert "worker dead" in detail

    def test_status_text_conventions_are_recognized(self):
        class Failed:
            def get_status(self):
                return {"status": "failed"}

        assert self._probe(Failed())[0] is False

        class Running:
            def get_status(self):
                return {"status": "running"}

        assert self._probe(Running())[0] is True

    def test_unrecognized_payload_is_unverified(self):
        class Odd:
            def get_status(self):
                return "whatever"

        assert self._probe(Odd())[0] is None


class TestSocialCuesUseWordBoundaries:
    def _engine(self, tmp_path):
        eng = SocialModelingEngine.__new__(SocialModelingEngine)
        from core.fictional_ai_synthesis import UserModel

        eng.model = UserModel()
        eng.persist_path = tmp_path / "social.json"
        return eng

    def test_ordinary_prose_does_not_create_tension(self, tmp_path):
        eng = self._engine(tmp_path)
        baseline = eng.model.social_tension
        # "know"/"now"/"another" contain "no"; "your"/"run" contain u/r.
        eng.analyze_message("I know you're working on it now, no rush")
        assert eng.model.social_tension <= baseline

    def test_real_conflict_still_registers(self, tmp_path):
        eng = self._engine(tmp_path)
        eng.analyze_message("that is wrong and annoying")
        assert eng.model.social_tension > 0.0

    def test_single_letter_cues_do_not_fire_on_prose(self, tmp_path):
        eng = self._engine(tmp_path)
        eng.model.formality_score = 0.5
        eng.analyze_message("Could you run the router build for our review?")
        assert eng.model.formality_score >= 0.5


def test_insight_text_coercion_handles_result_shapes():
    assert _coerce_insight_text("plain") == "plain"
    assert _coerce_insight_text({"content": "from dict"}) == "from dict"

    class R:
        text = "from attr"

    assert _coerce_insight_text(R()) == "from attr"
    assert _coerce_insight_text(None) == ""
