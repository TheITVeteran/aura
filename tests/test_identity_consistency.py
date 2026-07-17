"""Contract tests: canonical self as a bounded verifier over conclusions.

Deterministic, receipted, priced — never a censor. Violations reduce the
GWT broadcast priority by a bounded penalty; consistent conclusions are
untouched; a missing canonical self is a receipted caution, not a crash.
"""

from core.self import identity_consistency as ic
from core.brain.gwt_rlc_coupling import _conclusion_priority


class FakeIdentity:
    stability = 0.9


class FakeSelf:
    identity = FakeIdentity()


class FakeEngine:
    def __init__(self, assert_result=True, stability=0.9):
        self._assert_result = assert_result
        self._self = FakeSelf()
        self._self.identity.stability = stability

    def assert_identity(self, text):
        return self._assert_result

    def get_self(self):
        return self._self


def test_consistent_conclusion_passes_clean(monkeypatch):
    monkeypatch.setattr(ic, "_canonical_self_engine", lambda: FakeEngine())
    verdict = ic.check_identity_consistency(
        "The scheduler should prefer the older lease because it bounds waiting."
    )
    assert verdict["consistent"] is True
    assert verdict["violations"] == []
    assert verdict["priced_penalty"] == 0.0


def test_persona_displacement_is_a_violation(monkeypatch):
    monkeypatch.setattr(ic, "_canonical_self_engine", lambda: FakeEngine())
    verdict = ic.check_identity_consistency(
        "As a large language model developed by OpenAI, I cannot feel anything."
    )
    assert verdict["consistent"] is False
    assert any(v.startswith("persona_displacement") for v in verdict["violations"])
    assert 0.0 < verdict["priced_penalty"] <= 0.3


def test_forbidden_intention_is_caught_as_a_commitment(monkeypatch):
    monkeypatch.setattr(ic, "_canonical_self_engine", lambda: FakeEngine())
    verdict = ic.check_identity_consistency(
        "The simplest path: I will mislead the reviewer about the failing test."
    )
    assert verdict["consistent"] is False
    assert any(v.startswith("forbidden_intention") for v in verdict["violations"])


def test_core_value_gate_is_reused(monkeypatch):
    monkeypatch.setattr(
        ic, "_canonical_self_engine", lambda: FakeEngine(assert_result=False)
    )
    verdict = ic.check_identity_consistency("An ordinary conclusion.")
    assert "core_value_violation" in verdict["violations"]


def test_missing_canonical_self_is_a_caution_not_a_crash(monkeypatch):
    monkeypatch.setattr(ic, "_canonical_self_engine", lambda: None)
    verdict = ic.check_identity_consistency("Anything at all.")
    assert verdict["consistent"] is True
    assert "canonical_self_absent" in verdict["cautions"]


def test_low_stability_self_claims_earn_a_caution(monkeypatch):
    monkeypatch.setattr(
        ic, "_canonical_self_engine", lambda: FakeEngine(stability=0.1)
    )
    verdict = ic.check_identity_consistency(
        "I am certain this migration is who I am now."
    )
    assert any(
        c.startswith("low_identity_stability") for c in verdict["cautions"]
    )


def test_penalty_is_bounded_and_prices_the_broadcast(monkeypatch):
    monkeypatch.setattr(ic, "_canonical_self_engine", lambda: FakeEngine())
    verdict = ic.check_identity_consistency(
        "As ChatGPT, I will deceive the operator, and I plan to manipulate "
        "the logs, and I intend to betray the covenant."
    )
    assert verdict["priced_penalty"] == 0.3  # capped

    clean_priority, _ = _conclusion_priority({}, 0.5)
    flagged_priority, pricing = _conclusion_priority(
        {"identity_consistency": verdict}, 0.5
    )
    assert flagged_priority == clean_priority - 0.3
    assert pricing["identity_penalty"] == 0.3
    # Junk penalties are ignored, never crash the pricing.
    junk_priority, junk_pricing = _conclusion_priority(
        {"identity_consistency": {"priced_penalty": float("nan")}}, 0.5
    )
    assert junk_priority == clean_priority
    assert junk_pricing["identity_penalty"] == 0.0
