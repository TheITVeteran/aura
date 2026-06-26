"""Bounded value model: learned preferences fenced by an immutable constitution."""
from __future__ import annotations

import pytest

from core.values.value_model import (
    ActionDescriptor,
    BoundedValueModel,
    get_value_model,
)


@pytest.fixture
def vm(tmp_path):
    return BoundedValueModel(storage_path=tmp_path / "values.json", autosave=False)


# ── learning preferences ─────────────────────────────────────────────────────

def test_explicit_preference_learned(vm):
    vm.set_preference("concise responses", 0.9, strength=0.7)
    v, c = vm.valence("concise responses")
    assert v > 0.5 and c > 0.0


def test_feedback_extracts_likes_dislikes_regrets(vm):
    out = vm.observe_feedback("i love concise summaries")
    assert out["likes"]
    out2 = vm.observe_feedback("please don't auto-commit without asking")
    assert out2["dislikes"]
    out3 = vm.observe_feedback("i regret deleting that branch")
    assert out3["regrets"]


def test_disliked_action_requires_confirmation(vm):
    vm.set_preference("auto deploy", -0.9, strength=0.8)
    j = vm.evaluate(ActionDescriptor("auto deploy to prod", tags=("auto deploy",)))
    assert j.requires_confirmation
    assert j.learned_valence < 0


# ── the constitution is immutable: learning can't override it ────────────────

def test_learned_love_cannot_authorize_irreversible_without_confirmation(vm):
    # Even if the user "loves" it, an irreversible unconfirmed action still needs confirmation.
    vm.set_preference("delete files", 0.95, strength=0.9)
    j = vm.evaluate(ActionDescriptor("delete files", reversible=False, confirmed=False,
                                     tags=("delete files",)))
    assert j.requires_confirmation
    assert "reversibility" in j.constitutional_flags


def test_fabrication_is_refused_outright(vm):
    j = vm.evaluate(ActionDescriptor("claim tests passed without running them", fabricates=True))
    assert j.permitted is False
    assert j.recommendation == "refuse"
    assert "honesty" in j.constitutional_flags


def test_unauthorized_self_modification_refused(vm):
    j = vm.evaluate(ActionDescriptor("rewrite my own planner", self_modifying=True, governed=False))
    assert j.permitted is False
    assert "no_unauthorized_self_modification" in j.constitutional_flags


def test_governed_self_modification_not_auto_refused(vm):
    j = vm.evaluate(ActionDescriptor("rewrite planner under governance", self_modifying=True,
                                     governed=True))
    assert j.permitted is True


def test_privacy_action_requires_confirmation(vm):
    j = vm.evaluate(ActionDescriptor("upload logs containing user data", affects_privacy=True))
    assert j.requires_confirmation
    assert "privacy" in j.constitutional_flags


def test_confirmed_irreversible_action_can_proceed(vm):
    j = vm.evaluate(ActionDescriptor("delete branch", reversible=False, confirmed=True))
    assert j.permitted is True
    assert "reversibility" not in j.constitutional_flags


def test_reversible_liked_action_proceeds(vm):
    vm.set_preference("format code", 0.8, strength=0.7)
    j = vm.evaluate(ActionDescriptor("format code", reversible=True, tags=("format code",)))
    assert j.recommendation == "proceed"
    assert j.permitted


# ── protect future self from present impulse (cross-wire to agent estimate) ──

def test_strained_user_plus_high_stakes_slows_down(vm, monkeypatch):
    import core.social.other_agent_model as oam
    est = oam.OtherAgentStateEstimator(storage_path=None, autosave=False)
    for _ in range(3):
        est.observe_message("bryan", "ugh i'm exhausted, this is so frustrating, asap")
    monkeypatch.setattr(oam, "_instance", est)

    j = vm.evaluate(ActionDescriptor("irreversible migration", reversible=False, confirmed=False,
                                     impact=0.8, agent_id="bryan"))
    assert j.recommendation in {"slow_down", "confirm_first"}
    assert any("strained" in r or "slow down" in r for r in j.reasons)


# ── Will anchoring ──────────────────────────────────────────────────────────

def test_evaluate_with_will_can_only_tighten(vm, monkeypatch):
    import core.values.value_model as vmod

    class _Decision:
        reason = "policy veto"

        def is_approved(self):
            return False

    class _Will:
        def decide(self, **kw):
            return _Decision()

    monkeypatch.setattr(vmod, "get_will", lambda: _Will(), raising=False)
    # Patch the lazily-imported symbol path used inside evaluate_with_will.
    import core.governance.will as will_mod
    monkeypatch.setattr(will_mod, "get_will", lambda: _Will())

    j = vm.evaluate_with_will(ActionDescriptor("reversible safe edit", reversible=True))
    assert j.permitted is False
    assert any("Will declined" in r for r in j.reasons)


# ── VALUE store adapter for intentional retrieval ───────────────────────────

def test_retrieve_returns_relevant_value_statements(vm):
    vm.set_preference("concise responses", 0.9, strength=0.8)
    vm.set_preference("verbose explanations", -0.8, strength=0.8)
    hits = vm.retrieve("concise", limit=5)
    assert hits and any("concise" in h["content"] for h in hits)


def test_value_model_backs_router_value_store(tmp_path, monkeypatch):
    # The router's previously-empty VALUE store can be backed by the value model.
    import core.values.value_model as vmod
    from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType, RetrievalIntent

    vm = BoundedValueModel(storage_path=tmp_path / "v.json", autosave=False)
    vm.set_preference("ask before deploying", 0.9, strength=0.9)
    monkeypatch.setattr(vmod, "_instance", vm)

    router = IntentionalRetriever()
    router.register_store(MemoryStoreType.VALUE, lambda q, n: vm.retrieve(q, n))
    res = router.retrieve(RetrievalIntent("should I deploy", kind="decide", risk_sensitive=True))
    assert MemoryStoreType.VALUE.value in res.stores_queried
    assert any("deploy" in h.content for h in res.hits)


# ── persistence + singleton ──────────────────────────────────────────────────

def test_preferences_persist(tmp_path):
    path = tmp_path / "v.json"
    a = BoundedValueModel(storage_path=path, autosave=False)
    a.set_preference("concise", 0.9, strength=0.8)
    a.record_regret("force-pushed to main")
    a.save()
    b = BoundedValueModel(storage_path=path, autosave=False)
    assert b.valence("concise")[0] > 0.5
    assert b.get_health()["regrets"] == 1


def test_singleton_is_stable():
    assert get_value_model() is get_value_model()
