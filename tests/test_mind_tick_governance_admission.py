from __future__ import annotations

from types import SimpleNamespace

from core.mind_tick import _authorize_state_mutation_through_will


def test_mind_tick_state_mutation_uses_one_source_bound_canonical_admission(monkeypatch):
    calls = []
    decision = SimpleNamespace(
        is_approved=lambda: True,
        receipt_id="will-mind-tick-receipt",
    )

    def authorize(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(decision=decision)

    monkeypatch.setattr(
        "core.runtime.action_executor.ActionExecutor.authorize_action",
        authorize,
    )

    observed = _authorize_state_mutation_through_will(
        "consolidate working memory",
        "mind_tick.dream_consolidation",
        priority=0.55,
        context={
            "effect_scope": "internal_restoration",
            "no_external_effects": True,
        },
    )

    assert observed is decision
    assert len(calls) == 1
    assert calls[0]["source"] == "mind_tick.dream_consolidation"
    assert calls[0]["context"] == {
        "source": "mind_tick.dream_consolidation",
        "effect_scope": "internal_restoration",
        "no_external_effects": True,
    }


def test_mind_tick_state_mutation_rejects_conflicting_context_source(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "core.runtime.action_executor.ActionExecutor.authorize_action",
        lambda **kwargs: calls.append(kwargs),
    )

    observed = _authorize_state_mutation_through_will(
        "consolidate working memory",
        "mind_tick.dream_consolidation",
        context={
            "source": "untrusted.self_label",
            "effect_scope": "internal_restoration",
            "no_external_effects": True,
        },
    )

    assert observed is None
    assert calls == []
