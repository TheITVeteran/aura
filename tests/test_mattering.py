"""Mattering: a learned, decaying sense of what matters that reweights the workspace."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.cognition.mattering import MatteringModel, get_mattering_model


def test_intrinsic_topics_matter_by_default():
    m = MatteringModel()
    assert m.learned_importance("a question about safety and harm") > 0.5
    assert m.learned_importance("the weather is mild today") < 0.5


def test_learns_what_mattered():
    m = MatteringModel()
    before = m.learned_importance("the deployment pipeline broke")
    m.note_mattered("the deployment pipeline broke", weight=0.8)
    after = m.learned_importance("the deployment pipeline broke")
    assert after > before


def test_importance_decays(monkeypatch):
    m = MatteringModel(half_life_s=10.0)
    t = 1000.0
    m.note_mattered("flaky cache layer", weight=0.9, now=t)
    near = m.learned_importance("flaky cache layer", now=t)
    far = m.learned_importance("flaky cache layer", now=t + 60.0)
    assert far < near


def test_strong_feeling_raises_mattering():
    m = MatteringModel()
    calm = m.score("a routine note", affective_charge=0.0).score
    charged = m.score("a routine note", affective_charge=0.9).score
    assert charged > calm


def test_reweight_raises_salience_of_what_matters():
    m = MatteringModel()
    m.note_mattered("identity continuity", weight=0.9)

    important = SimpleNamespace(summary="identity continuity at risk", salience=0.3,
                               affective_charge=0.5, action_relevance=0.6, content_id="a")
    trivial = SimpleNamespace(summary="ambient background hum", salience=0.3,
                             affective_charge=0.0, action_relevance=0.1, content_id="b")
    # BoundContent is frozen in prod; here we use a frozen-like dataclass replacement path,
    # so give these __dataclass_fields__ via a real dataclass:
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class C:
        summary: str
        salience: float
        affective_charge: float
        action_relevance: float
        content_id: str

    important = C("identity continuity at risk", 0.3, 0.5, 0.6, "a")
    trivial = C("ambient background hum", 0.3, 0.0, 0.1, "b")

    out = m.reweight_contents([important, trivial])
    by_id = {c.content_id: c.salience for c in out}
    assert by_id["a"] > by_id["b"]   # what matters rose above what doesn't


def test_singleton_stable():
    assert get_mattering_model() is get_mattering_model()


def test_unity_applies_mattering_reweight():
    # End to end: the unified gather reweights by mattering and sorts important-first.
    from core.unity.runtime import UnityRuntime
    m = get_mattering_model()
    m.note_mattered("safety and harm", weight=0.95)

    rt = UnityRuntime()
    state = SimpleNamespace(
        cognition=SimpleNamespace(current_objective="is this safe or could it cause harm",
                                  current_origin="", current_partner=""),
        affect=None, working_memory=None, world_state=None,
    )
    contents = rt.gather_contents(state)
    assert contents, "expected at least the objective content"
    # salience is sorted descending after mattering reweight
    sals = [c.salience for c in contents]
    assert sals == sorted(sals, reverse=True)
