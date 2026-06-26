"""Unified world model: one query surface routing to the four complementary facets."""
from __future__ import annotations

import pytest

from core.world_model.unified_world_model import UnifiedWorldModel, get_unified_world_model


# ── fakes for the four facets (so routing is tested without heavy model init) ──


class _FakeLearned:
    def __init__(self):
        self.config = type("Cfg", (), {"latent_dim": 4})()
        self.h = [0.0, 0.0, 0.0, 0.0]

    def observe(self, obs, action=None, *, learn=True):
        return type("P", (), {"to_dict": lambda self: {"surprise": 0.5, "learned": learn}})()

    def get_surprise(self):
        return 0.42

    def imagine(self, obs, seq):
        return [type("P", (), {"to_dict": lambda self: {"step": i}})() for i in range(len(seq))]


class _FakeCausal:
    def __init__(self):
        self.calls = []

    def add_observation(self, source, target, correlation):
        self.calls.append((source, target, correlation))

    def predict_effects(self, source):
        return [("fire", 0.8)]

    def simulate_counterfactual(self, do, steps=3):
        return {"smoke": 0.9}

    def analyze_preventative_actions(self, node):
        return [("remove_fuel", 0.7)]


class _FakeOutcome:
    def predict(self, obs, act):
        return type("OP", (), {"to_dict": lambda self: {"reward": 0.6, "harm": 0.1}})()

    def observe_episode(self, ep):
        return type("OP", (), {"to_dict": lambda self: {"reward": 0.5}})()

    def get_status(self):
        return {"episodes": 3}


@pytest.fixture
def model():
    return UnifiedWorldModel(learned=_FakeLearned(), causal=_FakeCausal(), outcome=_FakeOutcome())


# ── routing per facet ─────────────────────────────────────────────────────

def test_observe_routes_to_dynamics(model):
    out = model.observe([0.1, 0.2])
    assert out["surprise"] == 0.5


def test_surprise_reads_dynamics(model):
    assert model.surprise() == pytest.approx(0.42)


def test_imagine_rolls_forward(model):
    traj = model.imagine([0.0], [[1.0], [1.0], [1.0]])
    assert len(traj) == 3


def test_predict_outcome_builds_schemas_and_routes(model):
    out = model.predict_outcome("filesystem", {"path": "/x"}, action_kind="delete")
    assert out["reward"] == 0.6


def test_causal_observe_and_effects(model):
    assert model.observe_causal("spark", "fire", 0.9) is True
    assert model.causal_effects("spark") == [("fire", 0.8)]


def test_counterfactual_and_prevent(model):
    assert model.counterfactual({"spark": 1.0})["smoke"] == 0.9
    assert model.preventative_actions("fire") == [("remove_fuel", 0.7)]


# ── the single dispatch surface ───────────────────────────────────────────

def test_query_dispatches_by_intent(model):
    r = model.query("causal_effects", source="spark")
    assert r["facet"] == "causal" and r["available"]
    assert r["result"] == [("fire", 0.8)]


def test_query_unknown_intent_is_graceful(model):
    r = model.query("teleport")
    assert r["available"] is False and r["error"] == "unknown_intent"


def test_query_bad_args_is_graceful(model):
    r = model.query("causal_effects")  # missing 'source'
    assert r["result"] is None and r["error"].startswith("bad_args")


# ── fault isolation: a missing facet degrades only its own paths ──────────

def test_missing_facet_returns_none_not_crash():
    m = UnifiedWorldModel(learned=None, causal=_FakeCausal(), outcome=None)
    m._failed.update({"learned": True, "outcome": True})  # force "unavailable"
    assert m.surprise() is None              # dynamics facet gone
    assert m.predict_outcome("d", {}, action_kind="x") is None  # outcome facet gone
    assert m.causal_effects("spark") == [("fire", 0.8)]         # causal still works


def test_status_reports_facet_availability(model):
    st = model.status()
    assert st["facets"]["learned"]["available"]
    assert st["facets"]["outcome"]["detail"] == {"episodes": 3}


# ── real integration: the facets actually instantiate ─────────────────────

def test_real_facets_instantiate(tmp_path):
    # No injection → lazy-loads the real specialist classes. Confirms the facade wires to the
    # genuine engines, not just to fakes. Redirect the causal graph's persistence at a temp
    # file so the test never touches the live causal_world.json.
    m = UnifiedWorldModel()
    assert m.causal is not None
    m.causal.data_path = tmp_path / "causal_world.json"
    assert m.observe_causal("unit_test_src", "unit_test_dst", 0.5) is True
    effects = m.causal_effects("unit_test_src")
    assert isinstance(effects, list)


# ── singleton ─────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_unified_world_model() is get_unified_world_model()
