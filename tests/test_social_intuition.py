"""Social intuition: forward-simulate how a candidate action lands socially."""
from __future__ import annotations

import pytest

from core.social.other_agent_model import OtherAgentStateEstimator


@pytest.fixture
def model(tmp_path):
    return OtherAgentStateEstimator(storage_path=tmp_path / "oam.json", autosave=False)


def test_warm_reliable_action_builds_trust(model):
    f = model.forecast_social_consequence("bryan", warmth=0.9, directness=0.3,
                                          reliability=0.9, fulfills_expectation=0.6)
    assert f["trust_delta"] > 0
    assert f["recommendation"] in {"proceed", "proceed_with_care"}


def test_blunt_action_on_frustrated_agent_predicts_rupture(model):
    for _ in range(6):
        model.observe_signal("bryan", threat=0.8)  # builds frustration
    blunt = model.forecast_social_consequence("bryan", warmth=0.1, directness=0.9,
                                              reliability=0.4, fulfills_expectation=-0.3)
    calm = model.forecast_social_consequence("bryan", warmth=0.9, directness=0.2,
                                             reliability=0.9, fulfills_expectation=0.3)
    assert blunt["rupture_delta"] > calm["rupture_delta"]
    assert blunt["projected_rupture_risk"] > calm["projected_rupture_risk"]


def test_breaking_expectation_costs_trust(model):
    keep = model.forecast_social_consequence("bryan", reliability=0.8, fulfills_expectation=0.5)
    break_ = model.forecast_social_consequence("bryan", reliability=0.8, fulfills_expectation=-0.8)
    assert break_["trust_delta"] < keep["trust_delta"]


def test_high_projected_rupture_recommends_repair(model):
    for _ in range(8):
        model.observe_signal("bryan", threat=0.9)
    f = model.forecast_social_consequence("bryan", warmth=0.0, directness=1.0,
                                          reliability=0.1, fulfills_expectation=-1.0)
    assert f["recommendation"] in {"repair_first", "soften_before_acting"}
