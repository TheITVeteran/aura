from __future__ import annotations

import hashlib

import pytest

from core.social.other_agent_model import OtherAgentStateEstimator
from core.social.relational_memory import RelationalMemoryAuthority


@pytest.fixture()
def model(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"i" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="social-intuition-consent",
    )
    return OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_person_specific_forecast_abstains_without_calibrated_outcome_model(model):
    forecast = model.forecast_social_consequence(
        "bryan",
        warmth=0.9,
        directness=0.3,
        reliability=0.9,
        fulfills_expectation=0.6,
    )

    assert forecast == {
        "prediction": "unknown",
        "confidence": 0.0,
        "abstained": True,
        "reason": "no calibrated person-specific outcome model",
    }


def test_action_parameter_changes_cannot_fabricate_trust_or_rupture_delta(model):
    warm = model.forecast_social_consequence("bryan", warmth=1.0, reliability=1.0)
    blunt = model.forecast_social_consequence("bryan", warmth=0.0, reliability=0.0)

    assert warm == blunt
    assert "trust_delta" not in warm
    assert "rupture_delta" not in warm


def test_unverified_sensor_threat_cannot_create_person_state(model):
    assert model.observe_signal(
        "bryan",
        evidence_digest=_digest("sensor"),
        source="unverified_sensor",
        threat=1.0,
    ) is False

    estimate = model.estimate("bryan")
    assert estimate.overall_confidence == 0.0
    assert estimate.social_rupture_risk == 0.0


def test_cognitive_snapshot_exposes_forecast_abstention(model):
    model.observe_message(
        "bryan",
        "I am frustrated",
        evidence_digest=_digest("turn"),
    )

    snapshot = model.cognitive_snapshot("bryan")

    assert snapshot["predicted_impacts"]["abstained"] is True
    assert snapshot["likely_goals"] == []
