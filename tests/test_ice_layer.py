import logging
from types import SimpleNamespace

import pytest

from core.cybernetics.ice_layer import ICELayer


def test_classify_anomaly_exposes_legacy_and_canonical_description_keys():
    ice = ICELayer()

    anomaly = ice.classify_anomaly("identity drift detected")

    assert anomaly["type"] == "SEMANTIC_DRIFT"
    assert anomaly["desc"] == "Loss of identity coherence."
    assert anomaly["description"] == anomaly["desc"]
    assert anomaly["containment"] == "RELOAD_CORE_NARRATIVE"


@pytest.mark.asyncio
async def test_executive_violation_uses_compatible_description_schema(caplog):
    ice = ICELayer()

    with caplog.at_level(logging.WARNING, logger="Aura.Cybernetics.ICE"):
        await ice._on_executive_violation({"label": "identity drift"})

    assert ice.get_status()["threat_level"] == pytest.approx(0.25)
    assert any(
        "Loss of identity coherence." in record.message
        for record in caplog.records
    )


class _NoveltyDetector:
    async def observe(self, _event):
        return SimpleNamespace(threat_probability=0.99)

    @staticmethod
    def get_threat_level():
        return 0.99


@pytest.mark.asyncio
async def test_statistical_novelty_alone_cannot_create_breach(caplog):
    ice = ICELayer()
    ice._anomaly_detector = _NoveltyDetector()

    with caplog.at_level(logging.WARNING, logger="Aura.Cybernetics.ICE"):
        for _ in range(5):
            await ice._on_audit({"drift": 0.05, "status": "STABLE"})

    status = ice.get_status()
    assert status["statistical_novelty"] == pytest.approx(0.99)
    assert status["is_breached"] is False
    assert status["threat_level"] == pytest.approx(0.0)
    assert not [record for record in caplog.records if record.levelno >= logging.CRITICAL]


@pytest.mark.asyncio
async def test_corroborated_drift_opens_one_incident_and_stable_audits_clear_it(caplog):
    ice = ICELayer()
    ice._anomaly_detector = _NoveltyDetector()

    with caplog.at_level(logging.INFO, logger="Aura.Cybernetics.ICE"):
        await ice._on_audit({"drift": 0.9, "status": "UNCANNY_VALLEY_DETECTED"})
        await ice._on_audit({"drift": 0.9, "status": "UNCANNY_VALLEY_DETECTED"})
        await ice._on_audit({"drift": 0.9, "status": "UNCANNY_VALLEY_DETECTED"})

    opened = ice.get_status()
    assert opened["is_breached"] is True
    assert opened["incident"]["state"] == "contained"
    critical = [record for record in caplog.records if record.levelno >= logging.CRITICAL]
    assert len(critical) == 1

    for _ in range(3):
        await ice._on_audit({"drift": 0.05, "status": "STABLE"})

    recovered = ice.get_status()
    assert recovered["is_breached"] is False
    assert recovered["incident"]["state"] == "recovered"


def test_authority_gateway_enforces_ice_containment(monkeypatch):
    from core.executive.authority_gateway import AuthorityGateway

    ice = SimpleNamespace(
        get_status=lambda: {
            "is_breached": True,
            "incident": {"incident_id": "ice-1", "reason": "corroborated"},
        }
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.ServiceContainer.get",
        lambda name, default=None: ice if name == "ice_layer" else default,
    )

    blocked = AuthorityGateway._security_containment_gate(
        source="desktop_ui",
        effect_scope="external_io",
        domain="tool_execution",
    )
    observed = AuthorityGateway._security_containment_gate(
        source="desktop_ui",
        effect_scope="read_only",
        domain="tool_execution",
    )

    assert blocked is not None
    assert blocked.reason == "security_containment_active"
    assert blocked.constraints["incident_id"] == "ice-1"
    assert observed is None
