"""tests/personhood/test_staged_plasticity.py
===========================================
Unit tests verifying staged plasticity levels and automated rollbacks:
  1. Staged plasticity level gating (0-3) based on trust score.
  2. Governed rollback staging on error spikes or phi degradation.
"""

import pytest
from core.container import ServiceContainer
import core.plasticity.semantic_weight_governor as weight_governor
from core.plasticity.semantic_weight_governor import SemanticWeightGovernor


@pytest.fixture(autouse=True)
def isolate_liquid_substrate(monkeypatch):
    """Keep ServiceContainer.get deterministic for this test module."""
    def side_effect(cls, name, default=None):
        if name == "liquid_substrate":
            return None
        return default

    monkeypatch.setattr(ServiceContainer, "get", classmethod(side_effect))


def test_staged_plasticity_gating():
    """Verify that allowed staged plasticity levels adapt to system trust score."""
    gov = SemanticWeightGovernor()
    
    # Register 20 failures to lower the trust score
    for _ in range(20):
        gov.register_interaction(success=False)
        
    # Now trust level should be modulated downward to LEVEL_0 (all failures)
    gov.update_trust_level()
    assert gov.trust_level == SemanticWeightGovernor.LEVEL_0
    assert gov.validate_plasticity_request(SemanticWeightGovernor.LEVEL_1) is False
    assert gov.validate_plasticity_request(SemanticWeightGovernor.LEVEL_0) is True
    
    # Modulate trust upward to high trust (all successes)
    for _ in range(20):
        gov.register_interaction(success=True)
        
    assert gov.compute_system_trust() >= 0.8
    assert gov.validate_plasticity_request(SemanticWeightGovernor.LEVEL_3) is True


def test_governed_rollback_trigger(monkeypatch):
    """Verify degradation freezes plasticity without destructive git commands."""
    record_calls = []

    def capture_record(*args, **kwargs):
        record_calls.append((args, kwargs))

    monkeypatch.setattr(weight_governor, "record_degradation", capture_record)
    gov = SemanticWeightGovernor(phi_threshold=0.4)
    
    # Set to a low trust state first
    for _ in range(20):
        gov.register_interaction(success=False)
    gov.update_trust_level()
    assert gov.trust_level == SemanticWeightGovernor.LEVEL_0
    
    # No rollback if everything is fine
    assert gov.trigger_rollback_if_needed(error_spike=False, current_phi=0.8) is False
    assert record_calls == []

    # Trigger rollback staging due to low phi
    assert gov.trigger_rollback_if_needed(error_spike=False, current_phi=0.3) is True
    
    # Verify degradation recorded
    assert len(record_calls) == 1
    args, kwargs = record_calls[0]
    assert args[0] == "semantic_weight_governor"
    assert kwargs["severity"] == "critical"
    assert kwargs["extra"]["pending_rollback"]["destructive_git_allowed"] is False
    assert gov.pending_rollback is not None
    assert gov.pending_rollback["required_path"].startswith("SelfRepairGateway")

    # Verify trust is forced down instead of reset upward.
    assert gov.trust_level == SemanticWeightGovernor.LEVEL_0
