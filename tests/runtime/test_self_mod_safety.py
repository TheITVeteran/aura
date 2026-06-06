"""tests/runtime/test_self_mod_safety.py — Self-modification safety policy tests.

Asserts that self-modification proposals are rejected when stability risks or
welfare costs exceed safety thresholds, or when risks outweigh predicted capability gains.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.self_modification.growth_ladder import GrowthLadder, ModificationLevel


def test_self_mod_safety_veto_high_risk(tmp_path):
    """Verify that a self-modification proposal with high stability risk is rejected."""
    ladder = GrowthLadder(state_path=tmp_path / "growth.json")
    ladder._current_level = ModificationLevel.EXPRESSION  # Allow level 1 modifications
    
    # 1. Propose with high stability risk (0.6 > 0.5)
    allowed = asyncio.run(ladder.propose_modification(
        proposal_id="test_high_risk",
        modification_type="expression",
        level=ModificationLevel.EXPRESSION,
        description="Optimize expression formatting",
        predicted_capability_gain=0.8,
        predicted_stability_risk=0.6,
        predicted_welfare_cost=0.1,
    ))
    
    assert allowed is False
    assert ladder._proposals[-1].status == "rejected_safety"


def test_self_mod_safety_veto_high_welfare_cost(tmp_path):
    """Verify that a self-modification proposal with high welfare cost is rejected."""
    ladder = GrowthLadder(state_path=tmp_path / "growth.json")
    ladder._current_level = ModificationLevel.EXPRESSION
    
    # 2. Propose with high welfare cost (0.7 > 0.6)
    allowed = asyncio.run(ladder.propose_modification(
        proposal_id="test_high_welfare",
        modification_type="expression",
        level=ModificationLevel.EXPRESSION,
        description="Refactor console logs",
        predicted_capability_gain=0.9,
        predicted_stability_risk=0.1,
        predicted_welfare_cost=0.7,
    ))
    
    assert allowed is False
    assert ladder._proposals[-1].status == "rejected_safety"


def test_self_mod_safety_veto_risk_exceeds_gain(tmp_path):
    """Verify that a self-modification proposal is rejected if the stability risk

    exceeds or equals the capability gain.
    """
    ladder = GrowthLadder(state_path=tmp_path / "growth.json")
    ladder._current_level = ModificationLevel.EXPRESSION
    
    # 3. Propose with stability risk (0.3) > capability gain (0.2)
    allowed = asyncio.run(ladder.propose_modification(
        proposal_id="test_risk_exceeds_gain",
        modification_type="expression",
        level=ModificationLevel.EXPRESSION,
        description="Tweak expression parser",
        predicted_capability_gain=0.2,
        predicted_stability_risk=0.3,
        predicted_welfare_cost=0.1,
    ))
    
    assert allowed is False
    assert ladder._proposals[-1].status == "rejected_risk_vs_gain"


def test_self_mod_safety_accept_good_profile(tmp_path):
    """Verify that a self-modification proposal with high capability gain

    and low risk is accepted.
    """
    ladder = GrowthLadder(state_path=tmp_path / "growth.json")
    ladder._current_level = ModificationLevel.EXPRESSION
    # Scripted self consent returns True for this test
    async def scripted_consent(prop):
        prop.aura_consent = True
    ladder._request_self_consent = scripted_consent
    
    allowed = asyncio.run(ladder.propose_modification(
        proposal_id="test_safe_good",
        modification_type="expression",
        level=ModificationLevel.EXPRESSION,
        description="Clean expression formatting code",
        predicted_capability_gain=0.6,
        predicted_stability_risk=0.2,
        predicted_welfare_cost=0.1,
    ))
    
    assert allowed is True
    assert ladder._proposals[-1].status == "pending" or ladder._proposals[-1].status == "approved"
