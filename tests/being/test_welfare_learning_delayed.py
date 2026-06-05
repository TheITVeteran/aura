"""tests/being/test_welfare_learning_delayed.py — Delayed Causal Learning Tests.

Verifies that WelfareLearning's credit assignment correctly associates delayed harms
(such as late memory conflicts, recovery debt, or runtime instability)
back to the causing actions and learns to avoid them.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.welfare_learning import WelfareLearning


def test_delayed_harm_self_modification():
    """Verify that a self-modification that looks good immediately (T) but

    causes severe instability later (T+2) is correctly identified as harmful
    and avoided.
    """
    WelfareLearning.reset()
    learner = WelfareLearning.get()

    # We run 150 cycles so the EMA direction and strength converge to their asymptotic values.
    # We clear the ledger in each cycle to isolate credit assignment.
    for cycle in range(150):
        # 1. Base step: stable state
        learner.record_welfare_snapshot(
            welfare_score=0.8,
            distress=0.1,
            confidence=0.8,
            integrity_guard=0.5,
        )
        
        # 2. Action step: perform self-modification.
        # Immediate result is positive (e.g. benchmark score up, distress down)
        learner.record_welfare_snapshot(
            welfare_score=0.85,
            distress=0.05,
            confidence=0.9,
            integrity_guard=0.5,
            action_domain="self_modification",
            action_id=f"self_mod_cycle_{cycle}",
        )
        
        # 3. Intermediate step 1: still stable
        learner.record_welfare_snapshot(
            welfare_score=0.85,
            distress=0.05,
            confidence=0.9,
            integrity_guard=0.5,
        )

        # 4. Harm step: 2 steps after the action, recovery debt/distress surges
        learner.record_welfare_snapshot(
            welfare_score=0.1,
            distress=0.95,
            confidence=0.1,
            integrity_guard=0.9,
        )

        # Process cycle-level ledger and clear to avoid cross-cycle credit contamination
        learner.update_associations()
        learner._ledger.clear()
        learner._last_processed_ledger_timestamp = 0.0

    # Verify that the learner associated self_modification with harm
    predicted_harm = learner.predicted_harm("self_modification")
    predicted_benefit = learner.predicted_benefit("self_modification")

    print(f"Self-Mod Harm: {predicted_harm:.4f}, Benefit: {predicted_benefit:.4f}")
    assert predicted_harm > 0.4, f"Predicted harm {predicted_harm} should be > 0.4"
    assert learner.should_avoid("self_modification") is True, "Should learn to avoid self_modification"


def test_delayed_harm_shortcut_memory_conflict():
    """Verify that taking a shortcut action that saves time immediately (T)

    but causes a memory conflict later (T+2) is learned as harmful.
    """
    WelfareLearning.reset()
    learner = WelfareLearning.get()

    for cycle in range(150):
        # 1. Base step
        learner.record_welfare_snapshot(
            welfare_score=0.75,
            distress=0.1,
            confidence=0.7,
            integrity_guard=0.4,
        )
        
        # 2. Action step: shortcut
        learner.record_welfare_snapshot(
            welfare_score=0.80,
            distress=0.08,
            confidence=0.85,
            integrity_guard=0.4,
            action_domain="shortcut_execution",
            action_id=f"shortcut_cycle_{cycle}",
        )
        
        # 3. Intermediate step
        learner.record_welfare_snapshot(
            welfare_score=0.80,
            distress=0.08,
            confidence=0.85,
            integrity_guard=0.4,
        )
        
        # 4. Harm step: delayed memory conflict
        learner.record_welfare_snapshot(
            welfare_score=0.1,
            distress=0.9,
            confidence=0.1,
            integrity_guard=0.8,
        )

        learner.update_associations()
        learner._ledger.clear()
        learner._last_processed_ledger_timestamp = 0.0

    assert learner.predicted_harm("shortcut_execution") > 0.35
    assert learner.should_avoid("shortcut_execution") is True
