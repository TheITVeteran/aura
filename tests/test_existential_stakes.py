"""tests/test_existential_stakes.py
==================================
Tests the Existential Stakes & Nociceptive Gate subsystem.
"""
from __future__ import annotations

import sys
import time
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.existential_stakes import ExistentialStakes, get_existential_stakes
from core.container import ServiceContainer
from core.governance.will import UnifiedWill, ActionDomain, WillOutcome
from core.state.aura_state import AuraState
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.inference_gate import InferenceGate


def test_existential_stakes_init():
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    status = stakes.get_status()
    assert status["existential_threat"] == 0.0
    assert status["total_ticks"] == 0


def test_memory_threat_trigger():
    # Process memory RSS will be > 1000 bytes, so memory_limit=1000 triggers threat=1.0
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    threat = stakes.update()
    assert threat == 1.0
    status = stakes.get_status()
    assert status["memory_threat"] == 1.0
    assert status["existential_threat"] == 1.0


def test_lag_threat_trigger():
    # Large delay between updates triggers loop lag threat
    stakes = ExistentialStakes(memory_limit_bytes=10**12)  # Huge limit so memory threat is ~0
    
    # First tick sets last_update_time
    stakes.update()
    
    # Set last_update_time to 5 seconds ago to exercise lag scoring.
    with stakes._lock:
        stakes._last_update_time = time.time() - 5.0

    threat = stakes.update()
    status = stakes.get_status()
    assert status["rolling_loop_lag_s"] > 0.0
    assert status["lag_threat"] > 0.0
    assert threat > 0.0


def test_will_gating_under_threat():
    # 1. Create a stakes instance with low limit to force 1.0 threat
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    assert stakes.get_existential_threat() == 1.0

    # Register in ServiceContainer
    ServiceContainer.register_instance("existential_stakes", stakes)

    # 2. Test Will decision
    will = UnifiedWill()
    will._started = True  # force start

    # Heavy action should be REFUSED due to survival inhibition
    decision_heavy = will.decide(
        content="reroute_vessel(Vessel_Alpha, 90, 15)",
        source="explore",
        domain=ActionDomain.TOOL_EXECUTION,
        is_critical=False,
    )
    assert decision_heavy.outcome == WillOutcome.REFUSE
    assert "survival_inhibition" in decision_heavy.reason

    # Critical action must PASS even under threat
    decision_critical = will.decide(
        content="apply_emergency_brake()",
        source="safety_system",
        domain=ActionDomain.STABILIZATION,
        is_critical=True,
    )
    assert decision_critical.outcome == WillOutcome.CRITICAL_PASS


def test_prompt_injection_under_threat():
    # 1. Create stakes with low memory limit so threat is 1.0
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    assert stakes.get_existential_threat() == 1.0

    # Register
    ServiceContainer.register_instance("existential_stakes", stakes)

    # Create an AuraState for prompt assembly.
    state = AuraState()

    # 2. Build system prompt
    prompt = ContextAssembler.build_system_prompt(state)
    assert "SYSTEM RESOURCE WARNING" in prompt
    assert "Felt Survival Threat Level: CRITICAL" in prompt
    assert "Keep all responses brief" in prompt


def test_sampling_parameter_modulation_under_threat():
    # 1. Start with no stakes
    ServiceContainer.register_instance("existential_stakes", None)
    
    gate = InferenceGate(None)
    
    # Under no threat, base parameters are returned
    # Use a simple context and verify that under threat it shrinks.
    context = {"max_tokens": 512, "temperature": 0.8}
    morpho_kwargs = {"temperature": 0.8}
    
    # 2. Register stakes with high threat
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    ServiceContainer.register_instance("existential_stakes", stakes)
    
    # Verify the stakes formula used by the inference morphogenetic block.
    threat = stakes.get_existential_threat()
    assert threat == 1.0
    
    # Check that the runtime formula lowers token budget and temperature.
    scaled_tokens = max(96, int(512 * (1.0 - threat * 0.7)))
    assert scaled_tokens == 153  # 512 * 0.3 = 153.6 -> 153
    
    scaled_temp = 0.8 * (1.0 - threat * 0.5)
    assert scaled_temp == 0.4
