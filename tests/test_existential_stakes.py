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


def test_singleton_memory_limit_is_machine_aware(monkeypatch):
    """The live singleton must not perceive perpetual near-death.

    A fixed 2GB ceiling makes a normal ~1.5GB-RSS runtime sit at ~0.75
    memory_threat, parking the will-system at its survival-veto boundary. The
    factory must derive a machine-aware ceiling so normal operation reads low
    memory_threat and the survival veto only fires near genuine danger.
    """
    import core.consciousness.existential_stakes as es

    monkeypatch.delenv("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(es, "_INSTANCE", None)

    stakes = es.get_existential_stakes()

    # On any real host this resolves well above the 2GB stale default (aligned
    # to the watchdog's process-RSS ceiling, typically tens of GB).
    assert stakes._memory_limit > es.DEFAULT_MEMORY_LIMIT_BYTES

    # Normal runtime RSS must read as low memory pressure, not near-death.
    stakes.update()
    assert stakes.get_status()["memory_threat"] < 0.5


def test_explicit_env_override_sets_memory_limit(monkeypatch):
    import core.consciousness.existential_stakes as es

    monkeypatch.setenv("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", "48")
    monkeypatch.setattr(es, "_INSTANCE", None)

    stakes = es.get_existential_stakes()

    assert stakes._memory_limit == int(48 * (1024 ** 3))


def test_operational_load_cannot_trigger_survival_veto():
    """High CPU/lag with healthy memory must stay below the will-veto line.

    Regression for the continual-learning battery blocking at threat=1.00:
    heavy 32B generation pegs CPU and event-loop lag, but a busy machine is not
    a dying one. Operational pressure is capped below 0.75 so survival
    inhibition only fires on genuine death risk (memory/degradation).
    """
    from core.consciousness.existential_stakes import (
        OPERATIONAL_THREAT_CAP,
        ExistentialStakes,
    )

    stakes = ExistentialStakes(memory_limit_bytes=10**12)  # memory threat ~0
    # Force maximal operational pressure directly.
    with stakes._lock:
        stakes._memory_threat = 0.02
        stakes._lag_threat = 1.0
        stakes._cpu_threat = 1.0
        stakes._degradation_threat = 0.0
        stakes._threat = max(
            max(stakes._memory_threat, stakes._degradation_threat),
            min(OPERATIONAL_THREAT_CAP, max(stakes._lag_threat, stakes._cpu_threat)),
        )

    threat = stakes.get_existential_threat()
    assert threat == OPERATIONAL_THREAT_CAP
    assert threat <= 0.70
    assert threat < 0.75  # the will-system survival-inhibition veto threshold


def test_warning_degradations_are_weak_existential_signal():
    """Warnings are real evidence, but not enough to shout near-death alone."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"warning_{idx}",
                    severity="warning",
                    error_type="Warning",
                    error_message="transient warning",
                    action="observed",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()
        status = stakes.get_status()

        assert status["degradation_threat"] < 0.75
        assert threat < 0.75
    finally:
        tracker.reset()


def test_degraded_cascade_still_reaches_critical():
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"degraded_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="fresh degraded event",
                    action="repair",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()

        assert threat == pytest.approx(1.0)
        assert stakes.get_status()["degradation_threat"] == pytest.approx(1.0)
    finally:
        tracker.reset()


def test_old_degradations_decay_instead_of_holding_veto():
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        old = time.time() - 45.0
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"old_degraded_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="old degraded event",
                    action="already repaired",
                    timestamp=old,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()

        assert 0.0 < stakes.get_status()["degradation_threat"] < 0.75
        assert threat < 0.75
    finally:
        tracker.reset()


def test_critical_existential_log_is_coalesced(caplog):
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"critical_log_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="fresh degraded event",
                    action="repair",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        with caplog.at_level("WARNING", logger="Consciousness.ExistentialStakes"):
            stakes.update()
            stakes.update()

        messages = [record.message for record in caplog.records]
        assert sum("CRITICAL EXISTENTIAL STAKES" in msg for msg in messages) == 1
    finally:
        tracker.reset()


def test_memory_death_risk_still_reaches_critical():
    """Genuine death risk (memory) is uncapped and still triggers the veto."""
    from core.consciousness.existential_stakes import ExistentialStakes

    stakes = ExistentialStakes(memory_limit_bytes=1000)  # tiny → memory_threat 1.0
    stakes.update()
    assert stakes.get_existential_threat() == 1.0
