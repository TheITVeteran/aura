"""tests/test_organism_life_loop.py
Unit tests for Aura's Canonical Organism loop, state ticking, and cycle clocks.
"""
import pytest
import asyncio
from core.organism.life_state import LifeState
from core.organism.life_tick import LifeTickProcessor
from core.organism.cycle_clock import CycleClock
from core.organism.life_loop import LifeLoop


def test_life_state_initialization():
    state = LifeState()
    assert state.tick_count == 0
    assert state.welfare.energy == 100.0
    assert state.body.is_sleeping is False
    assert state.cognition.active_attention == "idle"


def test_cycle_clock_diurnal():
    clock = CycleClock(tick_rate_hz=1.0)
    assert clock.tick_rate_hz == 1.0
    
    # Calculate sleep pressure
    pressure = clock.calculate_sleep_pressure(sleep_debt=4.0, hours_awake=16.0)
    assert 0.0 <= pressure <= 1.0


@pytest.mark.anyio
async def test_life_tick_execution():
    state = LifeState()
    processor = LifeTickProcessor()
    
    # Execute a single tick cycle
    await processor.execute_tick(state)
    assert state.tick_count == 1
    # Confirm that welfare bus evaluated and updated sleep debt/energy
    assert state.welfare.sleep_debt > 0.0


@pytest.mark.asyncio
async def test_life_loop_records_failure_and_exposes_unhealthy_status(monkeypatch):
    loop = LifeLoop(tick_rate_hz=100.0)
    tick_attempts = []

    class FailingProcessor:
        async def execute_tick(self, _state):
            tick_attempts.append("attempted")
            raise RuntimeError("tick failed")

    async def stop_after_backoff(_seconds):
        loop._running = False

    loop.processor = FailingProcessor()
    loop._running = True
    monkeypatch.setattr("core.organism.life_loop.asyncio.sleep", stop_after_backoff)

    await loop._loop_run()

    status = loop.get_health_status()
    assert status["healthy"] is False
    assert status["consecutive_failures"] == 1
    assert "tick failed" in status["last_error"]
    assert tick_attempts == ["attempted"]


@pytest.mark.asyncio
async def test_sleep_cycle_failure_is_observable_and_does_not_restore_energy():
    from core.sleep.sleep_cycle import SleepManager

    state = LifeState()
    state.welfare.energy = 7.0
    state.welfare.sleep_debt = 20.0
    manager = SleepManager()
    consolidation_attempts = []

    async def fail_consolidation(_state):
        consolidation_attempts.append("attempted")
        raise RuntimeError("memory consolidation failed")

    manager.memory_consolidator.consolidate_logs = fail_consolidation

    result = await manager.execute_sleep_cycle(state)

    assert result is False
    assert state.body.is_sleeping is False
    assert state.welfare.energy == 7.0
    assert state.welfare.sleep_debt == 20.0
    assert state.world_model["last_sleep_cycle_error"]["error_type"] == "RuntimeError"
    assert consolidation_attempts == ["attempted"]
