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
