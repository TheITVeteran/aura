"""tests/test_autobiographical_memory.py
Unit tests for Aura's autobiographical memory, episode stores, and truth court.
"""
import pytest
import os
import time
from core.organism.life_state import LifeState
from core.memory.autobiography import AutobiographyEngine
from core.memory.memory_court import MemoryCourt


@pytest.mark.anyio
async def test_autobiography_recording():
    state = LifeState()
    engine = AutobiographyEngine()
    
    receipt = {"status": "success", "channel": "desktop", "intent": {"test": "data"}}
    
    # Record event trace
    await engine.record_tick_event(state, receipt)
    
    # Assert state memory updated
    assert len(state.autobiographical_memory) == 1
    assert state.autobiographical_memory[0]["did"] == {"action": "desktop"}
    
    # Assert log file generated
    assert os.path.exists(engine.store.db_path)


@pytest.mark.anyio
async def test_memory_court_vetting():
    court = MemoryCourt()
    
    existing = {
        "user_profile": {"key": "user_profile", "value": "Bryan", "confidence": 0.80}
    }
    
    # Vet fact from high trust source (confidence 0.99 > 0.80)
    resolved = await court.vet_fact("user_profile", "New Bryan", "user_direct", existing)
    assert resolved["value"] == "New Bryan"
    assert resolved["confidence"] == 0.99

    # Vet fact from low trust source (confidence 0.20 < 0.80)
    rejected = await court.vet_fact("user_profile", "False Info", "hallucinated_inference", existing)
    assert rejected is None
