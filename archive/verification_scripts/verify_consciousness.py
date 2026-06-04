
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from core.consciousness import ConsciousnessSystem

# Mock Orchestrator
class MockOrchestrator:
    def __init__(self):
        self.cognitive = "MockCognitiveEngine"
        self.affect_engine = MockAffect()
        self.embodiment = MockEmbodiment()
        self.drive_engine = MockDrive()

class MockAffect:
    async def get(self):
        class S:
            valence=0.5
            arousal=0.7
            engagement=0.6
            dominant_emotion="excitement"
        return S()

class MockEmbodiment:
    def update(self):
        return {"integrity": 95.0, "heat": 40.0}

class MockDrive:
    async def get_status(self):
        return {"curiosity": {"level": 20.0}, "energy": {"level": 80.0}}

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Verification")
    
    orch = MockOrchestrator()
    sys = ConsciousnessSystem(orch)
    
    logger.info("Starting Consciousness System...")
    await sys.start()
    
    logger.info("Running for 5 seconds...")
    await asyncio.sleep(5)
    
    state = sys.get_state()
    logger.info(f"Heartbeat Ticks: {state['heartbeat_tick']}")
    logger.info(f"Last GW Winner: {state['workspace']['last_winner']}")
    logger.info(f"Narrative: {await sys.temporal_binding.get_narrative()}")
    
    await sys.stop()
    
    if state['heartbeat_tick'] >= 4:
        print("VERIFICATION SUCCESS")
    else:
        print("VERIFICATION FAILED: Not enough ticks")

if __name__ == "__main__":
    asyncio.run(main())
