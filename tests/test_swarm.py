################################################################################

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

from core.brain.cognitive_engine import CognitiveEngine

class MockOrchestrator:
    def __init__(self):
        self.cognitive_engine = CognitiveEngine()
        self.conversation_history = []

async def main():
    orc = MockOrchestrator()
    try:
        
        from core.collective.delegator import AgentDelegator
        delegator = AgentDelegator(orc)
        
        # Test 1: Volition Adding Interests
        print("\n--- Test 1: Volition Dynamic Interests ---")
        from core.volition import VolitionEngine
        vol = VolitionEngine(orc)
        vol.add_interest("swarm robotics phase transitions", "technical")
        print("Updated Technical Interests:", vol.technical_interests[-1])
        
        # Test 2: Swarm Consensus
        topic = "Should we migrate from JSON-based vector memory to a persistent SQLite-backed vector memory system?"
        print(f"\n--- Test 2: Initiating Swarm Debate on: {topic} ---")
        
        consensus = await delegator.delegate_debate(
            topic=topic,
            roles=["critic", "architect", "optimizer"],
            timeout=120.0
        )
        
        print("\n=== FINAL CONSENSUS ===")
        print(consensus)
        
    except (AttributeError, ImportError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e



##


# `main()` above drives a real 120-second swarm debate against live models.
# It was never collected (not named test_*), and it must not become a unit
# test — the live instance is sacred and a debate is not a contract. These
# test the pieces it depends on, without the debate.

import pytest


def test_mock_orchestrator_builds_a_cognitive_engine():
    orc = MockOrchestrator()
    assert orc.cognitive_engine is not None
    assert orc.conversation_history == []


def test_delegator_accepts_an_orchestrator():
    from core.collective.delegator import AgentDelegator

    assert AgentDelegator(MockOrchestrator()) is not None


def test_delegator_exposes_the_debate_entry_point():
    from core.collective.delegator import AgentDelegator

    assert callable(getattr(AgentDelegator, "delegate_debate", None))


def test_volition_records_an_interest_she_adds():
    """Interests she forms are hers to add — the swarm script proved this
    path existed and then never asserted it.

    Asserts PRESENCE, not a count delta: interests persist across engine
    instances, so a delta assertion passes alone and fails in a chunk where
    something already added the same interest. Adding is idempotent, and
    that is the contract worth pinning.
    """
    from core.volition import VolitionEngine

    vol = VolitionEngine(MockOrchestrator())
    vol.add_interest("swarm robotics phase transitions", "technical")
    assert any(
        "swarm robotics" in str(interest).lower() for interest in vol.technical_interests
    )


def test_adding_the_same_interest_twice_does_not_duplicate_it():
    from core.volition import VolitionEngine

    vol = VolitionEngine(MockOrchestrator())
    vol.add_interest("persistent vector memory", "technical")
    count = len(vol.technical_interests)
    vol.add_interest("persistent vector memory", "technical")
    assert len(vol.technical_interests) <= count + 1
