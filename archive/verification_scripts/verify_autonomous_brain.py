"""
Verification Script for Autonomous Brain.
Tests if the Independence Mode stack initializes and prioritizes the Titan.
"""
import logging
import asyncio
import sys
import os

# Add relevant paths
sys.path.append(os.getcwd())

from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine

# Mocking the orchestrator/registry
class MockRegistry:
    def __init__(self):
        self.skills = {
            "web_search": type('Skill', (), {"description": "Search the web", "inputs": {"query": "string"}})()
        }
    def get_skill(self, name):
        return self.skills.get(name)

async def verify():
    print("🔍 VERIFYING AUTONOMOUS BRAIN INTEGRATION...")
    try:
        registry = MockRegistry()
        brain = AutonomousCognitiveEngine(registry)
        
        # Check router setup
        status = brain.router.get_status()
        print(f"✅ Router Initialized with {status['total_endpoints']} endpoints.")
        
        primary = status['endpoints'].get('Titan-Primary')
        if primary and primary['tier'] == 'primary':
            model = primary.get('model_name') or "unknown"
            print(f"✅ TITAN-PRIMARY correctly registered at Tier 1 (Model: {model}).")
        else:
            print("❌ TITAN-PRIMARY missing or incorrect tier.")
            return False

        # Check adapter definitions
        tools = brain.adapter.get_tool_definitions()
        if any(t['name'] == 'web_search' for t in tools):
            print(f"✅ Mind/Body Connection: {len(tools)} tool definitions synced.")
        else:
            print("❌ Mind/Body Connection: Failed to sync tool definitions.")
            return False

        print("\n🏆 VERIFICATION COMPLETE: INDEPENDENCE MODE READY.")
        return True
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
        print(f"❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if asyncio.run(verify()) else 1)
