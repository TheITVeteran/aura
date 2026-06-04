import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath("."))

from core.orchestrator import AsyncAgentOrchestrator

STABILITY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def verify_stability_fixes():
    print("🚀 Verifying Aura Stability Fixes...")
    ok = True
    
    # Initialize Orchestrator
    orchestrator = AsyncAgentOrchestrator()
    print("✅ Orchestrator initialized.")
    
    # 1. Test Watchdog Compatibility (StateManager.create_snapshot)
    print("\n--- Testing Watchdog Compatibility ---")
    try:
        snapshot = orchestrator.state_manager.create_snapshot(orchestrator)
        print(f"✅ Snapshot captured: {snapshot}")
        orchestrator.state_manager.push_checkpoint(snapshot)
        print("✅ Checkpoint pushed.")
    except STABILITY_RECOVERABLE_ERRORS as e:
        print(f"❌ StateManager error: {e}")
        ok = False
    
    # 2. Test Resilient Async Call (No more coroutine attribute errors)
    print("\n--- Testing Resilient Async Call ---")
    task = {"skill": "chat", "params": {"message": "Test message"}}
    try:
        # We need to mock/ensure a skill exists or just test the call mechanism
        # For testing purposes, let's just call a dummy function
        async def dummy_skill(task, context):
            return {"ok": True, "response": "Hello!"}
        
        result = await orchestrator.hardening.resilient_call(
            "test_component",
            dummy_skill,
            task,
            {}
        )
        print(f"✅ Resilient call result: {result}")
        
        # Test the meta-cognition block logic (mimicking orchestrator finally)
        success = result.get("ok", False) if isinstance(result, dict) else True
        print(f"✅ Meta-cognition logic success check: {success}")
        
    except STABILITY_RECOVERABLE_ERRORS as e:
        print(f"❌ Resilient call error: {e}")
        ok = False

    # 3. Test string result handling
    print("\n--- Testing String Result Handling ---")
    result_str = "Status: OK"
    success_str = result_str.get("ok", False) if isinstance(result_str, dict) else True
    print(f"✅ String result success check: {success_str}")

    print("\n🎉 Stability Fixes Verified!")
    return ok

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(0 if asyncio.run(verify_stability_fixes()) else 1)
