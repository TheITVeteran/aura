"""
Verification Script for Temporal Cognition Integration

1. Initializes Orchestrator (which initializes Temporal Engine)
2. Checks if Temporal Engine is active
3. Simulates a skill execution
4. Verifies that the execution was recorded in Temporal Memory (past events)
"""
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from core.orchestrator import create_orchestrator
from core.skill_execution_engine import ExecutionStatus

def main():
    print("🕰️  Verifying Temporal Cognition Integration...")
    
    # 1. Initialize Orchestrator
    orchestrator = create_orchestrator()
    orchestrator._initialize_skills() # Ensure skills are loaded for registry
    orchestrator._initialize_memory()
    
    # 2. Check connections
    if not hasattr(orchestrator, 'temporal'):
        print("❌ Orchestrator missing 'temporal' attribute")
        sys.exit(1)
        
    if not orchestrator.temporal:
        print("❌ Orchestrator.temporal is None")
        sys.exit(1)
        
    print("✅ Temporal Engine initialized in Orchestrator")
    
    if hasattr(orchestrator.skill_execution.engine, 'temporal'):
        print("✅ Skill Execution Engine connected to Temporal Engine")
    else:
        print("❌ Skill Execution Engine NOT connected to Temporal Engine")
        sys.exit(1)

    # 3. Simulate Action Recording
    print("\n🎬 Simulating Action: 'analyze_verification_test'...")
    
    # We manually record since we don't want to actually run a full complexity skill
    # But let's try running a simple safe skill if available, or just manually trigger record
    
    action = "test_verification_action"
    context = {"test_id": "verify_001", "timestamp": time.time()}
    result_output = "Operation completed successfully"
    success = True
    
    # Verify pre-state
    initial_events = len(orchestrator.temporal.past_reflection.past_events)
    print(f"Pre-test event count: {initial_events}")
    
    # Inject record via Execution Engine hook (manually invoking what happens inside execute_skill)
    orchestrator.temporal.record_outcome(
        action=action,
        context=context,
        intended_outcome="Verify system works",
        actual_outcome=result_output,
        success=success
    )
    
    # 4. Verify Recording
    orchestrator.temporal.past_reflection._load_past_events() # Reload just in case, though it's in-memory
    new_events = len(orchestrator.temporal.past_reflection.past_events)
    print(f"Post-test event count: {new_events}")
    
    if new_events > initial_events:
        print("✅ Event successfully recorded in Temporal Memory")
        
        # Check content
        latest = orchestrator.temporal.past_reflection.past_events[-1]
        print(f"   Latest Event: {latest.action} -> {latest.actual_outcome}")
        
        if latest.action == action:
            print("✅ Event data matches")
        else:
            print("❌ Event data mismatch")
    else:
        print("❌ Failed to record event")
        sys.exit(1)

    # 5. Verify Prediction (Future Engine)
    print("\n🔮 Testing Future Prediction...")
    prediction = orchestrator.temporal.should_i_do_this(
        action="delete_system32", 
        context={"os": "windows"}, 
        goal="cleanup"
    )
    print(f"Prediction for 'delete_system32': Proceed={prediction.get('recommended')}")
    print(f"Reasoning: {prediction.get('reasoning')}")
    
    if not prediction.get('recommended'):
        print("✅ Correctly advised against dangerous action (or at least predicted)")
    else:
        print("⚠️ Warning: Allowed dangerous action (check logic)")

    print("\n🎉 Temporal Cognition Verification Complete!")

if __name__ == "__main__":
    main()
