"""
verify_syntax_gating.py
──────────────────────
Verifies that SafeSelfModification blocks syntactically invalid code.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modification.safe_modification import SafeSelfModification

@dataclass
class MockFix:
    target_file: str
    replacement_content: str
    risk_level: int = 1
    lines_changed: int = 1

def test_syntax_gating():
    # We don't need a real event bus for this test
    engine = SafeSelfModification(str(PROJECT_ROOT))
    
    # 1. Valid Code Fix
    valid_fix = MockFix(
        target_file="core/test.py",
        replacement_content="def hello():\n    print('world')\n"
    )
    allowed, reason = engine.validate_proposal(valid_fix)
    print(f"🔍 Testing Valid Fix: Allowed={allowed}, Reason='{reason}'")
    if not allowed:
        print("❌ FAILURE: Valid code was blocked.")
    else:
        print("✅ SUCCESS: Valid code allowed.")

    # 2. Invalid Syntax Fix
    invalid_fix = MockFix(
        target_file="core/test.py",
        replacement_content="def broken():\n    print('world')\n    if: # SYNTAX ERROR\n"
    )
    allowed, reason = engine.validate_proposal(invalid_fix)
    print(f"\n🔍 Testing Invalid Syntax Fix: Allowed={allowed}, Reason='{reason}'")
    if not allowed and "syntax error" in reason.lower():
        print("✅ SUCCESS: Invalid syntax was correctly blocked.")
    else:
        print("❌ FAILURE: Invalid syntax was NOT blocked or reason was wrong.")

if __name__ == "__main__":
    test_syntax_gating()
