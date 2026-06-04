import asyncio
import sys
import tempfile
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text

# Fix paths
sys.path.insert(0, str(Path.cwd()))

ZENITH_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def test_zenith_fixes():
    print("🧪 Testing Zenith Audit Fixes...")
    ok = True

    # 1. Test Mycelial Network (Cycle Detection)
    print("\n1. Testing Mycelial Network...")
    try:
        from core.mycelial_graph import get_mycelial
        mycelium = get_mycelial()
        # Add a safe edge
        await mycelium.add_edge("memory_A", "skill_B")
        # Try to create a cycle
        success = await mycelium.add_edge("skill_B", "memory_A")
        if not success:
            print("✅ Cycle detection blocked correctly.")
        else:
            print("❌ Cycle detection failed!")
            ok = False
    except ZENITH_RECOVERABLE_ERRORS as e:
        print(f"❌ Mycelial test error: {e}")
        ok = False

    # 2. Test Safety Registry
    print("\n2. Testing Safety Registry...")
    try:
        from core.agency.safety_registry import get_safety_registry
        safety = get_safety_registry()
        await safety.disable_skill("dangerous_skill")
        is_allowed = await safety.is_allowed("dangerous_skill")
        if not is_allowed:
            print("✅ Skill revocation working.")
        else:
            print("❌ Skill revocation failed!")
            ok = False
    except ZENITH_RECOVERABLE_ERRORS as e:
        print(f"❌ Safety registry test error: {e}")
        ok = False

    # 3. Test Hybrid Store (Unicode & Pruning)
    print("\n3. Testing Hybrid Store...")
    try:
        from core.memory.hybrid_store import get_hybrid_store
        store = get_hybrid_store()
        await store.store("Testing Zenith memory fix with emoji 🛡️", {"confidence": 0.9})
        results = await store.retrieve("Zenith")
        if results and "🛡️" in results[0]['content']:
            print("✅ Hybrid store unicode-safe and retrieving.")
        else:
            print("❌ Hybrid store retrieval failed.")
            ok = False
    except ZENITH_RECOVERABLE_ERRORS as e:
        print(f"❌ Hybrid store test error: {e}")
        ok = False

    # 4. Test Safe Optimizer (Safety checks)
    print("\n4. Testing Safe Optimizer...")
    try:
        from core.adaptation.safe_optimizer import get_safe_optimizer
        opt = get_safe_optimizer()
        # Mock a small file
        test_data = Path(tempfile.gettempdir()) / "lora_test.txt"
        atomic_write_text(test_data, "dummy dataset content")
        await opt.optimize_lora(str(test_data), "base_model")
        print("✅ Safe optimizer executed without crash.")
    except ZENITH_RECOVERABLE_ERRORS as e:
        print(f"❌ Safe optimizer test error: {e}")
        ok = False

    print("\n✨ Zenith Audit Fixes Verification Complete.")
    return ok

if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(test_zenith_fixes()) else 1)
