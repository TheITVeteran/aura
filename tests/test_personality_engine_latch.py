"""The personality-engine singleton latch and its test-reset seam.

2026-07-12 order-dependence register: get_personality_engine() captures the
container's current value into a module global. A test that registers a
bare double, triggers any code path that resolves the engine, then clears
the container leaves the double LATCHED for every later test in the
process — local_agent_client and memory_facade victims died on a stale
SimpleNamespace. These tests pin the mechanism, the reset seam, and the
autouse teardown wiring that makes the leak impossible.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.container import ServiceContainer


class TestLatchMechanism:
    def test_double_latches_across_container_clear_without_reset(self):
        from core.brain import personality_engine as pe

        pe.reset_personality_engine_for_test()
        double = SimpleNamespace(name="latch-proof-double")
        ServiceContainer.register_instance("personality_engine", double, required=False)
        try:
            assert pe.get_personality_engine() is double, "container value latches"
            ServiceContainer.clear()
            # THE DEFECT SHAPE: cleared container, latch still serves the double.
            assert pe.get_personality_engine() is double, (
                "the latch survives ServiceContainer.clear() — this is exactly "
                "why the reset seam and conftest wiring exist"
            )
        finally:
            pe.reset_personality_engine_for_test()
            ServiceContainer.clear()

    def test_reset_purges_the_latch(self):
        from core.brain import personality_engine as pe

        double = SimpleNamespace(name="latch-proof-double-2")
        ServiceContainer.register_instance("personality_engine", double, required=False)
        try:
            assert pe.get_personality_engine() is double
            ServiceContainer.clear()
            pe.reset_personality_engine_for_test()
            resolved = pe.get_personality_engine()
            assert resolved is not double, "reset must purge the latched double"
            assert hasattr(resolved, "get_personality_prompt"), (
                "post-reset resolution must yield a REAL engine"
            )
        finally:
            pe.reset_personality_engine_for_test()
            ServiceContainer.clear()


def test_conftest_service_container_teardown_wires_the_reset():
    """The fixture that clears the container must also purge the latch —
    otherwise every service_container test that touches personality paths
    re-arms the leak."""
    from pathlib import Path

    conftest = (Path(__file__).parent / "conftest.py").read_text()
    assert "reset_personality_engine_for_test" in conftest
