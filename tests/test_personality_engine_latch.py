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


def _engine_double(name: str) -> SimpleNamespace:
    """A double that can plausibly BE the personality engine.

    CP126 b9015f4d: get_personality_engine now refuses a registration that
    cannot answer the identity-causal interface, because a stale or hostile
    object must not become the identity service. These tests are about
    LATCHING, not validity, so the double satisfies the interface — what
    they assert (read-through, no caching into the module global) is
    unchanged.
    """
    return SimpleNamespace(
        name=name,
        get_personality_prompt=lambda *a, **k: "",
        current_mood=lambda *a, **k: "neutral",
        filter_response=lambda text, *a, **k: text,
    )


class TestLatchMechanism:
    def test_container_value_is_read_through_not_latched(self):
        """The historical latch defect is FIXED: get_personality_engine reads
        the container through on every call and never caches the registered
        object into the module global, so clearing the container automatically
        stops serving a registered double even WITHOUT an explicit reset."""
        from core.brain import personality_engine as pe

        pe.reset_personality_engine_for_test()
        double = _engine_double("latch-proof-double")
        ServiceContainer.register_instance("personality_engine", double, required=False)
        try:
            assert pe.get_personality_engine() is double, "registered value wins while present"
            ServiceContainer.clear()
            # FIXED SHAPE: the read-through purges the double the moment the
            # container is cleared — no stale SimpleNamespace poisons later callers.
            resolved = pe.get_personality_engine()
            assert resolved is not double, "read-through must not latch the double past clear"
            assert hasattr(resolved, "get_personality_prompt"), "clear falls back to the real engine"
        finally:
            pe.reset_personality_engine_for_test()
            ServiceContainer.clear()

    def test_reset_purges_the_latch(self):
        from core.brain import personality_engine as pe

        double = _engine_double("latch-proof-double-2")
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
