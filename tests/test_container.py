################################################################################

"""
Unit tests for core.container.ServiceContainer.
"""
import asyncio

import pytest

from core.container import ServiceContainer, ServiceLifetime


@pytest.fixture
def clean_container():
    """Provides a fresh container for each test, and leaves one behind.

    Clearing only on the way IN meant the last test in this file left its
    registrations installed for the whole session. ``register_instance``
    defaults to ``required=True``, which descriptors turn into a
    ``fail-closed`` failure policy — so the mycelium instance registered by
    the shutdown test below silently made ``mycelium`` a fail-closed
    subsystem for every test that ran afterwards. 37 mycelium tests that
    inject a failure and assert graceful degradation then saw
    ``record_degradation`` escalate and raise instead. They passed alone and
    failed in-suite, which is the signature of leaked global state, not of a
    defect in the code they were testing.
    """
    ServiceContainer.clear()
    try:
        yield ServiceContainer
    finally:
        ServiceContainer.clear()


class TestServiceRegistration:
    def test_register_and_get(self, clean_container):
        """Services can be registered and retrieved."""
        clean_container.register("test_svc", lambda: {"status": "ok"})
        result = clean_container.get("test_svc")
        assert result == {"status": "ok"}

    def test_singleton_returns_same_instance(self, clean_container):
        """Singleton lifetime returns the same object on repeated gets."""
        clean_container.register(
            "counter", lambda: {"count": 0}, lifetime=ServiceLifetime.SINGLETON
        )
        a = clean_container.get("counter")
        b = clean_container.get("counter")
        assert a is b

    def test_transient_returns_different_instances(self, clean_container):
        """Transient lifetime creates a new object each time."""
        clean_container.register(
            "ephemeral", lambda: {"count": 0}, lifetime=ServiceLifetime.TRANSIENT
        )
        a = clean_container.get("ephemeral")
        b = clean_container.get("ephemeral")
        assert a is not b

    def test_register_instance(self, clean_container):
        """Pre-created instances can be registered directly."""
        obj = {"pre_made": True}
        clean_container.register_instance("preset", obj)
        assert clean_container.get("preset") is obj

    def test_unregister_instance_cannot_remove_a_replacement(self, clean_container):
        stale_owner = object()
        replacement = object()
        clean_container.register_instance("replaceable", stale_owner)
        clean_container.register_instance("replaceable", replacement)

        assert clean_container.unregister_instance(
            "replaceable",
            expected_instance=stale_owner,
        ) is False
        assert clean_container.get("replaceable") is replacement
        assert clean_container.unregister_instance(
            "replaceable",
            expected_instance=replacement,
        ) is True
        assert clean_container.get("replaceable", default=None) is None

    def test_register_normalizes_legacy_instance_input(self, clean_container):
        """Legacy callers that pass an instance to register() should still resolve cleanly."""
        obj = object()
        clean_container.register("legacy", obj)
        assert clean_container.get("legacy") is obj

    def test_get_missing_service_raises(self, clean_container):
        """Accessing an unregistered service raises ServiceNotFoundError."""
        from core.exceptions import ServiceNotFoundError
        with pytest.raises(ServiceNotFoundError, match="not found"):
            clean_container.get("nonexistent")

    def test_get_missing_with_default(self, clean_container):
        """Accessing missing service with default returns the default."""
        result = clean_container.get("nonexistent", default="fallback")
        assert result == "fallback"


class TestCircularDependency:
    def test_circular_dependency_detected(self, clean_container):
        """Circular dependencies raise CircularDependencyError."""
        from core.exceptions import CircularDependencyError
        clean_container.register("a", lambda b: None, dependencies=["b"])
        clean_container.register("b", lambda a: None, dependencies=["a"])
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            clean_container.get("a")


class TestValidation:
    def test_validate_success(self, clean_container):
        """Validation passes when all required services resolve."""
        clean_container.register("svc", lambda: "hello")
        ok, errors = clean_container.validate()
        assert ok is True
        assert errors == []

    def test_validate_missing_dependency(self, clean_container):
        """Validation reports missing dependency."""
        clean_container.register("svc", lambda dep: None, dependencies=["dep"])
        ok, errors = clean_container.validate()
        assert ok is False
        assert any("dep" in e for e in errors)

    def test_health_report(self, clean_container):
        """Health report includes registered services."""
        clean_container.register("test", lambda: "val")
        clean_container.get("test")  # Materialize it
        report = clean_container.get_health_report()
        assert "test" in report["services"]
        assert report["services"]["test"]["status"] == "online"

    def test_health_report_marks_invalid_sovereignty_seal_degraded(self, clean_container, tmp_path, monkeypatch):
        """Seal drift should surface in the health report."""
        seal_path = tmp_path / "sovereignty_seal.json"
        monkeypatch.setattr(ServiceContainer, "_seal_path", classmethod(lambda cls: seal_path))

        clean_container.register_instance("alpha", object())
        clean_container.write_sovereignty_seal()
        clean_container.register_instance("beta", object())

        report = clean_container.get_health_report()

        assert report["status"] == "degraded"
        assert report["sovereignty_seal"]["present"] is True
        assert report["sovereignty_seal"]["valid"] is False

    @pytest.mark.asyncio
    async def test_shutdown_stops_runtime_hygiene_after_owned_services(self, clean_container):
        """Runtime hygiene should audit after thread/process owners have stopped."""
        order = []

        class OwnedService:
            def on_stop(self):
                order.append("owned")

        class RuntimeHygiene:
            async def on_stop_async(self):
                await asyncio.sleep(0)
                order.append("runtime_hygiene")

        clean_container.register_instance("runtime_hygiene", RuntimeHygiene())
        clean_container.register_instance("owned_service", OwnedService())

        await clean_container.shutdown()

        assert order == ["owned", "runtime_hygiene"]

    @pytest.mark.asyncio
    async def test_shutdown_owns_mycelium_lifecycle_once_across_aliases(
        self, clean_container
    ):
        from core.mycelium import MycelialNetwork

        MycelialNetwork._instance = None
        MycelialNetwork._initialized = False
        network = MycelialNetwork()
        clean_container.register_instance("mycelial_network", network)
        clean_container.register_instance("mycelium", network)

        report = await clean_container.shutdown()

        assert MycelialNetwork._instance is None
        assert MycelialNetwork._initialized is False
        assert network._stop_event.is_set() is True
        assert report["coalesced_aliases"] in (
            {"mycelial_network": "mycelium"},
            {"mycelium": "mycelial_network"},
        )


##
