"""Protection is a property of the service, not of the name used to reach it.

CP126: "Alias registration can bypass protected-service checks. Instance
registration resolves aliases for descriptor replacement while protection
decisions use the raw incoming name."

register_instance resolved the alias to find the descriptor, then compared
the ALIAS against _PROTECTED_CORE_SERVICES. No match, so it took the
hot-path upsert and swapped the instance of a service the container exists
to refuse to overwrite — after the registry was locked.
"""
from __future__ import annotations

import pytest

from core.container import _PROTECTED_CORE_SERVICES, ServiceContainer


@pytest.fixture
def sealed_registry():
    was_locked = ServiceContainer._registration_locked
    yield
    ServiceContainer._registration_locked = was_locked


def _a_protected_service() -> str:
    assert _PROTECTED_CORE_SERVICES, "no protected services to test against"
    return sorted(_PROTECTED_CORE_SERVICES)[0]


def test_an_alias_cannot_overwrite_the_protected_service_it_points_at(sealed_registry):
    protected = _a_protected_service()
    real, impostor = object(), object()

    ServiceContainer.register_instance(protected, real, required=False)
    ServiceContainer.register_alias("test_alias_backdoor", protected)
    ServiceContainer.lock_registration()

    ServiceContainer.register_instance("test_alias_backdoor", impostor, required=False)

    assert ServiceContainer.get(protected, default=None) is not impostor, (
        "an alias replaced a protected core service"
    )


def test_the_direct_name_is_still_blocked(sealed_registry):
    """The control: the guard that already worked must keep working."""
    protected = _a_protected_service()
    real, impostor = object(), object()

    ServiceContainer.register_instance(protected, real, required=False)
    ServiceContainer.lock_registration()
    ServiceContainer.register_instance(protected, impostor, required=False)

    assert ServiceContainer.get(protected, default=None) is not impostor


def test_unprotected_services_can_still_be_republished(sealed_registry):
    """The hot path must survive: aura_now is re-published constantly."""
    first, second = object(), object()
    ServiceContainer.register_instance("test_unprotected_svc", first, required=False)
    ServiceContainer.register_instance("test_unprotected_svc", second, required=False)
    assert ServiceContainer.get("test_unprotected_svc", default=None) is second


def test_unlock_requires_an_attributed_caller(sealed_registry):
    """The audit trail is called the primary defence; it has to be one."""
    with pytest.raises(ValueError, match="explicit caller identity"):
        ServiceContainer.unlock_registration()
    with pytest.raises(ValueError):
        ServiceContainer.unlock_registration(caller="   ")


def test_unlock_with_attribution_is_permitted(sealed_registry):
    ServiceContainer.lock_registration()
    ServiceContainer.unlock_registration(
        caller="tests.test_container_alias_cannot_bypass_protection",
        reason="verifying the attributed path still works",
    )
    assert ServiceContainer._registration_locked is False
