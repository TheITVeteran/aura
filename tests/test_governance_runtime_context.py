from __future__ import annotations

from core.container import ServiceContainer
from core.governance_context import governance_runtime_active


def test_governance_runtime_activates_when_core_service_exists_before_lock(monkeypatch):
    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)

    saved_services = dict(ServiceContainer._services)
    saved_aliases = dict(ServiceContainer._aliases)
    saved_locked = ServiceContainer._registration_locked
    try:
        ServiceContainer._services = {}
        ServiceContainer._aliases = {}
        ServiceContainer._registration_locked = False

        assert governance_runtime_active() is False

        ServiceContainer.register_instance("aura_kernel", object())

        assert governance_runtime_active() is True
    finally:
        ServiceContainer._services = saved_services
        ServiceContainer._aliases = saved_aliases
        ServiceContainer._registration_locked = saved_locked


def test_governance_runtime_activates_when_container_is_locked_without_services(monkeypatch):
    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)

    saved_services = dict(ServiceContainer._services)
    saved_aliases = dict(ServiceContainer._aliases)
    saved_locked = ServiceContainer._registration_locked
    try:
        ServiceContainer._services = {}
        ServiceContainer._aliases = {}
        ServiceContainer._registration_locked = True

        assert governance_runtime_active() is True
    finally:
        ServiceContainer._services = saved_services
        ServiceContainer._aliases = saved_aliases
        ServiceContainer._registration_locked = saved_locked
