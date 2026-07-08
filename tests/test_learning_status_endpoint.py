"""Contract tests for /api/system/learning — the learning stack's live view.

The endpoint must aggregate whatever learning services are registered, stay
200 when parts are missing (a degraded view is still a view), and never let
one broken section take down the whole report.
"""
from __future__ import annotations

import json

import pytest

import core.container as container_mod
import core.learning.verifiable_preference_harness as harness_mod
from interface.routes import system as system_routes

pytestmark = pytest.mark.unit


class FakeScheduler:
    def get_status(self):
        return {"service": "weight_compounding", "last_status": "promoted", "lineage": {"generations": 2}}


class FakeFlywheel:
    def get_status(self):
        return {"service": "selfplay_flywheel", "bursts": 7, "total_pairs": 21}


class FakeHarness:
    def stats(self):
        return {"total_pairs": 46, "pending": 0, "store_path": "test-store"}


class FakeLibrary:
    def stats(self):
        return {"registered": 1, "attached": "arithmetic_chain-specialist"}


@pytest.fixture
def wired(monkeypatch):
    services = {
        "weight_compounding": FakeScheduler(),
        "selfplay_flywheel": FakeFlywheel(),
    }
    monkeypatch.setattr(
        container_mod.ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )
    monkeypatch.setattr(harness_mod, "get_verifiable_preference_harness", FakeHarness)
    import core.brain.expert_lora_library as library_mod

    monkeypatch.setattr(library_mod, "get_expert_lora_library", FakeLibrary)
    return services


async def test_learning_status_aggregates_all_sections(wired):
    response = await system_routes.api_system_learning(request=None)
    payload = json.loads(response.body)
    assert payload["schema"] == "aura.learning_status.v1"
    assert payload["compounding"]["last_status"] == "promoted"
    assert payload["selfplay"]["total_pairs"] == 21
    assert payload["preference_store"]["total_pairs"] == 46
    assert payload["expert_library"]["registered"] == 1
    assert response.status_code == 200


async def test_learning_status_survives_missing_services(wired, monkeypatch):
    monkeypatch.setattr(
        container_mod.ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: default),
    )
    response = await system_routes.api_system_learning(request=None)
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert "compounding" not in payload          # absent, not fabricated
    assert payload["preference_store"]["total_pairs"] == 46   # what exists still reports


async def test_learning_status_isolates_broken_section(wired, monkeypatch):
    def broken():
        raise RuntimeError("store exploded")

    monkeypatch.setattr(harness_mod, "get_verifiable_preference_harness", broken)
    response = await system_routes.api_system_learning(request=None)
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["preference_store"] == {"error": "unavailable"}
    assert payload["compounding"]["last_status"] == "promoted"   # others unharmed
