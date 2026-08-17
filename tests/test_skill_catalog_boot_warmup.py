from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import core.capability_engine as capability_module
import core.utils.task_tracker as tracker_module
from core.container import ServiceContainer
from core.orchestrator.mixins.boot.boot_autonomy import BootAutonomyMixin


@pytest.fixture(autouse=True)
def clean_services():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


@pytest.mark.asyncio
async def test_skill_catalog_warmup_is_off_loop_private_and_reused(monkeypatch):
    main_thread = threading.get_ident()
    loaded_on = []
    constructions = []

    class Engine:
        def __init__(self, orchestrator=None):
            constructions.append(orchestrator)

        @property
        def skills(self):
            loaded_on.append(threading.get_ident())
            return {"search": object(), "files": object()}

    class Tracker:
        def create_task(self, coro, name):
            assert name == "orchestrator.skill_catalog_warmup"
            return asyncio.create_task(coro, name=name)

    monkeypatch.setattr(capability_module, "CapabilityEngine", Engine)
    monkeypatch.setattr(tracker_module, "get_task_tracker", lambda: Tracker())

    boot = BootAutonomyMixin()
    boot.status = SimpleNamespace(skills_loaded=0)
    boot._start_skill_catalog_warmup()

    assert ServiceContainer.get("capability_engine", default=None) is None
    engine, count = await boot._consume_skill_catalog_warmup()

    assert count == 2
    assert engine is not None
    assert constructions == [boot]
    assert loaded_on and loaded_on[0] != main_thread
    assert boot._skill_catalog_warmup_task is None
    assert boot._skill_catalog_warmup_engine is None


@pytest.mark.asyncio
async def test_skill_catalog_warmup_failure_retries_same_engine(monkeypatch):
    accesses = 0

    class Engine:
        @property
        def skills(self):
            nonlocal accesses
            accesses += 1
            return {"search": object()}

    async def failed_warmup():
        raise RuntimeError("worker interrupted")

    boot = BootAutonomyMixin()
    engine = Engine()
    boot._skill_catalog_warmup_engine = engine
    boot._skill_catalog_warmup_task = asyncio.create_task(failed_warmup())

    reused, count = await boot._consume_skill_catalog_warmup()

    assert reused is engine
    assert count == 1
    assert accesses == 1
