import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker


def test_self_awareness_source_has_no_raw_broad_exception_catches():
    source = Path("core/consciousness/self_awareness.py").read_text(encoding="utf-8")
    assert "except Exception" not in source
    assert "except BaseException" not in source


@pytest.mark.asyncio
async def test_consciousness_coordinator_skips_locked_late_registration(monkeypatch):
    from core.consciousness.coordinator import ConsciousnessCoordinator
    from core.container import ServiceContainer
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    coordinator = ConsciousnessCoordinator()
    coordinator._unified_self = SimpleNamespace()
    registered = []
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: default))
    monkeypatch.setattr(ServiceContainer, "has", classmethod(lambda cls, name: False))
    monkeypatch.setattr(
        ServiceContainer,
        "register_instance",
        classmethod(lambda cls, name, instance, **kwargs: registered.append((name, instance, kwargs))),
    )

    def _locked_register(cls, name, instance):
        raise AssertionError("factory registration should not be used for unified_self")

    monkeypatch.setattr(ServiceContainer, "register", classmethod(_locked_register))

    await coordinator._connect_subsystems()

    assert registered
    assert registered[0][0] == "unified_self"
    assert registered[0][1] is coordinator._unified_self
    assert registered[0][2]["failure_policy"] == "continue_with_local_unified_self"
    records = tracker.recent(subsystem="consciousness_coordinator.registration", limit=1)
    assert records == []
    tracker.reset()


@pytest.mark.asyncio
async def test_self_awareness_interaction_propagates_cancellation():
    from core.consciousness.self_awareness import SelfAwareness

    class _UnifiedSelf:
        async def interact(self):
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

    awareness = SelfAwareness()
    awareness._unified_self = _UnifiedSelf()

    with pytest.raises(asyncio.CancelledError):
        await awareness.on_interaction()


@pytest.mark.asyncio
async def test_self_awareness_signal_failure_is_recorded():
    from core.consciousness.self_awareness import SelfAwareness

    tracker = get_degradation_tracker()
    tracker.reset()

    def _fail_agency(_level):
        failed = True
        assert failed
        raise RuntimeError("agency sink offline")

    awareness = SelfAwareness()
    awareness._phenomenal_engine = SimpleNamespace(set_agency=_fail_agency)

    await awareness._signal_agency(0.7)

    records = tracker.recent(subsystem="self_awareness", limit=1)
    assert records
    assert records[-1].action == "skipped agency signal to phenomenal engine"
    tracker.reset()
