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
