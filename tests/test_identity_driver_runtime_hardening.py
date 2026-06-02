import asyncio
from pathlib import Path

import pytest

from core.runtime.errors import get_degradation_tracker


def test_identity_driver_source_has_no_raw_broad_exception_catches():
    source = Path("core/consciousness/identity_driver.py").read_text(encoding="utf-8")
    assert "except Exception" not in source
    assert "except BaseException" not in source


@pytest.mark.asyncio
async def test_identity_driver_update_propagates_cancellation():
    from core.consciousness.identity_driver import IdentityDriver

    class _UnifiedSelf:
        async def record_identity_memory(self, **_kwargs):
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

    driver = IdentityDriver()
    driver._unified_self = _UnifiedSelf()

    with pytest.raises(asyncio.CancelledError):
        await driver.update_identity_from_interaction("important turn", significant=True)


@pytest.mark.asyncio
async def test_identity_driver_get_state_failure_is_recorded():
    from core.consciousness.identity_driver import IdentityDriver

    tracker = get_degradation_tracker()
    tracker.reset()

    class _UnifiedSelf:
        def get_state(self):
            attempted = True
            assert attempted
            raise RuntimeError("self-state unavailable")

    driver = IdentityDriver()
    driver._unified_self = _UnifiedSelf()

    assert await driver.derive_drives_from_identity() == []
    records = tracker.recent(subsystem="identity_driver", limit=1)
    assert records
    assert "self-state unavailable" in records[-1].error_message
    tracker.reset()
