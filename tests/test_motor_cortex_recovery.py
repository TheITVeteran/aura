import asyncio
from types import SimpleNamespace

import pytest

from core.somatic import motor_cortex as mc


@pytest.mark.asyncio
async def test_health_reflex_offloads_host_telemetry(monkeypatch):
    calls = []

    async def _to_thread(fn):
        calls.append(fn)
        return 12.0, 34.0, 0.1

    monkeypatch.setattr(mc.asyncio, "to_thread", _to_thread)

    result = await mc._reflex_health_check({})

    assert calls == [mc._sample_reflex_health_telemetry]
    assert result == {
        "success": True,
        "summary": "health_ok",
        "cpu": 12.0,
        "mem_pct": 34.0,
        "thermal": 0.1,
    }


@pytest.mark.asyncio
async def test_health_reflex_contains_malformed_host_telemetry(monkeypatch):
    async def _to_thread(_fn):
        raise ValueError("malformed host sample")

    monkeypatch.setattr(mc.asyncio, "to_thread", _to_thread)

    result = await mc._reflex_health_check({})

    assert result["success"] is False
    assert result["summary"] == "health_check_failed: malformed host sample"


@pytest.mark.asyncio
async def test_screen_capture_reflex_supports_sync_capture(monkeypatch):
    vision = SimpleNamespace(capture_frame=lambda: b"frame")

    def get_service(name, default=None):
        if name == "continuous_vision":
            return vision
        return default

    monkeypatch.setattr(mc.ServiceContainer, "get", get_service)

    result = await mc._reflex_screen_capture({})

    assert result["success"] is True
    assert result["summary"] == "screen_captured"
    assert result["frame_size"] == 5


@pytest.mark.asyncio
async def test_execute_reflex_records_failed_receipt_for_handler_exception(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        mc,
        "_record_motor_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    cortex = mc.MotorCortex()
    cortex.issue_token(mc.ReflexClass.CUSTOM)

    boom_calls = []

    async def boom(_payload):
        boom_calls.append(_payload)
        raise RuntimeError("motor fault")

    cortex.register_handler("boom", boom)

    await cortex._execute_reflex(
        mc.ReflexAction(
            reflex_class=mc.ReflexClass.CUSTOM,
            handler_name="boom",
        )
    )

    receipt = cortex.get_recent_receipts(1)[0]
    assert len(boom_calls) == 1
    assert receipt["success"] is False
    assert receipt["summary"] == "handler_error"
    assert cortex._receipts[-1].error == "motor fault"
    assert recorded[0][1]["action"] == (
        "Converted reflex handler exception into a failed receipt and affect feedback"
    )


@pytest.mark.asyncio
async def test_execute_reflex_propagates_cancellation_without_false_receipt():
    cortex = mc.MotorCortex()
    cortex.issue_token(mc.ReflexClass.CUSTOM)

    cancel_calls = []

    async def cancel(_payload):
        cancel_calls.append(_payload)
        raise asyncio.CancelledError()

    cortex.register_handler("cancel", cancel)

    with pytest.raises(asyncio.CancelledError):
        await cortex._execute_reflex(
            mc.ReflexAction(
                reflex_class=mc.ReflexClass.CUSTOM,
                handler_name="cancel",
            )
        )
    assert len(cancel_calls) == 1

    assert cortex.get_recent_receipts(1) == []


def test_motor_cortex_status_exposes_loop_failure_state():
    cortex = mc.MotorCortex()
    cortex._loop_failure_count = 2
    cortex._last_loop_error = {"error": "loop fault"}

    status = cortex.get_status()

    assert status["loop_failures"] == 2
    assert status["last_loop_error"] == {"error": "loop fault"}
