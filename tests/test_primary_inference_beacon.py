"""The primary-lane inference beacon: cognition/foreground advertise 32B use so
slow background phenomenology yields instead of contending the single worker."""
from __future__ import annotations

import core.runtime.backpressure as bp
from core.runtime.backpressure import cognition_inference_active, primary_inference_lease


def test_inactive_by_default():
    assert cognition_inference_active() is False


def test_active_within_lease_then_released():
    assert not cognition_inference_active()
    with primary_inference_lease():
        assert cognition_inference_active() is True
    assert cognition_inference_active() is False


def test_reentrant_nesting():
    with primary_inference_lease():
        with primary_inference_lease():
            assert cognition_inference_active()
        assert cognition_inference_active()  # outer still holds it
    assert not cognition_inference_active()


def test_released_on_exception():
    try:
        with primary_inference_lease():
            raise ValueError("boom")
    except ValueError:
        pass
    assert not cognition_inference_active()


def test_stale_lease_auto_expires(monkeypatch):
    # Simulate a leaked lease older than the max age -> treated as inactive.
    with primary_inference_lease():
        assert cognition_inference_active()
        monkeypatch.setattr(bp, "_primary_lease_fresh_at", bp.time.monotonic() - (bp._PRIMARY_LEASE_MAX_AGE_S + 5))
        assert cognition_inference_active() is False
    # counter was reset by the staleness path; stays inactive
    assert cognition_inference_active() is False
