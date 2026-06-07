import pytest

from core.utils.concurrency import RobustLock


def test_robust_lock_blocks_force_release_without_explicit_opt_in(monkeypatch):
    from core.utils import concurrency as module

    records = []
    monkeypatch.setattr(
        module,
        "record_degradation",
        lambda *args, **kwargs: records.append((args, kwargs)),
    )
    lock = RobustLock("Safety.DefaultForceReleaseBlocked")
    assert lock._lock.acquire(timeout=0.0) is True

    try:
        assert lock.force_release() is False
        assert lock.locked() is True
        assert records[-1][0][0] == "concurrency"
        assert records[-1][1]["action"] == "blocked_lock_force_release_without_explicit_opt_in"
    finally:
        if lock.locked():
            lock._lock.release()


def test_robust_lock_allows_force_release_only_when_opted_in():
    lock = RobustLock("Safety.ExplicitForceRelease", force_release_on_stall=True)
    assert lock._lock.acquire(timeout=0.0) is True

    assert lock.force_release() is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_robust_lock_context_manager_fails_closed_when_acquire_times_out(monkeypatch):
    from core.utils import concurrency as module

    monkeypatch.setattr(module.random, "uniform", lambda *_args, **_kwargs: 0.0)
    lock = RobustLock(
        "Safety.ContextManagerAcquireTimeout",
        timeout_s=0.01,
        force_release_on_stall=False,
    )
    assert lock._lock.acquire(timeout=0.0) is True

    try:
        with pytest.raises(TimeoutError, match="failed to acquire robust lock"):
            async with lock:
                raise AssertionError("critical section should not run without lock ownership")
    finally:
        if lock.locked():
            lock._lock.release()


@pytest.mark.asyncio
async def test_robust_lock_context_manager_enters_after_successful_acquire():
    lock = RobustLock("Safety.ContextManagerAcquireSuccess", timeout_s=0.05)

    async with lock:
        assert lock.locked() is True

    assert lock.locked() is False
