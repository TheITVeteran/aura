"""Lockdep must never deadlock the process it is validating.

On 2026-08-03, instrumenting the skill catalog's locks made a latent defect
in lockdep itself reachable at boot. ``_splat_locked`` carried the docstring
"Caller holds self._lock" and then, still holding it, called ``logger.error``.
A log handler lazily imported a module whose import took a *checked* lock,
which re-entered ``on_acquire``, which waited for the non-reentrant validator
lock the reporting path had never released.

The live stack, bottom to top:

    reload_skills -> CheckedLock.__exit__ -> on_release -> _splat_locked
      -> logger.error -> omni_tracer.emit -> record_degraded_event
      -> guarded_import("core.terminal_monitor") -> state_root()
      -> CheckedLock.__enter__ -> on_acquire -> waits for self._lock

A deadlock detector that deadlocks is worse than none: it turns a report into
an outage. Reporting now runs with the validator lock released.
"""
from __future__ import annotations

import logging
import threading

import pytest

from core.runtime.lockdep import (
    LockRank,
    LockdepValidator,
    checked_lock,
    reset_lockdep_for_test,
)

REPORT_BUDGET_S = 20.0


@pytest.fixture(autouse=True)
def _clean_validator():
    reset_lockdep_for_test()
    yield
    reset_lockdep_for_test()


class _LockTakingHandler(logging.Handler):
    """A log handler that acquires a checked lock, like the live one did."""

    def __init__(self, lock):
        super().__init__()
        self._lock_to_take = lock
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock_to_take:
            self.records.append(record.getMessage())


def _run_with_deadline(target, budget_s: float = REPORT_BUDGET_S) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(budget_s)
    return thread


class TestReportingDoesNotReenterTheValidator:
    def test_a_handler_that_takes_a_checked_lock_does_not_wedge(self):
        """The exact live shape: reporting logs, the handler locks."""

        handler_lock = checked_lock("test.handler_lock", rank=LockRank.LEAF)
        handler = _LockTakingHandler(handler_lock)
        lockdep_logger = logging.getLogger("Aura.Lockdep")
        lockdep_logger.addHandler(handler)
        try:
            # Provoke a splat: a non-reentrant lock re-acquired by one context.
            offender = checked_lock("test.offender", reentrant=False)

            def provoke() -> None:
                offender.acquire()
                try:
                    # The second acquire is the self_deadlock finding. It must
                    # report — through the locking handler — and return, not
                    # park. Non-blocking so the test never depends on the very
                    # deadlock it is checking for.
                    assert offender.acquire(blocking=False) is False
                finally:
                    offender.release()

            thread = _run_with_deadline(provoke)
            assert not thread.is_alive(), (
                "lockdep's reporting path parked — a validator that deadlocks "
                "the runtime is the outage it exists to prevent"
            )
            assert any("LOCKDEP" in message for message in handler.records), (
                "the finding was never reported"
            )
        finally:
            lockdep_logger.removeHandler(handler)

    def test_the_finding_is_still_recorded(self):
        """Not deadlocking must not mean not reporting."""

        from core.runtime.lockdep import lockdep_report

        offender = checked_lock("test.offender_recorded", reentrant=False)
        offender.acquire()
        try:
            assert offender.acquire(blocking=False) is False
        finally:
            offender.release()

        splats = lockdep_report()["splats"]
        assert any(entry["kind"] == "self_deadlock" for entry in splats), (
            "the self-deadlock finding was dropped instead of reported"
        )


class TestReportingHoldsNoValidatorLock:
    def test_splat_bookkeeping_returns_the_finding_rather_than_logging_it(self):
        """The structural guarantee, not just the observed behaviour."""

        import inspect

        source = inspect.getsource(LockdepValidator._splat_locked)
        assert "logger.error" not in source, (
            "_splat_locked runs under the validator lock; logging from there is "
            "what wedged the live boot"
        )
        assert "record_degradation" not in source, (
            "_splat_locked runs under the validator lock; recording a degradation "
            "from there can re-enter the validator"
        )
        assert "return splat" in source, (
            "_splat_locked must hand the finding back for reporting outside the lock"
        )
