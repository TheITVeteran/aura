"""A state mutation whose receipt failed must not read as no mutation.

CP126 (high), core/being/closed_loop_controller.py: "Receipt failures are
silently ignored."

The emit was wrapped in ``except (...): return``. That produced the worst
possible pairing — the mutation still applied, and the only trace of it did
not. The audit trail then reads "no such change" for a change that is live
in her state, which is not a missing log line but a false negative in the
record used to reconstruct what she did.

Receipts must stay best-effort: a broken receipt store should not stop the
closed loop from running. So the emit still cannot raise; what changed is
that the failure is recorded and countable.
"""
from __future__ import annotations

import pytest

from core.being.closed_loop_controller import Main15ClosedLoopController


class _Store:
    def __init__(self, fails: Exception | None = None):
        self.fails = fails
        self.emitted: list = []

    def emit(self, receipt):
        if self.fails is not None:
            raise self.fails
        self.emitted.append(receipt)


class _Plasticity:
    def __init__(self, store):
        self.receipt_store = store


def _controller(store) -> Main15ClosedLoopController:
    c = Main15ClosedLoopController.__new__(Main15ClosedLoopController)
    c.plasticity = _Plasticity(store)
    c._receipts_emitted = 0
    c._receipts_failed = 0
    c._receipts_unstored = 0
    c._last_receipt_error = ""
    return c


class TestAFailedReceiptIsRecorded:
    @pytest.mark.parametrize(
        "error",
        [RuntimeError("store wedged"), OSError("disk full"), ValueError("bad payload")],
    )
    def test_the_failure_records_a_degradation(self, monkeypatch, error):
        import core.being.closed_loop_controller as mod

        recorded: list = []
        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: recorded.append(a))
        controller = _controller(_Store(fails=error))
        controller._emit_state_receipt(event="promote", payload={"x": 1})
        assert recorded, "a receipt failure was swallowed"

    def test_the_failure_is_counted(self, monkeypatch):
        import core.being.closed_loop_controller as mod

        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: None)
        controller = _controller(_Store(fails=RuntimeError("wedged")))
        controller._emit_state_receipt(event="promote", payload={})
        health = controller.receipt_health()
        assert health["failed"] == 1
        assert health["complete"] is False

    def test_the_last_error_is_retained(self, monkeypatch):
        import core.being.closed_loop_controller as mod

        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: None)
        controller = _controller(_Store(fails=RuntimeError("store wedged")))
        controller._emit_state_receipt(event="promote", payload={})
        assert "RuntimeError" in controller.receipt_health()["last_error"]

    def test_the_loop_still_runs(self, monkeypatch):
        """Best-effort is the right call — a broken store must not stop the
        closed loop. What was wrong is that it was also invisible."""
        import core.being.closed_loop_controller as mod

        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: None)
        controller = _controller(_Store(fails=RuntimeError("wedged")))
        controller._emit_state_receipt(event="promote", payload={})  # must not raise


class TestSuccessfulReceiptsAreCounted:
    def test_an_emitted_receipt_is_counted(self):
        store = _Store()
        controller = _controller(store)
        controller._emit_state_receipt(event="promote", payload={"x": 1})
        assert len(store.emitted) == 1
        assert controller.receipt_health()["emitted"] == 1
        assert controller.receipt_health()["complete"] is True

    def test_coverage_reflects_the_mix(self, monkeypatch):
        import core.being.closed_loop_controller as mod

        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: None)
        controller = _controller(_Store())
        controller._emit_state_receipt(event="a", payload={})
        controller.plasticity.receipt_store.fails = RuntimeError("wedged")
        controller._emit_state_receipt(event="b", payload={})
        assert controller.receipt_health()["coverage"] == pytest.approx(0.5)


class TestNoStoreIsDistinctFromBrokenStore:
    def test_an_absent_store_is_counted_separately(self):
        """"Unreceipted by design" and "receipting is broken" need different
        answers, so they must not share a counter."""
        controller = _controller(None)
        controller._emit_state_receipt(event="promote", payload={})
        health = controller.receipt_health()
        assert health["unstored"] == 1
        assert health["failed"] == 0
        assert health["complete"] is True

    def test_no_attempts_reports_zero_coverage_not_full(self):
        controller = _controller(None)
        assert controller.receipt_health()["coverage"] == 0.0

    def test_the_report_is_serializable(self):
        health = _controller(_Store()).receipt_health()
        assert health["schema"] == "aura.closed_loop_receipt_health.v1"
