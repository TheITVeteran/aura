"""A completed await must not mean "maybe delivered, maybe dropped".

CP126 (critical), core/bus/local_pipe_bus.py: "The public send API has no
delivery result. Not-running, closed, broken, suppression-window, and
locked-writer conditions all return normally without sending; RuntimeError
is also swallowed. A normal await completion therefore means neither
admission nor delivery, and drop counters live only inside the bus."

Six distinct drop paths returned None, exactly like a successful write.
Fire-and-forget is the right SEMANTIC — a caller should not block on the bus
— but that is a statement about waiting, not a licence to report a drop as a
send. Callers that want to know can now ask; callers that do not are
unaffected, because the return value was previously unused.
"""
from __future__ import annotations

import asyncio

import pytest

from core.bus.local_pipe_bus import LocalPipeBus, SendOutcome


def _bus(*, running=True, closed=False, broken=False, suppressed_until=0.0):
    bus = LocalPipeBus.__new__(LocalPipeBus)
    bus._is_running = running
    bus.write_conn = type("_Conn", (), {"closed": closed})()
    bus._pipe_broken = broken
    bus._write_suppressed_until = suppressed_until
    bus._write_backpressure_drops = 0
    return bus


class TestEachDropPathIsNamed:
    @pytest.mark.asyncio
    async def test_a_stopped_bus_reports_not_running(self):
        outcome = await _bus(running=False)._send_local("x", {})
        assert outcome.delivered is False
        assert outcome.reason == "bus_not_running"

    @pytest.mark.asyncio
    async def test_a_closed_connection_is_named(self):
        outcome = await _bus(closed=True)._send_local("x", {})
        assert outcome.reason == "connection_closed"

    @pytest.mark.asyncio
    async def test_a_broken_pipe_is_named(self):
        outcome = await _bus(broken=True)._send_local("x", {})
        assert outcome.reason == "pipe_broken"

    @pytest.mark.asyncio
    async def test_the_suppression_window_is_named(self):
        import time

        bus = _bus(suppressed_until=time.monotonic() + 60.0)
        outcome = await bus._send_local("x", {})
        assert outcome.reason == "write_suppression_window"

    @pytest.mark.asyncio
    async def test_the_reasons_are_distinct(self):
        """"The bus is not running" and "the writer is locked" demand
        different responses; collapsing them to None removed the choice."""
        import time

        reasons = {
            (await _bus(running=False)._send_local("x", {})).reason,
            (await _bus(closed=True)._send_local("x", {})).reason,
            (await _bus(broken=True)._send_local("x", {})).reason,
            (await _bus(suppressed_until=time.monotonic() + 60)._send_local("x", {})).reason,
        }
        assert len(reasons) == 4


class TestTheOutcomeIsEasyToUse:
    def test_it_is_falsy_when_dropped(self):
        assert not SendOutcome(False, "bus_not_running")

    def test_it_is_truthy_when_delivered(self):
        assert SendOutcome(True)

    def test_a_delivered_outcome_carries_no_reason(self):
        assert SendOutcome(True).reason == ""


class TestThePublicApiNeverRaises:
    @pytest.mark.asyncio
    async def test_a_transport_loop_error_becomes_an_outcome(self):
        """Fire-and-forget must not start raising just because it now
        reports — a caller that ignores the result is unchanged."""
        bus = _bus()

        async def _boom(_name, _fn):
            raise RuntimeError("transport loop gone")

        bus._run_on_transport_loop = _boom
        outcome = await bus.send("x", {})
        assert outcome.delivered is False
        assert "transport_loop_error" in outcome.reason

    @pytest.mark.asyncio
    async def test_a_legacy_none_return_is_treated_as_delivered(self):
        """Defensive: an inner path that predates SendOutcome must not be
        reported as a drop it never was."""
        bus = _bus()

        async def _quiet(_name, _fn):
            return None

        bus._run_on_transport_loop = _quiet
        assert (await bus.send("x", {})).delivered is True
