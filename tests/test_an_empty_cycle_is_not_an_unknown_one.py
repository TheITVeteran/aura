"""A cycle that produced nothing is an empty cycle, not an unknown one.

Live 2026-08-03, 231 times, escalating to emergency:

    [DEGRADATION] cognitive_engine (critical): TurnOutcomeError:
    unknown:nothing_recorded -> turn ended without establishing what happened
    CRITICAL SERVICE FAILURE: Subsystem 'cognitive_engine' failed with failure
    policy 'fail-closed'
    [DEGRADATION] metabolic_coordinator: ... long-term memory consolidation failed

Two separate mistakes stacked.

FIRST, the turn HAD established what happened. cognitive_engine.think ends an
empty cycle with mark_served("", state=NOTHING_SERVED) precisely so the ledger
knows. The resolver ignored that and reported "nothing_recorded" — reserved
for a turn where genuinely nothing was recorded at all.

SECOND, it recorded that at warning, and warning is the escalation floor for a
fail-closed subsystem. So an ordinary empty cycle was declared a critical
service failure, the healer dispatched repairs at severity=emergency, and
long-term memory consolidation went down with it.

An empty cycle is a real failure — the person got nothing — and a survivable
one. It is now named, still counted, and no longer fatal.
"""
from __future__ import annotations

import pytest

from core.runtime.turn_outcome import OutcomeStatus, TurnOutcome, UserVisibleState


def _empty_cycle() -> TurnOutcome:
    outcome = TurnOutcome(origin="chat")
    outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    return outcome


class TestTheResolverNamesIt:
    def test_an_empty_cycle_is_not_unknown(self):
        receipt = _empty_cycle().finalize(subsystem="test")
        assert receipt.rationale == "nothing_served"
        assert receipt.status is OutcomeStatus.RETRYABLE_FAILURE

    def test_it_is_still_a_failure(self):
        """The person got nothing. That is not success."""
        receipt = _empty_cycle().finalize(subsystem="test")
        assert not receipt.status.is_success

    def test_a_turn_that_recorded_nothing_is_still_unknown(self):
        """The genuine case keeps its name."""
        receipt = TurnOutcome(origin="chat").finalize(subsystem="test")
        assert receipt.status is OutcomeStatus.UNKNOWN
        assert receipt.rationale == "nothing_recorded"

    def test_a_served_turn_is_untouched(self):
        outcome = TurnOutcome(origin="chat")
        outcome.mark_served("here is your answer")
        receipt = outcome.finalize(subsystem="test")
        assert receipt.status is OutcomeStatus.SUCCEEDED
        assert receipt.rationale == "served"

    def test_a_held_answer_still_outranks_it(self):
        """An answer suppressed by a gate must not be filed as an empty cycle."""
        import inspect

        from core.runtime import turn_outcome

        body = inspect.getsource(turn_outcome.TurnOutcome._compute_status)
        held = body.index("answer_available_but_never_served")
        empty = body.index('RETRYABLE_FAILURE, "nothing_served"')
        assert held < empty, "the held-answer case must be checked first"


class TestItDoesNotDeclareTheSubsystemDead:
    def test_an_empty_cycle_records_below_the_escalation_floor(self, monkeypatch):
        """warning is the floor at which a fail-closed subsystem escalates."""
        from core.runtime import turn_outcome

        recorded: list[tuple[str, str]] = []

        def capture(subsystem, error, *, severity="warning", action="", **kwargs):
            recorded.append((severity, str(error)))

        monkeypatch.setattr(turn_outcome, "record_degradation", capture)
        _empty_cycle().finalize(subsystem="cognitive_engine")

        assert recorded, "the outcome must still be recorded"
        severity, message = recorded[0]
        assert "nothing_served" in message
        assert severity == "info", (
            "warning escalates to CRITICAL SERVICE FAILURE on a fail-closed "
            "subsystem, which is what took memory consolidation down"
        )

    def test_a_genuinely_unknown_turn_still_escalates(self, monkeypatch):
        from core.runtime import turn_outcome

        recorded: list[tuple[str, str]] = []

        def capture(subsystem, error, *, severity="warning", action="", **kwargs):
            recorded.append((severity, str(error)))

        monkeypatch.setattr(turn_outcome, "record_degradation", capture)
        TurnOutcome(origin="chat").finalize(subsystem="cognitive_engine")

        assert recorded
        assert recorded[0][0] == "warning"
        assert "nothing_recorded" in recorded[0][1]

    @pytest.mark.parametrize("subsystem", ["cognitive_engine", "turn_outcome"])
    def test_finalizing_an_empty_cycle_never_raises(self, subsystem):
        receipt = _empty_cycle().finalize(subsystem=subsystem)
        assert receipt.rationale == "nothing_served"
