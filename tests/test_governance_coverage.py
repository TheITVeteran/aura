"""A turn nobody governed must not look like a turn the Will approved.

The message handler asserted, in a comment, that ALL processing passes
through the Unified Will. Three things bypass it — one by design, two
silently — and the two silent ones left no trace, so the governance
boundary had the same defect this codebase keeps finding everywhere else:
the absence of a check reported as a passed check.
"""
from __future__ import annotations

import inspect

import pytest

from core.runtime.governance_coverage import (
    note_ungoverned_turn,
    reset_governance_coverage_for_test,
    ungoverned_turn_report,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_governance_coverage_for_test()
    yield
    reset_governance_coverage_for_test()


class TestTheCounter:
    def test_a_clean_runtime_reports_clean(self):
        report = ungoverned_turn_report()
        assert report["total"] == 0
        assert report["clean"] is True

    def test_an_ungoverned_turn_is_counted(self):
        note_ungoverned_turn("user_message", "will_not_started")
        report = ungoverned_turn_report()
        assert report["total"] == 1
        assert report["clean"] is False

    def test_the_two_causes_are_counted_apart(self):
        """"Governance is not up" and "the gate is erroring" need different fixes."""
        note_ungoverned_turn("user_message", "will_not_started")
        note_ungoverned_turn("user_message", "gate_error:RuntimeError")
        note_ungoverned_turn("user_message", "gate_error:RuntimeError")

        by_reason = ungoverned_turn_report()["by_reason"]
        assert by_reason["will_not_started:user_message"] == 1
        assert by_reason["gate_error:RuntimeError:user_message"] == 2

    def test_origins_are_counted_apart(self):
        note_ungoverned_turn("user_message", "will_not_started")
        note_ungoverned_turn("autonomous", "will_not_started")
        assert len(ungoverned_turn_report()["by_reason"]) == 2


class TestTheGate:
    def test_both_silent_bypasses_now_record(self):
        from core.orchestrator.mixins import message_handling

        source = inspect.getsource(message_handling)
        # The gate is skipped wholesale before the Will starts...
        assert 'note_ungoverned_turn(origin, "will_not_started")' in source
        # ...and continues degraded when it raises.
        assert 'note_ungoverned_turn(origin, f"gate_error:' in source

    def test_the_comment_no_longer_claims_the_gate_is_total(self):
        """A comment asserting an invariant the code does not hold is worse
        than no comment: it is what a reader trusts instead of reading."""
        from core.orchestrator.mixins import message_handling

        source = inspect.getsource(message_handling)
        assert "ALL processing — user-facing or internal — must pass through" not in source
        assert "It is not \"ALL\"" in source

    def test_the_designed_bypass_is_still_named(self):
        """The somatic reflex bypass is deliberate and must stay documented."""
        from core.orchestrator.mixins import message_handling

        source = inspect.getsource(message_handling)
        assert "Somatic Reflex Bypass" in source


def test_the_health_surface_carries_the_count():
    """A green verdict must be able to say it served ungoverned turns."""
    from core.runtime.health_contract import _runtime_integrity_block

    note_ungoverned_turn("user_message", "will_not_started")
    block = _runtime_integrity_block()

    assert "ungoverned_turns" in block, block.get("ungoverned_turns_error")
    assert block["ungoverned_turns"]["total"] >= 1
