"""Completion claims are decided by matching exact-turn effect evidence."""

from __future__ import annotations

import pytest

from core.conversation.grounded_claim_guard import verify_grounded_claims
from core.conversation.surface_disposition import record_tool_receipt
from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

pytestmark = pytest.mark.unit

CLAIM = "I wrote the report file to ~/Documents/report-a.txt."


def _record(**values: object) -> None:
    defaults = {
        "tool_name": "desktop_task",
        "action": "write_text_file",
        "object_ref": "~/Documents/report-a.txt",
        "ok": True,
        "effect_observed": True,
    }
    defaults.update(values)
    assert record_tool_receipt(**defaults)


def test_matching_observed_effect_supports_the_claim() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        _record()
        assert verify_grounded_claims(CLAIM).text == CLAIM


def test_unrelated_success_cannot_override_matching_failure() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        _record(
            tool_name="web_search",
            action="web_search",
            object_ref="orca cognition",
            ok=True,
            effect_observed=True,
        )
        _record(ok=False, effect_observed=False)
        result = verify_grounded_claims(CLAIM)
        assert "didn't go through" in result.text
        assert result.corrections


def test_success_for_the_wrong_object_cannot_override_matching_failure() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        _record(object_ref="~/Documents/report-b.txt")
        _record(ok=False, effect_observed=False)
        assert "didn't go through" in verify_grounded_claims(CLAIM).text


def test_dispatch_success_without_observed_effect_does_not_support_completion() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        _record(ok=True, effect_observed=False)
        assert "didn't go through" in verify_grounded_claims(CLAIM).text


def test_unrelated_receipt_alone_is_unknown_not_an_invented_confession() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        _record(
            tool_name="web_search",
            action="web_search",
            object_ref="orca cognition",
        )
        assert verify_grounded_claims(CLAIM).text == CLAIM
