"""Routing confidence must not rise on "nothing threw".

CP126 2462d3c5: reinforce() trained the routing ledger from a caller boolean
with no execution receipt and no independent outcome check. The live callers
made that concrete — the response lane passed success=True immediately after
a tool call that had merely not raised, so a tool RETURNING a failure still
strengthened the pathway that chose it.
"""
from __future__ import annotations

import pytest

from core.mycelium import HardwiredPathway, _evidence_verifies_outcome
from core.orchestrator.mixins.response_processing import _tool_result_succeeded


def _pathway():
    return HardwiredPathway(pathway_id="p1", pattern="x", skill_name="s")


# --- evidence, not assertion --------------------------------------------


@pytest.mark.parametrize("evidence", [None, True, False, "done", 42, {"text": "hi"}])
def test_a_claim_without_an_outcome_field_verifies_nothing(evidence):
    assert _evidence_verifies_outcome(evidence, True) is False


@pytest.mark.parametrize("key", ["ok", "success", "verified_success"])
def test_an_agreeing_outcome_field_verifies(key):
    assert _evidence_verifies_outcome({key: True}, True) is True
    assert _evidence_verifies_outcome({key: False}, False) is True


def test_evidence_contradicting_the_caller_is_not_evidence_for_them():
    assert _evidence_verifies_outcome({"ok": False}, True) is False
    assert _evidence_verifies_outcome({"ok": True}, False) is False


def test_an_object_with_an_outcome_attribute_verifies():
    class _Result:
        ok = True

    assert _evidence_verifies_outcome(_Result(), True) is True


# --- verified and asserted are counted apart ----------------------------


def test_an_unverified_success_earns_less_confidence():
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(True, verified=False)
    verified.reinforce(True, verified=True)

    assert verified.confidence > asserted.confidence


def test_a_failure_is_penalised_fully_either_way():
    """Discounting an unverified failure would keep a broken route alive."""
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(False, verified=False)
    verified.reinforce(False, verified=True)

    assert asserted.confidence == verified.confidence


def test_verified_outcomes_are_counted_separately():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    pathway.reinforce(True, verified=False)
    pathway.reinforce(False, verified=True)

    assert pathway.hit_count == 2 and pathway.miss_count == 1
    assert pathway.verified_hits == 1 and pathway.verified_misses == 1
    assert pathway.unverified_reinforcements == 1


def test_no_verified_evidence_is_none_not_zero():
    pathway = _pathway()
    assert pathway.verified_success_rate is None

    pathway.reinforce(True, verified=False)
    assert pathway.verified_success_rate is None  # still nothing checked


def test_the_verified_rate_ignores_asserted_outcomes():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    for _ in range(5):
        pathway.reinforce(False, verified=False)

    assert pathway.verified_success_rate == 1.0
    assert pathway.success_rate < 1.0


@pytest.mark.parametrize(
    "sequence,grade",
    [
        ([], "untested"),
        ([(True, False)], "asserted_only"),
        ([(True, True)], "verified"),
        ([(True, True), (True, False)], "mixed"),
    ],
)
def test_the_evidence_grade_reports_what_backs_the_record(sequence, grade):
    pathway = _pathway()
    for success, verified in sequence:
        pathway.reinforce(success, verified=verified)

    assert pathway.evidence_grade == grade


# --- the call site derives success from the result ----------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"ok": True}, True),
        ({"ok": False}, False),
        ({"success": False}, False),
        ({"error": "boom"}, False),
        ({"status": "failed"}, False),
        ({"status": "refused"}, False),
        ({"status": "ok"}, True),
        ({"text": "an answer"}, True),
        ("a bare string", True),
        (None, False),
        (False, False),
    ],
)
def test_a_returned_failure_is_not_a_success(result, expected):
    assert _tool_result_succeeded(result) is expected


def test_the_call_site_no_longer_hardcodes_success():
    import inspect

    from core.orchestrator.mixins import response_processing

    source = inspect.getsource(response_processing)
    assert "mycelium.reinforce(pw.pathway_id, success=True)" not in source
    assert "_tool_result_succeeded(result)" in source
