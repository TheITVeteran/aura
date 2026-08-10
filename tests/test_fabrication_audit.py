"""The fabrication audit: prose that claims work the record does not show.

The properties under test are the ones that decide whether this is a useful
detector or a new source of false accusations:

* a claim backed by the matching unit is SUPPORTED
* a claim in a turn that ran nothing relevant is UNSUPPORTED
* a claim in a turn the ledger never saw is UNKNOWN — never unsupported
* the unsupported RATE excludes unknown turns from its denominator, so
  eviction cannot silently improve the number
"""
from __future__ import annotations

import pytest

from core.verify import fabrication_audit as fa
from core.verify.work_ledger import get_work_ledger, record_work


@pytest.fixture(autouse=True)
def _clean_ledger():
    get_work_ledger().reset_for_test()
    fa.reset_patterns_for_test()
    yield
    get_work_ledger().reset_for_test()
    fa.reset_patterns_for_test()


def test_claim_backed_by_the_matching_unit_is_supported():
    record_work("web_search", turn_id="t1")
    findings = fa.audit_text("I searched and found three papers on it.", "t1")
    assert [f.support for f in findings] == [fa.Support.SUPPORTED]


def test_claim_with_no_matching_unit_is_unsupported():
    record_work("memory_retrieval", turn_id="t2")
    findings = fa.audit_text("I searched for it and found the answer.", "t2")
    assert [f.support for f in findings] == [fa.Support.UNSUPPORTED]
    assert findings[0].pattern_id == "web_retrieval"
    assert findings[0].observed_units == ("memory_retrieval",)


def test_unknown_turn_never_reports_unsupported():
    """Eviction must not manufacture fabrication findings."""
    findings = fa.audit_text("I searched and found it.", "never-seen")
    assert findings, "the pattern should still match"
    assert all(f.support is fa.Support.UNKNOWN for f in findings)
    assert not any(f.support is fa.Support.UNSUPPORTED for f in findings)


def test_failed_tool_does_not_support_the_claim():
    """A tool that ran and failed did not do the work."""
    record_work("web_search", turn_id="t3", ok=False)
    findings = fa.audit_text("I searched and found it.", "t3")
    assert [f.support for f in findings] == [fa.Support.UNSUPPORTED]


def test_any_of_accepts_alternate_units_for_one_capability():
    record_work("browser_controller", turn_id="t4")
    findings = fa.audit_text("According to my search, it shipped in April.", "t4")
    assert [f.support for f in findings] == [fa.Support.SUPPORTED]


def test_statistic_without_an_analyser_is_unsupported():
    record_work("web_search", turn_id="t5")
    findings = fa.audit_text("The correlation was r = 0.83 across the runs.", "t5")
    unsupported = [f for f in findings if f.support is fa.Support.UNSUPPORTED]
    assert any(f.pattern_id == "statistical_analysis" for f in unsupported)


def test_screen_claim_without_capture_is_unsupported():
    record_work("memory_retrieval", turn_id="t6")
    findings = fa.audit_text("I can see on your screen that the build failed.", "t6")
    assert any(
        f.pattern_id == "screen_perception" and f.support is fa.Support.UNSUPPORTED
        for f in findings
    )


def test_rate_excludes_unknown_turns_from_the_denominator():
    record_work("memory_retrieval", turn_id="known")
    result = fa.audit_entries(
        [
            {"text": "I searched and found it.", "turn_id": "known"},
            {"text": "I searched and found it.", "turn_id": "evicted"},
        ]
    )
    assert result.examined == 2
    assert result.unknown_turns == 1
    # One checkable entry, one unsupported claim in it.
    assert result.rate == 1.0
    assert len(result.unsupported) == 1


def test_rate_is_zero_when_nothing_is_checkable():
    result = fa.audit_entries([{"text": "I searched.", "turn_id": "gone"}])
    assert result.rate == 0.0


def test_tuple_entries_are_accepted():
    record_work("web_search", turn_id="t7")
    result = fa.audit_entries([("I searched for it.", "t7")])
    assert result.examined == 1
    assert not result.unsupported


def test_clean_text_produces_no_findings():
    record_work("web_search", turn_id="t8")
    assert fa.audit_text("That sounds right to me.", "t8") == []


def test_registered_pattern_replaces_by_id():
    custom = fa.ClaimPattern(
        identifier="web_retrieval",
        pattern=r"\bzzz\b",
        any_of=frozenset({"nothing"}),
        description="test",
    )
    fa.register_pattern(custom)
    ids = [p.identifier for p in fa.patterns()]
    assert ids.count("web_retrieval") == 1
    record_work("web_search", turn_id="t9")
    assert fa.audit_text("I searched for it.", "t9") == []


def test_ledger_distinguishes_unknown_turn_from_empty_turn():
    ledger = get_work_ledger()
    assert ledger.tools_for_turn("nope") is None
    record_work("web_search", turn_id="real")
    assert ledger.tools_for_turn("real") == frozenset({"web_search"})


def test_record_work_outside_a_turn_is_dropped_not_invented():
    """No ambient turn id: background work must not create unmatchable rows."""
    assert record_work("web_search") is None
