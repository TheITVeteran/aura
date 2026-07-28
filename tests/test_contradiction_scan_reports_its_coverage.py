"""A partial consistency audit must not report a clean bill of health.

CP126 (high), core/brain/cognitive/integrity_check.py: "Contradiction
scanning silently ignores most large graphs. Only the first 500 beliefs are
considered and reporting stops at 50 pairs, with no ordering guarantee,
pagination, coverage flag, omitted count, or continuation cursor."

The caps are defensible — this runs over a live belief graph and an
unbounded sweep with logging is not free. What was not defensible is that a
5,000-belief graph reported "contradictions=0" in exactly the same words as
a genuinely clean 400-belief one.

Measured directly: a 5,003-belief graph containing three real contradictions
reported zero, because all three lay past the 500 cap. An audit that quietly
examines a tenth of the evidence is worse than no audit, because it produces
a clean bill of health nobody thinks to question.

There was no ordering guarantee either, so the truncation was not merely
partial but arbitrarily partial — a contradiction could appear and vanish
between runs with no change to the beliefs.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive.integrity_check import AuditReport, IntegrityGuard


def _guard() -> IntegrityGuard:
    return IntegrityGuard.__new__(IntegrityGuard)


def _beliefs(count: int, contradictions: int = 0) -> list[dict]:
    out = [
        {"id": f"b{i:05d}", "content": f"fact {i}", "status": "active"}
        for i in range(count)
    ]
    # Sorted last by id, so they fall beyond a truncated prefix.
    out += [
        {"id": f"zneg{j}", "content": f"not fact {j}", "status": "active"}
        for j in range(contradictions)
    ]
    return out


class TestASmallGraphIsFullyAudited:
    def test_a_complete_scan_reports_complete(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(400, 3), report)
        assert report.contradiction_result_is_complete is True
        assert report.contradiction_coverage == 1.0

    def test_real_contradictions_are_still_found(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(400, 3), report)
        assert report.contradictions_found == 3

    def test_a_complete_summary_carries_no_qualifier(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(400, 3), report)
        assert "partial scan" not in str(report)


class TestALargeGraphSaysSoOutLoud:
    def test_a_truncated_scan_is_flagged(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(5000, 3), report)
        assert report.contradiction_scan_truncated is True
        assert report.contradiction_result_is_complete is False

    def test_the_omitted_count_is_available(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(5000, 3), report)
        assert report.contradiction_scan_total == 5003
        assert report.contradiction_scan_considered == 500
        assert report.contradiction_coverage == pytest.approx(500 / 5003)

    def test_the_summary_says_the_scan_was_partial(self):
        """The literal defect: this used to read identically to a clean audit."""
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(5000, 3), report)
        summary = str(report)
        assert "partial scan" in summary
        assert "500/5003" in summary

    def test_a_zero_from_a_partial_scan_is_not_a_clean_result(self):
        """The three contradictions here lie beyond the cap, so the count is
        zero — and the report must not present that as consistency."""
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(5000, 3), report)
        assert report.contradictions_found == 0
        assert report.contradiction_result_is_complete is False


class TestTheScanIsDeterministic:
    def test_two_runs_examine_the_same_subset(self):
        """Without an ordering guarantee a contradiction could appear and
        vanish between audits with no change to the beliefs."""
        beliefs = _beliefs(5000, 3)
        shuffled = list(reversed(beliefs))
        first, second = AuditReport(), AuditReport()
        _guard()._detect_contradictions(beliefs, first)
        _guard()._detect_contradictions(shuffled, second)
        assert first.contradictions_found == second.contradictions_found
        assert first.contradiction_scan_considered == second.contradiction_scan_considered


class TestReportingCap:
    def test_hitting_the_pair_cap_is_flagged(self):
        report = AuditReport()
        _guard()._detect_contradictions(_beliefs(200, 80), report)
        if report.contradictions_found >= 50:
            assert report.contradiction_reporting_capped is True
            assert report.contradiction_result_is_complete is False

    def test_quarantined_beliefs_are_excluded_from_the_total(self):
        beliefs = _beliefs(10)
        beliefs.append({"id": "q1", "content": "x", "status": "quarantined"})
        report = AuditReport()
        _guard()._detect_contradictions(beliefs, report)
        assert report.contradiction_scan_total == 10


class TestEmptyInputIsSafe:
    def test_no_beliefs_reports_zero_coverage_not_full(self):
        report = AuditReport()
        _guard()._detect_contradictions([], report)
        assert report.contradiction_scan_total == 0
        assert report.contradiction_coverage == 0.0
