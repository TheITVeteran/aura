"""CP126 7e008f5a: contamination has to be MEASURED, not asserted.

The certificate used to admit a trial on ``"contamination_scan_passed": True``.
A producer that never scanned emits exactly that field, and so does a producer
that scanned, saw a 40% overlap, and shipped anyway. The certificate could not
tell those apart from a real scan, which makes held-out status the weakest link
in a gate whose whole job is deciding whether a capability claim is publishable.

A receipt names the scanner that ran, the method, the threshold it was held to,
and the overlap it found — and the finding has to satisfy the threshold. These
tests pin both directions: a bare boolean no longer suffices, and a receipt
that fails on its own numbers is refused.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    _validate_contamination_receipt,
)
from tests.fixtures.latent_frontier import _bundle, _certify, _refresh_task_commitment


def _receipt(**overrides):
    receipt = {
        "scanner_implementation_sha256": "b" * 64,
        "method": "13gram_overlap",
        "max_overlap_threshold": 0.02,
        "max_overlap_observed": 0.0,
    }
    receipt.update(overrides)
    return receipt


class TestReceiptValidation:
    def test_a_complete_receipt_passes_and_returns_the_scanner(self):
        reasons: list[str] = []
        digest = _validate_contamination_receipt(
            {"contamination_scan": _receipt()}, "t1", reasons
        )
        assert reasons == []
        assert digest == "b" * 64

    def test_a_bare_boolean_no_longer_suffices(self):
        """The exact shape a producer emits when it never scanned."""
        reasons: list[str] = []
        digest = _validate_contamination_receipt(
            {"contamination_scan_passed": True}, "t1", reasons
        )
        assert reasons == ["t1:contamination_scan_receipt_missing"]
        assert digest == ""

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"scanner_implementation_sha256": "not-a-digest"}, "contamination_scanner_unproven"),
            ({"scanner_implementation_sha256": ""}, "contamination_scanner_unproven"),
            ({"method": "  "}, "contamination_method_missing"),
            ({"max_overlap_threshold": None}, "contamination_threshold_invalid"),
            ({"max_overlap_threshold": 1.5}, "contamination_threshold_invalid"),
            ({"max_overlap_threshold": -0.1}, "contamination_threshold_invalid"),
            ({"max_overlap_observed": None}, "contamination_overlap_unmeasured"),
            ({"max_overlap_observed": "low"}, "contamination_overlap_unmeasured"),
        ],
    )
    def test_each_missing_element_is_named(self, overrides, expected):
        reasons: list[str] = []
        _validate_contamination_receipt(
            {"contamination_scan": _receipt(**overrides)}, "t1", reasons
        )
        assert f"t1:{expected}" in reasons

    def test_measured_overlap_above_the_threshold_is_refused(self):
        """The case the boolean was hiding: a scan that ran and FAILED."""
        reasons: list[str] = []
        _validate_contamination_receipt(
            {
                "contamination_scan": _receipt(
                    max_overlap_threshold=0.02, max_overlap_observed=0.4
                )
            },
            "t1",
            reasons,
        )
        assert reasons == ["t1:contamination_overlap_exceeds_threshold"]

    def test_overlap_exactly_at_the_threshold_is_admitted(self):
        reasons: list[str] = []
        _validate_contamination_receipt(
            {
                "contamination_scan": _receipt(
                    max_overlap_threshold=0.02, max_overlap_observed=0.02
                )
            },
            "t1",
            reasons,
        )
        assert reasons == []


class TestCertificateRefusals:
    def test_a_trial_with_no_receipt_is_excluded_from_the_claim(self):
        bundle = _bundle()
        del bundle["trials"][0]["contamination_scan"]
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            "contamination_scan_receipt_missing" in reason
            for reason in certificate["reasons"]
        )

    def test_a_failing_scan_is_excluded_even_when_the_boolean_says_passed(self):
        bundle = _bundle()
        trial = bundle["trials"][0]
        assert trial["contamination_scan_passed"] is True
        trial["contamination_scan"] = copy.deepcopy(trial["contamination_scan"])
        trial["contamination_scan"]["max_overlap_observed"] = 0.9
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            "contamination_overlap_exceeds_threshold" in reason
            for reason in certificate["reasons"]
        )

    def test_per_trial_scanner_shopping_is_refused(self):
        """One scanner for the run, or the producer picked the instrument last.

        Rescanning a trial with a different scanner until it passes is choosing
        the measurement after seeing the reading. The trials each look clean;
        only the run-level view can see it.
        """
        bundle = _bundle()
        for offset, trial in enumerate(bundle["trials"][:2]):
            trial["contamination_scan"] = copy.deepcopy(trial["contamination_scan"])
            trial["contamination_scan"]["scanner_implementation_sha256"] = (
                str(offset) * 64
            )
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert "contamination_scanner_not_uniform" in certificate["reasons"]

    def test_the_unmodified_fixture_still_certifies(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
