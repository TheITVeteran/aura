"""Provider-independent privacy receipts for network-capable skills."""

from __future__ import annotations

import pytest

from core.brain.pii_scrubber import (
    SCRUBBER_VERSION,
    residual_pii_findings,
    scrub_for_egress_with_receipt,
)


def test_receipt_identifies_the_scrubber_and_payloads():
    _text, receipt = scrub_for_egress_with_receipt("hello")

    assert receipt["schema"] == "aura.egress.privacy_receipt.v1"
    assert receipt["scrubber_version"] == SCRUBBER_VERSION
    assert len(receipt["source_sha256"]) == 64
    assert len(receipt["scrubbed_sha256"]) == 64


@pytest.mark.parametrize(
    "text",
    (
        "reach me at someone@example.com",
        "my number is +1 415 555 0132",
        "the key is sk-ABCDEFGHIJKLMNOPQRSTUV",
    ),
)
def test_contact_details_and_credentials_are_removed_before_egress(text):
    scrubbed, receipt = scrub_for_egress_with_receipt(text)

    assert "REDACTED" in scrubbed
    assert receipt["residual_findings"] == []
    assert receipt["safe_to_send"] is True


def test_residual_scan_independently_catches_unscrubbed_data():
    assert residual_pii_findings("someone@example.com") == ["email"]


def test_ordinary_text_is_unchanged_but_still_receipted():
    source = "what is the capital of France?"
    scrubbed, receipt = scrub_for_egress_with_receipt(source)

    assert scrubbed == source
    assert receipt["changed"] is False
    assert receipt["safe_to_send"] is True


def test_empty_text_is_safe_and_unchanged():
    scrubbed, receipt = scrub_for_egress_with_receipt("")

    assert scrubbed == ""
    assert receipt["safe_to_send"] is True
