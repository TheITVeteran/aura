"""Regression lock for the memory-pin confirmation contract.

A memory-pin request needs the pinned CONTENT echoed back to count as a write
receipt. A future-tense receipt ("I will remember that <content>") is valid; a
content-less generic acknowledgement ("okay, I'll remember it") is not — the
payload-echo check is what distinguishes them, so accepting the base verb form
does not weaken the rejection of generic filler.

(Fixes the pre-existing test_prebuilt_desktop_contract... failure, where a valid
"I will remember that ..." reply was over-rejected as a generic acknowledgement
because only the past tense "remembered" was recognized.)
"""
from __future__ import annotations

from core.conversation.response_reliability import _matches_memory_pin_confirmation

_PIN_REQUEST = "Remember this note for later in this conversation: the blue lantern is under the desk."


def test_future_tense_receipt_with_content_is_accepted():
    assert _matches_memory_pin_confirmation(
        _PIN_REQUEST, "I will remember that the blue lantern is under the desk."
    ) is True


def test_past_tense_receipt_with_content_is_accepted():
    assert _matches_memory_pin_confirmation(
        _PIN_REQUEST, "Noted — I've recorded that the blue lantern is under the desk."
    ) is True


def test_generic_acknowledgement_without_content_is_rejected():
    # Confirmation verb present, but NO pinned content echoed → still generic.
    assert _matches_memory_pin_confirmation(_PIN_REQUEST, "Okay, I'll remember it.") is False
    assert _matches_memory_pin_confirmation(_PIN_REQUEST, "Got it, noted!") is False


def test_non_pin_request_is_not_a_confirmation():
    assert _matches_memory_pin_confirmation(
        "will you remember this conversation tomorrow?",
        "I will remember that the blue lantern is under the desk.",
    ) is False
