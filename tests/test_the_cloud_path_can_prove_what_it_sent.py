""""We scrub before sending" was a comment, not a claim anything checked.

The cloud fallback took two strings through `str(scrubber(x))` and sent them.
`str(None)` is `"None"` — a string that looks scrubbed and is a scrubber
failure. Nothing verified the scrub worked, so a name the private-name loader
could not read, or a shape no pattern covered, left the machine with the
privacy claim intact and no record of what had gone.

Three more on the same path:

- The system prompt was glued to the user's message as
  `"{system}\n\nUser: {user}\nAura:"`. The user can type those same labels, so
  their text competed with the instructions it was impersonating, and no
  provider metadata could show whether role separation survived. Gemini has
  accepted a `system_instruction` all along and nothing passed one.
- One shared 60-second deadline was set from any error string containing "429"
  or "quota", so one provider's rate limit suppressed cloud recovery through
  every other provider — and a Retry-After the provider actually sent was
  ignored in favour of a flat minute.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate
from core.brain.pii_scrubber import (
    SCRUBBER_VERSION,
    residual_pii_findings,
    scrub_for_cloud_with_receipt,
)


# ─────────────────────────────── the scrub is checkable


def test_the_receipt_names_the_scrubber_that_produced_it():
    _text, receipt = scrub_for_cloud_with_receipt("hello")

    assert receipt["scrubber_version"] == SCRUBBER_VERSION
    assert len(receipt["source_sha256"]) == 64
    assert len(receipt["scrubbed_sha256"]) == 64


@pytest.mark.parametrize(
    "text",
    [
        "reach me at someone@example.com",
        "my number is +1 415 555 0132",
        "the key is sk-ABCDEFGHIJKLMNOPQRSTUV",
    ],
)
def test_contact_details_and_keys_do_not_leave_the_machine(text):
    scrubbed, receipt = scrub_for_cloud_with_receipt(text)

    assert "REDACTED" in scrubbed
    assert receipt["residual_findings"] == []
    assert receipt["safe_to_send"] is True


def test_the_residual_scan_catches_what_the_scrubber_missed():
    """The scan is the check on the scrub, not a second scrubber."""
    assert residual_pii_findings("someone@example.com") == ["email"]


def test_ordinary_text_is_left_alone():
    scrubbed, receipt = scrub_for_cloud_with_receipt("what is the capital of France?")

    assert scrubbed == "what is the capital of France?"
    assert receipt["changed"] is False
    assert receipt["safe_to_send"] is True


def test_empty_text_is_safe_and_unchanged():
    scrubbed, receipt = scrub_for_cloud_with_receipt("")

    assert scrubbed == ""
    assert receipt["safe_to_send"] is True


def test_a_scrubber_returning_none_blocks_the_send(monkeypatch):
    """str(None) is "None" — a string that looks scrubbed."""
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_cloud_privacy_receipt = {}

    assert gate._scrub_cloud_payload("system", "prompt", scrubber=lambda _t: None) is None


def test_a_verified_payload_carries_a_receipt():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_cloud_privacy_receipt = {}

    payload = gate._scrub_cloud_payload("you are Aura", "what is 2+2?")

    assert payload is not None
    receipt = gate.cloud_privacy_receipt()
    assert receipt["residual_findings"] == []
    assert receipt["system"]["scrubber_version"] == SCRUBBER_VERSION
    assert receipt["prompt"]["scrubber_version"] == SCRUBBER_VERSION


def test_a_payload_that_does_not_verify_is_blocked(monkeypatch):
    import core.brain.pii_scrubber as scrubber_mod

    gate = InferenceGate.__new__(InferenceGate)
    gate._last_cloud_privacy_receipt = {}

    def _passthrough(text):
        return text, {
            "scrubber_version": SCRUBBER_VERSION,
            "source_sha256": "0" * 64,
            "scrubbed_sha256": "0" * 64,
            "source_chars": len(text),
            "scrubbed_chars": len(text),
            "changed": False,
            "residual_findings": ["private_name"],
            "safe_to_send": False,
        }

    monkeypatch.setattr(scrubber_mod, "scrub_for_cloud_with_receipt", _passthrough)

    assert gate._scrub_cloud_payload("Bryan is here", "hello") is None
    assert gate.cloud_privacy_receipt()["residual_findings"] == ["private_name"]


# ─────────────────────────────── roles are not flattened into a string


def test_the_gate_sends_the_system_prompt_as_a_system_instruction():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)

    assert '"system_instruction": cloud_system_prompt' in source
    assert 'adapter_prompt = f"{cloud_system_prompt}' not in source


def test_the_adapter_passes_the_system_instruction_through():
    import inspect

    import core.adapters.api_adapter as adapter_mod

    source = inspect.getsource(adapter_mod)

    assert "system_instruction=system_instruction" in source
    assert '"role_separation"' in source


# ─────────────────────────────── backoff is per provider


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._cloud_backoff_by_provider = {}
    return gate


def test_a_rate_limited_provider_does_not_suppress_the_others():
    gate = _gate()

    assert gate._note_cloud_provider_failure("api_adapter", "429 quota exceeded") is True
    assert gate._cloud_provider_in_backoff("api_adapter") is True
    assert gate._cloud_provider_in_backoff("health_router") is False


def test_an_ordinary_failure_is_not_a_rate_limit():
    gate = _gate()

    assert gate._note_cloud_provider_failure("api_adapter", "connection reset") is False
    assert gate._cloud_provider_in_backoff("api_adapter") is False


def test_a_provider_supplied_retry_after_is_honoured():
    gate = _gate()

    gate._note_cloud_provider_failure("api_adapter", "429; retry-after: 12")

    remaining = gate.cloud_backoff_state()["api_adapter"]
    assert 0 < remaining <= 12.0


def test_a_missing_retry_after_uses_the_default():
    gate = _gate()

    gate._note_cloud_provider_failure("api_adapter", "429 quota exceeded")

    assert gate.cloud_backoff_state()["api_adapter"] == pytest.approx(
        InferenceGate._CLOUD_BACKOFF_S, abs=1.0
    )


def test_an_absurd_retry_after_is_bounded():
    gate = _gate()

    gate._note_cloud_provider_failure("api_adapter", "429; retry-after: 999999")

    from core.brain.inference_gate import _MAX_HEALTH_WINDOW_S

    assert gate.cloud_backoff_state()["api_adapter"] <= _MAX_HEALTH_WINDOW_S


def test_a_longer_backoff_is_not_shortened_by_a_later_shorter_one():
    gate = _gate()

    gate._note_cloud_provider_failure("api_adapter", "429; retry-after: 300")
    gate._note_cloud_provider_failure("api_adapter", "429; retry-after: 5")

    assert gate.cloud_backoff_state()["api_adapter"] > 100.0


def test_an_expired_backoff_disappears_from_the_state():
    gate = _gate()
    gate._cloud_backoff_by_provider = {"api_adapter": 0.0}

    assert gate.cloud_backoff_state() == {}
    assert gate._cloud_provider_in_backoff("api_adapter") is False
