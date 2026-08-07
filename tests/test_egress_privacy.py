"""The outbound body boundary: what leaves, and what is refused.

The defect these cover is not "a regex did not match". It is that
``NetworkGateway`` decided whether a request could be made without ever
reading what was in it, so the audit row was redacted and the copy sent to a
third party was not.
"""
from __future__ import annotations

import json

import pytest

from core.security import egress_privacy
from core.security.egress_privacy import (
    Tier,
    destination_is_local,
    filter_outbound_body,
    tier_for,
)

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent"
MODEL_SOURCE = "llm_provider:gemini:gemini-2.0-flash"
TOOL_SOURCE = "skills.email_adapter"


@pytest.fixture(autouse=True)
def _reset_counters():
    egress_privacy.reset_egress_privacy_counters_for_test()
    yield
    egress_privacy.reset_egress_privacy_counters_for_test()


def _filter(url: str, body: bytes, source: str):
    return filter_outbound_body(url=url, body=body, source=source)


class TestDestinationClass:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/chat",
            "http://localhost:11434/api",
            "http://[::1]:8000/",
            "http://192.168.1.42:8080/pair",
            "http://10.0.0.5/",
            "http://aura.local:8000/",
            "http://169.254.10.1/",
        ],
    )
    def test_this_machine_and_its_lan_are_not_egress(self, url):
        assert destination_is_local(url) is True
        assert tier_for(url, MODEL_SOURCE) == Tier.LOCAL

    @pytest.mark.parametrize(
        "url", [GEMINI, "https://8.8.8.8/", "https://api.example.com/v1"]
    )
    def test_the_open_internet_is_egress(self, url):
        assert destination_is_local(url) is False

    def test_local_body_is_passed_through_untouched(self):
        body = b'{"prompt":"my key is sk-abcdefghijklmnopqrstuvwxyz01"}'
        result = _filter("http://127.0.0.1:8000/v1", body, "llm_provider:mlx")
        assert result.allowed is True
        assert result.body == body
        # Never claims an inspection it did not perform.
        assert result.inspected is False


class TestCredentialTier:
    """A credential is a leak at every destination, model or not."""

    @pytest.mark.parametrize(
        ("secret", "marker"),
        [
            ("sk-abcdefghijklmnopqrstuvwxyz01", "[REDACTED_API_KEY]"),
            # "Bearer <jwt>" is caught by the bearer pattern first — it runs
            # earlier — so the marker is BEARER, not JWT. Either way the
            # credential is gone, which is the property under test.
            ("Bearer eyJhbGciOiJIUzI1NiJ9.payloadpayload.signature", "[REDACTED_BEARER]"),
            ("eyJhbGciOiJIUzI1NiJ9.payloadpayload.signature", "[REDACTED_JWT]"),
            ("AKIAIOSFODNN7EXAMPLE", "[REDACTED_AWS_KEY]"),
            ("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "[REDACTED_GITHUB_TOKEN]"),
            ("xoxb-1234567890-abcdefghijkl", "[REDACTED_SLACK_TOKEN]"),
        ],
    )
    def test_secrets_never_reach_a_third_party_tool(self, secret, marker):
        body = json.dumps({"note": f"use {secret} to auth"}).encode()
        result = _filter("https://api.example.com/hook", body, TOOL_SOURCE)

        assert result.tier == Tier.CREDENTIALS
        assert result.allowed is True
        assert secret not in result.body.decode()
        assert marker in result.body.decode()

    def test_a_tool_may_still_send_the_person_it_is_about(self):
        # An email adapter whose recipient is stripped is a broken tool, and a
        # broken control is one somebody routes around.
        body = json.dumps({"to": "bryan@example.com", "subject": "hi"}).encode()
        result = _filter("https://api.example.com/send", body, TOOL_SOURCE)

        assert "bryan@example.com" in result.body.decode()


class TestModelProviderTier:
    """A turn's context becomes the provider's log line. Both tiers apply."""

    def test_personal_identifiers_are_held_back_from_a_model(self):
        body = json.dumps(
            {"contents": [{"parts": [{"text": "email bryan@example.com"}]}]}
        ).encode()
        result = _filter(GEMINI, body, MODEL_SOURCE)

        assert result.tier == Tier.FULL
        assert "bryan@example.com" not in result.body.decode()
        assert "[EMAIL_REDACTED]" in result.body.decode()
        assert "email_redacted" in result.kinds

    def test_health_router_probes_are_model_egress_too(self):
        assert tier_for(GEMINI, "llm_provider:health_router:gemini-flash") == Tier.FULL

    def test_owner_may_turn_the_personal_tier_off_but_not_the_credential_tier(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            egress_privacy, "_personal_redaction_enabled", lambda: False
        )
        body = json.dumps(
            {"text": "bryan@example.com key sk-abcdefghijklmnopqrstuvwxyz01"}
        ).encode()
        result = _filter(GEMINI, body, MODEL_SOURCE)

        assert result.tier == Tier.CREDENTIALS
        sent = result.body.decode()
        assert "bryan@example.com" in sent
        assert "sk-abcdefghijklmnopqrstuvwxyz01" not in sent


class TestTheRequestStillWorks:
    """A privacy control that corrupts working requests gets deleted."""

    def test_numbers_are_never_touched(self):
        # The phone pattern matches "1234.5678". Sweeping the serialized JSON
        # would turn an embedding into [PHONE_REDACTED] and the failure would
        # surface at the provider, looking like their bug. Redaction runs over
        # string leaves of the parsed document, so numbers cannot be reached.
        body = json.dumps(
            {"embedding": [1234.5678, 9876.5432], "temperature": 0.7, "top_k": 40}
        ).encode()
        result = _filter(GEMINI, body, MODEL_SOURCE)

        assert result.redactions == 0
        assert json.loads(result.body)["embedding"] == [1234.5678, 9876.5432]

    def test_a_clean_body_is_returned_byte_identical(self):
        body = b'{"q":"what is the capital of France"}'
        result = _filter(GEMINI, body, MODEL_SOURCE)

        assert result.body is body
        assert result.inspected is True
        assert result.redactions == 0

    def test_nested_structure_survives_redaction(self):
        body = json.dumps(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "key sk-aaaaaaaaaaaaaaaaaaaaaa"}]}
                ],
                "generationConfig": {"maxOutputTokens": 2048},
            }
        ).encode()
        result = _filter(GEMINI, body, MODEL_SOURCE)
        sent = json.loads(result.body)

        assert sent["generationConfig"]["maxOutputTokens"] == 2048
        assert sent["contents"][0]["role"] == "user"
        assert "[REDACTED_API_KEY]" in sent["contents"][0]["parts"][0]["text"]

    def test_a_deeply_nested_payload_does_not_hit_a_recursion_limit(self):
        payload: dict = {"text": "sk-abcdefghijklmnopqrstuvwxyz01"}
        for _ in range(400):
            payload = {"inner": payload}
        result = _filter(GEMINI, json.dumps(payload).encode(), MODEL_SOURCE)

        assert result.allowed is True
        assert result.redactions == 1

    def test_a_plain_text_body_is_filtered_as_text(self):
        result = _filter(
            GEMINI, b"prompt: my key is sk-abcdefghijklmnopqrstuvwxyz01", MODEL_SOURCE
        )
        assert "[REDACTED_API_KEY]" in result.body.decode()


class TestFailClosed:
    def test_an_unreadable_body_does_not_reach_a_model_provider(self):
        result = _filter(GEMINI, b"\xff\xfe\x00\x01binary", MODEL_SOURCE)

        assert result.allowed is False
        assert result.inspected is False
        assert result.body is None
        assert egress_privacy.egress_privacy_counters()["bodies_refused"] == 1

    def test_an_unreadable_body_to_a_tool_passes_and_says_it_was_not_read(self):
        # Refusing every binary upload would break working capability to
        # protect nothing this filter can see. The receipt carries the truth.
        result = _filter("https://api.example.com/upload", b"\xff\xd8\xffPNG", TOOL_SOURCE)

        assert result.allowed is True
        assert result.inspected is False
        assert "cannot be read" in result.reason

    def test_allowed_never_implies_inspected(self):
        # The failure class this module exists to stop: a reader inferring
        # "a check ran" from "the request went out".
        local = _filter("http://127.0.0.1:8000/", b"anything", MODEL_SOURCE)
        assert local.allowed is True and local.inspected is False

    def test_an_empty_body_is_honestly_inspected(self):
        result = _filter(GEMINI, b"", MODEL_SOURCE)
        assert result.allowed is True
        assert result.inspected is True

    def test_the_receipt_never_quotes_what_it_caught(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz01"
        result = _filter(GEMINI, json.dumps({"t": secret}).encode(), MODEL_SOURCE)
        rendered = json.dumps(result.to_dict())

        assert secret not in rendered
        assert result.redactions == 1


class TestGatewayWiring:
    """The gateway must not have a path to the socket that skips the filter."""

    def test_gateway_refuses_an_uninspectable_model_body(self, monkeypatch):
        from core.runtime import network_gateway

        def _never(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("request left the machine without inspection")

        monkeypatch.setattr(network_gateway.urllib.request, "urlopen", _never)
        monkeypatch.setattr(
            network_gateway,
            "_coerce_data",
            lambda data: b"\xff\xfe\x00binary",
        )

        result = network_gateway.NetworkGateway().request(
            "POST", GEMINI, data="x", source=MODEL_SOURCE, read_only=True
        )

        assert result["ok"] is False
        assert result["egress_privacy"]["allowed"] is False

    def test_gateway_fails_closed_for_models_when_the_filter_cannot_load(
        self, monkeypatch
    ):
        import sys

        from core.runtime import network_gateway

        monkeypatch.setitem(sys.modules, "core.security.egress_privacy", None)

        receipt = network_gateway._filter_outbound_body(
            url=GEMINI, body=b"{}", source=MODEL_SOURCE
        )
        assert receipt["allowed"] is False
        assert receipt["inspected"] is False

        tool_receipt = network_gateway._filter_outbound_body(
            url="https://api.example.com/x", body=b"{}", source=TOOL_SOURCE
        )
        assert tool_receipt["allowed"] is True
        assert tool_receipt["inspected"] is False
