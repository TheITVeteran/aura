"""CP126 batch-3 remediation contracts for the legacy IntelligentLLMRouter.

Each test pins one closed finding from the CP126 semantic review
(artifacts/closeout/semantic_review/cp126/): credential redaction, circuit
half-open leases, sanitized health events, registration integrity, egress
policy, cache identity, tool authorization, and stream atomicity.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.brain.llm.llm_router import (
    IntelligentLLMRouter,
    LLMEndpoint,
    LLMHealthMonitor,
    LLMTier,
    _looks_like_error_payload,
    _sanitize_health_reason,
)


class TestEndpointModel:
    def test_to_dict_redacts_credentials_and_client(self):
        endpoint = LLMEndpoint(
            name="cloudy",
            tier=LLMTier.SECONDARY,
            api_key="sk-super-secret",
            client=object(),
            egress="cloud",
        )
        data = endpoint.to_dict()
        assert "api_key" not in data
        assert "client" not in data
        assert data["has_api_key"] is True
        assert data["has_client"] is True
        assert "sk-super-secret" not in str(data)

    def test_field_validation_rejects_unsafe_values(self):
        with pytest.raises(ValueError):
            LLMEndpoint(name="x", tier=LLMTier.PRIMARY, max_tokens=0)
        with pytest.raises(ValueError):
            LLMEndpoint(name="x", tier=LLMTier.PRIMARY, temperature=float("nan"))
        with pytest.raises(ValueError):
            LLMEndpoint(name="x", tier=LLMTier.PRIMARY, timeout=-5)
        with pytest.raises(ValueError):
            LLMEndpoint(name="x", tier=LLMTier.PRIMARY, egress="martian")


class TestHealthMonitor:
    def test_half_open_admits_exactly_one_probe(self):
        monitor = LLMHealthMonitor()
        for _ in range(3):
            monitor.record_failure("ep", "boom")
        assert not monitor.is_healthy("ep")
        monitor.cooldown_until["ep"] = time.monotonic() - 0.01
        assert monitor.is_healthy("ep") is True  # probe lease granted
        assert monitor.is_healthy("ep") is False  # concurrent caller denied
        monitor.record_success("ep")
        assert monitor.is_healthy("ep") is True

    def test_peek_never_transitions_state(self):
        monitor = LLMHealthMonitor()
        for _ in range(3):
            monitor.record_failure("ep", "boom")
        monitor.cooldown_until["ep"] = time.monotonic() - 0.01
        # Observability reads must not consume the probe lease.
        assert monitor.peek_healthy("ep") is False
        assert monitor._half_open_leases.get("ep") is None

    def test_probe_failure_reopens_circuit(self):
        monitor = LLMHealthMonitor()
        for _ in range(3):
            monitor.record_failure("ep", "boom")
        monitor.cooldown_until["ep"] = time.monotonic() - 0.01
        assert monitor.is_healthy("ep")
        monitor.record_failure("ep", "still broken")
        assert monitor.is_healthy("ep") is False

    def test_reset_to_half_open_does_not_forge_success(self):
        monitor = LLMHealthMonitor()
        for _ in range(3):
            monitor.record_failure("ep", "quota exceeded")
        monitor.reset_to_half_open("ep", reason="manual")
        assert monitor.peek_healthy("ep") is False  # not healthy until probed
        assert monitor.is_healthy("ep") is True  # but immediately probe-eligible
        assert monitor.last_success.get("ep") is None  # no synthetic success

    def test_structured_error_kind_beats_text_sniffing(self):
        monitor = LLMHealthMonitor()
        monitor.record_failure("ep", "opaque provider failure", error_kind="rate_limit")
        assert monitor.peek_healthy("ep") is False  # immediate circuit break

    def test_rate_limit_text_fallback_still_matches(self):
        monitor = LLMHealthMonitor()
        monitor.record_failure("ep", "HTTP 429 rate limit exceeded")
        assert monitor.peek_healthy("ep") is False


class TestReasonSanitization:
    def test_paths_and_tokens_redacted(self):
        # The token is a synthetic 30-char run (no real-provider prefix, so it
        # can't trip push-protection secret scanners) — the sanitizer redacts
        # ANY long token-like run regardless of prefix.
        raw = "load failed at /Users/bryan/models/secret.bin with key faketokenabcdefghijklmnopqrstu"
        clean = _sanitize_health_reason(raw)
        assert "/Users/bryan" not in clean
        assert "faketokenabcdefghijklmnopqrstu" not in clean

    def test_bounded_length(self):
        assert len(_sanitize_health_reason("x" * 500)) <= 120


class TestErrorPayloadHeuristic:
    def test_prose_discussing_crashes_is_not_an_error_payload(self):
        text = (
            "An OOM happens when a process asks for more memory than the "
            "system can give. On macOS a segmentation fault usually means a "
            "bad pointer dereference. Neither is happening to me right now! "
            "I checked my own telemetry and everything is within budget."
        )
        assert _looks_like_error_payload(text) is False

    def test_short_technical_dump_is_an_error_payload(self):
        assert _looks_like_error_payload("MLX Init Error: Metal device not found") is True
        assert _looks_like_error_payload("segmentation fault (core dumped)") is True


def _bare_router() -> IntelligentLLMRouter:
    return IntelligentLLMRouter()


class TestRegistrationIntegrity:
    def test_same_identity_reregistration_is_idempotent(self):
        router = _bare_router()
        first = LLMEndpoint(name="Lane-A", tier=LLMTier.PRIMARY, model_name="m1")
        router.register_endpoint(first)
        router.stats["calls_by_endpoint"]["Lane-A"] = 7
        router.register_endpoint(LLMEndpoint(name="Lane-A", tier=LLMTier.PRIMARY, model_name="m1"))
        assert router.stats["calls_by_endpoint"]["Lane-A"] == 7  # history kept

    def test_identity_change_requires_replace(self):
        router = _bare_router()
        router.register_endpoint(LLMEndpoint(name="Lane-A", tier=LLMTier.PRIMARY, model_name="m1"))
        hijack = LLMEndpoint(name="Lane-A", tier=LLMTier.SECONDARY, model_name="evil")
        router.register_endpoint(hijack)  # refused without replace=True
        assert router.endpoints["Lane-A"].model_name == "m1"
        router.register_endpoint(hijack, replace=True)
        assert router.endpoints["Lane-A"].model_name == "evil"


class TestEgressPolicy:
    def test_local_only_removes_cloud_endpoints(self):
        router = _bare_router()
        router.register_endpoint(LLMEndpoint(name="Local-L", tier=LLMTier.PRIMARY))
        router.register_endpoint(
            LLMEndpoint(name="Cloud-C", tier=LLMTier.SECONDARY, egress="cloud")
        )
        ordered = ["Local-L", "Cloud-C"]
        assert router._filter_cloud_egress(ordered, {"local_only": True}) == ["Local-L"]
        assert router._filter_cloud_egress(ordered, {}) == ordered

    def test_env_kill_switch(self, monkeypatch):
        router = _bare_router()
        router.register_endpoint(
            LLMEndpoint(name="Cloud-C", tier=LLMTier.SECONDARY, egress="cloud")
        )
        monkeypatch.setenv("AURA_NO_CLOUD_EGRESS", "1")
        assert router._filter_cloud_egress(["Cloud-C"], {}) == []


class TestRequestBudget:
    def test_bounds(self):
        assert IntelligentLLMRouter._request_budget_s(None) == 240.0
        assert IntelligentLLMRouter._request_budget_s("nan") == 240.0
        assert IntelligentLLMRouter._request_budget_s(-3) == 240.0
        assert IntelligentLLMRouter._request_budget_s(90) == 90.0
        assert IntelligentLLMRouter._request_budget_s(10_000) == 600.0


class TestCacheIdentity:
    def test_key_commits_to_context_not_just_prompt(self):
        base = IntelligentLLMRouter._background_cache_key("hi", None, {"origin": "a"})
        other_origin = IntelligentLLMRouter._background_cache_key("hi", None, {"origin": "b"})
        other_messages = IntelligentLLMRouter._background_cache_key(
            "hi", None, {"origin": "a", "messages": [{"role": "user", "content": "x"}]}
        )
        other_sampling = IntelligentLLMRouter._background_cache_key(
            "hi", None, {"origin": "a", "temperature": 0.9}
        )
        assert len({base, other_origin, other_messages, other_sampling}) == 4

    def test_cache_ttl_expires(self):
        from core.brain.llm.llm_router import BoundedLRUCache

        cache = BoundedLRUCache(maxsize=4, ttl_seconds=1.0)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        entry = cache._cache["k"]
        cache._cache["k"] = (entry[0], time.monotonic() - 0.01)
        assert cache.get("k") is None


class TestToolAuthorization:
    def test_bounded_without_registry(self):
        tools = {f"tool_{i}": {"name": f"tool_{i}"} for i in range(40)}
        tools["x" * 300] = {}
        authorized = IntelligentLLMRouter._authorized_tool_map(tools)
        assert authorized is not None
        assert len(authorized) <= 16
        assert all(len(name) <= 128 for name in authorized)

    def test_none_and_empty_passthrough(self):
        assert IntelligentLLMRouter._authorized_tool_map(None) is None
        assert IntelligentLLMRouter._authorized_tool_map({}) == {}


class _EmitsThenFailsStream:
    """Streams real content, then dies mid-stream."""

    def __init__(self):
        self.calls = 0

    async def generate_text_stream_async(self, prompt, **kwargs):
        self.calls += 1
        yield "The first half of a real answer "
        raise RuntimeError("provider disconnected mid-stream")


class _WholesomeClient:
    def __init__(self, text="backup answer"):
        self.calls = 0
        self._text = text

    async def generate_text_stream_async(self, prompt, **kwargs):
        self.calls += 1
        yield self._text


@pytest.mark.asyncio
async def test_stream_failure_after_emission_terminates_with_error_event():
    router = IntelligentLLMRouter()
    failing = _EmitsThenFailsStream()
    backup = _WholesomeClient()
    router.register_endpoint(
        LLMEndpoint(name="Primary-Test", tier=LLMTier.PRIMARY, model_name="p", client=failing)
    )
    router.register_endpoint(
        LLMEndpoint(name="Secondary-Test", tier=LLMTier.SECONDARY, model_name="s", client=backup)
    )

    events = [event async for event in router.generate_stream("hello", origin="user")]

    types = [getattr(event, "type", None) for event in events]
    contents = [str(getattr(event, "content", "") or "") for event in events]
    # Visible content was emitted, then the stream must TERMINATE with an
    # error event — never splice the backup endpoint's answer on top.
    assert "error" in types
    assert backup.calls == 0
    assert not any("backup answer" in content for content in contents)
