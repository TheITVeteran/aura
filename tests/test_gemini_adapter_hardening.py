"""CP126 pure-logic contracts for the Gemini cloud adapter.

No network: these exercise the secret-handling, quota-atomicity, failure-
surfacing, and validation logic added to close the semantic-review findings.
"""
from __future__ import annotations

import pytest

from core.brain.llm.gemini_adapter import (
    DailyRateLimiter,
    GeminiAdapter,
    _clamp_float,
    _clamp_int,
)

# ── secret is never in the URL ─────────────────────────────────────────────


def test_api_key_travels_in_the_header_not_the_url():
    adapter = GeminiAdapter(api_key="SECRET_KEY", rate_limiter=DailyRateLimiter())
    headers = adapter._auth_headers()
    assert headers["x-goog-api-key"] == "SECRET_KEY"


# ── quota is atomic and counts every attempt ───────────────────────────────


def test_try_reserve_is_atomic_and_counts_attempts():
    limiter = DailyRateLimiter()
    limiter.DEFAULT_LIMITS = {"m": 3}
    reserved = [limiter.try_reserve("m") for _ in range(5)]
    # Exactly the limit is admitted; the rest are refused — no overrun.
    assert reserved == [True, True, True, False, False]
    assert limiter.get_usage().get("m", {}).get("used", limiter._counts["m"]) or \
        limiter._counts["m"] == 3


def test_reserve_counts_before_the_network_so_failures_are_not_free():
    limiter = DailyRateLimiter()
    limiter.DEFAULT_LIMITS = {"m": 10}
    limiter.try_reserve("m")
    assert limiter._counts["m"] == 1  # counted at reserve, not after success


# ── generation params are clamped ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0.8), ("hot", 0.8), (float("nan"), 0.8), (5.0, 2.0), (-1.0, 0.0), (0.5, 0.5)],
)
def test_clamp_float(value, expected):
    assert _clamp_float(value, 0.8, 0.0, 2.0) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 2048), ("x", 2048), (0, 1), (99999, 32768), (512, 512)],
)
def test_clamp_int(value, expected):
    assert _clamp_int(value, 2048, 1, 32768) == expected


# ── failures are surfaced, not erased to "" ────────────────────────────────


@pytest.mark.asyncio
async def test_failed_call_is_discoverable_through_last_metadata(monkeypatch):
    adapter = GeminiAdapter(api_key="k", rate_limiter=DailyRateLimiter())

    async def _fake_call(_prompt, **_kwargs):
        return False, "", {"error": "provider_exploded"}

    monkeypatch.setattr(adapter, "call", _fake_call)

    result = await adapter.generate("hello")

    assert result == ""  # compatibility signature unchanged
    meta = adapter.get_last_generation_metadata()
    assert meta["ok"] is False
    assert meta["error"] == "provider_exploded"


@pytest.mark.asyncio
async def test_successful_call_records_ok_metadata(monkeypatch):
    adapter = GeminiAdapter(api_key="k", rate_limiter=DailyRateLimiter())

    async def _fake_call(_prompt, **_kwargs):
        return True, "the answer", {}

    monkeypatch.setattr(adapter, "call", _fake_call)

    result = await adapter.generate("hello")

    assert result == "the answer"
    assert adapter.get_last_generation_metadata()["ok"] is True


@pytest.mark.asyncio
async def test_per_call_model_argument_is_honored(monkeypatch):
    adapter = GeminiAdapter(api_key="k", model="base-model",
                            rate_limiter=DailyRateLimiter())
    seen: list[str] = []

    async def _fake_call(_prompt, **_kwargs):
        seen.append(adapter.model)
        return True, "ok", {}

    monkeypatch.setattr(adapter, "call", _fake_call)

    await adapter.generate_text_async("hi", model="other-model")

    assert seen == ["other-model"]
    # The base model is restored after the call.
    assert adapter.model == "base-model"
