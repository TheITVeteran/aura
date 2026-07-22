"""CP126 hardening contracts for core/adapters/api_adapter.py."""
from __future__ import annotations

import asyncio

import pytest

from core.adapters.api_adapter import (
    _MAX_OUTPUT_TOKENS,
    APIAdapter,
    _bounded_float,
    _bounded_int,
)


class TestAdmissionBounds:
    def test_non_finite_temperature_falls_back(self):
        assert _bounded_float(float("nan"), default=0.7, low=0.0, high=2.0) == 0.7
        assert _bounded_float(float("inf"), default=0.7, low=0.0, high=2.0) == 0.7

    def test_out_of_range_clamps(self):
        assert _bounded_float(99.0, default=0.7, low=0.0, high=2.0) == 2.0
        assert _bounded_float(-5.0, default=0.7, low=0.0, high=2.0) == 0.0

    def test_garbage_types_fall_back(self):
        assert _bounded_float("hot", default=0.7, low=0.0, high=2.0) == 0.7
        assert _bounded_int([1], default=800, low=1, high=100) == 800

    def test_token_ceiling_is_enforced(self):
        assert _bounded_int(10**9, default=800, low=1, high=_MAX_OUTPUT_TOKENS) == _MAX_OUTPUT_TOKENS


class TestErrorVersusEmpty:
    def test_all_backend_failure_raises_instead_of_returning_empty(self):
        adapter = APIAdapter()

        async def _failed(*_a, **_k):
            return {"ok": False, "text": "", "error": "all_backends_failed"}

        adapter.generate_with_metadata = _failed

        with pytest.raises(RuntimeError, match="api_adapter_generation_failed"):
            asyncio.run(adapter.generate("hello"))

    def test_successful_empty_generation_is_not_an_error(self):
        adapter = APIAdapter()

        async def _ok_empty(*_a, **_k):
            return {"ok": True, "text": "", "error": ""}

        adapter.generate_with_metadata = _ok_empty
        assert asyncio.run(adapter.generate("hello")) == ""

    def test_oversized_prompt_is_refused_with_reason(self):
        adapter = APIAdapter()
        result = asyncio.run(
            adapter.generate_with_metadata("x" * 600_000, {"model_tier": "local"})
        )
        assert result["ok"] is False
        assert "prompt_too_large" in result["error"]


class TestStopClearsCapability:
    def test_stop_revokes_advertised_capability(self):
        adapter = APIAdapter()
        adapter.has_gemini = True
        adapter.has_local = True

        asyncio.run(adapter.stop())

        assert adapter.has_gemini is False
        assert adapter.has_local is False
        assert adapter.get_status()["gemini"] is False
        assert adapter.get_available_tiers() == []


class TestStatusIsACopy:
    def test_counters_cannot_be_mutated_through_status(self):
        adapter = APIAdapter()
        status = adapter.get_status()
        status["calls"]["gemini"] = 999
        assert adapter._call_count["gemini"] == 0


class TestEmbeddingSpaceIsIdentified:
    def test_local_fallback_declares_its_vector_space(self):
        adapter = APIAdapter()
        adapter.has_gemini = False

        vector = asyncio.run(adapter.embed_async("hello world"))

        assert isinstance(vector, list) and vector
        # A lexical vector must be distinguishable from a cloud embedding so
        # callers never mix incomparable spaces in one index.
        assert adapter.last_embedding_space() == APIAdapter.LOCAL_EMBED_SPACE
        assert adapter.last_embedding_space() != APIAdapter.CLOUD_EMBED_SPACE
