"""Local API-adapter streams terminate honestly and never select a remote model."""

from __future__ import annotations

import asyncio

import pytest

from core.adapters.api_adapter import APIAdapter, _StreamFailed
from core.schemas import ChatStreamEvent


def _adapter() -> APIAdapter:
    adapter = APIAdapter()
    adapter.has_local = False
    adapter._local_client = None
    return adapter


async def _drain(source) -> list[ChatStreamEvent]:
    return [event async for event in source]


def _types(events: list[ChatStreamEvent]) -> list[str]:
    return [str(event.type) for event in events]


@pytest.mark.asyncio
async def test_a_failed_local_stream_ends_with_an_error_not_silence():
    adapter = _adapter()
    adapter.has_local = True

    async def _broken(*_args, **_kwargs):
        raise _StreamFailed("local worker died mid-answer")
        yield  # pragma: no cover

    adapter._local_stream = _broken  # type: ignore[method-assign]
    events = await _drain(adapter._route_stream("hi", "local", 0.7, 100))

    assert _types(events) == ["error"]


@pytest.mark.asyncio
async def test_a_working_local_stream_ends_exactly_once():
    adapter = _adapter()
    adapter.has_local = True

    async def _ok(*_args, **_kwargs):
        yield ChatStreamEvent(type="token", content="one ")
        yield ChatStreamEvent(type="token", content="two")

    adapter._local_stream = _ok  # type: ignore[method-assign]
    events = await _drain(adapter._route_stream("hi", "api_fast", 0.7, 100))

    assert _types(events) == ["token", "token", "end"]


@pytest.mark.asyncio
async def test_a_partial_local_stream_is_never_restarted_as_a_second_answer():
    adapter = _adapter()
    adapter.has_local = True

    async def _partial(*_args, **_kwargs):
        yield ChatStreamEvent(type="token", content="the answer begins")
        raise _StreamFailed("worker reset")

    adapter._local_stream = _partial  # type: ignore[method-assign]
    events = await _drain(adapter._route_stream("hi", "local", 0.7, 100))

    assert _types(events) == ["token", "error"]
    assert "ended early" in str(events[-1].content)


@pytest.mark.asyncio
async def test_a_silent_local_stream_hits_the_inactivity_deadline(monkeypatch):
    adapter = _adapter()
    adapter.has_local = True
    monkeypatch.setattr(APIAdapter, "STREAM_INACTIVITY_TIMEOUT_S", 0.02)

    async def _stalls(*_args, **_kwargs):
        await asyncio.sleep(30)
        yield  # pragma: no cover

    adapter._local_stream = _stalls  # type: ignore[method-assign]
    events = await asyncio.wait_for(
        _drain(adapter._route_stream("hi", "local", 0.7, 100)),
        timeout=2.0,
    )

    assert _types(events) == ["error"]


@pytest.mark.asyncio
async def test_legacy_api_tiers_resolve_to_local_generation():
    adapter = _adapter()
    adapter.has_local = True

    async def _local(*_args, **_kwargs):
        return "local answer"

    adapter._local_generate = _local  # type: ignore[method-assign]
    result = await adapter._route_generate_with_metadata(
        "hi",
        "api_deep",
        0.2,
        100,
    )

    assert result["ok"] is True
    assert result["provider"] == "local"
    assert result["is_local"] is True
    assert result["tier_resolved"] == "local"


@pytest.mark.asyncio
async def test_remote_only_request_returns_the_retired_provider_contract():
    result = await _adapter()._route_generate_with_metadata(
        "hi",
        "api_deep",
        0.2,
        100,
        config={"cloud_only": True},
    )

    assert result["ok"] is False
    assert result["error"] == "remote_model_provider_removed"
    assert result["provider"] == "none"


def test_status_exposes_no_remote_model_provider():
    status = _adapter().get_status()

    assert status["remote_model_providers"] == ()
    assert _adapter().get_available_tiers() == []
