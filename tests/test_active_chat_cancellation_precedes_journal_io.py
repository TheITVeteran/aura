"""An authenticated local Stop remains available while durable receipt reads are stalled."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.responses import JSONResponse

from core.runtime.chat_delivery_journal import ChatDeliveryJournal, ChatDeliveryJournalUnavailable
from interface.routes import chat, chat_delivery
from tests.test_chat_delivery_journal import _patch_route_identity, _payload, _request

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_active_stop_does_not_wait_on_journal_io_and_keeps_its_authenticated_identity(tmp_path, monkeypatch):
    journal = ChatDeliveryJournal(tmp_path / "delivery.sqlite3", poll_interval_s=.01)
    _patch_route_identity(monkeypatch, journal)
    entered = asyncio.Event()

    @chat_delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        entered.set()
        await asyncio.Event().wait()
        return JSONResponse({"response": "should not finish"})

    key = "journal-blocked-stop"
    request = _request(key)
    body = chat.ChatRequest(message="Continue this task", session_id="session-1")
    task = asyncio.create_task(handler(body=body, request=request))
    await asyncio.wait_for(entered.wait(), timeout=2)

    def blocked():
        raise ChatDeliveryJournalUnavailable("receipt IO unavailable")

    monkeypatch.setattr(chat, "get_chat_delivery_journal", blocked)
    try:
        other = await chat.api_chat_delivery_cancel(key, request, session_id="other")
        assert other.status_code == 503 and not task.cancelling()
        monkeypatch.setattr(chat_delivery, "_authenticated_chat_principal", lambda _request: "owner:someone-else")
        other_owner = await chat.api_chat_delivery_cancel(key, request, session_id="session-1")
        assert other_owner.status_code == 503 and not task.cancelling()
        monkeypatch.setattr(chat_delivery, "_authenticated_chat_principal", lambda _request: "owner:bryan")
        acknowledged = await chat.api_chat_delivery_cancel(key, request, session_id="session-1")
        payload = _payload(acknowledged)
        assert acknowledged.status_code == 202
        assert payload["cancellation_status"] == "cancellation_requested"
        assert payload["terminal"] is False and "result" not in payload
        assert payload["idempotency_key"] == key
        response = await asyncio.wait_for(task, timeout=2)
        assert _payload(response)["status"] == "cancelled_by_user"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
