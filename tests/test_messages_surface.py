from __future__ import annotations

import asyncio

from fastapi.responses import JSONResponse
from starlette.requests import Request


def _request(*, host: str = "127.0.0.1", surface: str = "messages") -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/chat",
            "raw_path": b"/api/chat",
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1:8000"),
                (b"x-aura-surface", surface.encode("ascii")),
            ],
            "client": (host, 0),
            "server": ("127.0.0.1", 8000),
        }
    )


def test_messages_surface_is_local_owner_only_and_requires_cognitive_engine() -> None:
    from interface import auth
    from interface.routes import chat

    local = _request()
    remote = _request(host="203.0.113.7")

    assert auth._allow_local_without_token(local, protected_route=True) is True
    assert auth._allow_local_without_token(remote, protected_route=True) is False
    assert chat._request_requires_cognitive_engine(local) == (True, "messages")


def test_governed_messages_turn_reuses_complete_chat_handler(monkeypatch) -> None:
    async def exercise() -> None:
        from interface.routes import chat

        observed: dict[str, object] = {}
        monkeypatch.setattr(
            chat,
            "validate_runtime_security_request",
            lambda request: observed.update(security_path=request.url.path),
        )
        monkeypatch.setattr(
            chat,
            "_require_internal",
            lambda request: observed.update(internal_path=request.url.path),
        )
        monkeypatch.setattr(
            chat,
            "_check_rate_limit",
            lambda request: observed.update(rate_path=request.url.path),
        )

        async def fake_api_chat(*, body, request, _, __):
            observed["message"] = body.message
            observed["session_id"] = body.session_id
            observed["surface"] = request.headers["x-aura-response-surface"]
            observed["idempotency"] = request.headers["x-idempotency-key"]
            observed["context"] = chat._INTERNAL_SURFACE_CONTEXT.get()
            return JSONResponse({"response": "canonical reply", "status": "ok"})

        monkeypatch.setattr(chat, "api_chat", fake_api_chat)
        reply = await chat.run_governed_surface_chat_turn(
            "hello",
            surface="messages",
            surface_context="[private owner message]",
            session_id="messages-primary_operator",
            timeout_s=2.0,
            idempotency_key="messages-in-proof",
        )

        assert reply == "canonical reply"
        assert observed == {
            "message": "hello",
            "session_id": "messages-primary_operator",
            "surface": "messages",
            "idempotency": "messages-in-proof",
            "context": "[private owner message]",
            "security_path": "/api/chat",
            "internal_path": "/api/chat",
            "rate_path": "/api/chat",
        }

    asyncio.run(exercise())
