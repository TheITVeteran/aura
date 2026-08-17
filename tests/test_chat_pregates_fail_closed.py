from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import interface.routes.chat_preflight as _chat_preflight


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        headers={},
        # Non-Starlette test requests from the canonical "test" host are
        # authenticated as the synthetic owner surface by request_access_profile.
        client=SimpleNamespace(host="test"),
    )


@pytest.mark.asyncio
async def test_broken_defensive_preflight_never_reaches_cognition(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    delivered: list[tuple[str, dict[str, str]]] = []

    def capture_delivery(text: str, **identity: str) -> None:
        delivered.append((text, identity))

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("firewall internals must not leak")

    monkeypatch.setattr(
        "core.security.defensive_runtime.inspect_chat_ingress",
        unavailable,
    )
    monkeypatch.setattr(
        "core.conversation.surface_delivery.note_route_delivered",
        capture_delivery,
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_collect_conversation_lane_status",
        lambda: pytest.fail("cognition admission ran after a failed security preflight"),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Hello, Aura."),
        _request(),
        None,
        None,
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["gate"] == "defensive_runtime"
    assert payload["processed"] is False
    assert payload["retryable"] is True
    assert "firewall internals" not in response.body.decode()
    assert delivered[0][0] == payload["response"]
    assert delivered[0][1]["conversation_id"]
    assert delivered[0][1]["turn_id"] == response.headers["X-Aura-Turn-ID"]


@pytest.mark.asyncio
async def test_broken_conscience_preflight_never_reaches_cognition(monkeypatch):
    from core.security.defensive_runtime import IngressDecision
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        "core.security.defensive_runtime.inspect_chat_ingress",
        lambda *_args, **_kwargs: IngressDecision(allowed=True),
    )
    monkeypatch.setattr(
        "core.ethics.conscience.get_conscience",
        lambda: (_ for _ in ()).throw(RuntimeError("conscience internals")),
    )
    monkeypatch.setattr(
        _chat_preflight,
        "_collect_conversation_lane_status",
        lambda: pytest.fail("cognition admission ran after a failed conscience preflight"),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Hello, Aura."),
        _request(),
        None,
        None,
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["gate"] == "conscience"
    assert payload["processed"] is False
    assert payload["response_confidence"] == "fail_closed"
    assert "conscience internals" not in response.body.decode()
