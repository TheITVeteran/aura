import json

import pytest
from fastapi import HTTPException

from interface.routes import subsystems


class PayloadRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_terminal_send_preserves_client_input_errors():
    with pytest.raises(HTTPException) as caught:
        await subsystems.api_terminal_send(PayloadRequest({"text": "   "}))

    assert caught.value.status_code == 400
    assert caught.value.detail == "text required"


@pytest.mark.asyncio
async def test_skill_execute_returns_structured_failure_for_router_runtime_error(monkeypatch):
    recorded = []

    class Router:
        def __init__(self):
            self.calls = []

        async def route_execution(self, skill_name, params, engine):
            self.calls.append((skill_name, params, engine))
            raise RuntimeError(f"{skill_name} route unavailable")

    class Engine:
        pass

    def service_get(name, default=None):
        if name == "intent_router":
            return Router()
        if name == "capability_engine":
            return Engine()
        return default

    monkeypatch.setattr(subsystems.ServiceContainer, "get", staticmethod(service_get))
    monkeypatch.setattr(
        subsystems,
        "record_degradation",
        lambda subsystem, error: recorded.append((subsystem, str(error))),
    )

    response = await subsystems.api_skill_execute("research", {"query": "hello"}, None, None)
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload == {"ok": False, "error": "research route unavailable"}
    assert recorded == [("subsystems", "research route unavailable")]
