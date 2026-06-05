from __future__ import annotations

import urllib.request

import pytest

from core.runtime.network_gateway import NetworkGateway


class _FakeResponse:
    status = 200
    url = "https://example.test/feed"

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def info(self) -> dict[str, str]:
        return {"content-type": "text/plain"}

    def read(self) -> bytes:
        return b"gateway-ok"


@pytest.mark.asyncio
async def test_request_async_routes_through_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("core.runtime.network_gateway.urllib.request.urlopen", fake_urlopen)

    response = await NetworkGateway().request_async(
        "GET",
        "https://example.test/feed",
        timeout=4,
        read_only=True,
        source="tests.network_gateway.async",
    )

    assert response["ok"] is True
    assert response["content"] == b"gateway-ok"
    assert captured == {
        "url": "https://example.test/feed",
        "method": "GET",
        "timeout": 4.0,
    }
