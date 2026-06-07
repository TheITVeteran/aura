from __future__ import annotations

import urllib.request

import pytest
import urllib.error

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


def test_suppressed_readiness_failure_does_not_record_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, BaseException, dict[str, object]]] = []

    class FailingUrlOpen:
        def __init__(self) -> None:
            self.calls: list[tuple[urllib.request.Request, float]] = []

        def __call__(self, req: urllib.request.Request, timeout: float) -> _FakeResponse:
            self.calls.append((req, timeout))
            raise urllib.error.URLError("connection refused during startup")

    failing_urlopen = FailingUrlOpen()

    def fake_record_degradation(
        subsystem: str,
        exc: BaseException,
        **kwargs: object,
    ) -> None:
        recorded.append((subsystem, exc, kwargs))

    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("core.runtime.network_gateway.urllib.request.urlopen", failing_urlopen)
    monkeypatch.setattr(
        "core.runtime.network_gateway.record_degradation",
        fake_record_degradation,
    )

    response = NetworkGateway().request(
        "GET",
        "http://127.0.0.1:8000/api/health",
        timeout=1,
        read_only=True,
        source="maintenance_tooling:server_health_wait",
        suppress_degradation=True,
    )

    assert response["ok"] is False
    assert response["status_code"] == 0
    assert recorded == []
    assert len(failing_urlopen.calls) == 1
