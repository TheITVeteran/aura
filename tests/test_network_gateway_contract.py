from __future__ import annotations

import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from core.runtime.errors import NetworkEffectDenied
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


class _FakeWebSocket:
    def __init__(self, peer: str) -> None:
        self.transport = SimpleNamespace(get_extra_info=lambda name: (peer, 9090))
        self.closed: list[tuple[int, str]] = []

    async def close(self, *, code: int, reason: str) -> None:
        self.closed.append((code, reason))


@pytest.mark.asyncio
async def test_websocket_admission_pins_the_authorized_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    socket = _FakeWebSocket("192.168.50.8")

    async def resolve(host: str, port: int, *, timeout_s: float) -> tuple[str, ...]:
        captured["resolution"] = (host, port, timeout_s)
        return ("192.168.50.8",)

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        captured["url"] = url
        captured["connect"] = kwargs
        return socket

    monkeypatch.setattr("core.runtime.network_gateway._resolve_websocket_addresses", resolve)
    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("websockets.connect", connect)

    admission = await NetworkGateway().connect_websocket(
        "wss://robot.example.test:9090/bridge",
        headers={"Authorization": "secret"},
        source="tests.robot",
        read_only=True,
        allow_private_target=True,
    )

    assert admission.connection is socket
    assert admission.peer_address == "192.168.50.8"
    assert admission.destination_host == "robot.example.test"
    assert captured["resolution"] == ("robot.example.test", 9090, 10.0)
    options = captured["connect"]
    assert isinstance(options, dict)
    assert options["host"] == "192.168.50.8"
    assert options["port"] == 9090
    assert options["proxy"] is None
    assert options["additional_headers"] == {"Authorization": "secret"}


@pytest.mark.asyncio
async def test_websocket_admission_rejects_private_resolution_without_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    async def resolve(host: str, port: int, *, timeout_s: float) -> tuple[str, ...]:
        return ("127.0.0.1",)

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        nonlocal attempted
        attempted = True
        return _FakeWebSocket("127.0.0.1")

    monkeypatch.setattr("core.runtime.network_gateway._resolve_websocket_addresses", resolve)
    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("websockets.connect", connect)

    with pytest.raises(NetworkEffectDenied, match="private_target_requires_explicit_scope"):
        await NetworkGateway().connect_websocket(
            "ws://localhost:9090",
            source="tests.public",
            read_only=True,
        )

    assert attempted is False


@pytest.mark.asyncio
async def test_websocket_admission_rejects_peer_address_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeWebSocket("192.168.50.9")

    async def resolve(host: str, port: int, *, timeout_s: float) -> tuple[str, ...]:
        return ("192.168.50.8",)

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        return socket

    monkeypatch.setattr("core.runtime.network_gateway._resolve_websocket_addresses", resolve)
    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("websockets.connect", connect)

    with pytest.raises(NetworkEffectDenied, match="peer_address_mismatch"):
        await NetworkGateway().connect_websocket(
            "ws://robot.local:9090",
            source="tests.robot",
            read_only=True,
            allow_private_target=True,
        )

    assert socket.closed == [(1008, "network address changed")]


@pytest.mark.asyncio
async def test_websocket_admission_requires_governance_for_mutating_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def require(operation: str, **kwargs: object) -> None:
        calls.append((operation, kwargs))
        raise RuntimeError("no governed context")

    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: True)
    monkeypatch.setattr("core.runtime.network_gateway.require_governance", require)

    with pytest.raises(RuntimeError, match="no governed context"):
        await NetworkGateway().connect_websocket(
            "wss://robot.example.test/bridge",
            source="tests.mutating",
            read_only=False,
        )

    assert calls[0][0] == "network_gateway.connect_websocket:tests.mutating"


@pytest.mark.asyncio
async def test_websocket_admission_fails_closed_on_absent_defensive_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.security.defensive_runtime.validate_outbound_network",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)

    with pytest.raises(NetworkEffectDenied, match="websocket_admission_denied"):
        await NetworkGateway().connect_websocket(
            "wss://robot.example.test/bridge",
            source="tests.absent-verdict",
            read_only=True,
        )


@pytest.mark.asyncio
async def test_websocket_admission_never_connects_to_cloud_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    async def resolve(host: str, port: int, *, timeout_s: float) -> tuple[str, ...]:
        return ("169.254.169.254",)

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        nonlocal attempted
        attempted = True
        return _FakeWebSocket("169.254.169.254")

    monkeypatch.setattr("core.runtime.network_gateway._resolve_websocket_addresses", resolve)
    monkeypatch.setattr("core.runtime.network_gateway.governance_runtime_active", lambda: False)
    monkeypatch.setattr("websockets.connect", connect)

    with pytest.raises(NetworkEffectDenied, match="cloud_metadata_target_denied"):
        await NetworkGateway().connect_websocket(
            "ws://metadata.robot.local:9090",
            source="tests.robot",
            read_only=True,
            allow_private_target=True,
        )

    assert attempted is False
