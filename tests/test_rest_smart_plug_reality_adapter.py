from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.actuation.robotics_actuator import RoboticsActuator
from core.embodiment.hardware_manager import HardwareManager
from core.embodiment.mock_iot_plug import RestSmartPlug
from core.reality_reach.live import RealityReachService
from core.reality_reach.transactions import RealityActuationCoordinator


class _Gateway:
    def __init__(self) -> None:
        self.state = "off"
        self.calls: list[dict[str, Any]] = []

    async def request_async(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": method, "url": url, **kwargs})
        return {
            "ok": True,
            "status_code": 200,
            "content": json.dumps(
                {"state": self.state, "power_draw_watts": 0.0}
            ).encode(),
        }


class _Container:
    services: dict[str, Any] = {}

    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        return cls.services.get(name, default)


async def _executor(**kwargs: Any) -> dict[str, Any]:
    try:
        dispatched = dict(
            await kwargs["effect_handler"]({"will_receipt_id": "test.will.iot.1"})
        )
    except Exception as exc:  # noqa: BLE001 - emulate ActionExecutor containment
        return {"ok": False, "effect_verified": False, "error": str(exc)}
    verified = dict(await kwargs["effect_verifier"]({"result": dispatched}))
    return {**dispatched, **verified, "ok": verified.get("effect_verified") is True}


@pytest.mark.asyncio
async def test_rest_smart_plug_runs_through_reality_reach_and_fresh_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_IOT_ENDPOINT", "https://relay.example.test/aura")
    monkeypatch.setenv("AURA_IOT_KEY", "test-secret")
    gateway = _Gateway()
    monkeypatch.setattr(
        "core.embodiment.mock_iot_plug.get_network_gateway",
        lambda: gateway,
    )

    command_calls: list[dict[str, Any]] = []

    async def command_transport(**kwargs: Any) -> dict[str, Any]:
        command_calls.append(kwargs)
        body = json.loads(kwargs["data"])
        gateway.state = str(body["action"])
        return {"ok": True, "status_code": 204, "content": b""}

    monkeypatch.setattr(
        "core.embodiment.mock_iot_plug.ActionExecutor.request_network_transport",
        staticmethod(command_transport),
    )
    service = RealityReachService()
    manager = HardwareManager()
    manager.bind_reality_reach(service)
    device = RestSmartPlug("relay-1")
    manager.register_device(device)
    await manager.start()
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    _Container.services = {
        "hardware_manager": manager,
        "reality_actuation": coordinator,
    }
    monkeypatch.setattr("core.container.ServiceContainer", _Container)

    result = await RoboticsActuator.command_device(
        "relay-1",
        "turn_on",
        {},
        idempotency_key="test.relay.turn-on.1",
    )

    assert result["effect_verified"] is True
    assert result["reality_reach_transaction"]["state"] == "effect_verified"
    assert device.power_state is True
    assert command_calls[0]["source"] == "world_bridge:hardware.rest_smart_plug.apply"
    assert command_calls[0]["read_only"] is False
    assert "test-secret" not in command_calls[0]["data"]
    assert len(gateway.calls) >= 4  # connect, compile, prepare, and effect readback
    assert all(call["read_only"] is True for call in gateway.calls)

    with pytest.raises(RuntimeError, match="active physical adapter"):
        manager.unregister_device("relay-1")
    await manager.stop()
    manager.unregister_device("relay-1")
    assert manager.get_device("relay-1") is None
    assert service.status()["channel_count"] == 0


@pytest.mark.asyncio
async def test_failed_status_handshake_never_claims_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_IOT_ENDPOINT", "https://relay.example.test/aura")

    class FailedGateway:
        async def request_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"ok": False, "status_code": 503, "error": "unavailable"}

    monkeypatch.setattr(
        "core.embodiment.mock_iot_plug.get_network_gateway",
        lambda: FailedGateway(),
    )
    device = RestSmartPlug("relay-2")

    assert await device.connect() is False
    assert device.is_connected is False
    status = await device.get_status()
    assert status["ok"] is False


def test_manifest_is_explicit_and_transport_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_IOT_ENDPOINT", "https://relay.example.test/aura")
    manifest = RestSmartPlug("relay-3").reality_manifest()

    assert manifest.command_transport_id != manifest.readback_transport_id
    assert manifest.capability.observation_channels == (
        manifest.observation.channel_id,
    )
    assert {command.command for command in manifest.commands} == {
        "turn_on",
        "turn_off",
        "emergency_stop",
    }
