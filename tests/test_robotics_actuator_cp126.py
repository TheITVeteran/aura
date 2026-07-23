"""CP126 robotics_actuator — the last check before something physical moves."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.actuation.robotics_actuator import (
    MAX_COMMAND_DEADLINE_S,
    RoboticsActuationError,
    RoboticsActuator,
)


class _RecordingActuator:
    def __init__(self):
        self.calls = []

    async def actuate(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class _Registry:
    def __init__(self, devices):
        self._devices = devices

    def get_device(self, device_id):
        return self._devices.get(device_id)


@pytest.fixture
def actuator(monkeypatch):
    recorder = _RecordingActuator()
    monkeypatch.setattr(
        "core.actuation.robotics_actuator.get_world_actuator", lambda: recorder
    )
    return recorder


@pytest.fixture
def registry(monkeypatch):
    devices = {
        "arm-1": SimpleNamespace(
            device_id="arm-1", device_type="arm", status="idle", connected=True
        )
    }
    reg = _Registry(devices)

    class _Container:
        @staticmethod
        def get(name, default=None):
            return reg if name == "hardware_manager" else default

    monkeypatch.setattr("core.container.ServiceContainer", _Container)
    return devices


class TestPhysicalCommandsAreHighRisk:
    """67b7a5b1: physical effects are never ordinary risk."""

    @pytest.mark.asyncio
    async def test_every_command_is_high_risk(self, actuator, registry):
        await RoboticsActuator.command_device("arm-1", "move_to 0 0 1", {})
        assert actuator.calls[0]["high_risk_flag"] is True

    @pytest.mark.asyncio
    async def test_emergency_stop_is_high_risk(self, actuator, registry):
        await RoboticsActuator.emergency_stop("arm-1")
        assert actuator.calls[0]["high_risk_flag"] is True
        assert actuator.calls[0]["params"]["command"] == "emergency_stop"


class TestParameterSmuggling:
    """94b4cea8: params cannot rewrite the governed device or command."""

    @pytest.mark.asyncio
    async def test_smuggled_device_and_command_are_ignored(self, actuator, registry):
        await RoboticsActuator.command_device(
            "arm-1", "halt", {"device_id": "arm-999", "command": "full_speed"}
        )
        params = actuator.calls[0]["params"]
        assert params["device_id"] == "arm-1"
        assert params["command"] == "halt"

    @pytest.mark.asyncio
    async def test_param_count_is_bounded(self, actuator, registry):
        with pytest.raises(RoboticsActuationError, match="too many parameters"):
            await RoboticsActuator.command_device(
                "arm-1", "move", {f"k{i}": i for i in range(64)}
            )


class TestDeviceCapability:
    """15c5b221: the identifier must name a REGISTERED device."""

    @pytest.mark.asyncio
    async def test_unregistered_device_refused(self, actuator, registry):
        with pytest.raises(RoboticsActuationError, match="not registered"):
            await RoboticsActuator.command_device("ghost-arm", "move", {})
        assert actuator.calls == []

    @pytest.mark.asyncio
    async def test_missing_registry_refuses_rather_than_passes(self, actuator, monkeypatch):
        class _Empty:
            @staticmethod
            def get(name, default=None):
                return default

        monkeypatch.setattr("core.container.ServiceContainer", _Empty)
        with pytest.raises(RoboticsActuationError, match="not running"):
            await RoboticsActuator.command_device("arm-1", "move", {})
        assert actuator.calls == []

    @pytest.mark.asyncio
    async def test_empty_device_id_refused(self, actuator, registry):
        with pytest.raises(RoboticsActuationError, match="device_id is empty"):
            await RoboticsActuator.command_device("  ", "move", {})


class TestInterlockAndPreconditions:
    """384bd18d: state snapshot, interlock, deadline, idempotency, stop action."""

    @pytest.mark.asyncio
    async def test_disconnected_device_refused(self, actuator, registry):
        registry["arm-1"].connected = False
        with pytest.raises(RoboticsActuationError, match="not_connected"):
            await RoboticsActuator.command_device("arm-1", "move", {})

    @pytest.mark.asyncio
    async def test_faulted_device_refused(self, actuator, registry):
        registry["arm-1"].status = "fault"
        with pytest.raises(RoboticsActuationError, match="device_status:fault"):
            await RoboticsActuator.command_device("arm-1", "move", {})

    @pytest.mark.asyncio
    async def test_engaged_interlock_refused(self, actuator, registry):
        registry["arm-1"].interlock = True
        with pytest.raises(RoboticsActuationError, match="interlock_engaged"):
            await RoboticsActuator.command_device("arm-1", "move", {})

    @pytest.mark.asyncio
    async def test_contract_fields_are_present(self, actuator, registry):
        await RoboticsActuator.command_device("arm-1", "move", {})
        params = actuator.calls[0]["params"]
        assert params["device_state_before"]["status"] == "idle"
        assert params["idempotency_key"]
        assert params["requires_acknowledgement"] is True
        assert params["compensating_action"]["action"] == "emergency_stop"
        assert params["compensating_action"]["device_id"] == "arm-1"

    @pytest.mark.asyncio
    async def test_deadline_is_bounded_and_forwarded(self, actuator, registry):
        await RoboticsActuator.command_device(
            "arm-1", "move", {}, deadline_s=10_000.0
        )
        call = actuator.calls[0]
        assert call["deadline_s"] <= MAX_COMMAND_DEADLINE_S
        assert call["params"]["deadline_s"] == call["deadline_s"]

    @pytest.mark.asyncio
    async def test_empty_command_refused(self, actuator, registry):
        with pytest.raises(RoboticsActuationError, match="command is empty"):
            await RoboticsActuator.command_device("arm-1", "   ", {})

    @pytest.mark.asyncio
    async def test_oversized_command_refused(self, actuator, registry):
        with pytest.raises(RoboticsActuationError, match="exceeds"):
            await RoboticsActuator.command_device("arm-1", "x" * 5000, {})
