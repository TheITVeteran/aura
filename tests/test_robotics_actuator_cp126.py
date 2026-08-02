"""Robotics commands must enter the physical Reality Reach transaction path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from core.actuation.robotics_actuator import (
    MAX_COMMAND_DEADLINE_S,
    RoboticsActuationError,
    RoboticsActuator,
)
from core.embodiment.base_device import BaseHardwareDevice
from core.embodiment.hardware_manager import HardwareManager
from core.embodiment.reality_adapter import (
    HardwareCommandContract,
    HardwareRealityManifest,
)
from core.reality_reach import (
    ActuatorCapability,
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
    RealityReachService,
    Reversibility,
)
from core.reality_reach.transactions import RealityActuationCoordinator
from core.runtime.audit_chain import canonical_json, sha256_hex


def _manifest() -> HardwareRealityManifest:
    actuator = ChannelDeclaration(
        channel_id="test.arm.position.command",
        kind=ChannelKind.ACTUATOR,
        observable="arm_position",
        unit="meter",
        domain=NumericDomain(0.0, 1.0),
        coupling=CouplingClass.MECHANICAL,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT),
        evidence_level=EvidenceLevel.P1,
        owner="tests.arm",
        stale_after_s=5.0,
        coupling_validated=True,
    )
    observation = ChannelDeclaration(
        channel_id="test.arm.position.readback",
        kind=ChannelKind.SENSOR,
        observable="arm_position",
        unit="meter",
        domain=NumericDomain(0.0, 1.0),
        coupling=CouplingClass.MECHANICAL,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT),
        evidence_level=EvidenceLevel.P2,
        owner="tests.arm.encoder",
        resolution=0.001,
        sample_rate_hz=20.0,
        max_latency_s=0.05,
        stale_after_s=5.0,
        reference_id="test.arm.independent_encoder",
        coupling_validated=True,
    )
    capability = ActuatorCapability(
        adapter_id="test.arm.adapter",
        channel_id=actuator.channel_id,
        reversibility=Reversibility.REVERSIBLE,
        magnitude_domain=NumericDomain(0.0, 1.0),
        max_commands_per_minute=30,
        observation_channels=(observation.channel_id,),
        required_permissions=("hardware.motion",),
        failure_modes=("interlock", "encoder_mismatch"),
        watchdog_timeout_s=1.0,
        compensation_action="emergency_stop",
    )
    stop = HardwareCommandContract(
        command="emergency_stop",
        target=0.0,
        magnitude=0.0,
        tolerance=0.0,
        safe_envelope=NumericDomain(0.0, 0.0),
        allowed_parameters=("reason",),
        expected_effects=("arm_stopped",),
        rollback_command="emergency_stop",
    )
    move = HardwareCommandContract(
        command="move",
        target_parameter="target",
        magnitude_parameter="magnitude",
        tolerance=0.001,
        safe_envelope=NumericDomain(0.0, 1.0),
        allowed_parameters=("magnitude", "target"),
        expected_effects=("arm_position_observed",),
        abort_predicates=("encoder_mismatch",),
        rollback_command="emergency_stop",
    )
    return HardwareRealityManifest(
        adapter_id=capability.adapter_id,
        actuator=actuator,
        observation=observation,
        capability=capability,
        observation_field="position",
        command_transport_id="test.arm.motor_transport",
        readback_transport_id="test.arm.encoder_transport",
        commands=(move, stop),
        safe_state_command="emergency_stop",
        safe_state_target=0.0,
    )


class ArmDevice(BaseHardwareDevice):
    def __init__(self) -> None:
        super().__init__("arm-1", "Test arm", "test.arm")
        self.position = 0.0
        self.interlock = False
        self.execute_calls = 0
        self.interlock_checks = 0

    async def connect(self) -> bool:
        self.is_connected = True
        return True

    async def disconnect(self) -> bool:
        self.is_connected = False
        return True

    async def get_status(self) -> dict[str, Any]:
        return {
            "ok": self.is_connected,
            "position": self.position,
            "interlock": self.interlock,
        }

    async def check_interlocks(
        self,
        command: str,
        parameters: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.interlock_checks += 1
        body = {
            "command": command,
            "connected": self.is_connected,
            "interlock": bool(status.get("interlock")),
        }
        return {
            "ok": self.is_connected and not bool(status.get("interlock")),
            "reason": "interlock_engaged" if status.get("interlock") else "",
            "interlock_sha256": str(sha256_hex(canonical_json(body))),
        }

    async def execute_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        self.execute_calls += 1
        if command == "move":
            self.position = float(kwargs["target"])
        elif command == "emergency_stop":
            self.position = 0.0
        else:
            return {"ok": False, "error": "unknown_command"}
        return {"ok": True, "transport_completed": True}

    def reality_manifest(self) -> HardwareRealityManifest:
        return _manifest()


class RecordingCoordinator:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    async def execute(self, command: Any) -> dict[str, Any]:
        self.commands.append(command)
        return {"ok": True, "effect_verified": True}


class Container:
    services: dict[str, Any] = {}

    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        return cls.services.get(name, default)


@pytest_asyncio.fixture
async def hardware_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[
    HardwareManager,
    ArmDevice,
    RecordingCoordinator,
]:
    service = RealityReachService()
    manager = HardwareManager()
    manager.bind_reality_reach(service)
    device = ArmDevice()
    manager.register_device(device)
    await manager.start()
    coordinator = RecordingCoordinator()
    Container.services = {
        "hardware_manager": manager,
        "reality_actuation": coordinator,
    }
    monkeypatch.setattr("core.container.ServiceContainer", Container)
    return manager, device, coordinator


@pytest.mark.asyncio
async def test_command_compiles_into_reality_reach_not_generic_world_actuator(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, _device, coordinator = hardware_runtime

    result = await RoboticsActuator.command_device(
        "arm-1",
        "move",
        {"target": 0.5, "magnitude": 0.25},
        idempotency_key="test.robotics.command.1",
    )

    assert result["effect_verified"] is True
    command = coordinator.commands[0]
    assert command.adapter_id == "test.arm.adapter"
    assert command.channel_id == "test.arm.position.command"
    assert command.parameters["device_command"] == "move"
    assert command.parameters["device_id"] == "arm-1"
    assert command.inventory_sha256


@pytest.mark.asyncio
async def test_parameter_smuggling_and_undeclared_parameters_are_refused(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, _device, coordinator = hardware_runtime
    with pytest.raises(RoboticsActuationError, match="parameters_undeclared"):
        await RoboticsActuator.command_device(
            "arm-1",
            "move",
            {
                "target": 0.5,
                "magnitude": 0.25,
                "device_id": "arm-999",
                "command": "full_speed",
                "speed_override": 100,
            },
        )
    assert coordinator.commands == []


@pytest.mark.asyncio
async def test_unregistered_and_manifestless_devices_are_refused(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    manager, _device, coordinator = hardware_runtime
    with pytest.raises(RoboticsActuationError, match="not registered"):
        await RoboticsActuator.command_device("ghost-arm", "move", {})

    inventory_only = ArmDevice()
    inventory_only.device_id = "inventory-only"
    inventory_only.reality_manifest = lambda: None  # type: ignore[method-assign]
    manager.register_device(inventory_only)
    with pytest.raises(RoboticsActuationError, match="no registered explicit"):
        await RoboticsActuator.command_device("inventory-only", "move", {})
    assert coordinator.commands == []


@pytest.mark.asyncio
async def test_registry_and_boundary_validation_remain_fail_closed(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, _device, coordinator = hardware_runtime
    with pytest.raises(RoboticsActuationError, match="device_id is empty"):
        await RoboticsActuator.command_device("  ", "move", {})
    with pytest.raises(RoboticsActuationError, match="command is empty"):
        await RoboticsActuator.command_device("arm-1", "  ", {})
    with pytest.raises(RoboticsActuationError, match="exceeds"):
        await RoboticsActuator.command_device("arm-1", "x" * 5000, {})
    with pytest.raises(RoboticsActuationError, match="too many parameters"):
        await RoboticsActuator.command_device(
            "arm-1",
            "move",
            {f"k{i}": i for i in range(64)},
        )
    assert coordinator.commands == []


@pytest.mark.asyncio
async def test_stale_interlock_snapshot_cannot_cross_safe_execute(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, device, coordinator = hardware_runtime
    device.interlock = True
    with pytest.raises(RoboticsActuationError, match="interlock_engaged"):
        await RoboticsActuator.command_device(
            "arm-1",
            "move",
            {"target": 0.5, "magnitude": 0.25},
        )
    assert coordinator.commands == []


@pytest.mark.asyncio
async def test_deadline_is_bounded_in_typed_command(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, _device, coordinator = hardware_runtime
    before_ns = __import__("time").time_ns()
    await RoboticsActuator.command_device(
        "arm-1",
        "move",
        {"target": 0.5, "magnitude": 0.25},
        deadline_s=10_000.0,
    )
    delta_s = (coordinator.commands[0].deadline_ns - before_ns) / 1_000_000_000
    assert delta_s <= MAX_COMMAND_DEADLINE_S + 0.1


async def _executor(**kwargs: Any) -> dict[str, Any]:
    try:
        dispatched = dict(
            await kwargs["effect_handler"]({"will_receipt_id": "test.will.1"})
        )
    except Exception as exc:  # noqa: BLE001 - emulate ActionExecutor containment
        return {
            "ok": False,
            "effect_verified": False,
            "error": f"effect_handler_failed:{type(exc).__name__}",
        }
    verified = dict(await kwargs["effect_verifier"]({"result": dispatched}))
    return {**dispatched, **verified, "ok": verified.get("effect_verified") is True}


@pytest.mark.asyncio
async def test_real_coordinator_executes_through_safe_execute_and_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RealityReachService()
    manager = HardwareManager()
    manager.bind_reality_reach(service)
    device = ArmDevice()
    manager.register_device(device)
    await manager.start()
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    Container.services = {
        "hardware_manager": manager,
        "reality_actuation": coordinator,
    }
    monkeypatch.setattr("core.container.ServiceContainer", Container)

    first = await RoboticsActuator.command_device(
        "arm-1",
        "move",
        {"target": 0.75, "magnitude": 0.5},
        idempotency_key="test.robotics.durable.1",
    )
    assert first["effect_verified"] is True
    assert first["reality_reach_transaction"]["state"] == "effect_verified"
    assert device.execute_calls == 1
    assert device.interlock_checks >= 2
    assert device.position == 0.75


@pytest.mark.asyncio
async def test_interlock_change_between_prepare_and_transport_prevents_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RealityReachService()
    manager = HardwareManager()
    manager.bind_reality_reach(service)
    device = ArmDevice()
    original_check = device.check_interlocks
    checks = 0

    async def changing_interlock(
        command: str,
        parameters: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        nonlocal checks
        checks += 1
        result = await original_check(command, parameters, status)
        if checks == 1:
            device.interlock = True
        return result

    device.check_interlocks = changing_interlock  # type: ignore[method-assign]
    manager.register_device(device)
    await manager.start()
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    Container.services = {
        "hardware_manager": manager,
        "reality_actuation": coordinator,
    }
    monkeypatch.setattr("core.container.ServiceContainer", Container)

    result = await RoboticsActuator.command_device(
        "arm-1",
        "move",
        {"target": 0.75, "magnitude": 0.5},
        idempotency_key="test.robotics.interlock-race.1",
    )

    assert result["effect_verified"] is False
    assert result["reality_reach_transaction"]["state"] == "failed"
    assert device.execute_calls == 0


@pytest.mark.asyncio
async def test_emergency_stop_uses_the_same_governed_path(
    hardware_runtime: tuple[HardwareManager, ArmDevice, RecordingCoordinator],
) -> None:
    _manager, _device, coordinator = hardware_runtime
    await RoboticsActuator.emergency_stop("arm-1", reason="operator_stop")
    command = coordinator.commands[0]
    assert command.parameters["device_command"] == "emergency_stop"
    assert command.target == 0.0
