from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import pytest

from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.embodiment.ros2_connector import (
    ROS2Connector,
    ROS2ConnectorError,
    ROS2ManagedAdapter,
    ROS2NodeSpec,
    ROS2Transport,
    RosbridgeWebSocketTransport,
    ROSGraphSnapshot,
    ROSTopicSample,
    parse_ros2_node_manifest,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.live import RealityReachService
from core.reality_reach.middleware import RealityMiddlewareRuntime
from core.reality_reach.middleware_contracts import (
    ActionContext,
    ActionState,
    PhysicalEffectIndeterminateError,
)


def _manifest() -> dict[str, Any]:
    return {
        "node_id": "robot.orca",
        "device_id": "orca",
        "display_name": "Orca Research Robot",
        "telemetry": [
            {
                "endpoint_id": "robot.orca.battery",
                "channel_id": "robot.orca.battery_percent",
                "topic": "/robot/battery",
                "message_type": "sensor_msgs/msg/BatteryState",
                "value_pointer": "/percentage",
                "observable": "battery_charge",
                "unit": "percent",
                "minimum": 0.0,
                "maximum": 1.0,
                "resolution": 0.001,
                "sample_period_s": 0.1,
                "stale_after_s": 2.0,
                "qos": {"reliability": "reliable", "depth": 4},
            }
        ],
        "services": [
            {
                "endpoint_id": "robot.orca.inspect",
                "service": "/robot/inspect",
                "service_type": "orca_msgs/srv/Inspect",
                "read_only": True,
            },
        ],
        "actions": [
            {
                "endpoint_id": "robot.orca.navigate",
                "action": "/robot/navigate",
                "action_type": "nav2_msgs/action/NavigateToPose",
                "verification_service": "/robot/verify_pose",
                "verification_service_type": "orca_msgs/srv/VerifyPose",
                "verification_request": {"target": "reef"},
                "verification_pointer": "/at_goal",
                "verification_expected": True,
                "feedback_progress_pointer": "/progress",
                "reconciliation_service": "/robot/reconcile",
                "reconciliation_service_type": "orca_msgs/srv/ReconcileGoal",
                "reconciliation_state_pointer": "/state",
                "timeout_s": 5.0,
                "cancel_timeout_s": 0.5,
            },
            {
                "endpoint_id": "robot.orca.set_mode",
                "transport_kind": "service",
                "command_service": "/robot/set_mode",
                "command_service_type": "orca_msgs/srv/SetMode",
                "preemptible": False,
                "verification_service": "/robot/mode",
                "verification_service_type": "orca_msgs/srv/GetMode",
                "verification_request": {},
                "verification_pointer": "/mode",
                "verification_expected": "survey",
            },
        ],
    }


class FakeROS2Transport:
    transport_id = "rosbridge.fake"
    server_identity_sha256 = "sha256:" + "a" * 64
    identity_stable = True

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.subscribed: set[str] = set()
        self.service_calls: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.action_requests: list[tuple[str, str, dict[str, Any]]] = []
        self.action_events: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self.cancelled: list[str] = []
        self.mode = "survey"
        self.at_goal = True
        self.graph = ROSGraphSnapshot(
            topics={"/robot/battery": "sensor_msgs/msg/BatteryState"},
            services={
                "/robot/inspect": "orca_msgs/srv/Inspect",
                "/robot/set_mode": "orca_msgs/srv/SetMode",
                "/robot/mode": "orca_msgs/srv/GetMode",
                "/robot/verify_pose": "orca_msgs/srv/VerifyPose",
                "/robot/reconcile": "orca_msgs/srv/ReconcileGoal",
                "/robot/navigate/_action/send_goal": "action_transport",
                "/robot/navigate/_action/get_result": "action_transport",
                "/robot/navigate/_action/cancel_goal": "action_transport",
            },
        )

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def subscribe(self, spec) -> None:
        self.subscribed.add(spec.topic)

    async def unsubscribe(self, spec) -> None:
        self.subscribed.discard(spec.topic)

    async def latest(self, spec, *, timeout_s: float) -> ROSTopicSample:
        assert timeout_s > 0
        return ROSTopicSample(
            topic=spec.topic,
            message={"percentage": 0.73},
            captured_at_ns=time.time_ns(),
            source_sequence=7,
        )

    async def graph_snapshot(self, spec: ROS2NodeSpec) -> ROSGraphSnapshot:
        assert spec.device_id == "orca"
        return self.graph

    async def call_service(
        self,
        service: str,
        service_type: str,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        assert timeout_s > 0
        self.service_calls.append((service, service_type, dict(request), request_id))
        if service == "/robot/inspect":
            return {"healthy": True, "echo": request.get("probe")}
        if service == "/robot/set_mode":
            return {"accepted": True}
        if service == "/robot/mode":
            return {"mode": self.mode}
        if service == "/robot/verify_pose":
            return {"at_goal": self.at_goal}
        if service == "/robot/reconcile":
            return {"state": "succeeded"}
        raise LookupError(service)

    async def send_action_goal(self, spec, goal_id, request) -> None:
        self.action_requests.append((spec.action, goal_id, dict(request)))
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        queue.put_nowait(
            {
                "op": "action_feedback",
                "id": goal_id,
                "values": {"progress": 0.4, "phase": "moving"},
            }
        )
        queue.put_nowait(
            {
                "op": "action_result",
                "id": goal_id,
                "status": 4,
                "result": True,
                "values": {"distance": 12.5},
            }
        )
        self.action_events[goal_id] = queue

    async def next_action_event(self, goal_id: str, *, timeout_s: float) -> Mapping[str, Any]:
        return await asyncio.wait_for(self.action_events[goal_id].get(), timeout=timeout_s)

    async def cancel_action_goal(self, spec, goal_id, *, timeout_s: float) -> bool:
        assert timeout_s > 0
        self.cancelled.append(goal_id)
        return spec.preemptible


def _adapter(
    transport: FakeROS2Transport | None = None,
) -> tuple[ROS2ManagedAdapter, FakeROS2Transport]:
    transport = transport or FakeROS2Transport()
    spec = parse_ros2_node_manifest(_manifest())
    adapter = ROS2ManagedAdapter(
        transport,
        spec,
        installation_id="lab",
        initial_samples={
            "/robot/battery": ROSTopicSample(
                topic="/robot/battery",
                message={"percentage": 0.7},
                captured_at_ns=time.time_ns(),
                source_sequence=1,
            )
        },
    )
    return adapter, transport


def test_manifest_requires_verification_for_every_mutating_service() -> None:
    manifest = _manifest()
    manifest["services"][0]["read_only"] = False

    with pytest.raises(ValueError, match="must be declared as verified actions"):
        parse_ros2_node_manifest(manifest)


def test_manifest_rejects_non_mapping_verification_payload() -> None:
    manifest = _manifest()
    manifest["actions"][0]["verification_request"] = "not an object"

    with pytest.raises(ROS2ConnectorError, match="must_be_an_object"):
        parse_ros2_node_manifest(json.dumps(manifest))


@pytest.mark.asyncio
async def test_discovery_requires_exact_graph_and_live_measurement() -> None:
    transport = FakeROS2Transport()
    connector = ROS2Connector(
        transport,
        parse_ros2_node_manifest(_manifest()),
        installation_id="lab",
    )

    candidates = await connector.discover()

    assert len(candidates) == 1
    assert candidates[0].access == (
        AttachmentAccess.OBSERVE,
        AttachmentAccess.CONTROL,
    )
    assert candidates[0].persistent_identity is True
    assert candidates[0].metadata["verified_action_count"] == 2

    transport.graph = ROSGraphSnapshot(
        topics={"/robot/battery": "std_msgs/msg/String"},
        services=transport.graph.services,
    )
    assert await connector.discover() == ()


@pytest.mark.asyncio
async def test_observe_only_attachment_removes_mutating_endpoints() -> None:
    transport = FakeROS2Transport()
    connector = ROS2Connector(
        transport,
        parse_ros2_node_manifest(_manifest()),
        installation_id="lab",
    )
    candidate = (await connector.discover())[0]

    adapter = await connector.attach(candidate, (AttachmentAccess.OBSERVE,))
    lifecycle = adapter.lifecycle_declaration()

    assert [item.endpoint_id for item in lifecycle.services] == ["robot.orca.inspect"]
    assert lifecycle.actions == ()
    assert adapter.physical_identity_sha256 == candidate.identity_fingerprint


@pytest.mark.asyncio
async def test_telemetry_becomes_a_live_provenance_reading() -> None:
    adapter, _transport = _adapter()

    readings = await adapter.read_telemetry("robot.orca.battery")

    assert readings[0].value == 0.73
    assert readings[0].channel_id == "robot.orca.battery_percent"
    assert readings[0].source_sequence == 7
    assert readings[0].source_quality == "rosbridge_live_topic"
    assert adapter.read() == readings


@pytest.mark.asyncio
async def test_service_command_runs_as_action_and_requires_effect_readback() -> None:
    adapter, transport = _adapter()
    feedback: list[tuple[float, dict[str, Any]]] = []

    async def record(progress: float, payload: Mapping[str, Any]) -> None:
        feedback.append((progress, dict(payload)))

    result = await adapter.execute_action(
        "robot.orca.set_mode",
        {"mode": "survey"},
        ActionContext("goal-mode-1", asyncio.Event(), record),
    )

    assert result["effect_verified"] is True
    assert str(result["effect_receipt_sha256"]).startswith("sha256:")
    assert feedback == [
        (0.25, {"phase": "command_submitted"}),
        (1.0, {"phase": "effect_verified"}),
    ]
    assert [item[0] for item in transport.service_calls] == [
        "/robot/set_mode",
        "/robot/mode",
    ]

    transport.mode = "standby"
    with pytest.raises(
        PhysicalEffectIndeterminateError,
        match="effect_verification_failed",
    ):
        await adapter.execute_action(
            "robot.orca.set_mode",
            {"mode": "survey"},
            ActionContext("goal-mode-2", asyncio.Event(), record),
        )

    assert (
        await adapter.cancel_action(
            "robot.orca.set_mode",
            "goal-mode-2",
            "too_late",
        )
        is False
    )


@pytest.mark.asyncio
async def test_action_streams_feedback_and_requires_independent_effect_readback() -> None:
    adapter, transport = _adapter()
    feedback: list[tuple[float, dict[str, Any]]] = []

    async def record(progress: float, payload: Mapping[str, Any]) -> None:
        feedback.append((progress, dict(payload)))

    context = ActionContext("goal-orca-1", asyncio.Event(), record)
    result = await adapter.execute_action(
        "robot.orca.navigate",
        {"pose": "reef"},
        context,
    )

    assert feedback[0] == (0.4, {"progress": 0.4, "phase": "moving"})
    assert feedback[-1] == (1.0, {"phase": "effect_verified"})
    assert result["effect_verified"] is True
    assert result["result"] == {"distance": 12.5}
    assert transport.service_calls[-1][0] == "/robot/verify_pose"

    transport.at_goal = False
    with pytest.raises(
        PhysicalEffectIndeterminateError,
        match="effect_verification_failed",
    ):
        await adapter.execute_action(
            "robot.orca.navigate",
            {"pose": "reef"},
            ActionContext("goal-orca-2", asyncio.Event(), record),
        )


@pytest.mark.asyncio
async def test_restart_reconciliation_rechecks_physical_effect() -> None:
    adapter, transport = _adapter()
    record = {"goal_id": "goal-orca-restart"}

    result = await adapter.reconcile_action("robot.orca.navigate", record)

    assert result["state"] == ActionState.SUCCEEDED.value
    assert result["result"]["effect_verified"] is True

    transport.at_goal = False
    result = await adapter.reconcile_action("robot.orca.navigate", record)
    assert result["state"] == ActionState.INDETERMINATE.value


@pytest.mark.asyncio
async def test_action_cancel_is_forwarded_to_robot() -> None:
    adapter, transport = _adapter()

    assert (
        await adapter.cancel_action(
            "robot.orca.navigate",
            "goal-cancel",
            "user_request",
        )
        is True
    )
    assert transport.cancelled == ["goal-cancel"]


@pytest.mark.asyncio
async def test_adapter_runs_through_real_managed_middleware_contract(tmp_path) -> None:
    adapter, _transport = _adapter()
    service = RealityReachService(session_id="ros2-integration")
    service.register_adapter(adapter)
    runtime = RealityMiddlewareRuntime(
        service,
        state_path=tmp_path / "ros2-middleware.json",
    )
    await runtime.start()
    await runtime.register_adapter(adapter)

    receipt = await runtime.call_service(
        "robot.orca.inspect",
        {"probe": "health"},
        request_id="request.ros2.inspect",
    )
    accepted = await runtime.start_action(
        "robot.orca.set_mode",
        {"mode": "survey"},
        goal_id="goal.ros2.mode",
    )
    final = await runtime.wait_action(
        accepted["goal_id"],
        timeout_s=1.0,
        poll_interval_s=0.01,
    )

    assert receipt.ok is True
    assert receipt.response["healthy"] is True
    assert final["state"] == "succeeded"
    assert final["result"]["effect_verified"] is True
    assert runtime.node_status("robot.orca")["state"] == "active"
    await runtime.shutdown()


def test_websocket_transport_requires_tls_pin_or_explicit_plaintext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_ROSBRIDGE_INSTALLATION_ID", "lab")
    monkeypatch.setenv("AURA_ROSBRIDGE_URL", "wss://robot.local:9090")
    monkeypatch.delenv("AURA_ROSBRIDGE_SERVER_CERT_SHA256", raising=False)

    with pytest.raises(ROS2ConnectorError, match="certificate_pin_required"):
        RosbridgeWebSocketTransport()

    monkeypatch.setenv("AURA_ROSBRIDGE_URL", "ws://robot.local:9090")
    monkeypatch.delenv("AURA_ROSBRIDGE_ALLOW_PLAINTEXT", raising=False)
    with pytest.raises(ROS2ConnectorError, match="plaintext_requires_explicit_opt_in"):
        RosbridgeWebSocketTransport()


def test_boot_catalog_reports_ros2_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_ROSBRIDGE_INSTALLATION_ID", "lab")
    monkeypatch.setenv("AURA_ROSBRIDGE_URL", "ws://robot.local:9090")
    monkeypatch.setenv("AURA_ROSBRIDGE_ALLOW_PLAINTEXT", "1")
    monkeypatch.setenv(
        "AURA_ROSBRIDGE_NODE_MANIFEST_JSON",
        json.dumps(_manifest()),
    )

    status = build_configured_reality_connector_catalog().status()
    ros2 = next(item for item in status["connectors"] if item["connector_id"] == "ros2.rosbridge")

    assert ros2["configured"] is True
    assert ros2["state"] == "ready"


@pytest.mark.asyncio
async def test_rosbridge_terminal_result_is_visible_to_concurrent_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_ROSBRIDGE_INSTALLATION_ID", "lab")
    monkeypatch.setenv("AURA_ROSBRIDGE_URL", "ws://robot.local:9090")
    monkeypatch.setenv("AURA_ROSBRIDGE_ALLOW_PLAINTEXT", "1")
    transport = RosbridgeWebSocketTransport()
    loop = asyncio.get_running_loop()
    transport._action_queues["goal-shared"] = asyncio.Queue(maxsize=4)
    transport._action_results["goal-shared"] = loop.create_future()

    first = asyncio.create_task(transport.next_action_event("goal-shared", timeout_s=0.5))
    second = asyncio.create_task(transport.next_action_event("goal-shared", timeout_s=0.5))
    await asyncio.sleep(0)
    transport._dispatch(
        {
            "op": "action_result",
            "id": "goal-shared",
            "status": 5,
            "result": True,
            "values": {},
        }
    )

    observed = await asyncio.gather(first, second)
    assert [item["status"] for item in observed] == [5, 5]


@pytest.mark.asyncio
async def test_real_websocket_protocol_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import websockets

    observed: list[dict[str, Any]] = []
    held_goals: set[str] = set()
    service_types = {
        "/robot/inspect": "orca_msgs/srv/Inspect",
        "/robot/set_mode": "orca_msgs/srv/SetMode",
        "/robot/mode": "orca_msgs/srv/GetMode",
        "/robot/verify_pose": "orca_msgs/srv/VerifyPose",
        "/robot/reconcile": "orca_msgs/srv/ReconcileGoal",
    }
    action_services = {
        "/robot/navigate/_action/send_goal",
        "/robot/navigate/_action/get_result",
        "/robot/navigate/_action/cancel_goal",
    }

    async def handler(socket) -> None:
        async for raw in socket:
            body = json.loads(raw)
            observed.append(body)
            operation = body["op"]
            if operation == "subscribe":
                await socket.send(
                    json.dumps(
                        {
                            "op": "publish",
                            "topic": body["topic"],
                            "msg": {"percentage": 0.82},
                        }
                    )
                )
            elif operation == "call_service":
                service = body["service"]
                if service == "/rosapi/topics":
                    values = {
                        "topics": ["/robot/battery"],
                        "types": ["sensor_msgs/msg/BatteryState"],
                    }
                elif service == "/rosapi/services":
                    values = {
                        "services": sorted(set(service_types) | action_services)
                    }
                elif service == "/rosapi/service_type":
                    values = {"type": service_types[body["args"]["service"]]}
                elif service == "/robot/inspect":
                    values = {"healthy": True}
                else:
                    values = {"at_goal": True}
                await socket.send(
                    json.dumps(
                        {
                            "op": "service_response",
                            "id": body["id"],
                            "service": service,
                            "result": True,
                            "values": values,
                        }
                    )
                )
            elif operation == "send_action_goal":
                goal_id = body["id"]
                if body["args"].get("hold"):
                    held_goals.add(goal_id)
                    continue
                await socket.send(
                    json.dumps(
                        {
                            "op": "action_feedback",
                            "id": goal_id,
                            "action": body["action"],
                            "values": {"progress": 0.5},
                        }
                    )
                )
                await socket.send(
                    json.dumps(
                        {
                            "op": "action_result",
                            "id": goal_id,
                            "action": body["action"],
                            "status": 4,
                            "result": True,
                            "values": {"distance": 3.0},
                        }
                    )
                )
            elif operation == "cancel_action_goal":
                assert body["id"] in held_goals
                await socket.send(
                    json.dumps(
                        {
                            "op": "action_result",
                            "id": body["id"],
                            "action": body["action"],
                            "status": 5,
                            "result": True,
                            "values": {},
                        }
                    )
                )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("AURA_ROSBRIDGE_INSTALLATION_ID", "loopback")
        monkeypatch.setenv("AURA_ROSBRIDGE_URL", f"ws://127.0.0.1:{port}")
        monkeypatch.setenv("AURA_ROSBRIDGE_ALLOW_PLAINTEXT", "1")
        transport = RosbridgeWebSocketTransport()
        spec = parse_ros2_node_manifest(_manifest())

        graph = await transport.graph_snapshot(spec)
        sample = await transport.latest(spec.telemetry[0], timeout_s=0.5)
        service = await transport.call_service(
            "/robot/inspect",
            "orca_msgs/srv/Inspect",
            {},
            timeout_s=0.5,
            request_id="loopback.inspect",
        )
        action = spec.actions[0]
        await transport.send_action_goal(action, "goal-loopback", {"pose": "reef"})
        feedback = await transport.next_action_event("goal-loopback", timeout_s=0.5)
        result = await transport.next_action_event("goal-loopback", timeout_s=0.5)
        await transport.send_action_goal(action, "goal-cancel", {"hold": True})
        cancelled = await transport.cancel_action_goal(
            action,
            "goal-cancel",
            timeout_s=0.5,
        )
        await transport.close()

    assert graph.topics["/robot/battery"] == "sensor_msgs/msg/BatteryState"
    assert graph.services["/robot/inspect"] == "orca_msgs/srv/Inspect"
    assert sample.message["percentage"] == 0.82
    assert service == {"healthy": True}
    assert feedback["op"] == "action_feedback"
    assert result["status"] == 4
    assert cancelled is True
    subscription = next(item for item in observed if item["op"] == "subscribe")
    assert subscription["qos"]["reliability"] == "reliable"
    assert all("authorization" not in json.dumps(item).lower() for item in observed)


def test_fake_transport_satisfies_runtime_protocol() -> None:
    assert isinstance(FakeROS2Transport(), ROS2Transport)
