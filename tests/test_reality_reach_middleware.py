from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from core.bus.qos import Durability, QosBus, QosProfile, Reliability
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, RealityReachService
from core.reality_reach.middleware import (
    ActionContext,
    ActionEndpoint,
    ActionRecord,
    ActionState,
    ManagedAdapterDeclaration,
    PhysicalEffectIndeterminateError,
    RealityMiddlewareError,
    RealityMiddlewareRuntime,
    ServiceEndpoint,
    TelemetryEndpoint,
    TelemetryMode,
)
from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.audit_chain import canonical_json, sha256_hex

IDENTITY = "sha256:" + "a" * 64
EFFECT = "sha256:" + "b" * 64


@pytest.mark.asyncio
async def test_action_cancellation_wait_is_bounded_and_observable() -> None:
    event = asyncio.Event()

    async def feedback(_progress: float, _payload: dict[str, Any]) -> None:
        return None

    context = ActionContext("goal.cancel", event, feedback)
    assert await context.wait_cancelled(timeout_s=0.001) is False
    event.set()
    assert await context.wait_cancelled(timeout_s=0.1) is True


def _channel() -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id="test.sensor.temperature",
        kind=ChannelKind.SENSOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(-50.0, 150.0),
        coupling=CouplingClass.THERMAL,
        reality_layers=(RealityLayer.DIRECT,),
        evidence_level=EvidenceLevel.P2,
        owner="test",
        resolution=0.1,
        sample_rate_hz=10.0,
        max_latency_s=0.2,
        stale_after_s=2.0,
        reference_id="test.reference",
        coupling_validated=True,
    )


class ManagedAdapter:
    adapter_id = "test.adapter"
    physical_identity_sha256 = IDENTITY

    def __init__(self, *, action_delay: float = 0.01) -> None:
        self.declared_identity = IDENTITY
        self.configured = False
        self.active = False
        self.action_delay = action_delay
        self.service_calls = 0
        self.action_calls = 0
        self.cancel_calls = 0
        self.reconcile_calls = 0
        self.cancel_acknowledged = True
        self.effect_verified = True
        self.reconcile_state = ActionState.SUCCEEDED
        self.telemetry_reads = 0

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (_channel(),)

    def read(self) -> tuple[ChannelReading, ...]:
        return ()

    def lifecycle_declaration(self) -> ManagedAdapterDeclaration:
        return ManagedAdapterDeclaration(
            node_id="test.node",
            adapter_id=self.adapter_id,
            adapter_identity_sha256=self.declared_identity,
            telemetry=(
                TelemetryEndpoint(
                    endpoint_id="test.telemetry",
                    channel_ids=("test.sensor.temperature",),
                    qos=QosProfile(
                        reliability=Reliability.RELIABLE,
                        durability=Durability.TRANSIENT_LOCAL,
                        depth=4,
                        deadline_s=0.2,
                        liveliness_lease_s=0.3,
                    ),
                ),
            ),
            services=(ServiceEndpoint("test.inspect", timeout_s=0.1),),
            actions=(
                ActionEndpoint(
                    "test.move",
                    timeout_s=0.5,
                    cancel_timeout_s=0.1,
                    feedback_depth=3,
                ),
            ),
            transition_timeout_s=0.2,
        )

    async def on_configure(self) -> bool:
        self.configured = True
        return True

    async def on_activate(self) -> bool:
        assert self.configured
        self.active = True
        return True

    async def on_deactivate(self) -> bool:
        self.active = False
        return True

    async def on_cleanup(self) -> bool:
        self.configured = False
        return True

    async def on_shutdown(self) -> bool:
        self.active = False
        return True

    async def on_error(self) -> bool:
        self.active = False
        return True

    async def read_telemetry(
        self, endpoint_id: str
    ) -> dict[str, Any] | tuple[ChannelReading, ...]:
        assert endpoint_id == "test.telemetry"
        self.telemetry_reads += 1
        return {"temperature": 21.0, "unit": "celsius"}

    async def handle_service(self, endpoint_id: str, request: dict[str, Any]) -> dict[str, Any]:
        assert endpoint_id == "test.inspect"
        self.service_calls += 1
        if request.get("sleep"):
            await asyncio.sleep(float(request["sleep"]))
        return {"echo": request.get("value")}

    async def execute_action(self, endpoint_id, request, context):
        assert endpoint_id == "test.move"
        self.action_calls += 1
        await context.publish_feedback(0.25, {"phase": "started"})
        await asyncio.sleep(self.action_delay)
        if context.cancel_requested:
            return {"cancelled": True}
        await context.publish_feedback(1.0, {"phase": "verified"})
        return {
            "effect_verified": self.effect_verified,
            "effect_receipt_sha256": EFFECT,
            "position": request.get("position"),
        }

    async def cancel_action(self, endpoint_id: str, goal_id: str, reason: str) -> bool:
        assert endpoint_id == "test.move"
        assert goal_id
        assert reason
        self.cancel_calls += 1
        return self.cancel_acknowledged

    async def reconcile_action(self, endpoint_id: str, record: dict[str, Any]) -> dict[str, Any]:
        assert endpoint_id == "test.move"
        assert record["recovery_required"] is True
        self.reconcile_calls += 1
        return {
            "state": self.reconcile_state.value,
            "result": {
                "effect_verified": True,
                "effect_receipt_sha256": EFFECT,
                "reconciled": True,
            },
        }


async def _runtime(
    tmp_path: Path,
    *,
    adapter: ManagedAdapter | None = None,
    service: RealityReachService | None = None,
    state_name: str = "middleware.json",
) -> tuple[RealityMiddlewareRuntime, ManagedAdapter, QosBus, RealityReachService]:
    adapter = adapter or ManagedAdapter()
    service = service or RealityReachService(session_id=f"session-{time.time_ns()}")
    service.register_adapter(adapter)
    qos = QosBus()
    runtime = RealityMiddlewareRuntime(
        service,
        state_path=tmp_path / state_name,
        qos_bus=qos,
    )
    await runtime.start()
    await runtime.register_adapter(adapter)
    return runtime, adapter, qos, service


@pytest.mark.asyncio
async def test_managed_adapter_requires_exact_live_identity(tmp_path: Path) -> None:
    service = RealityReachService(session_id="identity-session")
    adapter = ManagedAdapter()
    service.register_adapter(adapter)
    adapter.declared_identity = "sha256:" + "c" * 64
    runtime = RealityMiddlewareRuntime(service, state_path=tmp_path / "state.json")
    await runtime.start()

    with pytest.raises(RealityMiddlewareError, match="identity differs"):
        await runtime.register_adapter(adapter)


@pytest.mark.asyncio
async def test_configure_activate_deactivate_and_shutdown_are_distinct(tmp_path: Path) -> None:
    runtime, adapter, _qos, _service = await _runtime(tmp_path)
    assert adapter.configured is True
    assert adapter.active is True
    assert runtime.node_status("test.node")["state"] == "active"

    assert await runtime.deactivate_node("test.node") is True
    assert adapter.active is False
    assert adapter.configured is True
    with pytest.raises(RealityMiddlewareError, match="service node is not active"):
        await runtime.call_service("test.inspect", {})

    assert await runtime.activate_node("test.node") is True
    await runtime.shutdown()
    assert runtime.is_alive() is False
    assert adapter.active is False


@pytest.mark.asyncio
async def test_push_telemetry_uses_declared_qos_and_retains_for_late_joiners(
    tmp_path: Path,
) -> None:
    runtime, _adapter, qos, _service = await _runtime(tmp_path)
    receipt = await runtime.publish_telemetry(
        "test.telemetry", {"temperature": 22.5, "unit": "celsius"}
    )
    assert receipt["published"] is True
    retained = qos.retained(
        "reality.telemetry.test.telemetry",
        profile=QosProfile(durability=Durability.TRANSIENT_LOCAL, depth=1),
    )
    assert retained[-1].data["payload"]["temperature"] == 22.5
    report = qos.report()["topics"]["reality.telemetry.test.telemetry"]
    assert report["profile"]["reliability"] == "reliable"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_pull_telemetry_runs_only_while_node_is_active(tmp_path: Path) -> None:
    class PullAdapter(ManagedAdapter):
        def lifecycle_declaration(self) -> ManagedAdapterDeclaration:
            declaration = super().lifecycle_declaration()
            return ManagedAdapterDeclaration(
                node_id=declaration.node_id,
                adapter_id=declaration.adapter_id,
                adapter_identity_sha256=declaration.adapter_identity_sha256,
                telemetry=(
                    TelemetryEndpoint(
                        endpoint_id="test.telemetry",
                        channel_ids=("test.sensor.temperature",),
                        qos=declaration.telemetry[0].qos,
                        mode=TelemetryMode.PULL,
                        sample_period_s=0.03,
                        sample_timeout_s=0.02,
                    ),
                ),
                services=declaration.services,
                actions=declaration.actions,
            )

    adapter = PullAdapter()
    runtime, adapter, _qos, _service = await _runtime(tmp_path, adapter=adapter)
    await asyncio.sleep(0.08)
    assert adapter.telemetry_reads >= 2
    await runtime.deactivate_node("test.node")
    stopped_at = adapter.telemetry_reads
    await asyncio.sleep(0.06)
    assert adapter.telemetry_reads == stopped_at
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_start_activates_pull_tasks_registered_during_staging(tmp_path: Path) -> None:
    class StagedPullAdapter(ManagedAdapter):
        def lifecycle_declaration(self) -> ManagedAdapterDeclaration:
            declaration = super().lifecycle_declaration()
            return ManagedAdapterDeclaration(
                node_id=declaration.node_id,
                adapter_id=declaration.adapter_id,
                adapter_identity_sha256=declaration.adapter_identity_sha256,
                telemetry=(
                    TelemetryEndpoint(
                        endpoint_id="test.telemetry",
                        channel_ids=("test.sensor.temperature",),
                        qos=declaration.telemetry[0].qos,
                        mode=TelemetryMode.PULL,
                        sample_period_s=0.03,
                        sample_timeout_s=0.02,
                    ),
                ),
                services=declaration.services,
                actions=declaration.actions,
            )

    adapter = StagedPullAdapter()
    service = RealityReachService(session_id="staged-session")
    service.register_adapter(adapter)
    runtime = RealityMiddlewareRuntime(service, state_path=tmp_path / "staged.json")
    await runtime.register_adapter(adapter)
    assert adapter.telemetry_reads == 0

    await runtime.start()
    await asyncio.sleep(0.05)

    assert adapter.telemetry_reads >= 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_service_request_is_bounded_timed_and_idempotent(tmp_path: Path) -> None:
    runtime, adapter, _qos, _service = await _runtime(tmp_path)
    first = await runtime.call_service(
        "test.inspect", {"value": 7}, request_id="request.same"
    )
    second = await runtime.call_service(
        "test.inspect", {"value": 7}, request_id="request.same"
    )
    assert first.to_dict() == second.to_dict()
    assert adapter.service_calls == 1

    with pytest.raises(RealityMiddlewareError, match="different content"):
        await runtime.call_service(
            "test.inspect", {"value": 8}, request_id="request.same"
        )

    timed = await runtime.call_service(
        "test.inspect",
        {"sleep": 0.05},
        request_id="request.timeout",
        timeout_s=0.01,
    )
    assert timed.ok is False
    assert timed.error.startswith("service_deadline_exceeded")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_service_idempotency_receipt_survives_restart(tmp_path: Path) -> None:
    runtime, adapter, _qos, _service = await _runtime(tmp_path)
    original = await runtime.call_service(
        "test.inspect", {"value": 17}, request_id="request.restart"
    )
    assert adapter.service_calls == 1
    await runtime.shutdown()

    replacement = ManagedAdapter()
    resumed, replacement, _qos, _service = await _runtime(
        tmp_path,
        adapter=replacement,
    )
    replay = await resumed.call_service(
        "test.inspect", {"value": 17}, request_id="request.restart"
    )
    assert replay.to_dict() == original.to_dict()
    assert replacement.service_calls == 0
    await resumed.shutdown()


@pytest.mark.asyncio
async def test_concurrent_duplicate_service_requests_are_singleflight(tmp_path: Path) -> None:
    runtime, adapter, _qos, _service = await _runtime(tmp_path)
    first, second = await asyncio.gather(
        runtime.call_service(
            "test.inspect",
            {"value": 23, "sleep": 0.03},
            request_id="request.concurrent",
        ),
        runtime.call_service(
            "test.inspect",
            {"value": 23, "sleep": 0.03},
            request_id="request.concurrent",
        ),
    )
    assert first.to_dict() == second.to_dict()
    assert adapter.service_calls == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_mutating_service_is_refused_in_favor_of_action_lane(tmp_path: Path) -> None:
    class MutatingAdapter(ManagedAdapter):
        def lifecycle_declaration(self) -> ManagedAdapterDeclaration:
            declaration = super().lifecycle_declaration()
            return ManagedAdapterDeclaration(
                node_id=declaration.node_id,
                adapter_id=declaration.adapter_id,
                adapter_identity_sha256=declaration.adapter_identity_sha256,
                telemetry=declaration.telemetry,
                services=(ServiceEndpoint("test.inspect", read_only=False),),
                actions=declaration.actions,
            )

    runtime, _adapter, _qos, _service = await _runtime(
        tmp_path, adapter=MutatingAdapter()
    )
    with pytest.raises(RealityMiddlewareError, match="declared as actions"):
        await runtime.call_service("test.inspect", {})
    await runtime.shutdown()


async def _wait_terminal(runtime: RealityMiddlewareRuntime, goal_id: str) -> dict[str, Any]:
    return await runtime.wait_action(goal_id, timeout_s=1.0, poll_interval_s=0.01)


@pytest.mark.asyncio
async def test_action_returns_immediate_handle_feedback_and_verified_result(
    tmp_path: Path,
) -> None:
    runtime, adapter, _qos, _service = await _runtime(tmp_path)
    accepted = await runtime.start_action(
        "test.move", {"position": 4}, goal_id="goal.verified"
    )
    assert accepted["state"] == "accepted"
    final = await _wait_terminal(runtime, "goal.verified")
    assert final["state"] == "succeeded"
    assert [item["progress"] for item in final["feedback"]] == [0.25, 1.0]
    assert runtime.action_feedback("goal.verified", after_sequence=1) == [
        final["feedback"][1]
    ]
    assert final["result"]["position"] == 4
    assert adapter.action_calls == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_wait_action_is_bounded_and_unknown_endpoints_are_typed(
    tmp_path: Path,
) -> None:
    runtime, _adapter, _qos, _service = await _runtime(
        tmp_path,
        adapter=ManagedAdapter(action_delay=1.0),
    )
    await runtime.start_action("test.move", {}, goal_id="goal.wait")
    with pytest.raises(TimeoutError, match="did not finish"):
        await runtime.wait_action("goal.wait", timeout_s=0.01)
    with pytest.raises(LookupError):
        await runtime.call_service("test.unknown", {})
    with pytest.raises(LookupError):
        await runtime.start_action("test.unknown", {})
    await runtime.cancel_action("goal.wait", reason="test_cleanup")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_action_cannot_succeed_without_effect_verification(tmp_path: Path) -> None:
    adapter = ManagedAdapter()
    adapter.effect_verified = False
    runtime, _adapter, _qos, _service = await _runtime(tmp_path, adapter=adapter)
    await runtime.start_action("test.move", {}, goal_id="goal.unverified")
    final = await _wait_terminal(runtime, "goal.unverified")
    assert final["state"] == "aborted"
    assert "effect_verified=true" in final["error"]
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unproven_physical_effect_requires_reconciliation(tmp_path: Path) -> None:
    class IndeterminateAdapter(ManagedAdapter):
        async def execute_action(self, endpoint_id, request, context):
            del endpoint_id, request, context
            raise PhysicalEffectIndeterminateError(
                "command accepted but independent readback timed out"
            )

    runtime, _adapter, _qos, _service = await _runtime(
        tmp_path,
        adapter=IndeterminateAdapter(),
    )
    await runtime.start_action("test.move", {}, goal_id="goal.indeterminate")

    final = await _wait_terminal(runtime, "goal.indeterminate")

    assert final["state"] == "indeterminate"
    assert final["recovery_required"] is True
    assert "independent readback timed out" in final["error"]
    with pytest.raises(RealityMiddlewareError, match="requiring reconciliation"):
        await runtime.start_action("test.move", {}, goal_id="goal.blocked")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_concurrent_action_admission_allows_one_effect(tmp_path: Path) -> None:
    runtime, adapter, _qos, _service = await _runtime(
        tmp_path,
        adapter=ManagedAdapter(action_delay=1.0),
    )
    outcomes = await asyncio.gather(
        runtime.start_action("test.move", {}, goal_id="goal.race-a"),
        runtime.start_action("test.move", {}, goal_id="goal.race-b"),
        return_exceptions=True,
    )
    accepted = [item for item in outcomes if isinstance(item, dict)]
    refused = [item for item in outcomes if isinstance(item, RealityMiddlewareError)]
    assert len(accepted) == 1
    assert len(refused) == 1
    await asyncio.sleep(0.02)
    assert adapter.action_calls == 1
    await runtime.cancel_action(accepted[0]["goal_id"], reason="test_cleanup")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_cancel_and_safe_preemption_are_explicit(tmp_path: Path) -> None:
    adapter = ManagedAdapter(action_delay=1.0)
    runtime, _adapter, _qos, _service = await _runtime(tmp_path, adapter=adapter)
    await runtime.start_action("test.move", {}, goal_id="goal.first")
    await asyncio.sleep(0.02)
    cancelled = await runtime.cancel_action("goal.first", reason="operator_cancel")
    assert cancelled["state"] == "cancelled"

    await runtime.start_action("test.move", {}, goal_id="goal.old")
    await asyncio.sleep(0.02)
    replacement = await runtime.start_action(
        "test.move", {}, goal_id="goal.new", preempt=True
    )
    assert replacement["state"] == "accepted"
    assert runtime.action_status("goal.old")["state"] == "preempted"
    await runtime.cancel_action("goal.new", reason="test_cleanup")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unconfirmed_preemption_blocks_replacement_effect(tmp_path: Path) -> None:
    adapter = ManagedAdapter(action_delay=1.0)
    adapter.cancel_acknowledged = False
    runtime, _adapter, _qos, _service = await _runtime(tmp_path, adapter=adapter)
    await runtime.start_action("test.move", {}, goal_id="goal.uncertain")
    await asyncio.sleep(0.02)

    with pytest.raises(RealityMiddlewareError, match="safe preemption"):
        await runtime.start_action(
            "test.move", {}, goal_id="goal.must-not-run", preempt=True
        )
    assert runtime.action_status("goal.uncertain")["state"] == "indeterminate"
    assert adapter.action_calls == 1
    assert runtime.is_ready() is False
    with pytest.raises(RealityMiddlewareError, match="requiring reconciliation"):
        await runtime.start_action("test.move", {}, goal_id="goal.after-uncertain")
    await runtime.shutdown()


def _write_interrupted_state(path: Path, session_id: str) -> None:
    now = time.time_ns()
    record = ActionRecord(
        goal_id="goal.restart",
        endpoint_id="test.move",
        node_id="test.node",
        adapter_id="test.adapter",
        adapter_identity_sha256=IDENTITY,
        request={"position": 9},
        request_sha256=str(sha256_hex(canonical_json({"position": 9}))),
        state=ActionState.EXECUTING,
        created_at_ns=now,
        updated_at_ns=now,
        deadline_at_ns=now + 10_000_000_000,
    )
    payload = {
        "service_session_id": session_id,
        "saved_at_ns": now,
        "desired_active": {"test.node": IDENTITY},
        "actions": [record.to_dict()],
    }
    payload["state_sha256"] = str(sha256_hex(canonical_json(payload)))
    atomic_write_json(
        path,
        payload,
        schema_version=1,
        schema_name="aura.reality_reach.middleware_state.v1",
    )


@pytest.mark.asyncio
async def test_restart_reconciles_without_reexecuting_effect(tmp_path: Path) -> None:
    state_path = tmp_path / "restart.json"
    _write_interrupted_state(state_path, "prior-session")
    adapter = ManagedAdapter()
    runtime, adapter, _qos, _service = await _runtime(
        tmp_path, adapter=adapter, state_name="restart.json"
    )
    recovered = runtime.action_status("goal.restart")
    assert recovered["state"] == "succeeded"
    assert recovered["result"]["reconciled"] is True
    assert adapter.reconcile_calls == 1
    assert adapter.action_calls == 0
    await runtime.shutdown()


def test_tampered_restart_state_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "tampered.json"
    _write_interrupted_state(path, "prior-session")
    envelope = read_json_envelope(path)
    envelope["payload"]["actions"][0]["request"]["position"] = 999
    path.write_text(json.dumps(envelope), encoding="utf-8")
    service = RealityReachService(session_id="new-session")

    with pytest.raises(RealityMiddlewareError, match="integrity check failed"):
        RealityMiddlewareRuntime(service, state_path=path)


def test_endpoint_contracts_are_bounded_and_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        ManagedAdapterDeclaration(
            node_id="test.node",
            adapter_id="test.adapter",
            adapter_identity_sha256=IDENTITY,
            services=(ServiceEndpoint("test.same"),),
            actions=(ActionEndpoint("test.same"),),
        )
    with pytest.raises(ValueError, match="must not exceed"):
        TelemetryEndpoint(
            endpoint_id="test.pull",
            channel_ids=("test.sensor",),
            qos=QosProfile(),
            mode=TelemetryMode.PULL,
            sample_period_s=0.1,
            sample_timeout_s=0.2,
        )
