from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.reality_reach.event_flow import RealityEventFlowError, RealityEventFlowRuntime
from core.reality_reach.event_flow_contracts import (
    BackpressurePolicy,
    DeliveryState,
    FlowContractError,
    FlowEdge,
    FlowGraph,
    FlowNode,
    FlowNodeKind,
    FlowPort,
    FlowValueType,
)


def _port(name: str, kind: FlowValueType = FlowValueType.INTEGER) -> FlowPort:
    return FlowPort(name, kind)


def _linear_graph(
    *,
    graph_id: str = "demo-flow",
    revision: int = 1,
    backpressure: BackpressurePolicy = BackpressurePolicy.REJECT,
    queue_depth: int = 4,
    processor_kind: FlowNodeKind = FlowNodeKind.PROCESSOR,
    processor_config: dict[str, Any] | None = None,
    max_attempts: int = 3,
    circuit_breaker_failures: int = 5,
) -> FlowGraph:
    value_type = (
        FlowValueType.OBJECT
        if processor_kind in {FlowNodeKind.SERVICE, FlowNodeKind.ACTION}
        else FlowValueType.INTEGER
    )
    return FlowGraph(
        graph_id=graph_id,
        revision=revision,
        nodes=(
            FlowNode(
                "input",
                FlowNodeKind.SOURCE,
                outputs=(_port("out", value_type),),
            ),
            FlowNode(
                "work",
                processor_kind,
                inputs=(_port("in", value_type),),
                outputs=(_port("out", value_type),),
                config=processor_config or {},
                max_attempts=max_attempts,
                circuit_breaker_failures=circuit_breaker_failures,
            ),
            FlowNode(
                "sink",
                FlowNodeKind.SINK,
                inputs=(_port("in", value_type),),
            ),
        ),
        edges=(
            FlowEdge(
                "input-work",
                "input",
                "out",
                "work",
                "in",
                queue_depth=queue_depth,
                backpressure=backpressure,
            ),
            FlowEdge("work-sink", "work", "out", "sink", "in"),
        ),
    )


@pytest.mark.asyncio
async def test_typed_flow_executes_to_sink_with_causal_receipts(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    graph = _linear_graph()
    seen: list[int] = []

    async def double(delivery, _node):
        return {"out": delivery.payload * 2}

    async def sink(delivery, _node):
        seen.append(delivery.payload)
        return None

    await runtime.deploy(graph)
    runtime.register_handler(graph.graph_id, "work", double)
    runtime.register_handler(graph.graph_id, "sink", sink)
    await runtime.start()
    root_ids = await runtime.emit("demo-flow", "input", "out", 21, event_id="event-root")
    receipts = await runtime.drain("demo-flow")

    assert seen == [42]
    assert len(root_ids) == 1
    assert [item.state for item in receipts] == [
        DeliveryState.SUCCEEDED,
        DeliveryState.SUCCEEDED,
    ]
    assert receipts[0].output_event_ids
    assert runtime.status()["queue_depth"] == 0
    assert runtime.status()["latency"]["sample_count"] == 2
    assert runtime.status()["latency"]["max_ms"] is not None
    assert runtime.receipt(root_ids[0])["state"] == "succeeded"


@pytest.mark.asyncio
async def test_live_worker_wakes_and_drains_deployed_graph(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(state_path=tmp_path / "live-flow.json")
    graph = _linear_graph()
    completed = asyncio.Event()

    async def pass_through(delivery, _node):
        return {"out": delivery.payload}

    async def sink(_delivery, _node):
        completed.set()
        return None

    await runtime.deploy(graph)
    runtime.register_handler("demo-flow", "work", pass_through)
    runtime.register_handler("demo-flow", "sink", sink)
    await runtime.start()
    try:
        await runtime.emit("demo-flow", "input", "out", 8)
        await asyncio.wait_for(completed.wait(), timeout=2.0)
        for _ in range(20):
            if runtime.status()["queue_depth"] == 0:
                break
            await asyncio.sleep(0.01)
        assert runtime.status()["queue_depth"] == 0
        assert runtime.status()["worker"] == {
            "enabled": True,
            "alive": True,
            "failure_streak": 0,
            "last_error": "",
        }
    finally:
        await runtime.shutdown()


def test_graph_rejects_type_mismatch_and_unbounded_cycle() -> None:
    source = FlowNode(
        "source",
        FlowNodeKind.SOURCE,
        outputs=(_port("out", FlowValueType.STRING),),
    )
    sink = FlowNode(
        "sink",
        FlowNodeKind.SINK,
        inputs=(_port("in", FlowValueType.INTEGER),),
    )
    with pytest.raises(FlowContractError, match="incompatible edge types"):
        FlowGraph("bad-types", 1, (source, sink), (FlowEdge("edge", "source", "out", "sink", "in"),))

    a = FlowNode("a", FlowNodeKind.PROCESSOR, (_port("in"),), (_port("out"),))
    b = FlowNode("b", FlowNodeKind.PROCESSOR, (_port("in"),), (_port("out"),))
    with pytest.raises(FlowContractError, match="cycles require"):
        FlowGraph(
            "bad-cycle",
            1,
            (a, b),
            (
                FlowEdge("a-b", "a", "out", "b", "in"),
                FlowEdge("b-a", "b", "out", "a", "in"),
            ),
        )


def test_cycle_is_legal_only_through_explicit_delay() -> None:
    source = FlowNode("source", FlowNodeKind.SOURCE, outputs=(_port("out"),))
    work = FlowNode("work", FlowNodeKind.PROCESSOR, (_port("in"),), (_port("out"),))
    delay = FlowNode("delay", FlowNodeKind.DELAY, (_port("in"),), (_port("out"),))
    graph = FlowGraph(
        "feedback",
        1,
        (source, work, delay),
        (
            FlowEdge("source-work", "source", "out", "work", "in"),
            FlowEdge("work-delay", "work", "out", "delay", "in"),
            FlowEdge("delay-work", "delay", "out", "work", "in"),
        ),
    )
    assert graph.sha256.startswith("sha256:")


@pytest.mark.asyncio
async def test_restart_requeues_processing_delivery_without_losing_identity(tmp_path) -> None:
    path = tmp_path / "flows.json"
    first = RealityEventFlowRuntime(state_path=path, worker_enabled=False)
    graph = _linear_graph()
    await first.deploy(graph)
    await first.start()
    delivery_id = (await first.emit("demo-flow", "input", "out", 7))[0]
    first._queue[0].state = DeliveryState.PROCESSING
    first._queue[0].attempt = 1
    await first._persist()

    recovered = RealityEventFlowRuntime(state_path=path, worker_enabled=False)
    await recovered.start()
    assert recovered.status()["counters"]["recovered_processing"] == 1
    assert recovered._queue[0].delivery_id == delivery_id
    assert recovered._queue[0].attempt == 1
    assert recovered._queue[0].state is DeliveryState.QUEUED


@pytest.mark.asyncio
async def test_claim_persistence_failure_rolls_back_in_memory_state(tmp_path, monkeypatch) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    await runtime.deploy(_linear_graph())
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 9)

    async def fail_persist():
        raise OSError("disk unavailable")

    monkeypatch.setattr(runtime, "_persist", fail_persist)
    with pytest.raises(OSError, match="disk unavailable"):
        await runtime.run_once()
    assert runtime._queue[0].state is DeliveryState.QUEUED
    assert runtime._queue[0].attempt == 0
    assert runtime._queue[0].error == "claim_persistence_failed"


@pytest.mark.asyncio
async def test_missing_restart_binding_waits_without_consuming_retry(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    graph = _linear_graph(max_attempts=1)
    await runtime.deploy(graph)
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 3)

    blocked = await runtime.run_once()
    assert blocked is not None and blocked.state is DeliveryState.QUEUED
    assert blocked.attempt == 0
    assert blocked.error.startswith("dependency_unavailable:")
    assert runtime.status()["dead_letter_depth"] == 0
    assert runtime.status()["ready"] is False

    async def bound(delivery, _node):
        return {"out": delivery.payload}

    runtime.register_handler("demo-flow", "work", bound)
    completed = await runtime.run_once()
    assert completed is not None and completed.state is DeliveryState.SUCCEEDED


@pytest.mark.asyncio
async def test_graph_revision_is_fenced_until_old_work_drains(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    await runtime.deploy(_linear_graph())
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 1)
    with pytest.raises(RealityEventFlowError, match="queued work"):
        await runtime.deploy(_linear_graph(revision=2))


@pytest.mark.asyncio
async def test_reject_backpressure_is_atomic(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    await runtime.deploy(_linear_graph(queue_depth=1))
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 1)
    with pytest.raises(RealityEventFlowError, match="queue is full"):
        await runtime.emit("demo-flow", "input", "out", 2)
    assert [item.payload for item in runtime._queue] == [1]
    assert runtime.status()["counters"]["backpressure_rejections"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "counter"),
    [
        (BackpressurePolicy.DROP_OLDEST, "dropped_oldest"),
        (BackpressurePolicy.COALESCE, "coalesced"),
    ],
)
async def test_lossy_backpressure_is_explicit_and_receipted(tmp_path, policy, counter) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / f"{counter}.json", worker_enabled=False
    )
    await runtime.deploy(_linear_graph(queue_depth=1, backpressure=policy))
    await runtime.start()
    first_id = (await runtime.emit("demo-flow", "input", "out", 1))[0]
    await runtime.emit("demo-flow", "input", "out", 2)

    assert [item.payload for item in runtime._queue] == [2]
    assert runtime.receipt(first_id)["state"] == "cancelled"
    assert runtime.status()["counters"][counter] == 1


@pytest.mark.asyncio
async def test_transient_failure_retries_then_succeeds(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    graph = _linear_graph()
    attempts = 0

    async def flaky(delivery, _node):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return {"out": delivery.payload}

    async def sink(_delivery, _node):
        return None

    await runtime.deploy(graph)
    runtime.register_handler("demo-flow", "work", flaky)
    runtime.register_handler("demo-flow", "sink", sink)
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 1)
    first = await runtime.run_once()
    second = await runtime.run_once()

    assert first is not None and first.state is DeliveryState.QUEUED
    assert second is not None and second.state is DeliveryState.SUCCEEDED
    assert runtime.status()["counters"]["retried"] == 1


@pytest.mark.asyncio
async def test_repeated_failure_opens_circuit_and_dead_letters(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    graph = _linear_graph(max_attempts=4, circuit_breaker_failures=2)

    async def broken(_delivery, _node):
        raise RuntimeError("broken")

    await runtime.deploy(graph)
    runtime.register_handler("demo-flow", "work", broken)
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 1)
    await runtime.run_once()
    receipt = await runtime.run_once()

    assert receipt is not None and receipt.state is DeliveryState.DEAD_LETTER
    assert runtime.status()["open_circuits"] == [
        {"graph_id": "demo-flow", "node_id": "work"}
    ]
    assert len(runtime.dead_letters("demo-flow")) == 1


@pytest.mark.asyncio
async def test_pause_resume_and_cancel_are_durable_lifecycle_operations(tmp_path) -> None:
    runtime = RealityEventFlowRuntime(
        state_path=tmp_path / "flows.json", worker_enabled=False
    )
    await runtime.deploy(_linear_graph())
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", 1)
    await runtime.pause("demo-flow")
    assert await runtime.run_once() is None
    await runtime.resume("demo-flow")
    assert await runtime.cancel_queued("demo-flow", reason="operator_drain") == 1
    assert runtime.status()["queue_depth"] == 0
    assert runtime.status()["counters"]["cancelled"] == 1


class _FakeMiddleware:
    def __init__(self) -> None:
        self.service_ids: list[str] = []
        self.goal_ids: list[str] = []

    async def call_service(self, _endpoint, payload, *, request_id, timeout_s):
        self.service_ids.append(request_id)
        return SimpleNamespace(ok=True, response=dict(payload), error="")

    async def start_action(self, _endpoint, _payload, *, goal_id, timeout_s, preempt):
        self.goal_ids.append(goal_id)
        return {"goal_id": goal_id, "state": "accepted"}

    async def wait_action(self, goal_id, *, timeout_s):
        return {
            "goal_id": goal_id,
            "state": "succeeded",
            "result": {
                "effect_verified": True,
                "effect_receipt_sha256": "sha256:" + "a" * 64,
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [FlowNodeKind.SERVICE, FlowNodeKind.ACTION])
async def test_physical_nodes_use_deterministic_middleware_effect_ids(tmp_path, kind) -> None:
    middleware = _FakeMiddleware()
    runtime = RealityEventFlowRuntime(
        middleware=middleware,  # type: ignore[arg-type]
        state_path=tmp_path / f"{kind.value}.json",
        worker_enabled=False,
    )
    graph = _linear_graph(
        processor_kind=kind,
        processor_config={"endpoint_id": f"physical-{kind.value}"},
    )
    await runtime.deploy(graph)
    await runtime.start()
    await runtime.emit("demo-flow", "input", "out", {"value": 1}, event_id="same-event")
    delivery = runtime._queue[0]

    await runtime._execute_node(graph, graph.node("work"), delivery)
    await runtime._execute_node(graph, graph.node("work"), delivery)
    ids = middleware.service_ids if kind is FlowNodeKind.SERVICE else middleware.goal_ids
    assert len(ids) == 2
    assert ids[0] == ids[1]
    assert ids[0].startswith("flow-")


def test_effectful_custom_sink_is_rejected() -> None:
    with pytest.raises(FlowContractError, match="governed service or action"):
        FlowNode(
            "unsafe",
            FlowNodeKind.SINK,
            inputs=(_port("in"),),
            config={"effectful": True},
        )
