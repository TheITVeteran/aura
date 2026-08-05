"""Typed contracts for durable Reality Reach event-flow programs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.reality_reach.middleware_contracts import (
    bounded_payload,
    bounded_seconds,
    canonical_identifier,
)
from core.runtime.audit_chain import canonical_json, sha256_hex


class FlowContractError(ValueError):
    """A graph, port, event, or receipt violates its declared contract."""


class FlowValueType(StrEnum):
    ANY = "any"
    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    NULL = "null"

    def accepts(self, value: Any) -> bool:
        if self is FlowValueType.ANY:
            return True
        if self is FlowValueType.OBJECT:
            return isinstance(value, Mapping)
        if self is FlowValueType.ARRAY:
            return isinstance(value, (list, tuple))
        if self is FlowValueType.STRING:
            return isinstance(value, str)
        if self is FlowValueType.NUMBER:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if self is FlowValueType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if self is FlowValueType.BOOLEAN:
            return isinstance(value, bool)
        return value is None

    def can_feed(self, target: FlowValueType) -> bool:
        return (
            self is FlowValueType.ANY
            or target is FlowValueType.ANY
            or self is target
            or (self is FlowValueType.INTEGER and target is FlowValueType.NUMBER)
        )


class FlowNodeKind(StrEnum):
    SOURCE = "source"
    PROCESSOR = "processor"
    FILTER = "filter"
    ROUTER = "router"
    DELAY = "delay"
    SERVICE = "service"
    ACTION = "action"
    SINK = "sink"


class BackpressurePolicy(StrEnum):
    REJECT = "reject"
    DROP_OLDEST = "drop_oldest"
    COALESCE = "coalesce"


class DeliveryState(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FlowPort:
    port_id: str
    value_type: FlowValueType = FlowValueType.ANY
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "port_id",
            canonical_identifier(self.port_id, name="port_id"),
        )
        if not isinstance(self.value_type, FlowValueType):
            raise TypeError("value_type must be a FlowValueType")

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "value_type": self.value_type.value,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowPort:
        return cls(
            port_id=str(value["port_id"]),
            value_type=FlowValueType(str(value.get("value_type") or "any")),
            required=bool(value.get("required", True)),
        )


@dataclass(frozen=True, slots=True)
class FlowNode:
    node_id: str
    kind: FlowNodeKind
    inputs: tuple[FlowPort, ...] = ()
    outputs: tuple[FlowPort, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 10.0
    max_attempts: int = 3
    circuit_breaker_failures: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            canonical_identifier(self.node_id, name="node_id"),
        )
        if not isinstance(self.kind, FlowNodeKind):
            raise TypeError("kind must be a FlowNodeKind")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        for name, ports in (("input", inputs), ("output", outputs)):
            ids = [item.port_id for item in ports]
            if len(ids) != len(set(ids)):
                raise FlowContractError(f"{name} port ids must be unique")
        if self.kind is FlowNodeKind.SOURCE and inputs:
            raise FlowContractError("source nodes cannot declare input ports")
        if self.kind is FlowNodeKind.SINK and outputs:
            raise FlowContractError("sink nodes cannot declare output ports")
        if self.kind is not FlowNodeKind.SOURCE and not inputs:
            raise FlowContractError("non-source nodes require at least one input port")
        if self.kind is not FlowNodeKind.SINK and not outputs:
            raise FlowContractError("non-sink nodes require at least one output port")
        config = bounded_payload(self.config, name="flow node config", maximum=65_536)
        if self.kind in {FlowNodeKind.SERVICE, FlowNodeKind.ACTION}:
            canonical_identifier(str(config.get("endpoint_id") or ""), name="endpoint_id")
            if len(outputs) != 1:
                raise FlowContractError("physical nodes require exactly one output port")
        if self.kind is FlowNodeKind.DELAY and (len(inputs) != 1 or len(outputs) != 1):
            raise FlowContractError("delay nodes require exactly one input and output port")
        if self.kind is FlowNodeKind.SINK and bool(config.get("effectful", False)):
            raise FlowContractError(
                "effectful sinks must be represented as governed service or action nodes"
            )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "config", config)
        object.__setattr__(
            self,
            "timeout_s",
            bounded_seconds(self.timeout_s, name="timeout_s", minimum=0.01, maximum=86400.0),
        )
        if isinstance(self.max_attempts, bool) or not 1 <= int(self.max_attempts) <= 32:
            raise FlowContractError("max_attempts must lie inside [1, 32]")
        if (
            isinstance(self.circuit_breaker_failures, bool)
            or not 1 <= int(self.circuit_breaker_failures) <= 128
        ):
            raise FlowContractError("circuit_breaker_failures must lie inside [1, 128]")

    def input(self, port_id: str) -> FlowPort:
        normalized = canonical_identifier(port_id, name="port_id")
        for port in self.inputs:
            if port.port_id == normalized:
                return port
        raise FlowContractError(f"unknown input port {self.node_id}.{normalized}")

    def output(self, port_id: str) -> FlowPort:
        normalized = canonical_identifier(port_id, name="port_id")
        for port in self.outputs:
            if port.port_id == normalized:
                return port
        raise FlowContractError(f"unknown output port {self.node_id}.{normalized}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "config": dict(self.config),
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "circuit_breaker_failures": self.circuit_breaker_failures,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowNode:
        return cls(
            node_id=str(value["node_id"]),
            kind=FlowNodeKind(str(value["kind"])),
            inputs=tuple(FlowPort.from_dict(item) for item in value.get("inputs") or ()),
            outputs=tuple(FlowPort.from_dict(item) for item in value.get("outputs") or ()),
            config=dict(value.get("config") or {}),
            timeout_s=float(value.get("timeout_s", 10.0)),
            max_attempts=int(value.get("max_attempts", 3)),
            circuit_breaker_failures=int(value.get("circuit_breaker_failures", 5)),
        )


@dataclass(frozen=True, slots=True)
class FlowEdge:
    edge_id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    queue_depth: int = 64
    backpressure: BackpressurePolicy = BackpressurePolicy.REJECT

    def __post_init__(self) -> None:
        for name in ("edge_id", "source_node", "source_port", "target_node", "target_port"):
            object.__setattr__(
                self,
                name,
                canonical_identifier(str(getattr(self, name)), name=name),
            )
        if self.source_node == self.target_node:
            raise FlowContractError("self-loop edges are not supported")
        if isinstance(self.queue_depth, bool) or not 1 <= int(self.queue_depth) <= 4096:
            raise FlowContractError("queue_depth must lie inside [1, 4096]")
        if not isinstance(self.backpressure, BackpressurePolicy):
            raise TypeError("backpressure must be a BackpressurePolicy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "queue_depth": self.queue_depth,
            "backpressure": self.backpressure.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowEdge:
        return cls(
            edge_id=str(value["edge_id"]),
            source_node=str(value["source_node"]),
            source_port=str(value["source_port"]),
            target_node=str(value["target_node"]),
            target_port=str(value["target_port"]),
            queue_depth=int(value.get("queue_depth", 64)),
            backpressure=BackpressurePolicy(str(value.get("backpressure") or "reject")),
        )


@dataclass(frozen=True, slots=True)
class FlowGraph:
    graph_id: str
    revision: int
    nodes: tuple[FlowNode, ...]
    edges: tuple[FlowEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_id",
            canonical_identifier(self.graph_id, name="graph_id"),
        )
        if isinstance(self.revision, bool) or int(self.revision) < 1:
            raise FlowContractError("revision must be a positive integer")
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if not nodes or len(nodes) > 512:
            raise FlowContractError("a flow graph must contain [1, 512] nodes")
        if len(edges) > 4096:
            raise FlowContractError("a flow graph may contain at most 4096 edges")
        by_id = {item.node_id: item for item in nodes}
        if len(by_id) != len(nodes):
            raise FlowContractError("node ids must be unique")
        if len({item.edge_id for item in edges}) != len(edges):
            raise FlowContractError("edge ids must be unique")
        incoming: dict[tuple[str, str], int] = defaultdict(int)
        for edge in edges:
            try:
                source = by_id[edge.source_node]
                target = by_id[edge.target_node]
            except KeyError as exc:
                raise FlowContractError("edge references an unknown node") from exc
            source_port = source.output(edge.source_port)
            target_port = target.input(edge.target_port)
            if not source_port.value_type.can_feed(target_port.value_type):
                raise FlowContractError(
                    f"incompatible edge types: {edge.source_node}.{edge.source_port} "
                    f"({source_port.value_type}) -> {edge.target_node}.{edge.target_port} "
                    f"({target_port.value_type})"
                )
            incoming[(edge.target_node, edge.target_port)] += 1
        for node in nodes:
            for port in node.inputs:
                if port.required and incoming[(node.node_id, port.port_id)] == 0:
                    raise FlowContractError(
                        f"required input has no edge: {node.node_id}.{port.port_id}"
                    )
        self._validate_cycles(nodes, edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @staticmethod
    def _validate_cycles(nodes: tuple[FlowNode, ...], edges: tuple[FlowEdge, ...]) -> None:
        kinds = {item.node_id: item.kind for item in nodes}
        graph: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if kinds[edge.source_node] is FlowNodeKind.DELAY:
                continue
            graph[edge.source_node].append(edge.target_node)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise FlowContractError("cycles require an explicit delay node")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in graph.get(node_id, ()):
                visit(target)
            visiting.remove(node_id)
            visited.add(node_id)

        for node in nodes:
            visit(node.node_id)

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))

    def node(self, node_id: str) -> FlowNode:
        normalized = canonical_identifier(node_id, name="node_id")
        for node in self.nodes:
            if node.node_id == normalized:
                return node
        raise FlowContractError(f"unknown flow node {normalized}")

    def outgoing(self, node_id: str, port_id: str) -> tuple[FlowEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.source_node == node_id and edge.source_port == port_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "revision": self.revision,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowGraph:
        return cls(
            graph_id=str(value["graph_id"]),
            revision=int(value["revision"]),
            nodes=tuple(FlowNode.from_dict(item) for item in value.get("nodes") or ()),
            edges=tuple(FlowEdge.from_dict(item) for item in value.get("edges") or ()),
        )


@dataclass(slots=True)
class FlowDelivery:
    delivery_id: str
    event_id: str
    trace_id: str
    parent_event_id: str | None
    graph_id: str
    graph_sha256: str
    edge_id: str
    target_node: str
    target_port: str
    payload: Any
    sequence: int
    emitted_at_ns: int
    attempt: int = 0
    state: DeliveryState = DeliveryState.QUEUED
    error: str = ""

    def __post_init__(self) -> None:
        for name in (
            "delivery_id",
            "event_id",
            "trace_id",
            "graph_id",
            "edge_id",
            "target_node",
            "target_port",
        ):
            setattr(self, name, canonical_identifier(str(getattr(self, name)), name=name))
        if self.parent_event_id is not None:
            self.parent_event_id = canonical_identifier(
                self.parent_event_id, name="parent_event_id"
            )
        if not str(self.graph_sha256).startswith("sha256:"):
            raise FlowContractError("graph_sha256 must be a sha256 digest")
        if isinstance(self.sequence, bool) or int(self.sequence) < 1:
            raise FlowContractError("sequence must be positive")
        if isinstance(self.attempt, bool) or int(self.attempt) < 0:
            raise FlowContractError("attempt must be non-negative")
        canonical_json(self.payload)
        if not isinstance(self.state, DeliveryState):
            raise TypeError("state must be a DeliveryState")

    @property
    def idempotency_key(self) -> str:
        body = (
            f"{self.graph_sha256}:{self.event_id}:{self.edge_id}:"
            f"{self.target_node}:{self.target_port}"
        )
        return str(sha256_hex(body.encode()))

    def to_dict(self) -> dict[str, Any]:
        body = {
            "delivery_id": self.delivery_id,
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "parent_event_id": self.parent_event_id,
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "edge_id": self.edge_id,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "payload": self.payload,
            "sequence": self.sequence,
            "emitted_at_ns": self.emitted_at_ns,
            "attempt": self.attempt,
            "state": self.state.value,
            "error": self.error[:1024],
        }
        return {**body, "delivery_sha256": str(sha256_hex(canonical_json(body)))}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowDelivery:
        body = dict(value)
        recorded = str(body.pop("delivery_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(body))):
            raise FlowContractError("flow delivery integrity check failed")
        return cls(
            delivery_id=str(body["delivery_id"]),
            event_id=str(body["event_id"]),
            trace_id=str(body["trace_id"]),
            parent_event_id=(
                None if body.get("parent_event_id") is None else str(body["parent_event_id"])
            ),
            graph_id=str(body["graph_id"]),
            graph_sha256=str(body["graph_sha256"]),
            edge_id=str(body["edge_id"]),
            target_node=str(body["target_node"]),
            target_port=str(body["target_port"]),
            payload=body.get("payload"),
            sequence=int(body["sequence"]),
            emitted_at_ns=int(body["emitted_at_ns"]),
            attempt=int(body.get("attempt", 0)),
            state=DeliveryState(str(body.get("state") or "queued")),
            error=str(body.get("error") or "")[:1024],
        )


@dataclass(frozen=True, slots=True)
class FlowReceipt:
    delivery_id: str
    graph_id: str
    graph_sha256: str
    node_id: str
    state: DeliveryState
    attempt: int
    output_event_ids: tuple[str, ...]
    started_at_ns: int
    completed_at_ns: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        body = {
            "delivery_id": self.delivery_id,
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "node_id": self.node_id,
            "state": self.state.value,
            "attempt": self.attempt,
            "output_event_ids": list(self.output_event_ids),
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "error": self.error[:1024],
        }
        return {**body, "receipt_sha256": str(sha256_hex(canonical_json(body)))}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowReceipt:
        body = dict(value)
        recorded = str(body.pop("receipt_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(body))):
            raise FlowContractError("flow receipt integrity check failed")
        return cls(
            delivery_id=canonical_identifier(str(body["delivery_id"]), name="delivery_id"),
            graph_id=canonical_identifier(str(body["graph_id"]), name="graph_id"),
            graph_sha256=str(body["graph_sha256"]),
            node_id=canonical_identifier(str(body["node_id"]), name="node_id"),
            state=DeliveryState(str(body["state"])),
            attempt=int(body["attempt"]),
            output_event_ids=tuple(
                canonical_identifier(str(item), name="event_id")
                for item in body.get("output_event_ids") or ()
            ),
            started_at_ns=int(body["started_at_ns"]),
            completed_at_ns=int(body["completed_at_ns"]),
            error=str(body.get("error") or "")[:1024],
        )


__all__ = [
    "BackpressurePolicy",
    "DeliveryState",
    "FlowContractError",
    "FlowDelivery",
    "FlowEdge",
    "FlowGraph",
    "FlowNode",
    "FlowNodeKind",
    "FlowPort",
    "FlowReceipt",
    "FlowValueType",
]
