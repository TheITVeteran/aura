from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.networking import hive_node as hive_module
from core.networking.hive_node import HiveNode, NodeInfo


def test_hive_node_defaults_to_loopback_and_derives_node_id(monkeypatch):
    monkeypatch.delenv("AURA_HIVE_HOST", raising=False)
    monkeypatch.delenv("AURA_HIVE_PORT", raising=False)

    node = HiveNode(SimpleNamespace(node_id="orchestrator-1"))

    assert node.node_id == "orchestrator-1"
    assert node.host == "127.0.0.1"
    assert node.port == 9999
    assert node.status()["peer_count"] == 0


def test_hive_node_start_failure_resets_running_and_records(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    async def failing_start_server(*_args, **_kwargs):
        attempted = True
        assert attempted
        raise OSError("bind denied")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(hive_module.asyncio, "start_server", failing_start_server)
    monkeypatch.setattr(hive_module, "record_degradation", record_degradation)

    node = HiveNode("node-a", host="127.0.0.1", port=0)
    asyncio.run(node.start())

    assert node.running is False
    assert "bind denied" in node.status()["last_error"]
    assert recorded[0][0] == "hive_node"
    assert recorded[0][1] == "OSError"
    assert "stayed offline" in str(recorded[0][2]["action"])


def test_hive_node_processes_valid_gossip_item(monkeypatch):
    from core.container import ServiceContainer

    published: list[dict[str, object]] = []

    class Workspace:
        async def publish(self, **kwargs):
            published.append(kwargs)

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: Workspace() if name == "global_workspace" else default))

    node = HiveNode("node-a")
    accepted = asyncio.run(
        node._process_gossip_item(
            {
                "id": "work-1",
                "priority": 2.0,
                "source": "peer-a",
                "payload": {"task": "inspect"},
                "reason": "shared task",
            }
        )
    )

    assert accepted is True
    assert "work-1" in node.known_work_ids
    assert published[0]["priority"] == 1.0
    assert published[0]["source"] == "hive_peer-a"


def test_hive_node_broadcast_removes_unreachable_peer(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    async def failing_open_connection(*_args, **_kwargs):
        attempted = True
        assert attempted
        raise OSError("peer offline")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(hive_module.asyncio, "open_connection", failing_open_connection)
    monkeypatch.setattr(hive_module, "record_degradation", record_degradation)

    node = HiveNode("node-a")
    node.peers["peer-1"] = NodeInfo(node_id="peer-1", ip="127.0.0.1", port=65500)

    result = asyncio.run(node.broadcast_work_item({"id": "work-1", "payload": {}}))

    assert result == {"sent": 0, "failed": 1}
    assert "peer-1" not in node.peers
    assert recorded[0][0] == "hive_node"
    assert "removed unreachable hive peer" in str(recorded[0][2]["action"])
