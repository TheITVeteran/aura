from __future__ import annotations

import asyncio
import json

import pytest


class RecordingFileGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def write_text(self, path, text, *, source, encoding="utf-8") -> None:
        self.calls.append((str(path), encoding, source))
        path.write_text(text, encoding=encoding)

    # Async lane delegators: production code now calls *_async; fakes
    # must mirror the gateway surface or every governed write breaks.
    async def write_text_async(self, *args, **kwargs):
        return self.write_text(*args, **kwargs)


def test_feature_flags_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.governance.feature_flags as feature_flags

    gateway = RecordingFileGateway()
    target = tmp_path / "feature_flags.json"
    monkeypatch.setattr(feature_flags, "get_file_write_gateway", lambda: gateway)

    flags = feature_flags.FeatureFlags(config_path=target)
    flags.set_flag("memory_dedup_on_write", False, reason="test")
    flags.save()

    assert json.loads(target.read_text(encoding="utf-8"))["memory_dedup_on_write"] is False
    assert gateway.calls == [(str(target), "utf-8", "core.governance.feature_flags.save")]


def test_outcome_ledger_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.environment.outcome.ledger as outcome_ledger

    gateway = RecordingFileGateway()
    target = tmp_path / "outcomes.json"
    monkeypatch.setattr(outcome_ledger, "get_file_write_gateway", lambda: gateway)

    ledger = outcome_ledger.OutcomeLedger(target)
    ledger.record_outcome("inspect", "env", "ctx", True, 1.0, ["door_opened"])
    ledger.save()

    assert "env::ctx::inspect" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.environment.outcome.ledger.save")]


def test_belief_graph_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.environment.belief_graph as belief_graph

    gateway = RecordingFileGateway()
    target = tmp_path / "belief.json"
    monkeypatch.setattr(belief_graph, "get_file_write_gateway", lambda: gateway)

    graph = belief_graph.EnvironmentBeliefGraph()
    graph.save(target)

    assert "nodes" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.environment.belief_graph.save")]


def test_semiotic_network_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.grounding.semiotic_network as semiotic_network

    gateway = RecordingFileGateway()
    target = tmp_path / "semiotic.json"
    monkeypatch.setattr(semiotic_network, "get_file_write_gateway", lambda: gateway)

    network = semiotic_network.SemioticNetwork(target)
    network.save()

    assert "methods" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.grounding.semiotic_network.save")]


def test_unity_receipts_artifact_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.unity.unity_receipts as unity_receipts

    gateway = RecordingFileGateway()
    target = tmp_path / "unity.json"
    monkeypatch.setattr(unity_receipts, "get_file_write_gateway", lambda: gateway)

    assert unity_receipts.write_unity_results_artifact(target, {"ok": True}) == target

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert gateway.calls == [
        (str(target), "utf-8", "core.unity.unity_receipts.write_results_artifact")
    ]


def test_proof_obligations_bytecode_check_uses_file_write_gateway(monkeypatch) -> None:
    import core.learning.proof_obligations as proof_obligations

    gateway = RecordingFileGateway()
    monkeypatch.setattr(proof_obligations, "get_file_write_gateway", lambda: gateway)

    ok, diagnostics = proof_obligations.ProofObligationEngine._bytecode_compiles(
        "x = 1\n",
        "candidate.py",
    )

    assert ok is True
    assert diagnostics == {"ok": True}
    assert gateway.calls
    assert gateway.calls[0][2] == "core.learning.proof_obligations.bytecode_compiles"


def test_plugin_allowlist_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.security.plugin_allowlist as plugin_allowlist

    gateway = RecordingFileGateway()
    allowlist_path = tmp_path / "allow.json"
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(plugin_allowlist, "get_file_write_gateway", lambda: gateway)

    allowlist = plugin_allowlist.PluginAllowlist(allowlist_path)
    allowlist.record(plugin_path, approved_by="test", reason="unit")

    assert "entries" in json.loads(allowlist_path.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(allowlist_path), "utf-8", "core.security.plugin_allowlist.save")]


def test_reddit_session_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.skills.reddit_adapter as reddit_adapter

    gateway = RecordingFileGateway()
    target = tmp_path / "reddit_state.json"
    monkeypatch.setattr(reddit_adapter, "_STORAGE_STATE_FILE", target)
    monkeypatch.setattr(reddit_adapter, "get_file_write_gateway", lambda: gateway)

    class Context:
        async def cookies(self):
            return [{"name": "session", "value": "abc"}]

    class Browser:
        context = Context()

    skill = reddit_adapter.RedditAdapterSkill()
    asyncio.run(skill._save_session(Browser()))

    assert json.loads(target.read_text(encoding="utf-8"))["cookies"][0]["name"] == "session"
    assert gateway.calls == [(str(target), "utf-8", "core.skills.reddit_adapter.save_session")]


def test_file_write_gateway_drain_text_atomically_removes_drained_file(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "queue.jsonl"

    gateway.append_text(target, '{"one": 1}\n', source="unit.append")
    gateway.append_text(target, '{"two": 2}\n', source="unit.append")

    drained = gateway.drain_text(target, source="unit.drain")

    assert drained.splitlines() == ['{"one": 1}', '{"two": 2}']
    assert not target.exists()
    assert gateway.drain_text(target, source="unit.drain") == ""


@pytest.mark.asyncio
async def test_file_write_gateway_atomically_replaces_directory_symlink(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "active"

    gateway.replace_symlink(link, first, source="unit.symlink")
    assert link.is_symlink()
    assert link.resolve() == first.resolve()

    await gateway.replace_symlink_async(link, second, source="unit.symlink")
    assert link.resolve() == second.resolve()
    assert list(tmp_path.glob(".*.symlink.tmp")) == []
    assert gateway.delete_file(link, source="unit.delete_symlink") is True
    assert not link.is_symlink()


def test_file_write_gateway_refuses_to_replace_real_directory(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "target"
    link = tmp_path / "active"
    target.mkdir()
    link.mkdir()

    with pytest.raises(IsADirectoryError, match="refusing to replace directory"):
        gateway.replace_symlink(link, target, source="unit.symlink")
