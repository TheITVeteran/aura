from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat

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


def test_file_write_gateway_batch_commits_private_files_with_receipt(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteBatchEntry, FileWriteGateway

    gateway = FileWriteGateway()
    key_path = tmp_path / "server.key"
    cert_path = tmp_path / "server.crt"

    receipt = gateway.write_bytes_batch(
        (
            FileWriteBatchEntry(key_path, b"private-key", mode=0o600),
            FileWriteBatchEntry(cert_path, b"certificate", mode=0o644),
        ),
        source="unit.batch",
    )

    assert key_path.read_bytes() == b"private-key"
    assert cert_path.read_bytes() == b"certificate"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cert_path.stat().st_mode) == 0o644
    assert receipt.paths == (str(key_path), str(cert_path))
    assert dict(receipt.sha256) == {
        str(key_path): hashlib.sha256(b"private-key").hexdigest(),
        str(cert_path): hashlib.sha256(b"certificate").hexdigest(),
    }
    assert len(receipt.transaction_id) == 32


def test_file_write_gateway_batch_restores_prior_targets_on_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as file_write_gateway

    gateway = file_write_gateway.FileWriteGateway()
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    first.chmod(0o640)
    second.chmod(0o600)
    real_atomic_write = file_write_gateway.atomic_write_bytes
    calls = 0

    def fail_second_write(path, payload, *, durable=True, mode=0o600):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second replacement failure")
        return real_atomic_write(path, payload, durable=durable, mode=mode)

    monkeypatch.setattr(file_write_gateway, "atomic_write_bytes", fail_second_write)

    with pytest.raises(file_write_gateway.FileWriteTransactionError) as exc_info:
        gateway.write_bytes_batch(
            (
                file_write_gateway.FileWriteBatchEntry(first, b"new-first"),
                file_write_gateway.FileWriteBatchEntry(second, b"new-second"),
            ),
            source="unit.batch.failure",
        )

    assert "prior targets restored" in str(exc_info.value)
    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert stat.S_IMODE(first.stat().st_mode) == 0o640
    assert stat.S_IMODE(second.stat().st_mode) == 0o600


def test_file_write_gateway_batch_rejects_ambiguous_targets(tmp_path) -> None:
    from core.runtime.file_write_gateway import (
        FileWriteBatchEntry,
        FileWriteGateway,
        FileWriteTransactionError,
    )

    gateway = FileWriteGateway()
    first = tmp_path / "first" / "value.bin"
    second = tmp_path / "second" / "value.bin"

    with pytest.raises(ValueError, match="share one directory"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(first, b"one"), FileWriteBatchEntry(second, b"two")),
            source="unit.batch.cross_directory",
        )

    target = tmp_path / "target.bin"
    with pytest.raises(ValueError, match="duplicate batch target"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(target, b"one"), FileWriteBatchEntry(target, b"two")),
            source="unit.batch.duplicate",
        )

    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"unchanged")
    link = tmp_path / "link.bin"
    link.symlink_to(backing)
    with pytest.raises(FileWriteTransactionError, match="symlink"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(link, b"replacement"),),
            source="unit.batch.symlink",
        )
    assert backing.read_bytes() == b"unchanged"

    lock_backing = tmp_path / "lock-backing.bin"
    lock_backing.write_bytes(b"do-not-follow")
    lock_path = tmp_path / ".aura_file_write_batch.lock"
    lock_path.symlink_to(lock_backing)
    with pytest.raises(OSError):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(target, b"replacement"),),
            source="unit.batch.lock_symlink",
        )
    assert lock_backing.read_bytes() == b"do-not-follow"


def test_file_write_gateway_owned_binary_is_narrow_private_and_no_follow(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "ring.bin"
    with gateway.open_owned_binary(
        target,
        mode="w+b",
        permissions=0o640,
        source="unit.owned_binary",
    ) as handle:
        handle.write(b"ring")
        handle.flush()
        os.fsync(handle.fileno())

    assert target.read_bytes() == b"ring"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    with pytest.raises(ValueError, match="unsupported owned binary mode"):
        gateway.open_owned_binary(target, mode="wb", source="unit.invalid_mode")

    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"unchanged")
    link = tmp_path / "ring-link.bin"
    link.symlink_to(backing)
    with pytest.raises(OSError, match="symlink"):
        gateway.open_owned_binary(link, mode="r+b", source="unit.owned_symlink")


def test_file_write_gateway_replace_file_durably_moves_source(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    assert gateway.replace_file(source, destination, source="unit.replace") == str(
        destination
    )
    assert not source.exists()
    assert destination.read_bytes() == b"new"

    new_source = tmp_path / "new-source.bin"
    backing = tmp_path / "backing.bin"
    link = tmp_path / "destination-link.bin"
    new_source.write_bytes(b"replacement")
    backing.write_bytes(b"unchanged")
    link.symlink_to(backing)
    with pytest.raises(OSError, match="symlink"):
        gateway.replace_file(new_source, link, source="unit.replace_symlink")
    assert new_source.read_bytes() == b"replacement"
    assert backing.read_bytes() == b"unchanged"
