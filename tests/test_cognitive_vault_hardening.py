import sqlite3

import pytest

from core.memory.cognitive_vault import CognitiveVault


@pytest.mark.asyncio
async def test_cognitive_vault_persists_allowlisted_memory_payload(tmp_path):
    db_path = tmp_path / "vault.db"
    vault = CognitiveVault(str(db_path))

    started = await vault.on_start_async()
    committed = await vault.commit(
        "memories",
        {
            "topic": "runtime",
            "content": "bounded queue write",
            "metadata": {"source": "test"},
        },
    )
    await vault._queue.join()
    stopped = await vault.on_stop_async()

    assert started is True
    assert committed is True
    assert stopped is True
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT topic, content, metadata FROM memories").fetchone()
    assert row[0] == "runtime"
    assert row[1] == "bounded queue write"
    assert '"source": "test"' in row[2]


@pytest.mark.asyncio
async def test_cognitive_vault_rejects_unallowlisted_table(tmp_path, monkeypatch):
    records = []
    monkeypatch.setattr(
        "core.memory.cognitive_vault.record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    vault = CognitiveVault(str(tmp_path / "vault.db"))
    await vault.on_start_async()

    committed = await vault.commit("memories; DROP TABLE memories", {"content": "x"})
    stopped = await vault.on_stop_async()

    assert committed is False
    assert stopped is True
    assert records[-1][0] == "cognitive_vault"
    assert "rejected invalid cognitive vault transaction" in records[-1][2]["action"]


@pytest.mark.asyncio
async def test_cognitive_vault_worker_marks_failed_transaction_done(tmp_path, monkeypatch):
    records = []
    monkeypatch.setattr(
        "core.memory.cognitive_vault.record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    vault = CognitiveVault(str(tmp_path / "vault.db"))
    await vault.on_start_async()

    def broken_execute(_tx):
        vault._failed_writes += 0
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(vault, "_execute_tx", broken_execute)

    committed = await vault.commit("audit_log", {"event": "e", "details": "d"})
    await vault._queue.join()
    stopped = await vault.on_stop_async()

    assert committed is True
    assert stopped is False
    assert vault._failed_writes == 1
    assert records[-1][0] == "cognitive_vault"
    assert "kept vault worker alive" in records[-1][2]["action"]
