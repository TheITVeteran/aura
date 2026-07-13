from __future__ import annotations

import pytest

from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway
from core.runtime.gateways import MemoryWriteRequest


def test_default_gateway_uses_hermetic_runtime_root(monkeypatch, tmp_path):
    runtime_root = tmp_path / "aura-runtime"
    monkeypatch.setenv("AURA_TEST_RUNTIME_ROOT", str(runtime_root))

    gateway = ConcreteMemoryWriteGateway(governance_decide=lambda **_kwargs: False)

    assert gateway.root == runtime_root / "memory"


@pytest.mark.asyncio
async def test_memory_write_gateway_denies_async_governance_outage(tmp_path):
    attempted = {"governance": False}

    async def _governance_down(**_kwargs):
        attempted["governance"] = True
        raise RuntimeError("will unavailable")

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=_governance_down)

    with pytest.raises(PermissionError):
        await gateway.write(
            MemoryWriteRequest(
                content={"note": "must not persist"},
                cause="unit-test",
                metadata={"family": "episodic", "record_id": "blocked"},
            )
        )

    assert not (tmp_path / "episodic" / "blocked.json").exists()
    assert attempted["governance"] is True


@pytest.mark.asyncio
async def test_memory_write_receipt_failure_restores_previous_record(
    tmp_path,
    monkeypatch,
):
    import core.memory.memory_write_gateway as memory_module
    from core.runtime.atomic_writer import read_json_envelope

    def approve(**_kwargs):
        return {"approved": True, "receipt_id": "will-memory"}

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    await gateway.write(
        MemoryWriteRequest(
            content="before",
            cause="unit-test",
            metadata={"family": "episodic", "record_id": "stable"},
        )
    )

    class FailingReceiptStore:
        def emit(self, _receipt):
            raise OSError("receipt disk unavailable")

    monkeypatch.setattr(memory_module, "get_receipt_store", lambda: FailingReceiptStore())
    with pytest.raises(RuntimeError, match="receipt_failed_rolled_back"):
        await gateway.write(
            MemoryWriteRequest(
                content="after",
                cause="unit-test",
                metadata={"family": "episodic", "record_id": "stable"},
            )
        )

    envelope = read_json_envelope(tmp_path / "episodic" / "stable.json")
    assert envelope["payload"]["content"] == "before"


@pytest.mark.asyncio
async def test_new_memory_write_receipt_failure_removes_unreceipted_record(
    tmp_path,
    monkeypatch,
):
    import core.memory.memory_write_gateway as memory_module

    def approve(**_kwargs):
        return {"approved": True, "receipt_id": "will-memory"}

    class FailingReceiptStore:
        def emit(self, _receipt):
            raise OSError("receipt disk unavailable")

    monkeypatch.setattr(memory_module, "get_receipt_store", lambda: FailingReceiptStore())
    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    with pytest.raises(RuntimeError, match="receipt_failed_rolled_back"):
        await gateway.write(
            MemoryWriteRequest(
                content="unreceipted",
                cause="unit-test",
                metadata={"family": "episodic", "record_id": "new-record"},
            )
        )

    assert not (tmp_path / "episodic" / "new-record.json").exists()


@pytest.mark.asyncio
async def test_memory_governance_context_redacts_secrets_and_hashes_content(tmp_path):
    captured = {}

    def approve(**kwargs):
        captured.update(kwargs)
        return {"approved": True, "receipt_id": "will-memory"}

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    await gateway.write(
        MemoryWriteRequest(
            content="private autobiographical content",
            cause="unit-test",
            metadata={
                "family": "episodic",
                "record_id": "private",
                "api_token": "metadata-secret",
                "source": "user",
            },
        )
    )

    context = captured["context"]
    assert context["content_length"] == len("private autobiographical content")
    assert len(context["content_sha256"]) == 64
    assert context["memory_metadata"]["api_token"] == "[REDACTED]"
    assert "private autobiographical content" not in str(captured)
    assert "metadata-secret" not in str(captured)


@pytest.mark.asyncio
async def test_memory_identifiers_reject_path_traversal(tmp_path):
    def approve(**_kwargs):
        return {"approved": True, "receipt_id": "will-memory"}

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    with pytest.raises(ValueError, match="memory family"):
        await gateway.write(
            MemoryWriteRequest(
                content="bad",
                cause="unit-test",
                metadata={"family": "../escape", "record_id": "record"},
            )
        )
    with pytest.raises(ValueError, match="memory record_id"):
        await gateway.quarantine("../escape", "bad")


@pytest.mark.asyncio
async def test_quarantine_receipt_failure_restores_source_record(
    tmp_path,
    monkeypatch,
):
    import core.memory.memory_write_gateway as memory_module

    def approve(**_kwargs):
        return {"approved": True, "receipt_id": "will-memory"}

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    await gateway.write(
        MemoryWriteRequest(
            content="contested",
            cause="unit-test",
            metadata={"family": "episodic", "record_id": "contested"},
        )
    )

    class FailingReceiptStore:
        def emit(self, _receipt):
            raise OSError("receipt disk unavailable")

    monkeypatch.setattr(memory_module, "get_receipt_store", lambda: FailingReceiptStore())
    with pytest.raises(RuntimeError, match="quarantine_receipt_failed_rolled_back"):
        await gateway.quarantine("contested", "conflicting evidence")

    assert (tmp_path / "episodic" / "contested.json").exists()
    assert not (tmp_path / "_quarantine" / "episodic_contested.json").exists()
