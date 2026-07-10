from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
from core.runtime.receipts import ReceiptStore
from tools.closeout.audit_cognitive_candidate_gates import audit


def _install_workspace_dependencies(monkeypatch, tmp_path, manager_ref):
    import core.consciousness.global_workspace as workspace_module

    def _get(name, default=None):
        if name == "inhibition_manager":
            return manager_ref["current"]
        return default

    store = ReceiptStore(tmp_path / "receipts")
    monkeypatch.setattr(workspace_module.ServiceContainer, "get", staticmethod(_get))
    monkeypatch.setattr(workspace_module, "get_receipt_store", lambda: store)
    return store


def test_cognitive_candidate_gate_inventory_matches_source():
    report = audit(Path.cwd())

    assert report["passed"] is True
    assert report["declared_count"] == report["discovered_count"] == 13
    assert report["issues"] == []
    assert REQUIRED_HEALTH_PROBE_GROUPS["workspace"] == (
        "inhibition_manager",
        "global_workspace",
    )
    assert REQUIRED_HEALTH_PROBE_GROUPS["attention"] == ("attention_schema",)


@pytest.mark.asyncio
async def test_workspace_revalidates_after_gate_instance_restart(monkeypatch, tmp_path):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class Inhibition:
        def __init__(self, instance_id, inhibited):
            self.instance_id = instance_id
            self._inhibited = inhibited

        async def is_inhibited(self, _source):
            return self._inhibited

        def is_ready(self):
            return True

    first = Inhibition("gate-before-restart", False)
    second = Inhibition("gate-after-restart", True)
    manager_ref = {"current": first}
    store = _install_workspace_dependencies(monkeypatch, tmp_path, manager_ref)
    workspace = GlobalWorkspace()

    candidate = CognitiveCandidate("stale approval", "memory", 0.9)
    assert await workspace.submit(candidate) is True
    assert candidate.gate_instance_id == "gate-before-restart"

    manager_ref["current"] = second
    assert await workspace.run_competition() is None
    assert workspace.last_winner is None
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert any(
        receipt.reason == "source_inhibited_before_competition"
        and receipt.gate_instance_id == "gate-after-restart"
        for receipt in receipts
    )


@pytest.mark.asyncio
async def test_workspace_submission_cancellation_rejects_and_receipts(
    monkeypatch, tmp_path
):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    started = asyncio.Event()
    never = asyncio.Event()

    class Inhibition:
        instance_id = "gate-cancelled"

        async def is_inhibited(self, _source):
            started.set()
            await never.wait()
            return False

        def is_ready(self):
            return True

    manager_ref = {"current": Inhibition()}
    store = _install_workspace_dependencies(monkeypatch, tmp_path, manager_ref)
    workspace = GlobalWorkspace()
    task = asyncio.create_task(
        workspace.submit(CognitiveCandidate("cancel me", "memory", 0.9))
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert workspace._candidates == []
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert any(
        receipt.reason == "gate_check_cancelled:CancelledError"
        and receipt.metadata["lane"] == "workspace_candidate_admission"
        for receipt in receipts
    )


@pytest.mark.asyncio
async def test_workspace_competition_cancellation_quarantines_approved_candidate(
    monkeypatch, tmp_path
):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    started = asyncio.Event()
    never = asyncio.Event()

    class Inhibition:
        instance_id = "gate-competition-cancelled"

        def __init__(self):
            self.calls = 0

        async def is_inhibited(self, _source):
            self.calls += 1
            if self.calls == 1:
                return False
            started.set()
            await never.wait()
            return False

        def is_ready(self):
            return True

    manager_ref = {"current": Inhibition()}
    store = _install_workspace_dependencies(monkeypatch, tmp_path, manager_ref)
    workspace = GlobalWorkspace()
    assert await workspace.submit(CognitiveCandidate("approved", "memory", 0.9))

    task = asyncio.create_task(workspace.run_competition())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert workspace._candidates == []
    assert workspace.last_winner is None
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert any(
        receipt.reason == "gate_revalidation_cancelled:CancelledError"
        for receipt in receipts
    )


@pytest.mark.asyncio
async def test_somatic_candidate_uses_same_inhibition_gate(monkeypatch, tmp_path):
    from core.consciousness.global_workspace import GlobalWorkspace

    class Inhibition:
        instance_id = "gate-somatic"

        async def is_inhibited(self, source):
            return source == "somatic_noise"

        def is_ready(self):
            return True

    monkeypatch.setenv("AURA_SOMATIC_NOISE", "1")
    monkeypatch.setenv("AURA_SOMATIC_NOISE_FORCE", "1")
    manager_ref = {"current": Inhibition()}
    store = _install_workspace_dependencies(monkeypatch, tmp_path, manager_ref)
    workspace = GlobalWorkspace()

    assert await workspace.run_competition() is None
    assert workspace.last_winner is None
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert any(
        receipt.candidate_source == "somatic_noise"
        and receipt.reason == "source_inhibited"
        for receipt in receipts
    )


@pytest.mark.asyncio
async def test_attention_focus_gate_failure_retains_focus_and_recovers(
    monkeypatch, tmp_path
):
    import core.consciousness.attention_schema as attention_module
    from core.consciousness.attention_schema import AttentionSchema

    store = ReceiptStore(tmp_path / "attention-receipts")
    monkeypatch.setattr(attention_module, "get_receipt_store", lambda: store)
    schema = AttentionSchema()
    original = await schema.set_focus("the current task", "user", 0.8)

    def _broken_signal():
        raise RuntimeError("free-energy gate offline")

    monkeypatch.setattr(attention_module, "_read_focus_rigidity_signal", _broken_signal)
    retained = await schema.set_focus("background novelty", "curiosity", 0.9)

    assert retained is original
    assert schema.current_focus is original
    assert schema.is_ready() is False
    receipts = store.query_recent_persisted("workspace_gate", limit=10)
    assert any(
        receipt.gate == "attention_focus_rigidity"
        and receipt.reason == "gate_check_failed:RuntimeError"
        and receipt.metadata["lane"] == "attention_focus"
        for receipt in receipts
    )

    monkeypatch.setattr(
        attention_module,
        "_read_focus_rigidity_signal",
        lambda: ("free-energy-recovered", 0.1),
    )
    recovered = await schema.set_focus("background novelty", "curiosity", 0.9)
    assert recovered.source == "curiosity"
    assert schema.is_ready() is True
    assert schema.get_status()["rigidity_gate"]["instance_id"] == "free-energy-recovered"


@pytest.mark.asyncio
async def test_attention_focus_gate_timeout_retains_focus(monkeypatch, tmp_path):
    import core.consciousness.attention_schema as attention_module
    from core.consciousness.attention_schema import AttentionSchema

    store = ReceiptStore(tmp_path / "attention-timeout-receipts")
    monkeypatch.setattr(attention_module, "get_receipt_store", lambda: store)
    monkeypatch.setenv("AURA_ATTENTION_RIGIDITY_GATE_TIMEOUT_S", "0.01")
    schema = AttentionSchema()
    original = await schema.set_focus("foreground work", "user", 0.8)

    def _wedged_signal():
        time.sleep(0.1)
        return "late-gate", 0.0

    monkeypatch.setattr(attention_module, "_read_focus_rigidity_signal", _wedged_signal)
    retained = await schema.set_focus("unrelated work", "background", 0.9)

    assert retained is original
    assert schema.current_focus is original
    assert schema.is_ready() is False
    receipt = store.query_recent_persisted("workspace_gate", limit=1)[0]
    assert receipt.reason == "gate_check_failed:TimeoutError"


@pytest.mark.asyncio
async def test_attention_focus_gate_cancellation_retains_focus_and_receipts(
    monkeypatch, tmp_path
):
    import core.consciousness.attention_schema as attention_module
    from core.consciousness.attention_schema import AttentionSchema

    store = ReceiptStore(tmp_path / "attention-cancel-receipts")
    monkeypatch.setattr(attention_module, "get_receipt_store", lambda: store)
    started = threading.Event()
    release = threading.Event()
    schema = AttentionSchema()
    original = await schema.set_focus("foreground work", "user", 0.8)

    def _blocked_signal():
        started.set()
        release.wait(timeout=0.5)
        return "late-gate", 0.0

    monkeypatch.setattr(attention_module, "_read_focus_rigidity_signal", _blocked_signal)
    task = asyncio.create_task(
        schema.set_focus("unrelated work", "background", 0.9)
    )
    await asyncio.to_thread(started.wait, 0.5)
    assert started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    assert schema.current_focus is original
    assert schema.is_ready() is False
    receipt = store.query_recent_persisted("workspace_gate", limit=1)[0]
    assert receipt.reason == "gate_check_cancelled:CancelledError"
