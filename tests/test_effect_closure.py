from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.constitution import ConstitutionalDecision, ProposalKind, ProposalOutcome
from core.governance_context import GovernanceViolation, governed_scope
from core.memory.vector_memory_engine import Memory, MemoryVault
from core.resilience.silent_failover import SilentFailover
from core.runtime.effect_boundary import get_registered_effect_sinks
from core.skills.base_skill import BaseSkill
from core.state.state_repository import StateRepository
from core.utils.output_gate import AutonomousOutputGate


class _ReplyQueue:
    def __init__(self) -> None:
        self.items: list[str] = []

    def put_nowait(self, value: str, **_kwargs) -> None:
        self.items.append(value)


class _AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _DemoSkill(BaseSkill):
    name = "demo_skill"
    description = "Demo governed skill"

    async def execute(self, params, context):
        return {"ok": True, "summary": f"echo:{params.get('value', 'x')}"}


def _live_runtime(service_container) -> None:
    service_container.register_instance("executive_core", object(), required=False)
    service_container.lock_registration()


@pytest.mark.asyncio
async def test_primary_output_sink_blocks_ungoverned_access_when_runtime_live(service_container):
    _live_runtime(service_container)
    gate = AutonomousOutputGate(orchestrator=SimpleNamespace(reply_queue=_ReplyQueue(), conversation_history=[]))

    with pytest.raises(GovernanceViolation):
        await gate._send_to_primary("hi", "system", {"suppress_bus": True})


@pytest.mark.asyncio
async def test_primary_output_sink_accepts_constitutional_receipt_shape(service_container):
    _live_runtime(service_container)
    queue = _ReplyQueue()
    gate = AutonomousOutputGate(orchestrator=SimpleNamespace(reply_queue=queue, conversation_history=[]))
    decision = ConstitutionalDecision(
        proposal_id="proposal-1",
        kind=ProposalKind.EXPRESSION,
        outcome=ProposalOutcome.APPROVED,
        reason="ok",
        source="test",
        constraints={"will_receipt_id": "will-123", "governance_domain": "expression"},
    )

    async with governed_scope(decision):
        receipt_id = await gate._send_to_primary(
            "hello",
            "system",
            {"suppress_bus": True},
        )

    assert queue.items == ["hello"]
    assert receipt_id


@pytest.mark.asyncio
async def test_primary_output_sink_withholds_receipt_without_an_accepted_transport(
    service_container,
):
    _live_runtime(service_container)
    gate = AutonomousOutputGate(orchestrator=None)
    decision = ConstitutionalDecision(
        proposal_id="proposal-no-sink",
        kind=ProposalKind.EXPRESSION,
        outcome=ProposalOutcome.APPROVED,
        reason="ok",
        source="test",
        constraints={"will_receipt_id": "will-no-sink"},
    )

    async with governed_scope(decision):
        receipt_id = await gate._send_to_primary(
            "hello",
            "system",
            {"suppress_bus": True, "voice": False},
        )

    assert receipt_id is None


@pytest.mark.asyncio
async def test_primary_output_sink_withholds_confirmation_when_receipt_persistence_fails(
    service_container,
    monkeypatch,
):
    _live_runtime(service_container)
    queue = _ReplyQueue()
    gate = AutonomousOutputGate(
        orchestrator=SimpleNamespace(reply_queue=queue, conversation_history=[])
    )
    decision = ConstitutionalDecision(
        proposal_id="proposal-receipt-failure",
        kind=ProposalKind.EXPRESSION,
        outcome=ProposalOutcome.APPROVED,
        reason="ok",
        source="test",
        constraints={"will_receipt_id": "will-receipt-failure"},
    )

    async def fail_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(gate, "_emit_output_receipt", fail_receipt)
    async with governed_scope(decision):
        receipt_id = await gate._send_to_primary(
            "hello",
            "system",
            {"suppress_bus": True, "voice": False},
        )

    assert queue.items == ["hello"]
    assert receipt_id is None


def test_effect_sink_registry_lists_critical_sinks():
    # Import modules so decorators register their sinks.
    _ = AutonomousOutputGate
    _ = StateRepository
    _ = MemoryVault
    from core.self_model import SelfModel  # noqa: F401

    sinks = get_registered_effect_sinks()
    for sink_id in {
        "output.primary",
        "state.sync_to_shm",
        "state.commit_to_db",
        "memory.vault_store",
        "belief.self_model_persist",
    }:
        assert sink_id in sinks


def test_memory_vault_store_requires_governance_when_runtime_live(service_container, tmp_path):
    _live_runtime(service_container)
    vault = MemoryVault(str(tmp_path / "vault"))
    memory = Memory(
        id="m1",
        content="hello",
        memory_type="episodic",
        timestamp=1.0,
        importance=0.5,
    )
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    with pytest.raises(GovernanceViolation):
        vault.store(memory, vector)


@pytest.mark.asyncio
async def test_state_repository_commit_sinks_require_governance_when_runtime_live(service_container, tmp_path):
    _live_runtime(service_container)
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)

    with pytest.raises(GovernanceViolation):
        await repo._commit_to_db(SimpleNamespace(), "{}")

    with pytest.raises(GovernanceViolation):
        await repo._sync_to_shm(SimpleNamespace(), "{}")


@pytest.mark.asyncio
async def test_base_skill_blocks_direct_execution_when_runtime_live(service_container):
    _live_runtime(service_container)
    result = await _DemoSkill().safe_execute({"value": "x"})

    assert result["ok"] is False
    assert "Ungoverned skill execution blocked" in result["error"]


@pytest.mark.asyncio
async def test_capability_engine_executes_skill_inside_governed_scope(service_container, monkeypatch):
    _live_runtime(service_container)
    begin_tool_execution = _AsyncCallRecorder(
        SimpleNamespace(
            approved=True,
            constraints={},
            capability_token_id="tok-1",
            decision=ConstitutionalDecision(
                proposal_id="tool-1",
                kind=ProposalKind.TOOL,
                outcome=ProposalOutcome.APPROVED,
                reason="ok",
                source="test",
                constraints={"will_receipt_id": "will-1", "governance_domain": "tool_execution"},
            ),
        )
    )
    finish_tool_execution = _AsyncCallRecorder()

    fake_constitution = SimpleNamespace(
        begin_tool_execution=begin_tool_execution,
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_constitution)
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: SimpleNamespace(verify_tool_access=lambda _skill, _token: True),
    )

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        "demo_skill": SkillMetadata(
            name="demo_skill",
            description="Demo governed skill",
            skill_class=_DemoSkill,
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *a, **k: None
    engine.max_retries = 1
    engine.retry_delay = 0.0
    engine.timeout = 5.0

    result = await CapabilityEngine.execute(engine, "demo_skill", {"value": "ok"}, context={})

    assert result["ok"] is True
    assert len(begin_tool_execution.calls) == 1
    assert len(finish_tool_execution.calls) == 1


def test_silent_failover_marks_fallback_as_inferred_and_uncommitted():
    failover = SilentFailover()
    result = failover.wrap_execution(
        lambda *_args, **_kwargs: {"ok": False, "error": "boom"},
        "web_search",
        {"query": "closure"},
        {},
    )

    assert result["response_class"] == "inferred_fallback"
    assert result["committed_action"] is False
    assert result["will_receipt_id"] is None


@pytest.mark.asyncio
async def test_self_model_deferred_update_persists_inside_governed_scope(service_container, monkeypatch, tmp_path):
    _live_runtime(service_container)
    from core.constitution import ConstitutionalDecision, ProposalKind, ProposalOutcome
    from core.self_model import SelfModel

    monkeypatch.setattr("core.self_model.DATA_FILE", tmp_path / "self_model.json")

    decision = ConstitutionalDecision(
        proposal_id="belief-1",
        kind=ProposalKind.BELIEF_MUTATION,
        outcome=ProposalOutcome.REJECTED,
        reason="executive_deferred",
        source="self_model",
        constraints={"will_receipt_id": "will-self-1", "governance_domain": "memory_write"},
    )

    fake_core = SimpleNamespace(
        belief_authority=SimpleNamespace(
            review_update=lambda namespace, key, value, note=None: SimpleNamespace(
                key=key,
                value=value,
                reason="accepted",
            )
        ),
        approve_belief_update_sync=lambda *args, **kwargs: (False, "executive_deferred", decision),
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    model = SelfModel(id="self-test")
    snapshot = await model.update_belief("executive_closure", {"phi": 0.12}, note="sync")

    assert snapshot.summary == "deferred update executive_closure"
    assert model.pending_updates
    assert (tmp_path / "self_model.json").exists()


@pytest.mark.asyncio
async def test_self_model_runtime_lesson_persists_inside_local_governance(
    service_container, monkeypatch, tmp_path
):
    _live_runtime(service_container)
    from core.self_model import SelfModel

    target = tmp_path / "self_model.json"
    monkeypatch.setattr("core.self_model.DATA_FILE", target)

    model = SelfModel(id="self-runtime-lesson")
    snap = model.record_runtime_lesson(
        source="boot_probe",
        lesson="boot probe observed a recoverable runtime event",
        confidence=0.8,
        evidence={"event": "probe"},
    )

    for _ in range(50):
        if target.exists():
            break
        await asyncio.sleep(0.01)

    assert snap.summary == "runtime lesson from boot_probe"
    assert target.exists()
