from pathlib import Path
from types import SimpleNamespace

import pytest

from core.middleware.capability_guard import CapabilityGuard
from core.self_modification.safe_modification import SafeSelfModification


def test_capability_guard_blocks_restricted_write_even_with_global_allow():
    guard = CapabilityGuard()
    # Force default restrictive capabilities — the installed manifest may be
    # wide-open, but this test validates the default-deny posture.
    guard.capabilities = guard._get_default_capabilities()
    protected = Path("core/security/trust_engine.py")
    allowed = Path("data/logs/test.log")

    assert guard.can_write_path(str(protected)) is False
    assert guard.can_write_path(str(allowed)) is True


def test_safe_self_modification_blocks_constitutionally_protected_paths(tmp_path):
    code_base = tmp_path / "repo"
    (code_base / "core" / "security").mkdir(parents=True)
    (code_base / "core" / "brain").mkdir(parents=True)
    (code_base / "core" / "security" / "trust_engine.py").write_text("x = 1\n", encoding="utf-8")
    (code_base / "core" / "brain" / "module.py").write_text("x = 1\n", encoding="utf-8")

    modifier = SafeSelfModification(code_base_path=str(code_base))

    protected_fix = SimpleNamespace(
        target_file="core/security/trust_engine.py",
        risk_level=1,
        lines_changed=1,
        replacement_content="x = 2\n",
        content="x = 2\n",
    )
    allowed_fix = SimpleNamespace(
        target_file="core/brain/module.py",
        risk_level=1,
        lines_changed=1,
        replacement_content="x = 2\n",
        content="x = 2\n",
    )

    allowed, reason = modifier.validate_proposal(protected_fix)
    assert allowed is False
    assert "constitutionally protected" in reason.lower()

    allowed, reason = modifier.validate_proposal(allowed_fix)
    assert allowed is True


def test_autonomous_self_modification_treats_security_as_protected():
    from core.autonomy.self_modification import AutonomousSelfModification, ModuleZone

    assert AutonomousSelfModification.classify_target("core/security/trust_engine.py") == ModuleZone.PROTECTED


@pytest.mark.asyncio
async def test_autonomous_self_modification_rejects_unsafe_code_patch_calls():
    from core.autonomy.self_modification import AutonomousSelfModification, ModificationProposal

    engine = AutonomousSelfModification.__new__(AutonomousSelfModification)
    proposal = ModificationProposal(
        proposal_id="p-unsafe",
        target_path="core/brain/example.py",
        description="unsafe patch",
        diff_summary="adds eval",
        changes={"type": "code_patch", "new_code": "def f():\n    return eval('1 + 1')\n"},
        source="test",
    )

    ok, detail = await engine._simulate(proposal)

    assert ok is False
    assert "Unsafe code patch call" in detail


@pytest.mark.asyncio
async def test_autonomous_self_modification_queues_code_patch_after_will_approval(monkeypatch):
    import core.will as will_module
    from core.autonomy.self_modification import (
        AutonomousSelfModification,
        ModificationProposal,
        ProposalOutcome,
    )

    receipts = []
    events = []
    domains = []

    class _Decision:
        receipt_id = "will-receipt"
        reason = "approved for staging"

        def is_approved(self):
            return True

    class _Will:
        def decide(self, **kwargs):
            domains.append(kwargs.get("domain"))
            return _Decision()

    monkeypatch.delenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", raising=False)
    monkeypatch.setattr(will_module, "get_will", lambda: _Will())
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_record_receipt",
        lambda self, receipt: receipts.append(receipt),
    )
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_publish_event",
        lambda self, topic, receipt: events.append((topic, receipt)),
    )

    engine = AutonomousSelfModification.__new__(AutonomousSelfModification)
    proposal = ModificationProposal(
        proposal_id="p-safe",
        target_path="core/brain/example.py",
        description="safe patch",
        diff_summary="returns a constant",
        changes={"type": "code_patch", "new_code": "def f():\n    return 2\n"},
        source="test",
    )

    receipt = await engine.propose(proposal)

    assert receipt.outcome == ProposalOutcome.QUEUED_FOR_PIPELINE
    assert receipts == [receipt]
    assert events == [("self_modification.queued", receipt)]
    assert domains
    assert str(domains[0]).endswith("self_modification")


@pytest.mark.asyncio
async def test_autonomous_self_modification_refuses_runtime_apply_without_audit_log(monkeypatch):
    import core.will as will_module
    from core.autonomy.self_modification import (
        AutonomousSelfModification,
        ModificationProposal,
        ProposalOutcome,
    )

    receipts = []
    events = []
    apply_called = False

    class _Decision:
        receipt_id = "will-runtime"
        reason = "approved runtime tuning"

        def is_approved(self):
            return True

    class _Will:
        def decide(self, **_kwargs):
            return _Decision()

    async def _apply_must_not_run(self, proposal):
        nonlocal apply_called
        apply_called = True
        raise AssertionError("runtime mutation must be blocked before apply when audit is unavailable")

    monkeypatch.setenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", "1")
    monkeypatch.setattr(will_module, "get_will", lambda: _Will())
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_audit_log_ready",
        staticmethod(lambda: (False, "disk full")),
    )
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_record_receipt",
        lambda self, receipt: receipts.append(receipt),
    )
    monkeypatch.setattr(AutonomousSelfModification, "_apply", _apply_must_not_run)
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_publish_event",
        lambda self, topic, receipt: events.append((topic, receipt)),
    )

    engine = AutonomousSelfModification.__new__(AutonomousSelfModification)
    proposal = ModificationProposal(
        proposal_id="p-runtime",
        target_path="core/affect/heartstone_values.py",
        description="adjust bounded value",
        diff_summary="sets curiosity drive",
        changes={
            "type": "value_adjustment",
            "target_system": "heartstone",
            "new_values": {"curiosity": 0.7},
        },
        source="test",
    )

    receipt = await engine.propose(proposal)

    assert receipt.outcome == ProposalOutcome.ERROR
    assert "Audit log unavailable" in receipt.will_reason
    assert apply_called is False
    assert receipts == [receipt]
    assert events == [("self_modification.refused", receipt)]


@pytest.mark.asyncio
async def test_autonomous_self_modification_queues_value_changes_even_with_runtime_flag(monkeypatch, tmp_path):
    import core.autonomy.self_modification as smod
    import core.will as will_module
    from core.autonomy.self_modification import (
        AutonomousSelfModification,
        ModificationProposal,
        ProposalOutcome,
    )

    # Keep the durable outbox/audit writes out of the live ~/.aura data dir.
    monkeypatch.setattr(smod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(smod, "_AUDIT_LOG_PATH", tmp_path / "audit_log.jsonl")
    monkeypatch.setattr(smod, "_OUTBOX_PATH", tmp_path / "pending_outbox.jsonl")

    events = []
    apply_called = False

    class _Decision:
        receipt_id = "will-runtime-queued"
        reason = "approved for staged evaluation"

        def is_approved(self):
            return True

    class _Will:
        def decide(self, **_kwargs):
            return _Decision()

    async def _apply_must_not_run(self, proposal):
        nonlocal apply_called
        apply_called = True
        return "should not happen"

    monkeypatch.setenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", "1")
    monkeypatch.setattr(will_module, "get_will", lambda: _Will())
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_audit_log_ready",
        staticmethod(lambda: (True, "audit log writable")),
    )
    monkeypatch.setattr(AutonomousSelfModification, "_record_receipt", lambda self, receipt: None)
    monkeypatch.setattr(AutonomousSelfModification, "_apply", _apply_must_not_run)
    monkeypatch.setattr(
        AutonomousSelfModification,
        "_publish_event",
        lambda self, topic, receipt: events.append((topic, receipt)),
    )

    engine = AutonomousSelfModification.__new__(AutonomousSelfModification)
    engine._pending = []
    engine._receipts = []
    proposal = ModificationProposal(
        proposal_id="p-value",
        target_path="core/affect/heartstone_values.py",
        description="adjust bounded value",
        diff_summary="sets curiosity drive",
        changes={
            "type": "value_adjustment",
            "target_system": "heartstone",
            "new_values": {"curiosity": 0.7},
        },
        source="test",
    )

    receipt = await engine.propose(proposal)

    assert receipt.outcome == ProposalOutcome.QUEUED_FOR_PIPELINE
    assert "SafeSelfModification quarantine" in receipt.simulation_result
    assert apply_called is False
    # The queued entry is a FROZEN snapshot bound to the authorized artifact,
    # not the live mutable proposal object (which a caller could still alter).
    assert len(engine._pending) == 1
    queued = engine._pending[0]
    assert queued.proposal_id == proposal.proposal_id
    assert queued.content_hash == proposal.content_hash()
    assert queued.will_receipt_id == "will-runtime-queued"
    assert events == [("self_modification.queued", receipt)]
