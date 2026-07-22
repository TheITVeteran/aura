"""CP126 hardening contracts for core/autonomy/self_modification.py.

Covers the self-modification safety pipeline: path-escape classification,
artifact-bound authorization, hermetic patch inspection, typed/finite
validation, the durable outbox + consumer API, overflow dead-lettering, the
tamper-evident audit chain, and the disabled live-apply helpers.
"""
from __future__ import annotations

import json
import math

import pytest

import core.autonomy.self_modification as smod
from core.autonomy.self_modification import (
    AutonomousSelfModification,
    ModificationProposal,
    ModuleZone,
    ProposalOutcome,
    QueuedProposal,
)


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Keep every durable write inside the test's tmp dir."""
    monkeypatch.setattr(smod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(smod, "_AUDIT_LOG_PATH", tmp_path / "audit_log.jsonl")
    monkeypatch.setattr(smod, "_OUTBOX_PATH", tmp_path / "pending_outbox.jsonl")
    yield


def _engine() -> AutonomousSelfModification:
    eng = AutonomousSelfModification.__new__(AutonomousSelfModification)
    eng._ensure_runtime_state()
    return eng


def _proposal(**kw) -> ModificationProposal:
    base = dict(
        proposal_id="p1",
        target_path="core/brain/example.py",
        description="a change",
        diff_summary="does a thing",
        changes={"type": "code_patch", "new_code": "def f():\n    return 2\n"},
        source="test",
    )
    base.update(kw)
    return ModificationProposal(**base)


# ── 462a4bd7: path escape via dot segments ─────────────────────────────────


def test_dotdot_escape_into_protected_is_caught():
    # Literal target is core/security/x — a protected module.
    assert (
        AutonomousSelfModification.classify_target("core/autonomy/../security/trust.py")
        == ModuleZone.PROTECTED
    )


def test_escape_out_of_root_fails_closed():
    assert (
        AutonomousSelfModification.classify_target("core/brain/../../../etc/passwd")
        == ModuleZone.PROTECTED
    )


def test_absolute_path_is_protected():
    assert AutonomousSelfModification.classify_target("/etc/passwd") == ModuleZone.PROTECTED


def test_legitimate_relative_target_still_modifiable():
    # ../ that resolves back into a modifiable tree is honored as its true target.
    assert (
        AutonomousSelfModification.classify_target("core/security/../brain/x.py")
        == ModuleZone.MODIFIABLE
    )


# ── 9cfc1cc1 + 757b9ff7: hash binds the changes; frozen snapshot is immutable


def test_content_hash_covers_the_changes():
    p1 = _proposal(changes={"type": "code_patch", "new_code": "def f():\n    return 1\n"})
    p2 = _proposal(changes={"type": "code_patch", "new_code": "def f():\n    return 999\n"})
    assert p1.content_hash() != p2.content_hash()


def test_frozen_snapshot_is_immutable_after_authorization():
    p = _proposal(changes={"type": "value_adjustment", "new_values": {"a": 0.5}})
    h = p.content_hash()
    frozen = p.frozen(content_hash=h, will_receipt_id="w1")
    # Mutating the live proposal must not change the frozen artifact or its hash.
    p.changes["new_values"]["a"] = 0.9
    assert json.loads(frozen.changes_json)["new_values"]["a"] == 0.5
    assert frozen.content_hash == h
    assert p.content_hash() != h  # the live object's hash moved; the frozen one did not


# ── b2bc3fa1: hermetic code-patch inspection ───────────────────────────────


@pytest.mark.parametrize(
    "code",
    [
        "import os\n",
        "from subprocess import run\n",
        "x = (1).__class__.__bases__\n",
        "y = getattr(obj, 'attr')\n",
        "f = open('/etc/passwd')\n",
        "def loop():\n    while True:\n        pass\n",
        "z = ().__class__.__subclasses__()\n",
    ],
)
def test_dangerous_code_patches_are_refused(code):
    ok, detail = AutonomousSelfModification._inspect_code_patch(code)
    assert ok is False, detail


def test_benign_patch_with_bounded_loop_passes():
    ok, detail = AutonomousSelfModification._inspect_code_patch(
        "def g(items):\n    while items:\n        items.pop()\n    return 0\n"
    )
    assert ok is True, detail


# ── 054a193b: unknown/config forms are not free passes ─────────────────────


@pytest.mark.asyncio
async def test_unknown_change_type_fails_closed():
    ok, detail = await _engine()._simulate(_proposal(changes={"type": "mystery"}))
    assert ok is False
    assert "Unknown change type" in detail


@pytest.mark.asyncio
async def test_config_update_requires_a_typed_schema():
    eng = _engine()
    ok, _ = await eng._simulate(_proposal(changes={"type": "config_update"}))
    assert ok is False  # missing key/value
    ok, _ = await eng._simulate(
        _proposal(changes={"type": "config_update", "config_key": "../etc", "config_value": 1})
    )
    assert ok is False  # traversal key rejected
    ok, detail = await eng._simulate(
        _proposal(changes={"type": "config_update", "config_key": "a.b.c", "config_value": 3})
    )
    assert ok is True, detail


# ── d1c690a2: non-finite values are refused ────────────────────────────────


@pytest.mark.asyncio
async def test_nan_value_adjustment_is_refused():
    eng = _engine()
    ok, _ = await eng._simulate(
        _proposal(changes={"type": "value_adjustment", "new_values": {"a": math.nan}})
    )
    assert ok is False
    ok, _ = await eng._simulate(
        _proposal(changes={"type": "threshold_adjustment", "new_threshold": math.inf})
    )
    assert ok is False


# ── 28126f78 + 20547554: audit readiness fails closed on more than OSError ─


def test_audit_ready_fails_closed_on_runtime_error(monkeypatch):
    class _Gateway:
        def append_text(self, *a, **k):
            raise RuntimeError("governance refused this write")

    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway", lambda: _Gateway()
    )
    ok, detail = AutonomousSelfModification._audit_log_ready()
    assert ok is False
    assert "governance refused" in detail


# ── 6ae6c708: dead apply helpers never mutate live values ──────────────────


@pytest.mark.asyncio
async def test_apply_helpers_refuse_to_mutate():
    eng = _engine()
    msg = await eng._apply(_proposal(changes={"type": "value_adjustment", "new_values": {"a": 0.5}}))
    assert "disabled" in msg.lower()
    msg2 = await eng._apply_value_adjustment({"new_values": {"a": 0.5}, "target_system": "heartstone"})
    assert "disabled" in msg2.lower()


# ── 06bdc857: audit log is a tamper-evident hash chain ─────────────────────


def test_audit_chain_detects_tampering():
    eng = _engine()
    for i in range(3):
        eng._record_receipt(
            smod.ModificationReceipt(
                proposal_id=f"p{i}",
                target_path="core/brain/x.py",
                description="d",
                diff_summary="s",
                source="t",
                outcome=ProposalOutcome.QUEUED_FOR_PIPELINE,
            )
        )
    ok, detail = eng.verify_audit_chain()
    assert ok is True, detail

    # Tamper with the middle entry on disk.
    path = smod._AUDIT_LOG_PATH
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["entry"]["description"] = "TAMPERED"
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")

    ok, detail = eng.verify_audit_chain()
    assert ok is False
    assert "mismatch" in detail


# ── c1cb078f: overflow is dead-lettered, not silently dropped ──────────────


def test_backlog_overflow_dead_letters_oldest():
    eng = _engine()
    dropped_events = []
    eng._publish_event = lambda topic, receipt: dropped_events.append((topic, receipt))  # type: ignore
    eng._MAX_PENDING = 3

    for i in range(5):
        p = _proposal(proposal_id=f"p{i}", changes={"type": "value_adjustment", "new_values": {"a": 0.5}})
        eng._queue_proposal(p.frozen(content_hash=p.content_hash(), will_receipt_id="w"))

    assert len(eng._pending) == 3
    dropped = [r for (t, r) in dropped_events if t == "self_modification.dropped"]
    assert len(dropped) == 2
    assert all(r.outcome == ProposalOutcome.DROPPED_OVERFLOW for r in dropped)


# ── 5e5f983f + 443cb5b6: durable outbox + consumer API + recovery ──────────


def test_queue_survives_restart_and_can_be_claimed():
    eng = _engine()
    p = _proposal(proposal_id="durable-1", changes={"type": "value_adjustment", "new_values": {"a": 0.5}})
    eng._queue_proposal(p.frozen(content_hash=p.content_hash(), will_receipt_id="w1"))

    # A fresh instance recovers the queued proposal from the durable outbox.
    revived = _engine()
    revived._recover_pending_from_outbox()
    assert [q.proposal_id for q in revived._pending] == ["durable-1"]

    claimed = revived.claim_next_pending()
    assert claimed is not None
    assert claimed["proposal_id"] == "durable-1"
    assert claimed["content_hash"] == p.content_hash()
    assert revived.claim_next_pending() is None  # nothing left

    assert revived.record_promotion_outcome("durable-1", "promoted", "ok") is True
    assert revived.record_promotion_outcome("durable-1", "not-a-state") is False


def test_claimed_proposal_is_not_re_handed_after_restart():
    eng = _engine()
    p = _proposal(proposal_id="once", changes={"type": "value_adjustment", "new_values": {"a": 0.5}})
    eng._queue_proposal(p.frozen(content_hash=p.content_hash(), will_receipt_id="w"))
    eng._recover_pending_from_outbox()
    assert eng.claim_next_pending() is not None

    # After the claim is durably recorded, a restart must not re-admit it.
    revived = _engine()
    revived._recover_pending_from_outbox()
    assert revived._pending == []


# ── 72afd1dc: a Will-approved queueing is not a zero approval rate ──────────


def test_authorization_rate_counts_queued_approvals():
    eng = _engine()
    for i in range(3):
        eng._record_receipt(
            smod.ModificationReceipt(
                proposal_id=f"ok{i}", target_path="core/brain/x.py", description="d",
                diff_summary="s", source="t", outcome=ProposalOutcome.QUEUED_FOR_PIPELINE,
            )
        )
    eng._record_receipt(
        smod.ModificationReceipt(
            proposal_id="no", target_path="core/security/x.py", description="d",
            diff_summary="s", source="t", outcome=ProposalOutcome.REFUSED_PROTECTED,
        )
    )
    status = eng.get_status()
    assert status["authorized"] == 3
    assert status["approval_rate"] == round(3 / 4, 4)
    assert status["approval_rate"] > 0.0


def test_queued_proposal_roundtrips_through_a_record():
    p = _proposal(changes={"type": "value_adjustment", "new_values": {"a": 0.25}})
    frozen = p.frozen(content_hash=p.content_hash(), will_receipt_id="w")
    restored = QueuedProposal.from_record(frozen.to_record())
    assert restored == frozen
