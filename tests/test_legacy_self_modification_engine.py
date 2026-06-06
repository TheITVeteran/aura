import importlib
import time


def _legacy_proposal(module, target):
    change = module.CodeChange(
        target_file=target,
        target_line=1,
        original_code="value = 1\n",
        modified_code="value = 2",
        change_type="modify",
        description="legacy write attempt",
        security_level=module.SecurityLevel.LOW,
        checksum="",
    )
    return module.ModificationProposal(
        id="legacy-direct-write",
        changes=[change],
        justification="prove deprecated path cannot mutate source",
        confidence=1.0,
        expected_impact="none",
        rollback_plan={},
        created_at=time.time(),
    )


def test_deprecated_self_modification_engine_blocks_public_source_mutation(tmp_path):
    legacy = importlib.import_module("core.self_modification_engine")
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    proposal = _legacy_proposal(legacy, target)
    engine = legacy.SelfModificationEngine(
        str(tmp_path),
        backup_dir=str(tmp_path / "backups"),
        require_human_approval=False,
    )

    result = engine.apply_proposal(proposal, force=True)

    assert result["success"] is False
    assert result["error"] == "legacy_self_modification_engine_direct_application_disabled"
    assert result["requires"] == "core.self_modification.safe_modification.SafeSelfModification"
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert engine.change_history == []
    assert engine.state is legacy.ModificationState.BLOCKED
    assert engine.audit_log[-1]["error"] == result["error"]


def test_deprecated_self_modification_engine_never_leaves_default_approval_path_applying(tmp_path):
    legacy = importlib.import_module("core.self_modification_engine")
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    proposal = _legacy_proposal(legacy, target)
    engine = legacy.SelfModificationEngine(str(tmp_path), backup_dir=str(tmp_path / "backups"))

    result = engine.apply_proposal(proposal)

    assert result["success"] is False
    assert result["error"] == "human_approval_required"
    assert result["requires"] == "core.self_modification.safe_modification.SafeSelfModification"
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert engine.active_proposal is None
    assert engine.state is legacy.ModificationState.IDLE


def test_deprecated_self_modification_engine_dry_run_is_analysis_only(tmp_path):
    legacy = importlib.import_module("core.self_modification_engine")
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    proposal = _legacy_proposal(legacy, target)
    engine = legacy.SelfModificationEngine(str(tmp_path), backup_dir=str(tmp_path / "backups"))

    result = engine.apply_proposal(proposal, dry_run=True)

    assert result == {"success": True, "dry_run": True}
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert engine.change_history == []
    assert engine.active_proposal is None
    assert engine.state is legacy.ModificationState.IDLE
    assert engine.audit_log[-1]["dry_run"] is True


def test_deprecated_self_modification_engine_private_apply_is_dry_run_only(tmp_path):
    legacy = importlib.import_module("core.self_modification_engine")
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    proposal = _legacy_proposal(legacy, target)
    engine = legacy.SelfModificationEngine(
        str(tmp_path),
        backup_dir=str(tmp_path / "backups"),
        require_human_approval=False,
    )
    change = proposal.changes[0]

    dry_run = engine._apply_change(change, dry_run=True)
    blocked = engine._apply_change(change, dry_run=False)

    assert dry_run["success"] is True
    assert dry_run["dry_run"] is True
    assert blocked["success"] is False
    assert blocked["error"] == "legacy_self_modification_engine_direct_application_disabled"
    assert target.read_text(encoding="utf-8") == "value = 1\n"
