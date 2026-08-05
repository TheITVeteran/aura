from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.brain.llm.latent_cortex.campaign_journal import CampaignJournal, CampaignPlan
from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from core.runtime.secure_path_custody import DirectoryCustody, path_custody_threat_model
from tools import run_resident_recurrent_sft_bootstrap_campaign as controller

_REAL_ACQUIRE_CAMPAIGN_CUSTODIES = controller._acquire_campaign_custodies
_REAL_RESIDENT_TRAINING_HOST_LEASE = controller._resident_training_host_lease


@pytest.fixture(autouse=True)
def _stub_campaign_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller, "_acquire_campaign_custodies", lambda _config: ())

    @contextmanager
    def _host_lease(**_kwargs):
        yield {"lease_sha256": "f" * 64}

    monkeypatch.setattr(controller, "_resident_training_host_lease", _host_lease)


def test_resident_training_host_lease_is_cross_campaign_and_crash_recoverable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "host.lock"
    first = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-first"
    second = "com.aura.resident-32b-recurrent-grpo-cp-second.post-training"

    with _REAL_RESIDENT_TRAINING_HOST_LEASE(
        label=first,
        config_sha256="a" * 64,
        lease_path=path,
    ) as lease:
        assert lease["active"] is True
        with pytest.raises(
            controller.ResidentSFTCampaignControllerError,
            match="resident_training_host_busy",
        ):
            with _REAL_RESIDENT_TRAINING_HOST_LEASE(
                label=second,
                config_sha256="b" * 64,
                lease_path=path,
            ):
                pass

    with _REAL_RESIDENT_TRAINING_HOST_LEASE(
        label=second,
        config_sha256="b" * 64,
        lease_path=path,
    ) as recovered:
        assert recovered["label"] == second

    persisted = json.loads(path.read_text())
    assert persisted["active"] is False
    assert persisted["label"] == second


def test_install_retirement_unloads_and_quarantines_every_superseded_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    quarantine = tmp_path / "quarantine"
    launch_agents.mkdir()
    active = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-active"
    stale_sft = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-stale"
    stale_grpo = "com.aura.resident-32b-recurrent-grpo-cp-stale.post-training"
    for label in (active, stale_sft, stale_grpo):
        (launch_agents / f"{label}.plist").write_bytes(
            controller.plistlib.dumps({"Label": label})
        )
    inventories = iter(
        [
            {active, stale_sft, stale_grpo},
            {active},
        ]
    )
    monkeypatch.setattr(
        controller,
        "_loaded_resident_training_labels",
        lambda: next(inventories),
    )
    calls: list[list[str]] = []

    def _run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(controller.subprocess, "run", _run)

    receipt = controller._retire_resident_training_jobs(
        active_label=active,
        launch_agents=launch_agents,
        quarantine_root=quarantine,
    )

    assert receipt["retired_labels"] == [stale_grpo, stale_sft]
    assert (launch_agents / f"{active}.plist").is_file()
    assert not (launch_agents / f"{stale_sft}.plist").exists()
    assert not (launch_agents / f"{stale_grpo}.plist").exists()
    assert len(list(quarantine.glob("*.plist"))) == 2
    assert calls == [
        ["/bin/launchctl", "bootout", f"gui/{controller.os.getuid()}/{stale_grpo}"],
        ["/bin/launchctl", "bootout", f"gui/{controller.os.getuid()}/{stale_sft}"],
    ]


def test_install_retirement_rejects_plist_filename_label_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    launch_agents.mkdir()
    active = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-active"
    substituted = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-other"
    (launch_agents / f"{active}.plist").write_bytes(
        controller.plistlib.dumps({"Label": substituted})
    )
    monkeypatch.setattr(controller, "_loaded_resident_training_labels", lambda: set())

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="resident_training_launchd_plist_label_mismatch",
    ):
        controller._retire_resident_training_jobs(
            active_label=active,
            launch_agents=launch_agents,
            quarantine_root=tmp_path / "quarantine",
        )


def _config() -> dict[str, Any]:
    body = {
        "schema": controller.CONTROLLER_CONFIG_SCHEMA,
        "campaign_id": "resident-32b-recurrent-sft-bootstrap-cp-test-canary",
        "profile": "canary",
        "source": {"branch": "main", "commit": "a" * 40, "origin_main": "a" * 40},
        "authority": {
            "path": "authority.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "semantic_sha256": "c" * 64,
        },
        "plan": {
            "path": "plan.json",
            "sha256": "d" * 64,
            "size_bytes": 1,
            "semantic_sha256": "e" * 64,
        },
        "paths": {
            "artifact_root": "artifacts/run",
            "training_output": "artifacts/run/training",
            "controller_root": "artifacts/run/controller",
            "journal": "artifacts/run/controller/campaign.journal.jsonl",
            "manifest": "artifacts/run/controller/campaign-manifest.json",
            "detached_attempts": "artifacts/run/controller/detached-attempts",
        },
        "path_custody": {
            "artifact_root": {"st_dev": 1, "st_ino": 2},
            "training_output": {"st_dev": 1, "st_ino": 3},
            "controller_root": {"st_dev": 1, "st_ino": 4},
        },
        "path_custody_threat_model": path_custody_threat_model(),
        "watchdog": {
            "schema": "aura.resident_recurrent_sft_controller_watchdog.v1",
            "max_attempts_per_cell": 6,
            "max_consecutive_no_progress_failures": 2,
            "poll_interval_s": 0.01,
            "heartbeat_stale_s": 1.0,
            "attempt_timeout_s": 10.0,
            "retry_backoff_s": 0.01,
            "resume_exact_checkpoint_only": True,
        },
        "launch": {
            "label": "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-test-canary",
            "launchd_required": True,
            "caffeinate_required": True,
        },
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        },
    }
    return {**body, "config_sha256": sha256_json(body)}


def _plan(config: dict[str, Any]) -> CampaignPlan:
    return CampaignPlan.build(
        config["campaign_id"],
        [
            {
                "invocation_ordinal": 1,
                "expected_start_step": 0,
                "required_end_step": 1,
            },
            {
                "invocation_ordinal": 2,
                "expected_start_step": 1,
                "required_end_step": 2,
            },
        ],
        metadata={"strict_execution_order": True},
    )


def _stub_verified_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controller,
        "_verify_execution_supervision",
        lambda *_args, **_kwargs: {
            "mode": "launchd_caffeinate",
            "launchd": True,
            "caffeinate": True,
        },
    )


def _snapshot(step: int, *, terminal: bool = False) -> dict[str, Any]:
    return {
        "present": step > 0,
        "step": step,
        "checkpoint_sequence": step + (1 if step else 0),
        "invocation_count": step,
        "terminal": terminal,
        "halt_reason": "max_steps" if terminal else None,
        "complete_sha256": f"{step + 1:064x}" if step else "",
        "model_identity_sha256": "f" * 64 if step else None,
    }


def test_attempt_record_requires_checkpoint_and_step_progress() -> None:
    config = _config()
    status = {"plan_sha256": "a" * 64, "receipt": {"returncode": 0, "receipt_sha256": "b" * 64}}
    record = controller._attempt_record(
        config=config,
        cell_id="cell",
        cell_ordinal=1,
        attempt_id="attempt",
        attempt_number=1,
        before=_snapshot(0),
        after=_snapshot(1),
        detached_status=status,
        required_end_step=1,
    )
    assert record["durable_progress"] is True
    assert record["required_end_reached"] is True
    assert record["terminal_success"] is True

    same = controller._attempt_record(
        config=config,
        cell_id="cell",
        cell_ordinal=1,
        attempt_id="attempt-2",
        attempt_number=2,
        before=_snapshot(1),
        after=_snapshot(1),
        detached_status=status,
        required_end_step=2,
    )
    assert same["durable_progress"] is False
    assert same["required_end_reached"] is False


def test_initial_checkpoint_accepts_only_the_exact_verified_migration_step() -> None:
    assert controller._initial_checkpoint_matches_plan(
        observed_step=0,
        required_start=0,
        required_end=4,
        migration_start=None,
    )
    assert controller._initial_checkpoint_matches_plan(
        observed_step=2,
        required_start=0,
        required_end=4,
        migration_start=2,
    )
    assert not controller._initial_checkpoint_matches_plan(
        observed_step=2,
        required_start=0,
        required_end=4,
        migration_start=None,
    )
    assert not controller._initial_checkpoint_matches_plan(
        observed_step=5,
        required_start=0,
        required_end=4,
        migration_start=5,
    )


def test_verified_migration_commits_covered_cells_before_partial_resume(tmp_path) -> None:
    config = _config()
    plan = _plan(config)
    journal_path = tmp_path / "campaign.journal.jsonl"
    snapshot = _snapshot(2)

    with CampaignJournal(journal_path, plan) as journal:
        first = plan.cell_ids[0]
        first_definition = plan.cell_definition(first)
        controller._commit_migration_covered_cell(
            journal,
            cell_id=first,
            required_start=int(first_definition["expected_start_step"]),
            required_end=int(first_definition["required_end_step"]),
            migration_start=2,
            snapshot=snapshot,
        )
        assert journal.resume().committed_cell_ids == (first,)
        second = plan.cell_ids[1]
        attempt_id = journal.start_cell(second)
        assert journal.attempt_status(second)["active_attempt_id"] == attempt_id

    with CampaignJournal(journal_path, plan) as recovered:
        assert recovered.resume().committed_cell_ids == (plan.cell_ids[0],)
        assert recovered.attempt_status(plan.cell_ids[1])["active_attempt_id"] is not None


def test_verified_migration_start_rejects_invalid_receipt(monkeypatch, tmp_path) -> None:
    receipt = tmp_path / "checkpoint-migration.json"
    receipt.write_text("{}", encoding="ascii")
    monkeypatch.setattr(controller, "_repo_path", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        controller,
        "verify_migration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            controller.ResidentSFTCheckpointMigrationError("invalid")
        ),
    )

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="checkpoint_migration_invalid",
    ):
        controller._verified_migration_start_step({"artifact_root": "ignored"})


def test_verified_migration_start_uses_historical_generation_after_progress(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "migration-step-5"
    checkpoint.mkdir(parents=True)
    (tmp_path / "checkpoint-migration.json").write_text("{}", encoding="ascii")
    migrated_state = {"step": 5}
    current_state = {"step": 21}
    descendant_calls: list[tuple[dict[str, int], dict[str, int]]] = []

    monkeypatch.setattr(controller, "_repo_path", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        controller,
        "verify_migration",
        lambda *_args, **_kwargs: {
            "destination": {"checkpoint": str(checkpoint)},
        },
    )
    monkeypatch.setattr(controller, "authority_state_bindings", lambda _authority: {})
    monkeypatch.setattr(
        controller,
        "inspect_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(state=migrated_state),
    )
    monkeypatch.setattr(
        controller,
        "inspect_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(state=current_state),
    )
    monkeypatch.setattr(
        controller,
        "validate_checkpoint_descendant",
        lambda earlier, later: descendant_calls.append((earlier, later)),
    )

    assert controller._verified_migration_start_step({"artifact_root": "ignored"}) == 5
    assert descendant_calls == [(migrated_state, current_state)]


def test_protected_source_changes_include_dynamic_and_declared_closure() -> None:
    changed = [
        "README.md",
        "core/brain/frontier_evidence_v5.py",
        "tools/train_resident_recurrent_sft_bootstrap.py",
    ]

    assert controller._protected_source_changes(
        changed,
        frozenset({"core/brain/frontier_evidence_v5.py"}),
    ) == (
        "core/brain/frontier_evidence_v5.py",
        "tools/train_resident_recurrent_sft_bootstrap.py",
    )


def test_live_trainer_import_closure_captures_transitive_source() -> None:
    closure = controller._trainer_import_closure()

    assert "tools/train_resident_recurrent_sft_bootstrap.py" in closure
    assert "core/brain/llm/latent_cortex/execution_spec.py" in closure
    assert "core/brain/frontier_evidence_v5.py" in closure

def test_launch_command_caps_partial_resume_to_exact_cell_remainder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    authority = {
        "authority_sha256": "c" * 64,
        "artifact_root": "artifacts/run/training",
        "runtime": {"interpreter": {"executable": str(Path(sys.executable).absolute())}},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    (tmp_path / "authority.json").write_text("{}")
    (tmp_path / "tools").mkdir()
    for name in (
        "run_resident_recurrent_sft_bootstrap_campaign.py",
        "train_resident_recurrent_sft_bootstrap.py",
    ):
        (tmp_path / "tools" / name).write_text("# bound\n")
    monkeypatch.setattr(
        controller,
        "__file__",
        str(tmp_path / "tools" / "run_resident_recurrent_sft_bootstrap_campaign.py"),
    )

    args = controller._launch_args(
        config_path=tmp_path / "config.json",
        config=config,
        authority=authority,
        run_dir=tmp_path / "run-dir",
        run_dir_identity={"st_dev": 1, "st_ino": 2},
        name="test",
        minimum_step=2,
        invocation_step_budget=2,
        required_end_step=4,
        resume=False,
    )

    budget_index = args.index("--invocation-step-budget")
    assert args[budget_index + 1] == "2"
    target_index = args.index("--required-end-step")
    assert args[target_index + 1] == "4"
    lexical_python = str(Path(sys.executable).absolute())
    resolved_python = str(Path(sys.executable).resolve())
    assert lexical_python in args
    if resolved_python != lexical_python:
        assert resolved_python not in args
    verifier = json.loads(args[args.index("--resume-verifier-json") + 1])
    assert verifier[-2:] == ["--minimum-step", "2"]


def test_config_rejects_nested_watchdog_or_launch_drift(tmp_path: Path) -> None:
    config = _config()
    config["watchdog"]["max_consecutive_no_progress_failures"] = 3
    body = dict(config)
    body.pop("config_sha256")
    config["config_sha256"] = sha256_json(body)
    path = tmp_path / "config.json"
    path.write_bytes(controller._canonical(config))
    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="watchdog_invalid",
    ):
        controller._load_config(path)


def test_config_accepts_published_worktree_branch(tmp_path: Path) -> None:
    config = _config()
    config["source"]["branch"] = "codex/rlc-control-candidate"
    body = dict(config)
    body.pop("config_sha256")
    config["config_sha256"] = sha256_json(body)
    path = tmp_path / "config.json"
    path.write_bytes(controller._canonical(config))

    assert controller._load_config(path)["source"]["branch"] == (
        "codex/rlc-control-candidate"
    )


def test_config_accepts_detached_source_bound_to_published_commit(
    tmp_path: Path,
) -> None:
    config = _config()
    config["source"]["branch"] = ""
    body = dict(config)
    body.pop("config_sha256")
    config["config_sha256"] = sha256_json(body)
    path = tmp_path / "config.json"
    path.write_bytes(controller._canonical(config))

    assert controller._load_config(path)["source"]["branch"] == ""


def test_source_lineage_accepts_worktree_at_published_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = "a" * 40

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = command[1:]
        if args == ["branch", "--show-current"]:
            stdout, returncode = "codex/rlc-control-candidate\n", 0
        elif args in (["rev-parse", "HEAD"], ["rev-parse", "origin/main"]):
            stdout, returncode = f"{frozen}\n", 0
        elif args[:2] == ["merge-base", "--is-ancestor"]:
            stdout, returncode = "", 0
        elif args[:2] == ["diff", "--name-only"]:
            stdout, returncode = "", 0
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller, "_trainer_import_closure", lambda: frozenset())

    assert controller._verify_source_lineage({"commit": frozen}) == {
        "branch": "codex/rlc-control-candidate",
        "frozen_commit": frozen,
        "observed_head": frozen,
        "observed_origin_main": frozen,
    }


def test_source_lineage_accepts_detached_checkout_at_published_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = "a" * 40

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = command[1:]
        if args == ["branch", "--show-current"]:
            stdout, returncode = "", 0
        elif args in (["rev-parse", "HEAD"], ["rev-parse", "origin/main"]):
            stdout, returncode = f"{frozen}\n", 0
        elif args[:2] == ["merge-base", "--is-ancestor"]:
            stdout, returncode = "", 0
        elif args[:2] == ["diff", "--name-only"]:
            stdout, returncode = "", 0
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller, "_trainer_import_closure", lambda: frozenset())

    assert controller._verify_source_lineage({"commit": frozen})["branch"] == ""


def test_config_rejects_changed_path_custody_threat_model(tmp_path: Path) -> None:
    config = _config()
    config["path_custody_threat_model"]["excluded_adversary"] = "none"
    body = dict(config)
    body.pop("config_sha256")
    config["config_sha256"] = sha256_json(body)
    path = tmp_path / "config.json"
    path.write_bytes(controller._canonical(config))

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="path_custody_threat_model_invalid",
    ):
        controller._load_config(path)


def test_controller_acquires_exact_prepared_directory_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    paths = {
        "artifact_root": tmp_path / "artifacts" / "run",
        "training_output": tmp_path / "artifacts" / "run" / "training",
        "controller_root": tmp_path / "artifacts" / "run" / "controller",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for role, path in paths.items():
        observed = path.stat()
        config["path_custody"][role] = {
            "st_dev": observed.st_dev,
            "st_ino": observed.st_ino,
        }

    custodies = _REAL_ACQUIRE_CAMPAIGN_CUSTODIES(config)
    try:
        controller._verify_campaign_custodies(custodies)
        assert [custody.path for custody in custodies] == list(paths.values())
    finally:
        for custody in reversed(custodies):
            custody.close()


def test_controller_completes_two_cells_and_never_promotes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verified_supervision(monkeypatch)
    config = _config()
    plan = _plan(config)
    authority = {
        "authority_sha256": "c" * 64,
        "campaign_scope": "canary_lifecycle",
        "artifact_root": "artifacts/run/training",
        "trainer": {"max_steps": 2},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_config", lambda _path: config)
    monkeypatch.setattr(controller, "_load_contracts", lambda _path: (config, authority, plan))
    monkeypatch.setattr(controller, "_verify_source_lineage", lambda _source: {})
    monkeypatch.setattr(controller, "_verify_authority_artifacts", lambda _authority: None)
    snapshots = iter(
        [
            _snapshot(0),
            _snapshot(0),
            _snapshot(1),
            _snapshot(1),
            _snapshot(1),
            _snapshot(2, terminal=True),
            _snapshot(2, terminal=True),
        ]
    )
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: next(snapshots))
    monkeypatch.setattr(
        controller,
        "_wait_attempt",
        lambda **_kwargs: {
            "plan_sha256": "a" * 64,
            "receipt": {"returncode": 0, "receipt_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        controller,
        "_invocation_receipt",
        lambda _authority, snapshot: {
            "canary_lifecycle_complete": snapshot["terminal"],
            "bootstrap_complete": False,
            "claim_state": {"resident_sft_complete": False},
        },
    )
    monkeypatch.setattr(
        controller,
        "_verify_commit_evidence",
        lambda **kwargs: (
            dict(kwargs["record"]["progress_after"]),
            {
                "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
                "attempt_sha256": kwargs["record"]["attempt_sha256"],
                "checkpoint_complete_sha256": kwargs["record"]["progress_after"]["complete_sha256"],
                "invocation_receipt_sha256": "a" * 64,
                "required_end_reached": True,
                "base_checkpoint_fingerprint": "f" * 64,
                "base_checkpoint_immutable": True,
            },
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    receipt = controller.run_controller(config_path, launchd_supervised=True)

    assert receipt["campaign_scope"] == "canary_lifecycle"
    assert receipt["canary_lifecycle_complete"] is True
    assert receipt["bootstrap_complete"] is False
    assert receipt["claim_state"]["reasoning_gain_proven"] is False
    assert receipt["claim_state"]["promotion_allowed"] is False
    root = tmp_path / "artifacts" / "run" / "controller"
    assert len(list((root / "attempt-results").glob("*.json"))) == 2
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "completed"


def test_controller_stops_after_two_consecutive_no_progress_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verified_supervision(monkeypatch)
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    authority = {
        "authority_sha256": "c" * 64,
        "artifact_root": "artifacts/run/training",
        "trainer": {"max_steps": 1},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_config", lambda _path: config)
    monkeypatch.setattr(controller, "_load_contracts", lambda _path: (config, authority, plan))
    monkeypatch.setattr(controller, "_verify_source_lineage", lambda _source: {})
    monkeypatch.setattr(controller, "_verify_authority_artifacts", lambda _authority: None)
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: _snapshot(0))
    monkeypatch.setattr(
        controller,
        "_wait_attempt",
        lambda **_kwargs: {
            "plan_sha256": "a" * 64,
            "receipt": {"returncode": 1, "receipt_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="no_progress_limit_exhausted",
    ):
        controller.run_controller(config_path, launchd_supervised=True)

    results = list(
        (tmp_path / "artifacts" / "run" / "controller" / "attempt-results").glob("*.json")
    )
    assert len(results) == 2


def test_reconcile_imports_staged_success_after_controller_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    root = tmp_path / "controller"
    root.mkdir()
    journal_path = root / "campaign.journal.jsonl"
    with CampaignJournal(journal_path, plan) as journal:
        cell_id = plan.cell_ids[0]
        attempt_id = journal.start_cell(cell_id)
        record = controller._attempt_record(
            config=config,
            cell_id=cell_id,
            cell_ordinal=1,
            attempt_id=attempt_id,
            attempt_number=1,
            before=_snapshot(0),
            after=_snapshot(1),
            detached_status={
                "plan_sha256": "a" * 64,
                "receipt": {"returncode": 0, "receipt_sha256": "b" * 64},
            },
            required_end_step=1,
        )
        controller._write_once(
            root / "attempt-results" / "cell-0001-attempt-0001.json",
            record,
        )
    with CampaignJournal(journal_path, plan) as recovered:
        monkeypatch.setattr(
            controller,
            "_verify_commit_evidence",
            lambda **kwargs: (
                dict(kwargs["record"]["progress_after"]),
                {
                    "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
                    "attempt_sha256": kwargs["record"]["attempt_sha256"],
                    "checkpoint_complete_sha256": kwargs["record"]["progress_after"][
                        "complete_sha256"
                    ],
                    "invocation_receipt_sha256": "a" * 64,
                    "required_end_reached": True,
                    "base_checkpoint_fingerprint": "f" * 64,
                    "base_checkpoint_immutable": True,
                },
            ),
        )
        controller._reconcile_staged_results(recovered, root, plan, {})
        assert recovered.resume().committed_cell_ids == plan.cell_ids


def test_reconcile_skips_target_checkpoint_with_failed_detached_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    root = tmp_path / "controller"
    root.mkdir()
    journal_path = root / "campaign.journal.jsonl"
    with CampaignJournal(journal_path, plan) as journal:
        cell_id = plan.cell_ids[0]
        attempt_id = journal.start_cell(cell_id)
        record = controller._attempt_record(
            config=config,
            cell_id=cell_id,
            cell_ordinal=1,
            attempt_id=attempt_id,
            attempt_number=1,
            before=_snapshot(0),
            after=_snapshot(1),
            detached_status={
                "plan_sha256": "a" * 64,
                "receipt": {"returncode": 1, "receipt_sha256": "b" * 64},
            },
            required_end_step=1,
        )
        controller._write_once(
            root / "attempt-results" / "cell-0001-attempt-0001.json",
            record,
        )
    monkeypatch.setattr(
        controller,
        "_verify_commit_evidence",
        lambda **_kwargs: pytest.fail("failed result must not be committed"),
    )
    with CampaignJournal(journal_path, plan) as recovered:
        controller._reconcile_staged_results(recovered, root, plan, {})
        assert recovered.resume().committed_cell_ids == ()


def test_restart_after_journal_start_without_reservation_fails_attempt_then_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verified_supervision(monkeypatch)
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    authority = {
        "authority_sha256": "c" * 64,
        "campaign_scope": "canary_lifecycle",
        "artifact_root": "artifacts/run/training",
        "trainer": {"max_steps": 1},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_config", lambda _path: config)
    monkeypatch.setattr(controller, "_load_contracts", lambda _path: (config, authority, plan))
    monkeypatch.setattr(controller, "_verify_source_lineage", lambda _source: {})
    monkeypatch.setattr(controller, "_verify_authority_artifacts", lambda _authority: None)
    root = tmp_path / "artifacts" / "run" / "controller"
    root.mkdir(parents=True)
    journal_path = root / "campaign.journal.jsonl"
    with CampaignJournal(journal_path, plan) as journal:
        journal.start_cell(plan.cell_ids[0])
    snapshots = iter(
        [
            _snapshot(0),
            _snapshot(0),
            _snapshot(0),
            _snapshot(1, terminal=True),
            _snapshot(1, terminal=True),
        ]
    )
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: next(snapshots))
    monkeypatch.setattr(
        controller,
        "_wait_attempt",
        lambda **_kwargs: {
            "plan_sha256": "a" * 64,
            "receipt": {"returncode": 0, "receipt_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        controller,
        "_invocation_receipt",
        lambda _authority, _snapshot: {
            "canary_lifecycle_complete": True,
            "bootstrap_complete": False,
            "claim_state": {"resident_sft_complete": False},
            "receipt_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        controller,
        "_verify_commit_evidence",
        lambda **kwargs: (
            dict(kwargs["record"]["progress_after"]),
            {
                "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
                "attempt_sha256": kwargs["record"]["attempt_sha256"],
                "checkpoint_complete_sha256": kwargs["record"]["progress_after"]["complete_sha256"],
                "invocation_receipt_sha256": "a" * 64,
                "required_end_reached": True,
                "base_checkpoint_fingerprint": "f" * 64,
                "base_checkpoint_immutable": True,
            },
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    receipt = controller.run_controller(config_path, launchd_supervised=True)

    assert receipt["canary_lifecycle_complete"] is True
    with CampaignJournal(journal_path, plan) as recovered:
        assert recovered.attempt_status(plan.cell_ids[0])["attempt_count"] == 2


def test_target_checkpoint_with_failed_receipt_is_certified_without_overshoot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verified_supervision(monkeypatch)
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    authority = {
        "authority_sha256": "c" * 64,
        "campaign_scope": "canary_lifecycle",
        "artifact_root": "artifacts/run/training",
        "trainer": {"max_steps": 1},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_config", lambda _path: config)
    monkeypatch.setattr(controller, "_load_contracts", lambda _path: (config, authority, plan))
    monkeypatch.setattr(controller, "_verify_source_lineage", lambda _source: {})
    monkeypatch.setattr(controller, "_verify_authority_artifacts", lambda _authority: None)
    snapshots = iter(
        [
            _snapshot(0),
            _snapshot(0),
            _snapshot(1, terminal=True),
            _snapshot(1, terminal=True),
            _snapshot(1, terminal=True),
            _snapshot(1, terminal=True),
        ]
    )
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: next(snapshots))
    waits: list[dict[str, Any]] = []

    def wait_attempt(**kwargs: Any) -> dict[str, Any]:
        waits.append(kwargs)
        return {
            "plan_sha256": "a" * 64,
            "receipt": {
                "returncode": 1 if len(waits) == 1 else 0,
                "receipt_sha256": "b" * 64,
            },
        }

    monkeypatch.setattr(controller, "_wait_attempt", wait_attempt)
    monkeypatch.setattr(
        controller,
        "_invocation_receipt",
        lambda _authority, _snapshot: {
            "canary_lifecycle_complete": True,
            "bootstrap_complete": False,
            "claim_state": {"resident_sft_complete": False},
            "receipt_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        controller,
        "_verify_commit_evidence",
        lambda **kwargs: (
            dict(kwargs["record"]["progress_after"]),
            {
                "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
                "attempt_sha256": kwargs["record"]["attempt_sha256"],
                "checkpoint_complete_sha256": kwargs["record"]["progress_after"]["complete_sha256"],
                "invocation_receipt_sha256": "a" * 64,
                "required_end_reached": True,
                "base_checkpoint_fingerprint": "f" * 64,
                "base_checkpoint_immutable": True,
            },
        ),
    )
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    receipt = controller.run_controller(config_path, launchd_supervised=True)

    assert receipt["canary_lifecycle_complete"] is True
    assert len(waits) == 2
    assert waits[0]["minimum_step"] == 0
    assert waits[1]["minimum_step"] == 1
    assert waits[1]["required_end_step"] == 1
    assert waits[1]["invocation_step_budget"] == 1


def test_stale_detached_heartbeat_requests_authenticated_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    run_dir = tmp_path / "detached"
    run_dir.mkdir()
    (run_dir / controller.detached.PLAN_FILE).write_text("{}")
    statuses = iter(
        [
            {
                "terminal": False,
                "completion_indeterminate": False,
                "heartbeat_at": time.time() - 30.0,
            },
            {
                "terminal": True,
                "completion_indeterminate": False,
                "heartbeat_at": time.time(),
                "receipt": {"returncode": 143},
            },
        ]
    )
    stops: list[Path] = []
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda _run_dir, **_kwargs: next(statuses),
    )
    monkeypatch.setattr(
        controller.detached,
        "_stop",
        lambda path, **_kwargs: stops.append(path) or {"stopped": True},
    )
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)

    status = controller._wait_attempt(
        config_path=tmp_path / "config.json",
        config=config,
        authority={"authority_sha256": "a" * 64, "artifact_root": "artifacts/run"},
        run_dir=run_dir,
        name="stale-test",
        minimum_step=0,
        invocation_step_budget=1,
        required_end_step=1,
    )

    assert status["terminal"] is True
    assert stops == [run_dir]


def test_every_profile_refuses_manual_controller_entrypoint() -> None:
    config = _config()
    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="requires_launchd_entrypoint",
    ):
        controller._verify_execution_supervision(config, launchd_supervised=False)


def test_full_profile_accepts_launchd_owned_controller_with_exact_caffeinate_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["profile"] = "full"
    config["launch"] = {
        "label": f"com.aura.resident-sft.{config['campaign_id']}",
        "launchd_required": True,
        "caffeinate_required": True,
    }
    monkeypatch.setattr(
        controller,
        "_launchd_job",
        lambda _label: {"target": "gui/501/test", "job_pid": 4242},
    )
    monkeypatch.setattr(controller.os, "getpid", lambda: 4242)
    monkeypatch.setattr(controller.os, "getppid", lambda: 1)
    monkeypatch.setattr(controller.sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(
        controller.sys,
        "argv",
        [
            str(controller.Path(controller.__file__).resolve()),
            "run",
            "--config",
            "/repo/controller-config.json",
            "--launchd-supervised",
        ],
    )
    expected = (
        "/usr/bin/caffeinate -i /venv/bin/python "
        f"{controller.Path(controller.__file__).resolve()} run --config "
        "/repo/controller-config.json --launchd-supervised\n"
    )
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"4243 4242 {expected}",
            stderr="",
        ),
    )

    supervision = controller._verify_execution_supervision(config, launchd_supervised=True)

    assert supervision["mode"] == "launchd_caffeinate"
    assert supervision["launchd_pid"] == 4242
    assert supervision["controller_pid"] == 4242
    assert supervision["caffeinate_pid"] == 4243


def test_full_profile_rejects_launchd_pid_that_is_not_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["profile"] = "full"
    monkeypatch.setattr(
        controller,
        "_launchd_job",
        lambda _label: {"target": "gui/501/test", "job_pid": 4242},
    )
    monkeypatch.setattr(controller.os, "getpid", lambda: 4343)

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="launchd_parent_mismatch",
    ):
        controller._verify_execution_supervision(config, launchd_supervised=True)


def test_full_profile_rejects_missing_exact_caffeinate_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["profile"] = "full"
    monkeypatch.setattr(
        controller,
        "_launchd_job",
        lambda _label: {"target": "gui/501/test", "job_pid": 4242},
    )
    monkeypatch.setattr(controller.os, "getpid", lambda: 4242)
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="4243 4242 /usr/bin/caffeinate -t 60\n",
            stderr="",
        ),
    )

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="caffeinate_parent_missing",
    ):
        controller._verify_execution_supervision(config, launchd_supervised=True)


def test_recovered_active_attempt_reattaches_from_partial_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verified_supervision(monkeypatch)
    config = _config()
    plan = CampaignPlan.build(
        config["campaign_id"],
        [{"expected_start_step": 0, "required_end_step": 2}],
        metadata={"strict_execution_order": True},
    )
    authority = {
        "authority_sha256": "c" * 64,
        "campaign_scope": "canary_lifecycle",
        "artifact_root": "artifacts/run/training",
        "trainer": {"max_steps": 2},
    }
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_config", lambda _path: config)
    monkeypatch.setattr(controller, "_load_contracts", lambda _path: (config, authority, plan))
    monkeypatch.setattr(controller, "_verify_source_lineage", lambda _source: {})
    monkeypatch.setattr(controller, "_verify_authority_artifacts", lambda _authority: None)
    root = tmp_path / "artifacts" / "run" / "controller"
    root.mkdir(parents=True)
    journal_path = root / "campaign.journal.jsonl"
    cell_id = plan.cell_ids[0]
    with CampaignJournal(journal_path, plan) as journal:
        attempt_id = journal.start_cell(cell_id)
        controller._reserve_attempt(
            root=root,
            config=config,
            cell_id=cell_id,
            cell_ordinal=1,
            attempt_id=attempt_id,
            attempt_number=1,
            before=_snapshot(0),
            required_end_step=2,
        )

    snapshots = iter([_snapshot(1), _snapshot(2, terminal=True), _snapshot(2, terminal=True)])
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: next(snapshots))
    launched: list[dict[str, Any]] = []
    monkeypatch.setattr(
        controller,
        "_wait_attempt",
        lambda **kwargs: (
            launched.append(kwargs)
            or {
                "plan_sha256": "a" * 64,
                "receipt": {"returncode": 0, "receipt_sha256": "b" * 64},
            }
        ),
    )
    monkeypatch.setattr(
        controller,
        "_invocation_receipt",
        lambda _authority, snapshot: {
            "canary_lifecycle_complete": snapshot["terminal"],
            "bootstrap_complete": False,
            "claim_state": {"resident_sft_complete": False},
            "receipt_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        controller,
        "_verify_commit_evidence",
        lambda **kwargs: (
            dict(kwargs["record"]["progress_after"]),
            {
                "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
                "attempt_sha256": kwargs["record"]["attempt_sha256"],
                "checkpoint_complete_sha256": kwargs["record"]["progress_after"]["complete_sha256"],
                "invocation_receipt_sha256": "a" * 64,
                "required_end_reached": True,
                "base_checkpoint_fingerprint": "f" * 64,
                "base_checkpoint_immutable": True,
            },
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    receipt = controller.run_controller(config_path, launchd_supervised=True)

    assert receipt["canary_lifecycle_complete"] is True
    assert receipt["bootstrap_complete"] is False
    assert len(launched) == 1
    assert launched[0]["minimum_step"] == 0
    assert launched[0]["required_end_step"] == 2


def test_custodied_controller_write_rejects_root_exchange_without_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "controller"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    custody = DirectoryCustody.acquire(root, create=True, private=True)
    monkeypatch.setattr(controller, "_ACTIVE_CUSTODIES", (custody,))
    displaced = tmp_path / "displaced"
    root.rename(displaced)
    replacement.rename(root)

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="custodied_write_failed",
    ):
        controller._write_once(root / "attempt-results" / "result.json", {"ok": True})

    custody.close()
    assert list(root.iterdir()) == []
    assert not (displaced / "attempt-results" / "result.json").exists()


def test_custodied_controller_write_rejects_nested_symlink_without_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "controller"
    outside = tmp_path / "outside"
    outside.mkdir()
    custody = DirectoryCustody.acquire(root, create=True, private=True)
    monkeypatch.setattr(controller, "_ACTIVE_CUSTODIES", (custody,))
    (root / "attempt-results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="custodied_write_failed",
    ):
        controller._write_once(root / "attempt-results" / "result.json", {"ok": True})

    custody.close()
    assert list(outside.iterdir()) == []


def test_launchd_supervised_unknown_exception_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        controller,
        "run_controller",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    assert (
        controller.main(
            ["run", "--config", str(tmp_path / "config.json"), "--launchd-supervised"]
        )
        == 0
    )


def test_detached_cli_exit_becomes_stable_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller.detached,
        "main",
        lambda _args: (_ for _ in ()).throw(SystemExit(2)),
    )

    with pytest.raises(
        controller.ResidentSFTCampaignControllerError,
        match="detached_cli_contract_invalid",
    ):
        controller._invoke_detached(
            ["launch"],
            failure_code="resident_sft_controller_detached_cli_contract_invalid",
        )


def test_main_does_not_publish_failure_after_controller_releases_custody(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        controller,
        "run_controller",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            controller.ResidentSFTCampaignControllerError(
                "resident_sft_controller_source_lineage_drift"
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_publish_status",
        lambda *_args, **_kwargs: pytest.fail("main must not republish after custody release"),
    )

    assert (
        controller.main(
            ["run", "--config", str(tmp_path / "config.json"), "--launchd-supervised"]
        )
        == 0
    )


def test_resume_verifier_emits_exact_in_band_detached_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    authority = {"model": {"path": "model", "base_checkpoint": {"fingerprint": "base"}}}
    snapshot = _snapshot(1)
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: snapshot)
    monkeypatch.setattr(
        controller,
        "_repo_path",
        lambda *_args, **_kwargs: Path("/tmp/model"),
    )
    monkeypatch.setattr(
        controller,
        "full_weight_checkpoint_identity",
        lambda _path: authority["model"]["base_checkpoint"],
    )
    monkeypatch.setenv("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT", "stdout-v3")
    monkeypatch.setenv("AURA_DETACHED_PLAN_SHA256", "1" * 64)
    monkeypatch.setenv("AURA_DETACHED_COMMAND_SHA256", "2" * 64)
    monkeypatch.setenv("AURA_DETACHED_PRIOR_ATTEMPT", "1")
    monkeypatch.setenv("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "3" * 64)

    verdict = controller._verify_resume_custodied(config, authority, 1)

    assert verdict["schema"] == "aura.detached_step.resume_verdict.v3"
    assert verdict["evidence"]["schema"] == "aura.detached_step.resume_evidence.v2"
    assert verdict["evidence"]["checkpoint"] == snapshot
    assert verdict["evidence_sha256"] == sha256_json(verdict["evidence"])


def test_controller_resume_verdict_is_accepted_by_detached_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    authority = {"model": {"path": "model", "base_checkpoint": {"fingerprint": "base"}}}
    snapshot = _snapshot(1)
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _authority: snapshot)
    monkeypatch.setattr(controller, "_repo_path", lambda *_args, **_kwargs: Path("/tmp/model"))
    monkeypatch.setattr(
        controller,
        "full_weight_checkpoint_identity",
        lambda _path: authority["model"]["base_checkpoint"],
    )
    monkeypatch.setattr(
        controller.detached,
        "_verify_execution_manifest_current",
        lambda _manifest: None,
    )
    plan = {
        "plan_sha256": "1" * 64,
        "command_sha256": "2" * 64,
        "cwd": str(tmp_path),
        "execution_environment": {},
        "resume_verifier_command": ["/usr/bin/true"],
        "resume_verifier_execution_manifest": {},
    }

    def run_verifier(*_args, **kwargs):
        environment = kwargs["env"]
        with monkeypatch.context() as context:
            for key, value in environment.items():
                context.setenv(key, value)
            verdict = controller._verify_resume_custodied(config, authority, 1)
        return subprocess.CompletedProcess(
            args=["/usr/bin/true"],
            returncode=0,
            stdout=json.dumps(verdict),
            stderr="",
        )

    monkeypatch.setattr(controller.detached.subprocess, "run", run_verifier)

    verdict = controller.detached._run_resume_verifier(
        plan,
        tmp_path,
        prior_attempt=1,
        prior_journal_head_sha256="3" * 64,
    )

    assert verdict["schema"] == "aura.detached_step.resume_verdict.v3"
    assert verdict["verdict"] == "safe_to_resume"
    assert not list(tmp_path.glob("resume_evidence*.json"))
