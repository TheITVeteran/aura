from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    REQUIRED_SOURCE_ROLES,
    sha256_bytes,
    sha256_json,
)
from core.runtime.secure_path_custody import path_custody_threat_model
from tools import prepare_resident_recurrent_sft_bootstrap_campaign as prepare
from tools import run_resident_recurrent_sft_bootstrap_campaign as controller


def _identity(digest_field: str, value: str) -> dict[str, Any]:
    return {digest_field: value}


def _source_bindings(root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_SOURCE_ROLES:
        path = root / f"source-{role}.py"
        payload = f"# {role}\n".encode()
        path.write_bytes(payload)
        bindings[role] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return bindings


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, profile: str) -> dict[str, Any]:
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"eos_token_id":2}')
    spec = RLCExecutionSpec(recurrent_steps=2)
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(prepare._canonical(spec.to_dict()))
    source_state = {"branch": "main", "commit": "a" * 40, "origin_main": "a" * 40}
    monkeypatch.setattr(prepare, "_git_source_state", lambda: source_state)
    monkeypatch.setattr(prepare, "_source_bindings", lambda: _source_bindings(tmp_path))
    monkeypatch.setattr(prepare, "load_resident_bootstrap_tokenizer", lambda _path: object())
    monkeypatch.setattr(
        prepare,
        "resident_bootstrap_tokenizer_identity",
        lambda _path, _tokenizer: {
            "identity_sha256": "1" * 64,
            "artifact_sha256": "2" * 64,
            "runtime_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        prepare,
        "full_weight_checkpoint_identity",
        lambda _path: {"fingerprint": "4" * 64, "method": "sha256", "files": 1},
    )
    monkeypatch.setattr(
        prepare,
        "model_behavior_bundle_identity",
        lambda _path: {"bundle_sha256": "5" * 64, "file_count": 0, "files": []},
    )
    monkeypatch.setattr(
        prepare,
        "absent_personality_identity",
        lambda: {
            "present": False,
            "bundle_sha256": "",
            "file_count": 0,
            "files": [],
            "identity_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(
        prepare,
        "resident_bootstrap_runtime_identity",
        lambda: {
            "identity_sha256": "7" * 64,
            "runtime": "test",
            "interpreter": {"executable": "/test/venv/bin/python"},
        },
    )
    monkeypatch.setattr(prepare, "_probe_training_entrypoint", lambda _runtime: None)
    return prepare.prepare_campaign(
        profile=profile,
        campaign_id=f"resident-32b-recurrent-sft-bootstrap-cp-test-{profile}",
        model="model",
        execution_spec="spec.json",
        artifact_root=f"artifacts/{profile}",
        seed=19,
        committed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_load_spec_accepts_pretty_json_while_preserving_exact_bytes(tmp_path: Path) -> None:
    spec = RLCExecutionSpec(recurrent_steps=2)
    payload = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path = tmp_path / "spec.json"
    path.write_bytes(payload)

    loaded, observed_payload = prepare._load_spec(path)

    assert loaded.to_dict() == spec.to_dict()
    assert observed_payload == payload


def test_load_spec_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text('{"schema":"aura.rlc_execution_spec.v1","schema":"duplicate"}')

    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="execution_spec_duplicate_key",
    ):
        prepare._load_spec(path)


@pytest.mark.parametrize(
    ("profile", "steps", "invocations", "train_count", "validation_count"),
    [("canary", 5, 5, 12, 12), ("full", 104, 26, 144, 72)],
)
def test_prepare_campaign_freezes_disjoint_profile_and_claim_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    steps: int,
    invocations: int,
    train_count: int,
    validation_count: int,
) -> None:
    receipt = _prepare(monkeypatch, tmp_path, profile=profile)
    root = tmp_path / "artifacts" / profile
    authority = json.loads((root / "inputs" / "authority.json").read_text())
    controller_config = json.loads((root / "controller-config.json").read_text())
    train = json.loads((root / "inputs" / "train.json").read_text())
    validation = json.loads((root / "inputs" / "validation.json").read_text())

    assert receipt["max_steps"] == steps
    assert receipt["invocation_count"] == invocations
    assert receipt["train_count"] == train_count
    assert receipt["validation_count"] == validation_count
    assert {row["task_id"] for row in train}.isdisjoint(row["task_id"] for row in validation)
    assert authority["claim_state"]["reasoning_gain_proven"] is False
    assert authority["campaign_scope"] == (
        "canary_lifecycle" if profile == "canary" else "full_bootstrap"
    )
    training_stat = (root / "training").stat()
    assert authority["artifact_root_identity"] == {
        "st_dev": training_stat.st_dev,
        "st_ino": training_stat.st_ino,
    }
    assert authority["post_training_gate"]["fresh_heldout_tasks_required"] is True
    assert controller_config["watchdog"]["max_consecutive_no_progress_failures"] == 2
    assert controller_config["claim_state"]["promotion_allowed"] is False
    assert controller_config["path_custody_threat_model"] == path_custody_threat_model()
    for role, path in {
        "artifact_root": root,
        "training_output": root / "training",
        "controller_root": root / "controller",
    }.items():
        observed = path.stat()
        assert controller_config["path_custody"][role] == {
            "st_dev": observed.st_dev,
            "st_ino": observed.st_ino,
        }

    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    loaded_controller = controller._load_config(root / "controller-config.json")
    assert loaded_controller["config_sha256"] == controller_config["config_sha256"]

    replay = _prepare(monkeypatch, tmp_path, profile=profile)
    assert replay == receipt


def test_prepare_refuses_dirty_or_unpublished_main(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        prepare.subprocess,
        "run",
        lambda *args, **kwargs: Result(" M tracked.py\n"),
    )
    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="tracked_source_dirty",
    ):
        prepare._git_source_state()


def test_prepare_accepts_clean_published_worktree_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    head = "a" * 40

    def run(command: list[str], **_kwargs: Any) -> Result:
        args = command[1:]
        if args[:3] == ["diff", "--name-only", "HEAD"]:
            return Result("")
        if args == ["rev-parse", "HEAD"]:
            return Result(f"{head}\n")
        if args == ["branch", "--show-current"]:
            return Result("codex/rlc-control-candidate\n")
        if args == ["rev-parse", "origin/main"]:
            return Result(f"{head}\n")
        raise AssertionError(args)

    monkeypatch.setattr(prepare.subprocess, "run", run)

    assert prepare._git_source_state() == {
        "branch": "codex/rlc-control-candidate",
        "commit": head,
        "origin_main": head,
    }


def test_prepare_accepts_clean_published_detached_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    head = "a" * 40

    def run(command: list[str], **_kwargs: Any) -> Result:
        args = command[1:]
        if args[:3] == ["diff", "--name-only", "HEAD"]:
            return Result("")
        if args in (["rev-parse", "HEAD"], ["rev-parse", "origin/main"]):
            return Result(f"{head}\n")
        if args == ["branch", "--show-current"]:
            return Result("")
        raise AssertionError(args)

    monkeypatch.setattr(prepare.subprocess, "run", run)

    assert prepare._git_source_state() == {
        "branch": "",
        "commit": head,
        "origin_main": head,
    }


def test_invalid_campaign_id_is_rejected_before_any_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    output = tmp_path / "artifacts" / "invalid"

    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="campaign_identity_invalid",
    ):
        prepare.prepare_campaign(
            profile="full",
            campaign_id="role-v6-full",
            model="missing-model",
            execution_spec="missing-spec.json",
            artifact_root="artifacts/invalid",
            seed=19,
            committed_at=datetime(2026, 8, 2, tzinfo=UTC),
        )

    assert not output.exists()


def test_profile_plan_covers_exact_steps_without_gaps() -> None:
    config, *_ = prepare._profile_config("full", seed=23)
    assert config.objective == prepare.OBJECTIVE_NAME_V4
    assert config.generated_rollin is not None
    assert config.branch_specialization is not None
    assert config.trajectory_objective is not None
    assert config.trajectory_objective.probe_steps == (1, 2, 4, 8)
    assert config.trajectory_objective.improvement_weight == 1.0
    assert config.role_conditioned_branches == 2
    assert config.structural_warmup_steps == 8
    assert config.validation_examples == 24
    assert config.intermediate_validation_examples == 4
    assert prepare.SOURCE_PATHS["objective_policy"].endswith(
        "recurrence_native_objective_v5.py"
    )
    assert prepare.SOURCE_PATHS["specialization_objective"].endswith(
        "recurrence_native_objective_v6.py"
    )
    assert prepare.SOURCE_PATHS["role_conditioned_adapter"].endswith(
        "role_conditioned_lora.py"
    )
    plan = prepare._build_plan(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test-full",
        profile="full",
        authority_sha256="a" * 64,
        config=config,
        source_commit="b" * 40,
    )
    ranges = [
        (
            plan.cell_definition(cell)["expected_start_step"],
            plan.cell_definition(cell)["required_end_step"],
        )
        for cell in plan.cell_ids
    ]
    assert ranges[0] == (0, 4)
    assert ranges[-1] == (100, 104)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False))
    assert plan.to_dict()["metadata"]["claim_eligible"] is False


def test_preparation_receipt_hash_is_self_consistent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = _prepare(monkeypatch, tmp_path, profile="canary")
    body = dict(receipt)
    claimed = body.pop("preparation_sha256")
    assert claimed == sha256_json(body)


def test_preparation_intent_recovers_original_timestamp_after_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    root = tmp_path / "artifacts" / "canary"
    root.mkdir(parents=True)
    committed_at = datetime(2026, 8, 1, 12, 34, tzinfo=UTC)
    with prepare.DirectoryCustody.acquire(root, private=True) as custody:
        prepare._preparation_intent(
            root=root,
            profile="canary",
            campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test-canary",
            model="model",
            execution_spec="spec.json",
            artifact_root="artifacts/canary",
            seed=19,
            committed_at=committed_at,
            custody=custody,
        )

    recovered = prepare._existing_intent_committed_at(
        profile="canary",
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test-canary",
        model="model",
        execution_spec="spec.json",
        artifact_root="artifacts/canary",
        seed=19,
    )

    assert recovered == committed_at


def test_full_profile_accepts_only_a_bounded_budget_extension() -> None:
    config, *_ = prepare._profile_config("full", seed=19, max_minutes=2_880.0)
    assert config.max_minutes == 2_880.0

    for invalid in (1_439.0, 10_081.0, float("nan"), float("inf")):
        with pytest.raises(
            prepare.ResidentSFTCampaignPreparationError,
            match="max_minutes|full_budget",
        ):
            prepare._profile_config("full", seed=19, max_minutes=invalid)


def test_canary_profile_refuses_a_budget_override() -> None:
    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="canary_budget_override_forbidden",
    ):
        prepare._profile_config("canary", seed=19, max_minutes=240.0)


def test_prepare_rejects_symlink_in_output_ancestry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="symlink_forbidden",
    ):
        prepare._repo_path("artifacts/canary", role="artifact_root", directory=None)


def test_source_binding_requires_tracked_bytes_at_exact_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prepare, "SOURCE_PATHS", {"source": "source.py"})
    monkeypatch.setattr(prepare, "REQUIRED_SOURCE_ROLES", frozenset({"source"}))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)

    assert prepare._source_bindings()["source"]["sha256"] == sha256_bytes(b"VALUE = 1\n")
    source.write_text("VALUE = 2\n")
    with pytest.raises(
        prepare.ResidentSFTCampaignPreparationError,
        match="source_not_exact_head",
    ):
        prepare._source_bindings()
