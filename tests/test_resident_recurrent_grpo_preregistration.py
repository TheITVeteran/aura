"""Contracts for the resident recurrent-GRPO preregistration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from tools import prepare_resident_recurrent_grpo_campaign as prereg

BASE_IDENTITY = {"method": "sha256", "fingerprint": "1" * 64, "files": 4}
BEHAVIOR_IDENTITY = {"bundle_sha256": "2" * 64, "file_count": 1, "files": []}


@pytest.fixture(autouse=True)
def _stub_fused_model_dir():
    """Hermetic model directory: these tests exercise contract logic only.

    ``build_contract`` requires the campaign's fused-model directory to
    *exist* (identities are injected, so nothing inside it is read). The real
    artifact is untracked (.git/info/exclude) and lives only in the main
    checkout, so in a fresh worktree we create an empty stub at the exact
    repo-relative path and remove precisely what we created afterwards.
    Where the real model is present this fixture does nothing.
    """
    target = prereg.REPO_ROOT / prereg.DEFAULT_MODEL
    created: list[Path] = []
    probe = target
    while not probe.exists():
        created.append(probe)
        probe = probe.parent
    if created:
        target.mkdir(parents=True)
    yield
    for path in created:  # leaf → root, only ever removing empty stub dirs
        try:
            path.rmdir()
        except OSError:
            break


def _contract():
    return prereg.build_contract(
        committed_at="2026-07-21T15:00:00-07:00",
        model_identity=BASE_IDENTITY,
        behavior_identity=BEHAVIOR_IDENTITY,
    )


def test_repo_path_resolves_artifact_stored_only_in_main_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    main = tmp_path / "main"
    worktree = tmp_path / "worktree"
    gitdir = main / ".git" / "worktrees" / "spark"
    gitdir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="ascii")
    relative = Path("training/fused-model/resident")
    resident = main / relative
    resident.mkdir(parents=True)
    monkeypatch.setattr(prereg, "REPO_ROOT", worktree)

    assert prereg._repo_path(relative.as_posix(), role="model") == resident


def test_repo_path_rejects_main_checkout_symlink_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    main = tmp_path / "main"
    worktree = tmp_path / "worktree"
    gitdir = main / ".git" / "worktrees" / "spark"
    gitdir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="ascii")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = main / "training" / "fused-model" / "resident"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(prereg, "REPO_ROOT", worktree)

    with pytest.raises(prereg.PreregistrationError, match="model_path_invalid"):
        prereg._repo_path("training/fused-model/resident", role="model")


def test_preregistration_binds_broad_training_and_powered_evaluation():
    contract = _contract()
    receipt = prereg.validate_contract(contract, verify_model=False)

    assert contract["training"]["parameters"]["domains"] == list(RECURRENCE_TRAINING_FAMILIES)
    assert contract["training"]["dataset"]["train_tasks"] == 288
    assert contract["training"]["dataset"]["holdout_tasks"] == 36
    assert contract["training"]["dataset"]["train_holdout_id_overlap"] == 0
    assert contract["training"]["parameters"]["trajectory_credit"] is False
    artifact = contract["training"]["verified_trajectory_config_artifact"]
    assert contract["training"]["parameters"]["group_size"] == 2
    assert artifact["config"]["intervention_config"]["lesion_steps"] == [1, 2, 4]
    assert artifact["config"]["intervention_config"]["stopping_steps"] == [1, 2, 4]
    assert artifact["sha256"] == artifact["semantic_sha256"]
    assert "--trajectory-credit" not in contract["training"]["argv"]
    assert "--verified-trajectory-config" in contract["training"]["argv"]
    assert contract["training"]["argv"][
        contract["training"]["argv"].index("--min-signal-groups") + 1
    ] == str(prereg.TRAINING_PARAMETERS["min_signal_groups"])
    mechanism = contract["evaluation"]["mechanism_attribution"]
    assert mechanism["required"] is True
    assert mechanism["claim_eligible"] is False
    assert "resident_full_stack" in mechanism["candidate_profiles"]
    assert "resident_full_stack_no_fast_weights" in mechanism["candidate_profiles"]
    assert "resident_full_stack > adapter_equal_compute" in mechanism["required_comparisons"]
    assert "fast_weight_erase_and_canary_receipts_required" in mechanism["acceptance_rules"]
    assert contract["evaluation"]["powered_confirmatory"]["task_count"] == 2877
    assert contract["evaluation"]["powered_confirmatory"]["cell_count"] == 17262
    assert receipt["claim_eligible"] is False


def test_update_canary_uses_exact_full_stack_with_bounded_nonclaim_dose():
    contract = prereg.build_contract(
        campaign_id="resident-32b-recurrent-grpo-cp420s14-update-canary",
        campaign_profile=prereg.UPDATE_CANARY_PROFILE,
        artifact_root=(
            "artifacts/closeout/latent_cortex/"
            "cp420s14_resident_32b_recurrent_grpo_update_canary"
        ),
        committed_at="2026-07-29T20:00:00-07:00",
        model_identity=BASE_IDENTITY,
        behavior_identity=BEHAVIOR_IDENTITY,
    )

    receipt = prereg.validate_contract(contract, verify_model=False)
    parameters = contract["training"]["parameters"]

    assert receipt["campaign_profile"] == prereg.UPDATE_CANARY_PROFILE
    assert parameters["domains"] == list(RECURRENCE_TRAINING_FAMILIES)
    assert parameters["depths"] == [4]
    assert parameters["train_per_cell"] == 1
    assert parameters["holdout_per_cell"] == 1
    assert parameters["max_steps"] == len(RECURRENCE_TRAINING_FAMILIES)
    assert parameters["eval_every"] == len(RECURRENCE_TRAINING_FAMILIES)
    assert parameters["group_size"] == prereg.TRAINING_PARAMETERS["group_size"]
    assert parameters["max_tokens"] == prereg.TRAINING_PARAMETERS["max_tokens"]
    assert parameters["lora_rank"] == prereg.TRAINING_PARAMETERS["lora_rank"]
    assert parameters["lora_layers"] == prereg.TRAINING_PARAMETERS["lora_layers"]
    assert parameters["lora_targets"] == prereg.TRAINING_PARAMETERS["lora_targets"]
    assert parameters["learning_rate"] == prereg.TRAINING_PARAMETERS["learning_rate"]
    assert parameters["fixed_update_canary"] is True
    assert parameters["calibrate"] is False
    assert "--fixed-update-canary" in contract["training"]["argv"]
    assert "--calibrate" not in contract["training"]["argv"]
    probe_argv = prereg._policy_probe_argv(contract)
    assert "--fixed-update-canary" not in probe_argv
    assert (
        probe_argv[probe_argv.index("--adapter-id") + 1]
        == f"{contract['campaign_id']}-initial-policy-probe"
    )
    assert contract["training"]["dataset"]["train_tasks"] == 12
    assert contract["training"]["dataset"]["holdout_tasks"] == 12
    assert (
        contract["training"]["completion_required"]["training_adequacy"]
        == prereg.recurrent_training_adequacy_policy()
    )
    assert contract["evaluation"]["engineering_canary"]["minimum_optimizer_updates"] == 3
    assert contract["evaluation"]["engineering_canary"]["reasoning_gain_claim_eligible"] is False
    assert contract["claim_state"]["resident_training_complete"] is False
    assert contract["claim_state"]["frontier_level_proven"] is False
    assert contract["required_stage_order"][-1] == "retire_canary_adapter"
    assert (
        contract["training"]["resource_envelope"]["detached_timeout_s"]
        == prereg.UPDATE_CANARY_RESOURCE_ENVELOPE["detached_timeout_s"]
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("campaign_profile",), prereg.FULL_TRAINING_PROFILE),
        (("training", "campaign_profile"), prereg.FULL_TRAINING_PROFILE),
        (("training", "parameters", "max_steps"), 1),
        (
            (
                "training",
                "completion_required",
                "training_adequacy",
                "minimum_optimizer_update_fraction",
            ),
            0.0,
        ),
        (("training", "resource_envelope", "detached_timeout_s"), 300),
        (("evaluation", "engineering_canary", "reasoning_gain_claim_eligible"), True),
    ],
)
def test_update_canary_rejects_profile_or_gate_rebinding(path, value):
    contract = prereg.build_contract(
        campaign_id="resident-32b-recurrent-grpo-cp420s14-update-canary",
        campaign_profile=prereg.UPDATE_CANARY_PROFILE,
        committed_at="2026-07-29T20:00:00-07:00",
        model_identity=BASE_IDENTITY,
        behavior_identity=BEHAVIOR_IDENTITY,
    )
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    material = dict(contract)
    material.pop("contract_sha256")
    contract["contract_sha256"] = prereg._document_sha(material)

    with pytest.raises(prereg.PreregistrationError):
        prereg.validate_contract(contract, verify_model=False)


def test_update_canary_verdict_recomputes_policy_lineage_and_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import train_grpo as trainer

    artifact_root = tmp_path / "canary"
    training = artifact_root / "training"
    checkpoint = training / "checkpoints" / "step-00000004-proof"
    campaign = artifact_root / "verified-launch" / "custody" / "campaign"
    checkpoint.mkdir(parents=True)
    campaign.mkdir(parents=True)
    initial = artifact_root / "verified-launch" / "initial_adapter.safetensors"
    initial.write_bytes(b"initial")
    final = training / "campaign_adapter" / "adapters.safetensors"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"updated")
    policies = [str(index) * 64 for index in range(1, 6)]
    step_receipts = []
    statuses = ["updated", "rejected", "rejected", "rejected"]
    for sequence, status in enumerate(statuses):
        before = policies[sequence] if sequence == 0 else step_receipts[-1]["policy_after_sha256"]
        after = policies[sequence + 1] if status == "updated" else before
        step_receipt_sha256 = f"{sequence + 5:x}" * 64
        step_receipts.append(
            {
                "step": sequence + 1,
                "task_id": f"task-{sequence + 1}",
                "step_kind": (
                    "verified_optimizer_update"
                    if status == "updated"
                    else "verified_rejected_group"
                ),
                "policy_before_sha256": before,
                "policy_after_sha256": after,
                "receipt_sha256": step_receipt_sha256,
            }
        )
        (campaign / f"group-{sequence:08d}.started.json").write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "admitted_at_unix_ns": sequence * 10_000_000_000 + 1_000_000_000,
                }
            ),
            encoding="ascii",
        )
        (campaign / f"group-{sequence:08d}.terminal.json").write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "status": status,
                    "finished_at_unix_ns": (
                        sequence * 10_000_000_000 + (sequence + 2) * 1_000_000_000
                    ),
                    "policy_before_sha256": before,
                    "policy_after_sha256": after,
                }
            ),
            encoding="ascii",
        )
    (campaign / "campaign.closed.json").write_text("{}\n", encoding="ascii")
    (training / "training_completion.json").write_text(
        json.dumps(
            {
                "schema": "aura.recurrent_grpo_training_completion.v1",
                "complete": True,
                "halt_reason": "max_steps",
                "step": 4,
                "adapter_sha256": "a" * 64,
            }
        ),
        encoding="ascii",
    )
    (training / "grpo_receipt.json").write_text(
        json.dumps(
            {
                "adapter_id": "resident-32b-recurrent-grpo-cp-test-canary",
                "steps": 4,
                "optimizer_updates": 1,
                "termination": {
                    "reason": "max_steps",
                    "completed_budget": True,
                    "signal": None,
                },
                "training_adequacy": {
                    "policy": prereg.recurrent_training_adequacy_policy(),
                    "admitted": True,
                },
                "step_receipts": step_receipts,
            }
        ),
        encoding="ascii",
    )
    (training / "training_protocol.json").write_text(
        json.dumps(
            {
                "personality_adapter": {"method": "none"},
                "runtime": {"runtime": "test"},
            }
        ),
        encoding="ascii",
    )
    protocol_sha256 = prereg._sha256(
        (training / "training_protocol.json").read_bytes()
    )
    timing_root = training / "update-canary-step-timings"
    timing_root.mkdir()
    for sequence, step_receipt in enumerate(step_receipts):
        step_checkpoint = (
            training / "checkpoints" / f"step-{sequence + 1:08d}-proof"
        )
        step_checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = json.dumps(
            {"step": sequence + 1},
            separators=(",", ":"),
        ).encode("ascii")
        (step_checkpoint / "complete.json").write_bytes(checkpoint_payload)
        timing_body = {
            "schema": "aura.recurrent_grpo.update_canary_step_timing.v1",
            "adapter_id": "resident-32b-recurrent-grpo-cp-test-canary",
            "protocol_sha256": protocol_sha256,
            "step": sequence + 1,
            "task_id": step_receipt["task_id"],
            "step_receipt_sha256": step_receipt["receipt_sha256"],
            "checkpoint": str(step_checkpoint.relative_to(training)),
            "checkpoint_complete_sha256": prereg._sha256(checkpoint_payload),
            "started_at_unix_ns": (sequence + 1) * 1_000_000_000,
            "finished_at_unix_ns": (sequence + 2) * 1_000_000_000,
            "elapsed_monotonic_ns": (sequence + 1) * 1_000_000_000,
            "includes_durable_checkpoint_publication": True,
        }
        timing = {
            **timing_body,
            "receipt_sha256": prereg._sha256(
                prereg.canonical_json_bytes(timing_body)
            ),
        }
        (timing_root / f"step-{sequence + 1:08d}.json").write_bytes(
            prereg.canonical_json_bytes(timing)
        )
    (training / "recurrence_adapter_manifest.json").write_text(
        json.dumps({"adapter": {"path": "campaign_adapter/adapters.safetensors"}}),
        encoding="ascii",
    )
    (training / "NON_PROMOTABLE_CANARY.json").write_text(
        json.dumps(
            {
                "schema": "aura.recurrent_grpo.non_promotable_canary.v1",
                "adapter_id": "resident-32b-recurrent-grpo-cp-test-canary",
                "adapter_sha256": "a" * 64,
                "runtime_promotion_allowed": False,
                "reasoning_gain_proven": False,
                "frontier_level_proven": False,
            }
        ),
        encoding="ascii",
    )
    containment_body = {
        "schema": "aura.resident_recurrent_grpo.canary_containment.v1",
        "campaign_id": "resident-32b-recurrent-grpo-cp-test-canary",
        "campaign_contract_sha256": "b" * 64,
        "training_completion_sha256": "d" * 64,
        "non_promotable_marker_sha256": "e" * 64,
        "base_checkpoint_sha256": BASE_IDENTITY["fingerprint"],
        "runtime_model_state_released": True,
        "runtime_promotion_allowed": False,
        "released_at_unix_ns": 1,
    }
    (training / "CANARY_CONTAINMENT.json").write_bytes(
        prereg.canonical_json_bytes(
            {
                **containment_body,
                "receipt_sha256": prereg._sha256(
                    prereg.canonical_json_bytes(containment_body)
                ),
            }
        )
    )
    (training / "latest.json").write_text(
        json.dumps({"checkpoint": "checkpoints/step-00000004-proof"}),
        encoding="ascii",
    )
    launch = artifact_root / "verified-launch" / "launch-bundle.json"
    launch.write_text(
        json.dumps({"campaign_ledger_root": str(campaign)}),
        encoding="ascii",
    )
    contract = {
        "campaign_id": "resident-32b-recurrent-grpo-cp-test-canary",
        "campaign_profile": prereg.UPDATE_CANARY_PROFILE,
        "contract_sha256": "b" * 64,
        "training": {"parameters": {"max_steps": 4}},
        "paths": {
            "artifact_root": str(artifact_root),
            "training_output": str(training),
            "verified_launch_bundle": str(launch),
        },
        "model": {
            "path": str(tmp_path / "model"),
            "base_checkpoint": BASE_IDENTITY,
            "behavior_bundle": BEHAVIOR_IDENTITY,
        },
    }
    monkeypatch.setattr(
        prereg,
        "validate_contract",
        lambda *_args, **_kwargs: {"claim_eligible": False},
    )
    monkeypatch.setattr(
        prereg,
        "_repo_path",
        lambda value, **_kwargs: Path(value),
    )
    monkeypatch.setattr(
        trainer,
        "_validate_published_recurrent_bundle",
        lambda *_args, **_kwargs: {
            "adapter_sha256": "a" * 64,
            "composite_identity_sha256": "c" * 64,
        },
    )

    verdict = prereg.build_update_canary_verdict(contract, verify_model=False)

    assert verdict["verdict"] == "pass"
    assert verdict["optimizer_updates"] == 1
    assert verdict["optimizer_update_fraction"] == 0.25
    assert verdict["optimizer_update_fraction_wilson_95"]["low"] > 0.0
    assert verdict["optimizer_update_fraction_wilson_95"]["high"] < 1.0
    assert verdict["latency_s"] == {
        "scope": "task_admission_through_durable_checkpoint_publication",
        "count": 4,
        "p50": 2.0,
        "p90": 4.0,
        "max": 4.0,
        "total": 10.0,
    }
    assert verdict["campaign_execution_latency_s"] == {
        "scope": "signed_group_admission_through_campaign_terminal",
        "count": 4,
        "p50": 2.0,
        "p90": 4.0,
        "max": 4.0,
        "total": 10.0,
    }
    assert verdict["process_containment_rollback"] is True
    assert verdict["reasoning_gain_proven"] is False
    assert verdict["frontier_level_proven"] is False


def test_preregistration_archives_enabled_verified_trajectory_config(
    monkeypatch: pytest.MonkeyPatch,
):
    parameters = dict(prereg.TRAINING_PARAMETERS)
    parameters["group_size"] = 2
    parameters["verified_trajectory_config"] = (
        "tests/fixtures/verified_trajectory_group_config.json"
    )
    monkeypatch.setattr(prereg, "TRAINING_PARAMETERS", parameters)

    contract = _contract()
    receipt = prereg.validate_contract(contract, verify_model=False)
    artifact = contract["training"]["verified_trajectory_config_artifact"]

    assert artifact["path"] == parameters["verified_trajectory_config"]
    assert artifact["sha256"] == artifact["semantic_sha256"]
    assert artifact["config"]["trajectory_config"]["probe_steps"] == [1, 2, 4]
    assert (
        contract["training"]["argv"][
            contract["training"]["argv"].index("--verified-trajectory-config") + 1
        ]
        == parameters["verified_trajectory_config"]
    )
    probe_argv = prereg._policy_probe_argv(contract)
    assert "--verified-trajectory-config" not in probe_argv
    assert probe_argv[-1] == "--initial-policy-probe"
    assert receipt["claim_eligible"] is False


def test_preregistration_archives_combined_intervention_config(
    monkeypatch: pytest.MonkeyPatch,
):
    parameters = dict(prereg.TRAINING_PARAMETERS)
    parameters["group_size"] = 2
    parameters["verified_trajectory_config"] = (
        "tests/fixtures/verified_intervention_group_config.json"
    )
    monkeypatch.setattr(prereg, "TRAINING_PARAMETERS", parameters)

    contract = _contract()
    receipt = prereg.validate_contract(contract, verify_model=False)
    artifact = contract["training"]["verified_trajectory_config_artifact"]

    assert artifact["config"]["schema"] == ("aura.recurrent_grpo.verified_trajectory_composite.v2")
    assert artifact["config"]["intervention_config"]["lesion_steps"] == [1, 2, 4]
    assert artifact["config"]["intervention_config"]["stopping_steps"] == [1, 2, 4]
    assert artifact["sha256"] == artifact["semantic_sha256"]
    assert receipt["claim_eligible"] is False


@pytest.mark.parametrize("field", ["lesion_steps", "stopping_steps"])
def test_preregistration_rejects_intervention_depth_beyond_execution_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    config = json.loads(
        (prereg.REPO_ROOT / "tests/fixtures/verified_intervention_group_config.json").read_text(
            encoding="ascii"
        )
    )
    config["intervention_config"][field] = [1, 2, 5]
    config_path = tmp_path / "intervention-config.json"
    config_path.write_bytes(prereg.canonical_json_bytes(config))
    parameters = {
        **prereg.TRAINING_PARAMETERS,
        "group_size": 2,
        "verified_trajectory_config": config_path.name,
    }
    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "TRAINING_PARAMETERS", parameters)
    spec = SimpleNamespace(
        branch_roles=("constructive_solution", "critical_verification"),
        recurrent_steps=4,
    )

    with pytest.raises(
        prereg.PreregistrationError,
        match="verified_trajectory_config_depth_invalid",
    ):
        prereg._verified_trajectory_config_commitment(spec)


def test_preregistration_rejects_trajectory_group_branch_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    parameters = dict(prereg.TRAINING_PARAMETERS)
    parameters["group_size"] = 4
    parameters["verified_trajectory_config"] = (
        "tests/fixtures/verified_trajectory_group_config.json"
    )
    monkeypatch.setattr(prereg, "TRAINING_PARAMETERS", parameters)

    with pytest.raises(
        prereg.PreregistrationError,
        match="verified_trajectory_group_branch_count_mismatch",
    ):
        _contract()


def test_trajectory_config_commitment_rejects_symlink_and_oversized_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_bytes = (
        prereg.REPO_ROOT / "tests/fixtures/verified_trajectory_group_config.json"
    ).read_bytes()
    real_config = tmp_path / "real-config.json"
    real_config.write_bytes(config_bytes)
    link = tmp_path / "trajectory-config.json"
    link.symlink_to(real_config.name)
    parameters = {
        **prereg.TRAINING_PARAMETERS,
        "group_size": 2,
        "verified_trajectory_config": link.name,
    }
    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "TRAINING_PARAMETERS", parameters)
    spec = SimpleNamespace(
        branch_roles=("constructive_solution", "critical_verification"),
        recurrent_steps=4,
    )

    with pytest.raises(
        prereg.PreregistrationError,
        match="verified_trajectory_config_file_invalid",
    ):
        prereg._verified_trajectory_config_commitment(spec)

    link.unlink()
    link.write_bytes(b" " * 65_537)
    with pytest.raises(
        prereg.PreregistrationError,
        match="verified_trajectory_config_invalid",
    ):
        prereg._verified_trajectory_config_commitment(spec)


def test_preregistration_can_bind_new_attempt_campaign_identity():
    contract = prereg.build_contract(
        campaign_id="resident-32b-recurrent-grpo-cp273",
        artifact_root="artifacts/closeout/latent_cortex/cp273_resident_32b_recurrent_grpo",
        committed_at="2026-07-21T15:00:00-07:00",
        model_identity=BASE_IDENTITY,
        behavior_identity=BEHAVIOR_IDENTITY,
    )
    receipt = prereg.validate_contract(contract, verify_model=False)

    assert contract["campaign_id"] == "resident-32b-recurrent-grpo-cp273"
    assert receipt["campaign_id"] == "resident-32b-recurrent-grpo-cp273"
    assert "resident-32b-recurrent-grpo-cp273" in contract["training"]["argv"]
    assert (
        contract["paths"]["artifact_root"]
        == "artifacts/closeout/latent_cortex/cp273_resident_32b_recurrent_grpo"
    )


def test_preregistration_rejects_command_or_claim_rebinding():
    contract = _contract()
    rebound = copy.deepcopy(contract)
    rebound["training"]["argv"][-1] = "999"
    material = dict(rebound)
    material.pop("contract_sha256")
    rebound["contract_sha256"] = prereg._document_sha(material)
    with pytest.raises(prereg.PreregistrationError, match="training_contract_mismatch"):
        prereg.validate_contract(rebound, verify_model=False)

    rebound = copy.deepcopy(contract)
    rebound["claim_state"]["frontier_level_proven"] = True
    material = dict(rebound)
    material.pop("contract_sha256")
    rebound["contract_sha256"] = prereg._document_sha(material)
    with pytest.raises(prereg.PreregistrationError, match="prelaunch_claim_state_invalid"):
        prereg.validate_contract(rebound, verify_model=False)


def test_preregistration_rejects_any_uncommitted_byte_change():
    contract = _contract()
    contract["training"]["parameters"]["max_tokens"] = 64

    with pytest.raises(prereg.PreregistrationError, match="contract_digest_mismatch"):
        prereg.validate_contract(contract, verify_model=False)


def test_resume_verdict_binds_one_complete_checkpoint(tmp_path, monkeypatch):
    contract = _contract()
    training = tmp_path / "training"
    checkpoint = training / "checkpoints" / "step-00000003-proof"
    checkpoint.mkdir(parents=True)
    protocol = b'{"protocol":"bound"}\n'
    dataset = b'{"dataset":"bound"}\n'
    adapter = b"adapter"
    optimizer = b"optimizer"
    (training / "training_protocol.json").write_bytes(protocol)
    (training / "dataset_manifest.json").write_bytes(dataset)
    (checkpoint / "adapter.safetensors").write_bytes(adapter)
    (checkpoint / "optimizer.safetensors").write_bytes(optimizer)
    contract["training"]["dataset"]["sha256"] = hashlib.sha256(dataset).hexdigest()
    contract["paths"]["training_output"] = str(training.relative_to(tmp_path))
    material = dict(contract)
    material.pop("contract_sha256")
    contract["contract_sha256"] = prereg._document_sha(material)
    complete = {
        "schema": "aura.grpo_checkpoint.v2",
        "checkpoint_id": checkpoint.name,
        "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
        "step": 3,
        "last_step_committed": True,
        "execution_mode": "recurrent",
        "execution_spec_sha256": contract["execution_spec"]["semantic_sha256"],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": hashlib.sha256(adapter).hexdigest(),
            "size_bytes": len(adapter),
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": hashlib.sha256(optimizer).hexdigest(),
            "size_bytes": len(optimizer),
        },
    }
    complete_raw = prereg.canonical_json_bytes(complete)
    (checkpoint / "complete.json").write_bytes(complete_raw)
    (training / "latest.json").write_text(
        json.dumps(
            {
                "schema": "aura.grpo_checkpoint_pointer.v1",
                "checkpoint": f"checkpoints/{checkpoint.name}",
                "complete_sha256": hashlib.sha256(complete_raw).hexdigest(),
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    evidence = tmp_path / "supervisor" / "resume.json"
    environment = {
        "AURA_DETACHED_PLAN_SHA256": "3" * 64,
        "AURA_DETACHED_COMMAND_SHA256": "4" * 64,
        "AURA_DETACHED_PRIOR_ATTEMPT": "1",
        "AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256": "5" * 64,
        "AURA_DETACHED_RESUME_EVIDENCE_PATH": str(evidence),
    }

    verdict = prereg.build_resume_verdict(
        contract,
        environment=environment,
        verify_model=False,
    )

    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 3
    assert verdict["evidence"]["adapter"]["sha256"] == hashlib.sha256(adapter).hexdigest()
    assert json.loads(evidence.read_text(encoding="ascii")) == verdict["evidence"]


def test_launch_training_preserves_virtualenv_launcher_path(tmp_path, monkeypatch):
    contract = _contract()
    contract["paths"]["detached_training"] = "artifacts/run"
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="ascii")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(__import__("sys").executable))
    captured: dict[str, object] = {}

    def fake_detached_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg.sys, "executable", str(venv_python))
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prereg.run_detached_step, "main", fake_detached_main)

    assert (
        prereg._launch_training(
            contract_path,
            resume=False,
            expected_launch_bundle_sha256="a" * 64,
        )
        == 0
    )

    argv = captured["argv"]
    assert isinstance(argv, list)
    verifier = json.loads(argv[argv.index("--resume-verifier-json") + 1])
    command = argv[argv.index("--resume-verifier-json") + 2 :]
    assert verifier[0] == str(venv_python)
    assert command[0] == str(venv_python)
    assert str(Path(venv_python).resolve()) not in verifier
    assert str(Path(venv_python).resolve()) not in command
    assert command[-2:] == [
        "--expected-launch-bundle-sha256",
        "a" * 64,
    ]


def test_training_watchdog_retries_only_into_exact_durable_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import run_verified_recurrent_grpo_training as runner

    training = tmp_path / "training"
    bundle = tmp_path / "bundle.json"
    digest_file = tmp_path / "bundle.sha256"
    bundle.write_text("{}\n", encoding="ascii")
    digest_file.write_text("a" * 64 + "\n", encoding="ascii")
    contract = {
        "campaign_id": "watchdog-test",
        "contract_sha256": "b" * 64,
        "launch_not_before_unix": 0,
        "paths": {
            "artifact_root": "artifacts/watchdog-test",
            "verified_launch_bundle": "bundle.json",
            "verified_launch_bundle_sha256": "bundle.sha256",
            "training_output": "training",
        },
        "training": {
            "argv": ["tools/train_grpo.py"],
            "parameters": {"max_steps": 288},
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
            },
            "dataset": {"sha256": hashlib.sha256(b"dataset\n").hexdigest()},
            "watchdog_policy": {
                **prereg.TRAINING_WATCHDOG_POLICY,
                "retry_backoff_s": 0.001,
            },
        },
    }
    calls = 0

    def run(_argv):
        nonlocal calls
        calls += 1
        training.mkdir(exist_ok=True)
        (training / "dataset_manifest.json").write_bytes(b"dataset\n")
        if calls == 1:
            (training / "baseline-progress.json").write_text(
                f'{{"completed":{calls}}}\n',
                encoding="ascii",
            )
            raise RuntimeError("transient resident failure")
        (training / "training_completion.json").write_bytes(
            prereg.canonical_json_bytes(
                {
                    "schema": "aura.recurrent_grpo_training_completion.v1",
                    "complete": True,
                    "halt_reason": "max_steps",
                    "step": 288,
                }
            )
        )
        return 0

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "main", run)
    monkeypatch.setattr(prereg, "_release_failed_training_runtime", lambda: None)
    monkeypatch.setattr(prereg.time, "sleep", lambda _seconds: None)

    assert prereg._run_training(contract, expected_launch_bundle_sha256="a" * 64) == 0
    assert calls == 2
    journal = json.loads(
        (tmp_path / "artifacts/watchdog-test/training-watchdog/attempts.json").read_text(
            encoding="ascii"
        )
    )
    assert [record["durable_progress"] for record in journal["records"]] == [True, True]
    status = json.loads(
        (tmp_path / "artifacts/watchdog-test/training-watchdog/status.json").read_text(
            encoding="ascii"
        )
    )
    assert status["state"] == "complete"


def test_training_watchdog_resumes_zero_exit_wall_clock_until_full_dose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import run_verified_recurrent_grpo_training as runner

    training = tmp_path / "training"
    bundle = tmp_path / "bundle.json"
    digest_file = tmp_path / "bundle.sha256"
    bundle.write_text("{}\n", encoding="ascii")
    digest_file.write_text("a" * 64 + "\n", encoding="ascii")
    contract = {
        "campaign_id": "watchdog-wall-clock-test",
        "contract_sha256": "b" * 64,
        "launch_not_before_unix": 0,
        "paths": {
            "artifact_root": "artifacts/watchdog-wall-clock-test",
            "verified_launch_bundle": "bundle.json",
            "verified_launch_bundle_sha256": "bundle.sha256",
            "training_output": "training",
        },
        "training": {
            "argv": ["tools/train_grpo.py"],
            "parameters": {"max_steps": 288},
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
            },
            "dataset": {"sha256": hashlib.sha256(b"dataset\n").hexdigest()},
            "watchdog_policy": {
                **prereg.TRAINING_WATCHDOG_POLICY,
                "retry_backoff_s": 0.001,
            },
        },
    }
    calls = 0

    def run(_argv):
        nonlocal calls
        calls += 1
        training.mkdir(exist_ok=True)
        (training / "dataset_manifest.json").write_bytes(b"dataset\n")
        if calls == 1:
            checkpoint = training / "checkpoints" / "step-00000120"
            checkpoint.mkdir(parents=True)
            (checkpoint / "complete.json").write_bytes(
                prereg.canonical_json_bytes({"step": 120})
            )
            (training / "latest.json").write_bytes(
                prereg.canonical_json_bytes(
                    {"checkpoint": "checkpoints/step-00000120"}
                )
            )
            (training / "grpo_receipt.json").write_bytes(
                prereg.canonical_json_bytes(
                    {
                        "steps": 120,
                        "termination": {
                            "reason": "wall_clock_budget",
                            "completed_budget": False,
                        },
                    }
                )
            )
            return 0
        (training / "training_completion.json").write_bytes(
            prereg.canonical_json_bytes(
                {
                    "schema": "aura.recurrent_grpo_training_completion.v1",
                    "complete": True,
                    "halt_reason": "max_steps",
                    "step": 288,
                }
            )
        )
        return 0

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "main", run)
    monkeypatch.setattr(prereg, "_release_failed_training_runtime", lambda: None)
    monkeypatch.setattr(prereg.time, "sleep", lambda _seconds: None)

    assert prereg._run_training(contract, expected_launch_bundle_sha256="a" * 64) == 0
    assert calls == 2
    journal = json.loads(
        (
            tmp_path
            / "artifacts/watchdog-wall-clock-test/training-watchdog/attempts.json"
        ).read_text(encoding="ascii")
    )
    assert [record["disposition"] for record in journal["records"]] == [
        "resume",
        "complete",
    ]


def test_training_watchdog_pause_releases_runtime_and_waits_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import run_verified_recurrent_grpo_training as runner

    training = tmp_path / "training"
    bundle = tmp_path / "bundle.json"
    digest_file = tmp_path / "bundle.sha256"
    bundle.write_text("{}\n", encoding="ascii")
    digest_file.write_text("a" * 64 + "\n", encoding="ascii")
    contract = {
        "campaign_id": "watchdog-pause-test",
        "contract_sha256": "b" * 64,
        "launch_not_before_unix": 0,
        "paths": {
            "artifact_root": "artifacts/watchdog-pause-test",
            "verified_launch_bundle": "bundle.json",
            "verified_launch_bundle_sha256": "bundle.sha256",
            "training_output": "training",
        },
        "training": {
            "argv": ["tools/train_grpo.py"],
            "parameters": {"max_steps": 12},
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
            },
            "dataset": {"sha256": hashlib.sha256(b"dataset\n").hexdigest()},
            "watchdog_policy": {
                **prereg.TRAINING_WATCHDOG_POLICY,
                "retry_backoff_s": 0.001,
            },
        },
    }
    calls = 0
    releases = 0
    resumes = 0

    def run(_argv):
        nonlocal calls
        calls += 1
        training.mkdir(exist_ok=True)
        (training / "dataset_manifest.json").write_bytes(b"dataset\n")
        if calls == 1:
            checkpoint = training / "checkpoints" / "step-00000004"
            checkpoint.mkdir(parents=True)
            (checkpoint / "complete.json").write_bytes(
                prereg.canonical_json_bytes({"step": 4})
            )
            (training / "latest.json").write_bytes(
                prereg.canonical_json_bytes(
                    {"checkpoint": "checkpoints/step-00000004"}
                )
            )
            (training / "grpo_receipt.json").write_bytes(
                prereg.canonical_json_bytes(
                    {
                        "steps": 4,
                        "termination": {
                            "reason": "operator_pause",
                            "completed_budget": False,
                        },
                    }
                )
            )
            return 0
        (training / "training_completion.json").write_bytes(
            prereg.canonical_json_bytes(
                {
                    "schema": "aura.recurrent_grpo_training_completion.v1",
                    "complete": True,
                    "halt_reason": "max_steps",
                    "step": 12,
                }
            )
        )
        return 0

    def release():
        nonlocal releases
        releases += 1

    def resume(_contract):
        nonlocal resumes
        resumes += 1
        return {"request_sha256": "c" * 64}

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "main", run)
    monkeypatch.setattr(prereg, "_release_failed_training_runtime", release)
    monkeypatch.setattr(prereg, "_wait_for_training_resume", resume)
    monkeypatch.setattr(prereg.time, "sleep", lambda _seconds: None)

    assert prereg._run_training(contract, expected_launch_bundle_sha256="a" * 64) == 0
    assert calls == 2
    assert releases == 1
    assert resumes == 1
    journal = json.loads(
        (
            tmp_path / "artifacts/watchdog-pause-test/training-watchdog/attempts.json"
        ).read_text(encoding="ascii")
    )
    assert [record["disposition"] for record in journal["records"]] == [
        "paused",
        "complete",
    ]


def test_answer_channel_preflight_command_is_bounded_and_source_separated():
    contract = _contract()

    argv = prereg._answer_channel_preflight_argv(contract)

    assert argv[0] == "tools/train_grpo.py"
    assert argv[argv.index("--model") + 1] == contract["model"]["path"]
    assert argv[argv.index("--execution-spec") + 1] == contract["execution_spec"]["path"]
    assert argv[argv.index("--task-source") + 1] == "answer_channel_curriculum"
    assert argv[argv.index("--domains") + 1] == "json_copy,typed_boolean,key_selection"
    assert argv[argv.index("--max-steps") + 1] == "1"
    assert argv[argv.index("--max-minutes") + 1] == "45.0"
    assert argv[argv.index("--calibrate-minutes") + 1] == "10.0"
    assert "--trajectory-credit" not in argv
    assert "recurrence_curriculum" not in argv
    assert "--read-only-answer-channel-preflight" in argv


def test_answer_channel_preflight_invokes_trainer_without_launching_detached(
    monkeypatch,
):
    contract = _contract()
    captured: dict[str, object] = {}

    def fake_train_main():
        captured["argv"] = list(prereg.sys.argv)
        return 7

    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    from tools import train_grpo

    monkeypatch.setattr(train_grpo, "main", fake_train_main)

    assert prereg._run_answer_channel_preflight(contract) == 7
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "tools/train_grpo.py"
    assert "answer_channel_curriculum" in argv
    assert "--read-only-answer-channel-preflight" in argv


def test_launch_initial_policy_probe_is_detached_and_nonresumable(tmp_path, monkeypatch):
    contract = _contract()
    contract["paths"]["artifact_root"] = "artifacts/probe"
    contract_path = tmp_path / "config" / "probe-contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract), encoding="ascii")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(__import__("sys").executable))
    captured: dict[str, object] = {}

    def fake_detached_main(argv):
        captured["argv"] = list(argv)
        return 19

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg.sys, "executable", str(venv_python))
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prereg.run_detached_step, "main", fake_detached_main)

    assert prereg._launch_initial_policy_probe(contract_path) == 19

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "launch"
    assert argv[argv.index("--resume-contract") + 1] == "none"
    assert argv[argv.index("--timeout") + 1] == "7200"
    assert argv[argv.index("--run-dir") + 1] == str(
        tmp_path / "artifacts" / "probe" / "detached-initial-policy-probe"
    )
    command = argv[argv.index("--resume-contract") + 2 :]
    assert command[0] == str(venv_python)
    assert command[2:4] == ["run-initial-policy-probe", "--contract"]


def test_launch_answer_channel_preflight_is_detached_and_source_bound(tmp_path, monkeypatch):
    contract = _contract()
    contract["paths"]["artifact_root"] = "artifacts/preflight"
    contract_path = tmp_path / "config" / "preflight-contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract), encoding="ascii")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(__import__("sys").executable))
    captured: dict[str, object] = {}

    def fake_detached_main(argv):
        captured["argv"] = list(argv)
        return 13

    monkeypatch.setattr(prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prereg.sys, "executable", str(venv_python))
    monkeypatch.setattr(prereg, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prereg.run_detached_step, "main", fake_detached_main)

    assert prereg._launch_answer_channel_preflight(contract_path) == 13

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "launch"
    assert argv[argv.index("--run-dir") + 1] == str(
        tmp_path / "artifacts" / "preflight" / "detached-answer-channel-preflight"
    )
    assert argv[argv.index("--name") + 1].endswith("-answer-channel-preflight")
    assert argv[argv.index("--cwd") + 1] == str(tmp_path)
    assert argv[argv.index("--timeout") + 1] == "5400"
    resume_index = argv.index("--resume-contract")
    assert argv[resume_index + 1] == "none"
    command = argv[resume_index + 2 :]
    assert command[0] == str(venv_python)
    assert command[1] == str(Path(prereg.__file__).resolve(strict=True))
    assert command[2:4] == ["run-answer-channel-preflight", "--contract"]
    assert command[4] == str(contract_path.resolve(strict=True))
