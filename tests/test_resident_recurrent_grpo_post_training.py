from __future__ import annotations

import json
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import run_resident_recurrent_grpo_post_training as post


def _contract(root: Path) -> dict:
    artifact = "artifacts/cp259"
    material = {
        "schema": post.prereg.CONTRACT_SCHEMA,
        "campaign_id": post.prereg.CAMPAIGN_ID,
        "launch_not_before_unix": 1,
        "model": {
            "path": "model",
            "base_checkpoint": {"fingerprint": "1" * 64},
            "behavior_bundle": {"bundle_sha256": "2" * 64},
        },
        "execution_spec": {"semantic_sha256": "3" * 64},
        "paths": {
            "artifact_root": artifact,
            "training_output": f"{artifact}/training",
            "detached_training": f"{artifact}/detached-training",
            "verified_launch_bundle": f"{artifact}/verified-launch/launch-bundle.json",
            "frozen_adapter": f"{artifact}/frozen-adapter",
            "directional_campaign": f"{artifact}/directional-campaign",
        },
        "training": {
            "parameters": {"max_steps": 288},
            "watchdog_policy": {
                "max_attempts": 3,
                "max_consecutive_no_progress_failures": 2,
                "retry_backoff_s": 0.0,
            },
            "dataset": {"sha256": "4" * 64},
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
                "complete": True,
                "halt_reason": "max_steps",
                "causal_gain_proven": False,
            },
        },
        "independent_custody": {"required_roles": ["task_issuer", "verifier"]},
    }
    return {**material, "contract_sha256": post.prereg._document_sha(material)}


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "REPO_ROOT", tmp_path)
    source = tmp_path / post.SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_text("# controller\n", encoding="ascii")
    (tmp_path / "model").mkdir()
    contract = _contract(tmp_path)
    launch_bundle = tmp_path / contract["paths"]["verified_launch_bundle"]
    launch_bundle.parent.mkdir(parents=True)
    launch_bundle.write_text("{}\n", encoding="ascii")
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(canonical_json_bytes(contract))
    monkeypatch.setattr(
        post.prereg,
        "validate_contract",
        lambda value, *, verify_model: {"contract_sha256": value["contract_sha256"]},
    )
    return tmp_path, contract_path, contract


def test_config_binds_nonclaiming_six_arm_directional_contract(isolated_repo):
    root, contract_path, contract = isolated_repo
    seeds = [(1 << 62) + index for index in range(8)]
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=seeds,
    )

    validated, observed = post.validate_config(
        config, require_live_preregistration=True
    )

    assert observed == contract
    assert validated["directional"]["profile"] == "full"
    assert validated["directional"]["rlc_profile"] == "resident_full_stack"
    assert validated["directional"]["seeds"] == seeds
    assert validated["directional"]["claim_eligible"] is False
    mechanism = validated["mechanism_attribution"]
    assert mechanism["required"] is True
    assert mechanism["claim_eligible"] is False
    assert mechanism["baseline_profile"] == "resident_full_stack"
    assert tuple(mechanism["profiles"]) == post._MECHANISM_PROFILES
    assert mechanism["seeds"] == seeds
    assert mechanism["domains"] == validated["directional"]["domains"]
    assert all(
        validated["claim_policy"][claim] is False
        for claim in (
            "reasoning_gain_proven",
            "positive_interaction_proven",
            "frontier_level_proven",
            "release_eligible",
        )
    )


def test_directional_command_runs_the_full_stack_profile(isolated_repo, monkeypatch):
    root, contract_path, contract = isolated_repo
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    monkeypatch.setattr(post.sys, "executable", str(venv_python))
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    (root / "artifacts/cp259/frozen-adapter").mkdir(parents=True)

    command = post.ControllerRun(config, contract).directional_command()

    assert command[0] == str(venv_python)
    profile_index = command.index("--rlc-profile") + 1
    assert command[profile_index] == "resident_full_stack"


@pytest.mark.parametrize("profile", post._MECHANISM_PROFILES)
def test_mechanism_command_runs_requested_ablation_profile(
    isolated_repo, monkeypatch, profile
):
    root, contract_path, contract = isolated_repo
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    monkeypatch.setattr(post.sys, "executable", str(venv_python))
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    (root / "artifacts/cp259/frozen-adapter").mkdir(parents=True)

    command = post.ControllerRun(config, contract).mechanism_command(profile)

    assert command[0] == str(venv_python)
    profile_index = command.index("--rlc-profile") + 1
    assert command[profile_index] == profile
    campaign_dir_index = command.index("--campaign-dir") + 1
    assert command[campaign_dir_index].endswith(f"mechanism-attribution/{profile}")


def test_config_rejects_rebound_frontier_claim(isolated_repo):
    root, contract_path, _contract_value = isolated_repo
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    config["claim_policy"]["frontier_level_proven"] = True
    material = dict(config)
    material.pop("config_sha256")
    config["config_sha256"] = post._document_sha(material)

    with pytest.raises(post.PostTrainingError, match="claim_policy_invalid"):
        post.validate_config(config, require_live_preregistration=False)


def test_training_completion_rejects_wall_clock_and_partial_runs(isolated_repo):
    _root, _contract_path, contract = isolated_repo
    valid = {
        "schema": "aura.recurrent_grpo_training_completion.v1",
        "complete": True,
        "halt_reason": "max_steps",
        "step": 288,
        "manifest_sha256": "5" * 64,
    }
    post._validate_training_completion(valid, contract)

    for mutation in (
        {"halt_reason": "wall_clock"},
        {"step": 287},
        {"complete": False},
    ):
        candidate = {**valid, **mutation}
        with pytest.raises(
            post.PostTrainingError, match="training_completion_not_admissible"
        ):
            post._validate_training_completion(candidate, contract)


def test_training_no_signal_stop_is_a_diagnostic_not_claim():
    receipt = {
        "termination": {
            "reason": "no_learning_signal",
            "completed_budget": False,
        },
        "learning_signal": {
            "learning_signal": False,
            "diagnosis": "tasks_too_hard: no gradients",
        },
        "verdict": {
            "had_signal": False,
            "causal_gain_proven": False,
        },
    }

    assert post._training_diagnostic_failure(receipt) == [
        "training:no_learning_signal",
        "diagnosis:tasks_too_hard: no gradients",
    ]

    claimed = dict(receipt)
    claimed["verdict"] = {**receipt["verdict"], "causal_gain_proven": True}
    with pytest.raises(post.PostTrainingError, match="diagnostic_claims"):
        post._training_diagnostic_failure(claimed)


def test_training_adequacy_stop_is_a_terminal_diagnostic_not_claim():
    receipt = {
        "termination": {
            "reason": "training_adequacy_failed",
            "completed_budget": True,
        },
        "training_adequacy": {
            "admitted": False,
            "failed_checks": [
                "distributed_update_activity",
                "learning_signal",
                "minimum_optimizer_updates",
            ],
        },
        "learning_signal": {
            "learning_signal": False,
            "diagnosis": "all_verified_transition_groups_rejected",
        },
        "verdict": {
            "had_signal": False,
            "causal_gain_proven": False,
        },
    }

    assert post._training_diagnostic_failure(receipt) == [
        "training:training_adequacy_failed",
        "diagnosis:all_verified_transition_groups_rejected",
        "training_adequacy:distributed_update_activity",
        "training_adequacy:learning_signal",
        "training_adequacy:minimum_optimizer_updates",
    ]

    incomplete = dict(receipt)
    incomplete["termination"] = {
        "reason": "training_adequacy_failed",
        "completed_budget": False,
    }
    with pytest.raises(post.PostTrainingError, match="diagnostic_claims"):
        post._training_diagnostic_failure(incomplete)


def test_detached_terminal_requires_empty_contained_lineage():
    receipt = {
        "returncode": 0,
        "containment_verified": True,
        "process_group_empty": True,
        "lineage_empty": True,
        "timed_out": False,
        "receipt_sha256": "6" * 64,
    }
    status = {
        "terminal": True,
        "completion_indeterminate": False,
        "supervisor_alive": False,
        "child_state": "dead",
        "receipt": receipt,
    }
    assert post._validate_detached_terminal(
        status, role="training", allowed_returncodes=frozenset({0})
    ) == receipt

    status["receipt"] = {**receipt, "returncode": 3}
    assert post._validate_detached_terminal(
        status, role="training", allowed_returncodes=frozenset({0, 3})
    )["returncode"] == 3

    status["receipt"] = {**receipt, "lineage_empty": False}
    with pytest.raises(
        post.PostTrainingError, match="training_detached_evidence_invalid"
    ):
        post._validate_detached_terminal(
            status, role="training", allowed_returncodes=frozenset({0})
        )


def test_controller_journal_detects_tampering(isolated_repo):
    root, _contract_path, contract = isolated_repo
    controller_root = root / "artifacts/cp259/post-training"
    config = {
        "output_root": str(controller_root.relative_to(root)),
        "config_sha256": "7" * 64,
    }
    run = post.ControllerRun(config, contract)
    run.set_stage("test_stage")
    events = run.journal_path.read_text(encoding="ascii").splitlines()
    event = json.loads(events[0])
    event["status"] = "forged"
    run.journal_path.write_bytes(canonical_json_bytes(event) + b"\n")

    with pytest.raises(post.PostTrainingError, match="controller_journal_invalid"):
        post.ControllerRun(config, contract)


def test_launchd_contract_restarts_only_unexpected_nonzero_exit(
    isolated_repo, monkeypatch
):
    root, contract_path, _contract_value = isolated_repo
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    monkeypatch.setattr(post.sys, "executable", str(venv_python))
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    config_path = root / "config.json"
    config_path.write_bytes(canonical_json_bytes(config))

    payload = plistlib.loads(post._launchd_payload(config_path, config))

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ThrottleInterval"] == 30
    assert payload["ProgramArguments"][0:2] == ["/usr/bin/caffeinate", "-i"]
    assert payload["ProgramArguments"][2] == str(venv_python)


def test_grpo_uses_same_cross_family_host_lease_as_resident_sft(tmp_path):
    path = tmp_path / "host.lock"
    grpo = "com.aura.resident-32b-recurrent-grpo-cp-test.post-training"
    sft = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-test"

    with post._resident_training_host_lease(
        label=grpo,
        config_sha256="a" * 64,
        lease_path=path,
    ):
        with pytest.raises(post.PostTrainingError, match="resident_training_host_busy"):
            with post._resident_training_host_lease(
                label=sft,
                config_sha256="b" * 64,
                lease_path=path,
            ):
                pass

    with post._resident_training_host_lease(
        label=sft,
        config_sha256="b" * 64,
        lease_path=path,
    ):
        assert json.loads(path.read_text())["active"] is True


def test_grpo_install_retires_stale_sft_and_grpo_jobs(monkeypatch, tmp_path):
    launch_agents = tmp_path / "LaunchAgents"
    quarantine = tmp_path / "quarantine"
    launch_agents.mkdir()
    active = "com.aura.resident-32b-recurrent-grpo-cp-active.post-training"
    stale_sft = "com.aura.resident-sft.resident-32b-recurrent-sft-bootstrap-cp-stale"
    stale_grpo = "com.aura.resident-32b-recurrent-grpo-cp-stale.post-training"
    for label in (active, stale_sft, stale_grpo):
        (launch_agents / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label}))
    inventories = iter([{active, stale_sft, stale_grpo}, {active}])
    monkeypatch.setattr(post, "_loaded_resident_training_labels", lambda: next(inventories))
    calls = []

    def _run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(post.subprocess, "run", _run)

    receipt = post._retire_resident_training_jobs(
        active_label=active,
        launch_agents=launch_agents,
        quarantine_root=quarantine,
    )

    assert receipt["retired_labels"] == [stale_grpo, stale_sft]
    assert len(list(quarantine.glob("*.plist"))) == 2
    assert (launch_agents / f"{active}.plist").is_file()
    assert len(calls) == 2


def _detached_status(returncode: int) -> dict:
    receipt = {
        "returncode": returncode,
        "containment_verified": True,
        "process_group_empty": True,
        "lineage_empty": True,
        "timed_out": False,
        "receipt_sha256": f"{abs(returncode):064x}"[-64:],
    }
    return {
        "terminal": True,
        "completion_indeterminate": False,
        "supervisor_alive": False,
        "child_state": "dead",
        "receipt": receipt,
    }


def test_training_controller_recovers_native_crash_from_verified_checkpoint(
    isolated_repo,
    monkeypatch,
):
    root, contract_path, contract = isolated_repo
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    run = post.ControllerRun(config, contract)
    snapshots = iter(
        [
            {"sha256": "1" * 64, "checkpoint_step": 0},
            {"sha256": "2" * 64, "checkpoint_step": 6},
            {"sha256": "2" * 64, "checkpoint_step": 6},
            {"sha256": "3" * 64, "checkpoint_step": 288},
        ]
    )
    statuses = iter([_detached_status(-5), _detached_status(0)])
    launched: list[int] = []

    def launch_or_attach(attempt):
        launched.append(attempt)
        return root / f"attempt-{attempt}", {
            "progress_before": next(snapshots),
        }

    monkeypatch.setattr(run, "_launch_or_attach_training_attempt", launch_or_attach)
    monkeypatch.setattr(run, "wait_detached", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr(
        post.prereg,
        "_training_progress_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        post.prereg,
        "validated_training_resume_checkpoint",
        lambda *_args, **_kwargs: {
            "checkpoint_sequence": 6,
            "checkpoint_evidence_sha256": "4" * 64,
        },
    )

    terminal = run.run_training_with_recovery()

    assert terminal["receipt"]["returncode"] == 0
    assert launched == [1, 2]
    failed = post._strict_json(run.root / "training_attempt_0001.json")
    assert failed["terminal_success"] is False
    assert failed["durable_progress"] is True
    assert failed["resume_checkpoint"]["checkpoint_sequence"] == 6
    succeeded = post._strict_json(run.root / "training_attempt_0002.json")
    assert succeeded["terminal_success"] is True


def test_training_controller_stops_after_repeated_no_progress(
    isolated_repo,
    monkeypatch,
):
    root, contract_path, contract = isolated_repo
    config = post.build_config(
        contract_path=contract_path,
        output_root=root / "artifacts/cp259/post-training",
        source_commit="a" * 40,
        seeds=[(1 << 62) + index for index in range(8)],
    )
    run = post.ControllerRun(config, contract)
    unchanged = {"sha256": "1" * 64, "checkpoint_step": 0}
    monkeypatch.setattr(
        run,
        "_launch_or_attach_training_attempt",
        lambda attempt: (root / f"attempt-{attempt}", {"progress_before": unchanged}),
    )
    monkeypatch.setattr(
        run,
        "wait_detached",
        lambda *_args, **_kwargs: _detached_status(-5),
    )
    monkeypatch.setattr(
        post.prereg,
        "_training_progress_snapshot",
        lambda *_args, **_kwargs: unchanged,
    )
    monkeypatch.setattr(
        post.prereg,
        "validated_training_resume_checkpoint",
        lambda *_args, **_kwargs: {
            "checkpoint_sequence": 0,
            "checkpoint_evidence_sha256": "4" * 64,
        },
    )

    with pytest.raises(
        post.PostTrainingError,
        match="training_no_progress_failure_limit_exhausted",
    ):
        run.run_training_with_recovery()

    assert (run.root / "training_attempt_0001.json").is_file()
    assert (run.root / "training_attempt_0002.json").is_file()


def test_external_custody_request_cannot_self_certify(isolated_repo):
    root, _contract_path, contract = isolated_repo
    controller_root = root / "artifacts/cp259/post-training"
    config = {
        "output_root": str(controller_root.relative_to(root)),
        "config_sha256": "7" * 64,
    }
    run = post.ControllerRun(config, contract)

    request = run.custody_request()

    assert request["distinct_keys_and_organizations_required"] is True
    assert request["producer_private_key_access_disqualifies_claim"] is True
    assert request["claim_state"] == {
        "external_trust_present": False,
        "positive_interaction_proven": False,
        "frontier_level_proven": False,
        "release_eligible": False,
    }
