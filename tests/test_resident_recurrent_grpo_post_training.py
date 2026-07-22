from __future__ import annotations

import json
import plistlib
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
            "frozen_adapter": f"{artifact}/frozen-adapter",
            "directional_campaign": f"{artifact}/directional-campaign",
        },
        "training": {
            "parameters": {"max_steps": 288},
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
    return {**material, "contract_sha256": post._document_sha(material)}


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "REPO_ROOT", tmp_path)
    source = tmp_path / post.SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_text("# controller\n", encoding="ascii")
    (tmp_path / "model").mkdir()
    contract = _contract(tmp_path)
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
    assert validated["directional"]["seeds"] == seeds
    assert validated["directional"]["claim_eligible"] is False
    assert all(
        validated["claim_policy"][claim] is False
        for claim in (
            "reasoning_gain_proven",
            "positive_interaction_proven",
            "frontier_level_proven",
            "release_eligible",
        )
    )


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


def test_launchd_contract_restarts_only_unexpected_nonzero_exit(isolated_repo):
    root, contract_path, _contract_value = isolated_repo
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
