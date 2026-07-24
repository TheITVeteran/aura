"""Contracts for the resident recurrent-GRPO preregistration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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


def test_preregistration_binds_broad_training_and_powered_evaluation():
    contract = _contract()
    receipt = prereg.validate_contract(contract, verify_model=False)

    assert contract["training"]["parameters"]["domains"] == list(
        RECURRENCE_TRAINING_FAMILIES
    )
    assert contract["training"]["dataset"]["train_tasks"] == 288
    assert contract["training"]["dataset"]["holdout_tasks"] == 36
    assert contract["training"]["dataset"]["train_holdout_id_overlap"] == 0
    assert contract["training"]["parameters"]["trajectory_credit"] is True
    assert "--trajectory-credit" in contract["training"]["argv"]
    mechanism = contract["evaluation"]["mechanism_attribution"]
    assert mechanism["required"] is True
    assert mechanism["claim_eligible"] is False
    assert "resident_full_stack" in mechanism["candidate_profiles"]
    assert "resident_full_stack_no_fast_weights" in mechanism["candidate_profiles"]
    assert "resident_full_stack > adapter_equal_compute" in mechanism[
        "required_comparisons"
    ]
    assert "fast_weight_erase_and_canary_receipts_required" in mechanism[
        "acceptance_rules"
    ]
    assert contract["evaluation"]["powered_confirmatory"]["task_count"] == 2877
    assert contract["evaluation"]["powered_confirmatory"]["cell_count"] == 17262
    assert receipt["claim_eligible"] is False


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
    assert verdict["evidence"]["adapter"]["sha256"] == hashlib.sha256(
        adapter
    ).hexdigest()
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

    assert prereg._launch_training(contract_path, resume=False) == 0

    argv = captured["argv"]
    assert isinstance(argv, list)
    verifier = json.loads(argv[argv.index("--resume-verifier-json") + 1])
    command = argv[argv.index("--resume-verifier-json") + 2 :]
    assert verifier[0] == str(venv_python)
    assert command[0] == str(venv_python)
    assert str(Path(venv_python).resolve()) not in verifier
    assert str(Path(venv_python).resolve()) not in command


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


def test_launch_answer_channel_preflight_is_detached_and_source_bound(
    tmp_path, monkeypatch
):
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
    assert argv[argv.index("--name") + 1].endswith(
        "-answer-channel-preflight"
    )
    assert argv[argv.index("--cwd") + 1] == str(tmp_path)
    assert argv[argv.index("--timeout") + 1] == "5400"
    resume_index = argv.index("--resume-contract")
    assert argv[resume_index + 1] == "none"
    command = argv[resume_index + 2 :]
    assert command[0] == str(venv_python)
    assert command[1] == str(Path(prereg.__file__).resolve(strict=True))
    assert command[2:4] == ["run-answer-channel-preflight", "--contract"]
    assert command[4] == str(contract_path.resolve(strict=True))
