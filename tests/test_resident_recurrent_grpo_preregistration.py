"""Contracts for the resident recurrent-GRPO preregistration."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from tools import prepare_resident_recurrent_grpo_campaign as prereg

BASE_IDENTITY = {"method": "sha256", "fingerprint": "1" * 64, "files": 4}
BEHAVIOR_IDENTITY = {"bundle_sha256": "2" * 64, "file_count": 1, "files": []}


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
    assert contract["evaluation"]["powered_confirmatory"]["task_count"] == 2877
    assert contract["evaluation"]["powered_confirmatory"]["cell_count"] == 17262
    assert receipt["claim_eligible"] is False


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
