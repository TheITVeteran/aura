"""Adversarial custody tests for the causal verified-transition ledger."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    build_role_attestation,
    policy_signed_payload,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.learning.recurrent_grpo import (
    ExactAdjointInterventionConfig,
    RecurrentGRPOConfig,
    VerifiedTrajectoryGroupConfig,
    recurrent_policy_optimizer_config,
)
from core.learning.verified_recurrent_transition_repository import (
    finalize_verified_recurrent_transition_campaign,
)
from core.learning.verified_transition_causal_campaign import (
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4,
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5,
    CAUSAL_CAMPAIGN_MANIFEST_SCHEMA,
    EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA,
    EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA_V3,
    EXTERNAL_POLICY_STATE_REPLAY_RESULT_SCHEMA,
    CausalCampaignScheduleEntry,
    VerifiedTransitionCausalCampaignError,
    VerifiedTransitionCausalCampaignLedger,
    build_causal_campaign_manifest,
    validate_causal_campaign_evidence_manifest,
    validate_causal_campaign_manifest,
    validate_external_evidence_verification_receipt,
    validate_external_policy_state_replay_result,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    TransitionGroupPlanEntry,
    build_transition_group_manifest,
)
from core.learning.verified_transition_measurement_chain import (
    recurrent_grpo_config_contract,
)
from core.learning.verified_transition_policy_probe import (
    build_initial_policy_state_custody,
)
from core.learning.verified_transition_policy_state_replay import (
    POLICY_STATE_REPLAY_RECEIPT_SCHEMA,
    build_policy_state_replay_contract,
)

BASE_SECOND = 1_800_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _evidence_manifest(
    material: dict[str, Any],
    statuses: list[str],
    *,
    pre_measurements: bool = False,
) -> dict[str, Any]:
    rows = [
        {
            "sequence": sequence,
            "status": status,
            "package_artifact": {
                "path": f"/private/replay/group-{sequence:08d}.json",
                "sha256": _sha(f"package-bytes-{sequence}"),
                "size_bytes": 128 + sequence,
            },
            "package_receipt_sha256": _sha(f"package-{sequence}"),
            "group_manifest_sha256": _sha(f"group-{sequence}"),
            "reward_receipt_sha256": _sha(f"reward-{sequence}"),
            "group_admission_sha256": (
                _sha(f"admission-{sequence}") if status == "updated" else None
            ),
            "update_receipt_sha256": (_sha(f"update-{sequence}") if status == "updated" else None),
            "trainer_step_receipt_sha256": _sha(f"trainer-step-{sequence}"),
            "sample_receipt_sha256s": [_sha(f"sample-{sequence}")],
            "evidence_receipt_sha256s": [_sha(f"evidence-{sequence}")],
        }
        for sequence, status in enumerate(statuses)
    ]
    if pre_measurements:
        for row in rows:
            row["pre_measurement_sha256"] = (
                _sha(f"pre-measurement-{row['sequence']}") if row["status"] == "updated" else None
            )
    body = {
        "schema": (
            CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
            if pre_measurements
            else CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA
        ),
        "contract_sha256": _sha("provider-contract"),
        "campaign_schedule_root_sha256": material["schedule_root"],
        "trust_policy_sha256": material["policy"].policy_sha256,
        "campaign_ledger_root": str(material["root"].resolve()),
        "transition_artifact_root": str(
            (material["root"].parent / "transition-artifacts").resolve()
        ),
        "update_journal_root": str((material["root"].parent / "updates").resolve()),
        "transaction_root": str((material["root"].parent / "transactions").resolve()),
        "completed_groups": len(rows),
        "halt_reason": "max_steps",
        "group_packages": rows,
        "updated_replay_sequences": [row["sequence"] for row in rows if row["status"] == "updated"],
        "created_at_unix_ns": (BASE_SECOND + 181) * 1_000_000_000,
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _verification_receipt(
    evidence_manifest: dict[str, Any],
    *,
    verifier_identity: str = "fixture-evidence-verifier",
    verified_at_unix: int = BASE_SECOND + 181,
) -> dict[str, Any]:
    observations = []
    for package in evidence_manifest["group_packages"]:
        observation = {
            "sequence": package["sequence"],
            "package_artifact": package["package_artifact"],
            "package_receipt_sha256": package["package_receipt_sha256"],
            "sample_receipt_sha256s": package["sample_receipt_sha256s"],
            "evidence_receipt_sha256s": package["evidence_receipt_sha256s"],
            "reward_receipt_sha256": package["reward_receipt_sha256"],
            "group_admission_sha256": package["group_admission_sha256"],
            "update_receipt_sha256": package["update_receipt_sha256"],
            "trainer_step_receipt_sha256": package["trainer_step_receipt_sha256"],
        }
        if evidence_manifest["schema"] in {
            CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4,
            CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5,
        }:
            observation["pre_measurement_sha256"] = package["pre_measurement_sha256"]
        if evidence_manifest["schema"] == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5:
            observation.update(
                {
                    "group_manifest_sha256": package["group_manifest_sha256"],
                    "policy_before_sha256": package["policy_before_sha256"],
                    "policy_after_sha256": package["policy_after_sha256"],
                    "objective_receipt_sha256": package["objective_receipt_sha256"],
                    "state_source_sha256": package["state_source_sha256"],
                    "post_state_transaction_stage_sha256": package[
                        "post_state_transaction_stage_sha256"
                    ],
                    "policy_state_replay_receipt_artifact": package[
                        "policy_state_replay_receipt_artifact"
                    ],
                    "policy_state_replay_receipt_sha256": package[
                        "policy_state_replay_receipt_sha256"
                    ],
                }
            )
        observations.append(observation)
    is_v5 = evidence_manifest["schema"] == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5
    body = {
        "schema": (
            EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA_V3
            if is_v5
            else EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA
        ),
        "evidence_manifest_sha256": evidence_manifest["manifest_sha256"],
        "verifier_identity": verifier_identity,
        "verified_package_count": len(observations),
        "artifact_observation_root_sha256": hashlib.sha256(
            canonical_json_bytes({"artifact_observations": observations})
        ).hexdigest(),
        "validation_profile": (
            "recurrent_transition_causal_replay.v4"
            if is_v5
            else (
                "recurrent_transition_causal_replay.v3"
                if evidence_manifest["schema"] == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
                else "recurrent_transition_causal_replay.v2"
            )
        ),
        "verified_at_unix": verified_at_unix,
        **(
            {
                "policy_state_replay_contract_sha256": evidence_manifest[
                    "policy_state_replay_contract_sha256"
                ],
                "verified_updated_transition_count": len(
                    evidence_manifest["updated_replay_sequences"]
                ),
                "policy_state_replay_receipt_root_sha256": evidence_manifest[
                    "policy_state_replay_receipt_root_sha256"
                ],
                "external_policy_state_replayed": True,
            }
            if is_v5
            else {}
        ),
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _write_bound_file(path: Path, payload: bytes, *, private: bool = False) -> Path:
    path.write_bytes(payload)
    if private:
        path.chmod(0o600)
    return path.resolve(strict=True)


def _file_binding(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _policy_state_replay_contract(
    root: Path,
    *,
    initial_policy: str | None = None,
) -> dict[str, Any]:
    root.mkdir()
    model = root / "model"
    model.mkdir()
    _write_bound_file(model / "weights.safetensors", b"resident-weights")
    for name, document in (
        ("config.json", {"model_type": "qwen2"}),
        ("tokenizer.json", {"version": "1.0"}),
        ("tokenizer_config.json", {"eos_token": "<eos>"}),
    ):
        _write_bound_file(model / name, canonical_json_bytes(document))

    spec = RLCExecutionSpec(recurrent_steps=4)
    spec_path = _write_bound_file(
        root / "execution-spec.json",
        json.dumps(
            spec.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
    )
    replay_source = _write_bound_file(
        root / "replay-source.py",
        b"REPLAY_VERSION = 1\n",
    )
    objective_source = _write_bound_file(
        root / "objective-source.py",
        b"OBJECTIVE_VERSION = 1\n",
    )
    adapter_path = _write_bound_file(
        root / "initial-adapter.safetensors",
        b"sealed-adapter",
        private=True,
    )
    optimizer_path = _write_bound_file(
        root / "initial-optimizer.safetensors",
        b"sealed-optimizer",
        private=True,
    )
    initial_policy = initial_policy or _sha("v5-initial-policy")
    adapter_keys = ["layer.lora_a", "layer.lora_b"]
    optimizer_keys = [
        "layer.lora_a.m",
        "layer.lora_a.v",
        "layer.lora_b.m",
        "layer.lora_b.v",
        "learning_rate",
        "step",
    ]
    custody = build_initial_policy_state_custody(
        initial_policy_probe_sha256=_sha("v5-policy-probe"),
        initial_policy_sha256=initial_policy,
        execution_spec_sha256=spec.sha256,
        adapter_initialization={
            "seed": 17,
            "rank": 8,
            "layers": 8,
            "targets": ["q_proj"],
        },
        optimizer_initialization=recurrent_policy_optimizer_config(1e-5),
        initial_adapter_artifact={
            "path": adapter_path.name,
            "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
            "size_bytes": adapter_path.stat().st_size,
            "tensor_count": len(adapter_keys),
            "tensor_keys": adapter_keys,
            "tensor_keys_sha256": hashlib.sha256(canonical_json_bytes(adapter_keys)).hexdigest(),
            "policy_sha256": initial_policy,
        },
        initial_optimizer_artifact={
            "path": optimizer_path.name,
            "sha256": hashlib.sha256(optimizer_path.read_bytes()).hexdigest(),
            "size_bytes": optimizer_path.stat().st_size,
            "tensor_count": len(optimizer_keys),
            "tensor_keys": optimizer_keys,
            "tensor_keys_sha256": hashlib.sha256(canonical_json_bytes(optimizer_keys)).hexdigest(),
        },
        initial_adapter_path=adapter_path,
        initial_optimizer_path=optimizer_path,
    )
    trajectory = VerifiedTrajectoryGroupConfig(
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.4,
            causality_margin=0.1,
            stopping_steps=(1, 2),
            stopping_weight=0.3,
            stopping_ponder_cost=0.01,
            stopping_temperature=0.2,
        )
    ).to_dict()
    return build_policy_state_replay_contract(
        preregistration_contract_sha256=_sha("v5-preregistration"),
        initial_policy_sha256=initial_policy,
        model_path=model,
        base_checkpoint=full_weight_checkpoint_identity(model),
        behavior_bundle=model_behavior_bundle_identity(model),
        execution_spec_path=spec_path,
        execution_spec_document=spec.to_dict(),
        source_bindings={
            "objective": _file_binding(objective_source),
            "replay": _file_binding(replay_source),
        },
        initial_policy_state_custody=custody,
        recurrent_grpo_config=recurrent_grpo_config_contract(
            RecurrentGRPOConfig(
                kl_coefficient=0.02,
                advantage_clip=4.0,
            )
        ),
        verified_trajectory_config=trajectory,
        external_verifier_max_seconds=21_600,
    )


def _tensor_identity(role: str, *, material: str) -> dict[str, Any]:
    names = ["layer.weight"]
    body = {
        "schema": "aura.tensor_map_identity.v1",
        "role": role,
        "tensor_count": 1,
        "tensor_keys_sha256": hashlib.sha256(canonical_json_bytes(names)).hexdigest(),
        "tensors": [
            {
                "name": names[0],
                "dtype": "float32",
                "shape": [1, 1],
                "value_sha256": _sha(f"{role}-{material}"),
            }
        ],
    }
    return {
        **body,
        "tensor_root_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _policy_state_replay_receipt(
    contract: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    optimizer_config = recurrent_policy_optimizer_config(1e-5)
    body = {
        "schema": POLICY_STATE_REPLAY_RECEIPT_SCHEMA,
        "execution_spec_sha256": contract["execution_spec"]["semantic_sha256"],
        "policy_before_sha256": expected["policy_before_sha256"],
        "policy_after_sha256": expected["policy_after_sha256"],
        "objective_receipt_sha256": expected["objective_receipt_sha256"],
        "optimizer_config": optimizer_config,
        "optimizer_config_sha256": hashlib.sha256(
            canonical_json_bytes(optimizer_config)
        ).hexdigest(),
        "pre_adapter_identity": _tensor_identity("pre_adapter", material="pre"),
        "pre_optimizer_identity": _tensor_identity("pre_optimizer", material="pre"),
        "gradient_identity": _tensor_identity("gradient", material="gradient"),
        "post_adapter_identity": _tensor_identity("post_adapter", material="post"),
        "post_optimizer_identity": _tensor_identity("post_optimizer", material="post"),
        "optimizer_update_count": 1,
        "objective_recomputed": True,
        "all_gradient_tensors_recomputed": True,
        "adapter_post_state_exact": True,
        "optimizer_post_state_exact": True,
        "external_policy_state_replayed": True,
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _external_policy_state_replay_result(
    contract: dict[str, Any],
    expected: dict[str, Any],
    *,
    verifier_identity: str = "fixture-external-policy-replayer",
    verified_at_unix: int = BASE_SECOND + 181,
) -> dict[str, Any]:
    producer_binding = {
        key: value
        for key, value in expected.items()
        if key
        not in {
            "policy_state_replay_receipt_artifact",
            "policy_state_replay_receipt_sha256",
        }
    }
    body = {
        "schema": EXTERNAL_POLICY_STATE_REPLAY_RESULT_SCHEMA,
        "policy_state_replay_contract_sha256": contract["contract_sha256"],
        **producer_binding,
        "execution_spec_sha256": contract["execution_spec"]["semantic_sha256"],
        "verifier_identity": verifier_identity,
        "verified_at_unix": verified_at_unix,
        "policy_state_replay_receipt": _policy_state_replay_receipt(
            contract,
            expected,
        ),
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _reseal_manifest(document: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    document["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return document


def _reseal_receipt(document: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    document["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return document


def _v5_evidence_manifest(
    material: dict[str, Any],
    contract: dict[str, Any],
    statuses: list[str],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    evidence = _evidence_manifest(
        material,
        statuses,
        pre_measurements=True,
    )
    evidence["schema"] = CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5
    policy_before = contract["initial_policy_sha256"]
    results: dict[int, dict[str, Any]] = {}
    expected_transitions: dict[int, dict[str, Any]] = {}
    for row in evidence["group_packages"]:
        sequence = row["sequence"]
        if row["status"] == "updated":
            policy_after = _sha(f"v5-policy-after-{sequence}")
            expected = {
                "provider_contract_sha256": evidence["contract_sha256"],
                "campaign_schedule_root_sha256": evidence["campaign_schedule_root_sha256"],
                "campaign_manifest_sha256": material["manifest"]["manifest_sha256"],
                "sequence": sequence,
                "group_manifest_sha256": row["group_manifest_sha256"],
                "group_admission_sha256": row["group_admission_sha256"],
                "update_receipt_sha256": row["update_receipt_sha256"],
                "pre_measurement_sha256": row["pre_measurement_sha256"],
                "state_source_sha256": _sha(f"state-source-{sequence}"),
                "post_state_transaction_stage_sha256": _sha(f"post-state-stage-{sequence}"),
                "objective_receipt_sha256": _sha(f"objective-{sequence}"),
                "policy_before_sha256": policy_before,
                "policy_after_sha256": policy_after,
            }
            result = _external_policy_state_replay_result(
                contract,
                expected,
            )
            payload = canonical_json_bytes(result)
            replay_artifact = {
                "path": (f"/private/replay/policy-state-replay-{sequence:08d}.json"),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            expected.update(
                {
                    "policy_state_replay_receipt_artifact": replay_artifact,
                    "policy_state_replay_receipt_sha256": result["receipt_sha256"],
                }
            )
            row.update(
                {
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    "objective_receipt_sha256": expected["objective_receipt_sha256"],
                    "state_source_sha256": expected["state_source_sha256"],
                    "post_state_transaction_stage_sha256": expected[
                        "post_state_transaction_stage_sha256"
                    ],
                    "policy_state_replay_receipt_artifact": replay_artifact,
                    "policy_state_replay_receipt_sha256": result["receipt_sha256"],
                }
            )
            results[sequence] = result
            expected_transitions[sequence] = expected
        else:
            policy_after = policy_before
            row.update(
                {
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    "objective_receipt_sha256": None,
                    "state_source_sha256": None,
                    "post_state_transaction_stage_sha256": None,
                    "policy_state_replay_receipt_artifact": None,
                    "policy_state_replay_receipt_sha256": None,
                }
            )
        policy_before = policy_after
    ordered = [
        {
            "sequence": sequence,
            "receipt_sha256": results[sequence]["receipt_sha256"],
        }
        for sequence in sorted(results)
    ]
    evidence.update(
        {
            "policy_state_replay_contract": contract,
            "policy_state_replay_contract_sha256": contract["contract_sha256"],
            "policy_state_replay_receipt_root_sha256": hashlib.sha256(
                canonical_json_bytes({"policy_state_replay_receipts": ordered})
            ).hexdigest(),
        }
    )
    return (
        _reseal_manifest(evidence),
        results,
        expected_transitions,
    )


def test_external_verifier_receipt_stays_compact_at_288_groups(
    material: dict[str, Any],
) -> None:
    evidence = _evidence_manifest(material, ["rejected"] * 288)
    receipt = _verification_receipt(evidence)
    validated = validate_external_evidence_verification_receipt(
        receipt,
        evidence_manifest=evidence,
    )
    assert validated["verified_package_count"] == 288
    assert len(canonical_json_bytes(validated)) < 1_024


def test_v4_manifest_requires_pre_measurement_only_for_updated_rows(
    material: dict[str, Any],
) -> None:
    evidence = _evidence_manifest(
        material,
        ["updated", "rejected"],
        pre_measurements=True,
    )
    assert validate_causal_campaign_evidence_manifest(evidence) == evidence
    receipt = _verification_receipt(evidence)
    assert (
        validate_external_evidence_verification_receipt(
            receipt,
            evidence_manifest=evidence,
        )["validation_profile"]
        == "recurrent_transition_causal_replay.v3"
    )

    tampered = copy.deepcopy(evidence)
    tampered["group_packages"][0]["pre_measurement_sha256"] = None
    unsigned = dict(tampered)
    unsigned.pop("manifest_sha256")
    tampered["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="causal_campaign_evidence_pre_measurement_status_invalid",
    ):
        validate_causal_campaign_evidence_manifest(tampered)


def test_v3_v4_validation_preserves_exact_canonical_documents(
    material: dict[str, Any],
) -> None:
    for pre_measurements in (False, True):
        evidence = _evidence_manifest(
            material,
            ["updated", "rejected"],
            pre_measurements=pre_measurements,
        )
        evidence_bytes = canonical_json_bytes(evidence)
        receipt = _verification_receipt(evidence)
        receipt_bytes = canonical_json_bytes(receipt)

        assert (
            canonical_json_bytes(validate_causal_campaign_evidence_manifest(evidence))
            == evidence_bytes
        )
        assert (
            canonical_json_bytes(
                validate_external_evidence_verification_receipt(
                    receipt,
                    evidence_manifest=evidence,
                )
            )
            == receipt_bytes
        )
        assert receipt["schema"] == EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA


def test_v5_manifest_transition_results_and_positive_aggregate_validate(
    material: dict[str, Any],
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / "v5-contract")
    evidence, results, expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated", "rejected"],
    )

    assert validate_causal_campaign_evidence_manifest(evidence) == evidence
    assert (
        validate_external_policy_state_replay_result(
            results[0],
            policy_state_replay_contract=contract,
            expected_transition=expected[0],
        )
        == results[0]
    )
    aggregate = _verification_receipt(evidence)
    assert (
        validate_external_evidence_verification_receipt(
            aggregate,
            evidence_manifest=evidence,
        )
        == aggregate
    )
    assert aggregate["external_policy_state_replayed"] is True
    assert aggregate["verified_updated_transition_count"] == 1
    assert (
        aggregate["policy_state_replay_receipt_root_sha256"]
        == evidence["policy_state_replay_receipt_root_sha256"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "provider_contract_sha256",
        "campaign_schedule_root_sha256",
        "campaign_manifest_sha256",
        "sequence",
        "group_manifest_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "pre_measurement_sha256",
        "state_source_sha256",
        "post_state_transaction_stage_sha256",
        "objective_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "execution_spec_sha256",
        "policy_state_replay_contract_sha256",
    ],
)
def test_external_policy_state_replay_result_rejects_cross_binding_drift(
    material: dict[str, Any],
    field: str,
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / f"result-drift-{field}")
    _evidence, results, expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated"],
    )
    tampered = copy.deepcopy(results[0])
    tampered[field] = tampered[field] + 1 if field == "sequence" else _sha(f"tampered-{field}")
    _reseal_receipt(tampered)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="external_policy_state_replay_result_invalid",
    ):
        validate_external_policy_state_replay_result(
            tampered,
            policy_state_replay_contract=contract,
            expected_transition=expected[0],
        )


def test_external_policy_state_replay_result_rejects_nested_receipt_claim_drift(
    material: dict[str, Any],
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / "nested-receipt-drift")
    _evidence, results, expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated"],
    )
    tampered = copy.deepcopy(results[0])
    nested = tampered["policy_state_replay_receipt"]
    nested["external_policy_state_replayed"] = False
    _reseal_receipt(nested)
    _reseal_receipt(tampered)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="external_policy_state_replay_result_nested_contract_invalid",
    ):
        validate_external_policy_state_replay_result(
            tampered,
            policy_state_replay_contract=contract,
            expected_transition=expected[0],
        )


def test_external_policy_state_replay_result_rejects_artifact_substitution(
    material: dict[str, Any],
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / "result-artifact-drift")
    _evidence, results, expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated"],
    )
    tampered_expected = copy.deepcopy(expected[0])
    tampered_expected["policy_state_replay_receipt_artifact"]["sha256"] = _sha(
        "substituted-result-artifact"
    )

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="external_policy_state_replay_result_invalid",
    ):
        validate_external_policy_state_replay_result(
            results[0],
            policy_state_replay_contract=contract,
            expected_transition=tampered_expected,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            "updated_missing_replay",
            "policy_state_replay_status_invalid",
        ),
        (
            "rejected_has_replay",
            "policy_state_replay_status_invalid",
        ),
        (
            "rejected_policy_changed",
            "policy_state_replay_status_invalid",
        ),
        (
            "policy_lineage_changed",
            "policy_state_replay_lineage_invalid",
        ),
        (
            "receipt_root_changed",
            "policy_state_replay_receipt_root_mismatch",
        ),
    ],
)
def test_v5_manifest_rejects_incomplete_or_substituted_replay_rows(
    material: dict[str, Any],
    mutation: str,
    error: str,
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / f"manifest-drift-{mutation}")
    evidence, _results, _expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated", "rejected"],
    )
    tampered = copy.deepcopy(evidence)
    if mutation == "updated_missing_replay":
        tampered["group_packages"][0]["policy_state_replay_receipt_artifact"] = None
    elif mutation == "rejected_has_replay":
        tampered["group_packages"][1]["objective_receipt_sha256"] = _sha(
            "forged-rejected-objective"
        )
    elif mutation == "rejected_policy_changed":
        tampered["group_packages"][1]["policy_after_sha256"] = _sha("forged-rejected-policy")
    elif mutation == "policy_lineage_changed":
        tampered["group_packages"][1]["policy_before_sha256"] = _sha("substituted-policy-lineage")
        tampered["group_packages"][1]["policy_after_sha256"] = tampered["group_packages"][1][
            "policy_before_sha256"
        ]
    else:
        tampered["policy_state_replay_receipt_root_sha256"] = _sha("substituted-replay-root")
    _reseal_manifest(tampered)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match=error,
    ):
        validate_causal_campaign_evidence_manifest(tampered)


def test_v5_manifest_rejects_duplicate_or_reordered_transition_receipts(
    material: dict[str, Any],
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / "duplicate-replay-receipts")
    evidence, _results, _expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated", "updated"],
    )
    duplicate = copy.deepcopy(evidence)
    duplicate["group_packages"][1]["policy_state_replay_receipt_sha256"] = duplicate[
        "group_packages"
    ][0]["policy_state_replay_receipt_sha256"]
    duplicate_pairs = [
        {
            "sequence": row["sequence"],
            "receipt_sha256": row["policy_state_replay_receipt_sha256"],
        }
        for row in duplicate["group_packages"]
    ]
    duplicate["policy_state_replay_receipt_root_sha256"] = hashlib.sha256(
        canonical_json_bytes({"policy_state_replay_receipts": duplicate_pairs})
    ).hexdigest()
    _reseal_manifest(duplicate)
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="policy_state_replay_receipt_duplicate",
    ):
        validate_causal_campaign_evidence_manifest(duplicate)

    reordered = copy.deepcopy(evidence)
    ordered_pairs = [
        {
            "sequence": row["sequence"],
            "receipt_sha256": row["policy_state_replay_receipt_sha256"],
        }
        for row in reversed(reordered["group_packages"])
    ]
    reordered["policy_state_replay_receipt_root_sha256"] = hashlib.sha256(
        canonical_json_bytes({"policy_state_replay_receipts": ordered_pairs})
    ).hexdigest()
    _reseal_manifest(reordered)
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="policy_state_replay_receipt_root_mismatch",
    ):
        validate_causal_campaign_evidence_manifest(reordered)


@pytest.mark.parametrize(
    "field",
    [
        "policy_state_replay_contract_sha256",
        "verified_updated_transition_count",
        "policy_state_replay_receipt_root_sha256",
        "external_policy_state_replayed",
    ],
)
def test_v5_aggregate_rejects_false_policy_replay_claims(
    material: dict[str, Any],
    field: str,
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / f"aggregate-drift-{field}")
    evidence, _results, _expected = _v5_evidence_manifest(
        material,
        contract,
        ["updated", "rejected"],
    )
    receipt = _verification_receipt(evidence)
    if field == "verified_updated_transition_count":
        receipt[field] = 0
    elif field == "external_policy_state_replayed":
        receipt[field] = False
    else:
        receipt[field] = _sha(f"substituted-{field}")
    _reseal_receipt(receipt)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="external_evidence_verification_receipt_invalid",
    ):
        validate_external_evidence_verification_receipt(
            receipt,
            evidence_manifest=evidence,
        )


def test_v5_positive_aggregate_requires_at_least_one_updated_replay(
    material: dict[str, Any],
) -> None:
    contract = _policy_state_replay_contract(material["root"].parent / "zero-updated-replays")
    evidence, _results, _expected = _v5_evidence_manifest(
        material,
        contract,
        ["rejected", "rejected"],
    )
    receipt = _verification_receipt(evidence)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="external_evidence_verification_receipt_invalid",
    ):
        validate_external_evidence_verification_receipt(
            receipt,
            evidence_manifest=evidence,
        )


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    public = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-external-organization",
        "public_key_b64": base64.b64encode(public).decode("ascii"),
        "key_id": hashlib.sha256(public).hexdigest(),
        "implementation_sha256": _sha(f"{role}-implementation"),
        "release_sha256": _sha(f"{role}-release"),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha(f"{role}-custody"),
    }


def _trust_material() -> tuple[Any, dict[str, Ed25519PrivateKey]]:
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "causal-transition-campaign-2026-07",
        "policy_revision": 1,
        "campaign_name": "causal-transition-campaign",
        "protocol_sha256": _sha("causal-protocol"),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": BASE_SECOND,
        "not_before_unix": BASE_SECOND + 100,
        "expires_at_unix": BASE_SECOND + 10_000,
        "roles": {role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    assert policy_signed_payload(document) == body
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="causal-transition-campaign",
        expected_protocol_sha256=_sha("causal-protocol"),
        now_unix=BASE_SECOND + 120,
    )
    return policy, role_keys


def _make_material(tmp_path: Path, *, group_count: int = 2) -> dict[str, Any]:
    policy, role_keys = _trust_material()
    initial_policy = _sha("initial-policy")
    schedule_root = _sha("causal-schedule-root")
    schedule = tuple(
        CausalCampaignScheduleEntry(
            sequence=sequence,
            task_id=f"task-{sequence}",
            task_commitment_sha256=_sha(f"task-commitment-{sequence}"),
        )
        for sequence in range(group_count)
    )
    planned_second = BASE_SECOND + 150
    manifest = build_causal_campaign_manifest(
        campaign_id="causal-campaign-001",
        provider_contract_sha256=_sha("provider-contract"),
        campaign_schedule_root_sha256=schedule_root,
        trust_policy_sha256=policy.policy_sha256,
        initial_policy_sha256=initial_policy,
        schedule=schedule,
        planned_at_unix_ns=planned_second * 1_000_000_000,
    )
    manifest_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=planned_second,
        private_key=role_keys[TASK_ISSUER],
    )
    root = tmp_path / "campaign"
    ledger = VerifiedTransitionCausalCampaignLedger.create(
        root,
        campaign_manifest=manifest,
        campaign_manifest_attestation=manifest_attestation,
        policy=policy,
    )
    return {
        "policy": policy,
        "role_keys": role_keys,
        "manifest": manifest,
        "initial_policy": initial_policy,
        "schedule_root": schedule_root,
        "schedule": schedule,
        "ledger": ledger,
        "root": root,
    }


@pytest.fixture
def material(tmp_path: Path) -> dict[str, Any]:
    return _make_material(tmp_path)


def _group(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_before: str,
    planned_second: int | None = None,
    manifest_signed_second: int | None = None,
    lineage_signed_second: int | None = None,
) -> dict[str, Any]:
    second = BASE_SECOND + 160 + sequence * 10 if planned_second is None else planned_second
    entries = tuple(
        TransitionGroupPlanEntry(
            episode_id=f"episode-{sequence}-{index}",
            task_id=f"task-{sequence}",
            rng_root_sha256=_sha(f"rng-{sequence}-{index}"),
            policy_sha256=policy_before,
            recurrent_execution_spec_sha256=_sha("execution-spec"),
            producing_branch_index=index,
            sample_seed=1000 + sequence * 10 + index,
            sampling_config_sha256=_sha(f"sampling-{sequence}-{index}"),
        )
        for index in range(2)
    )
    manifest = build_transition_group_manifest(
        group_id=f"group-{sequence}",
        task_id=f"task-{sequence}",
        entries=entries,
        reward_config_sha256=_sha("reward-config"),
        planned_at_unix_ns=second * 1_000_000_000,
    )
    lineage = {
        "schema": "aura.verified_transition.lineage_plan.v1",
        "contract_sha256": _sha("provider-contract"),
        "campaign_id": material["manifest"]["campaign_id"],
        "campaign_schedule_root_sha256": material["schedule_root"],
        "sequence": sequence,
        "task_commitment_sha256": material["schedule"][sequence].task_commitment_sha256,
        "policy_before_sha256": policy_before,
        "group_manifest_sha256": manifest["manifest_sha256"],
    }
    manifest_signed = second if manifest_signed_second is None else manifest_signed_second
    lineage_signed = second if lineage_signed_second is None else lineage_signed_second
    return {
        "manifest": manifest,
        "lineage": lineage,
        "manifest_attestation": build_role_attestation(
            material["policy"],
            role=TASK_ISSUER,
            payload=manifest,
            signed_at_unix=manifest_signed,
            private_key=material["role_keys"][TASK_ISSUER],
        ),
        "lineage_attestation": build_role_attestation(
            material["policy"],
            role=TASK_ISSUER,
            payload=lineage,
            signed_at_unix=lineage_signed,
            private_key=material["role_keys"][TASK_ISSUER],
        ),
        "planned_second": second,
    }


def _admit(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_before: str,
    planned_second: int | None = None,
    admitted_second: int | None = None,
    manifest_signed_second: int | None = None,
    lineage_signed_second: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group = _group(
        material,
        sequence=sequence,
        policy_before=policy_before,
        planned_second=planned_second,
        manifest_signed_second=manifest_signed_second,
        lineage_signed_second=lineage_signed_second,
    )
    admitted = group["planned_second"] + 1 if admitted_second is None else admitted_second
    start = material["ledger"].admit_group_plan(
        sequence=sequence,
        campaign_id=material["manifest"]["campaign_id"],
        campaign_schedule_root_sha256=material["schedule_root"],
        policy_before_sha256=policy_before,
        group_manifest=group["manifest"],
        group_manifest_attestation=group["manifest_attestation"],
        lineage_plan=group["lineage"],
        lineage_attestation=group["lineage_attestation"],
        policy=material["policy"],
        admitted_at_unix_ns=admitted * 1_000_000_000,
    )
    return dict(start), group


def _finish_updated(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_after: str,
) -> dict[str, Any]:
    terminal = material["ledger"].finish_group(
        sequence=sequence,
        status="updated",
        reward_receipt_sha256=_sha(f"reward-{sequence}"),
        group_admission_sha256=_sha(f"admission-{sequence}"),
        update_receipt_sha256=_sha(f"update-{sequence}"),
        policy_after_sha256=policy_after,
        terminal_reason="optimizer_update_committed",
        finished_at_unix_ns=(BASE_SECOND + 162 + sequence * 10) * 1_000_000_000,
    )
    return dict(terminal)


def _complete(material: dict[str, Any]) -> tuple[str, str]:
    policy_1 = _sha("actual-policy-after-0")
    policy_2 = _sha("actual-policy-after-1")
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    _finish_updated(material, sequence=0, policy_after=policy_1)
    _admit(material, sequence=1, policy_before=policy_1)
    _finish_updated(material, sequence=1, policy_after=policy_2)
    return policy_1, policy_2


def _close(material: dict[str, Any]) -> dict[str, Any]:
    completed_second = BASE_SECOND + 181
    evidence = _evidence_manifest(material, ["updated", "updated"])
    payload = material["ledger"].close_payload(
        completed_at_unix_ns=completed_second * 1_000_000_000,
        policy=material["policy"],
        evidence_manifest=evidence,
        external_evidence_verification_receipt=(_verification_receipt(evidence)),
    )
    attestation = build_role_attestation(
        material["policy"],
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][EVIDENCE_VERIFIER],
    )
    return dict(
        material["ledger"].close(
            close_payload=payload,
            evidence_verifier_attestation=attestation,
            policy=material["policy"],
        )
    )


def _rewrite_canonical(path: Path, document: dict[str, Any]) -> None:
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    document["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    path.write_bytes(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def test_manifest_precommits_only_knowable_schedule_facts(
    material: dict[str, Any],
) -> None:
    manifest = validate_causal_campaign_manifest(material["manifest"])
    encoded = canonical_json_bytes(manifest).decode("ascii")

    assert manifest["schema"] == CAUSAL_CAMPAIGN_MANIFEST_SCHEMA
    assert manifest["initial_policy_sha256"] == material["initial_policy"]
    assert "policy_before_sha256" not in encoded
    assert "policy_after_sha256" not in encoded
    assert "group_manifest_sha256" not in encoded
    assert "group_manifest" not in encoded

    attacked = copy.deepcopy(manifest)
    attacked["future_policy_sha256"] = _sha("unknowable-future-policy")
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="manifest_schema_invalid",
    ):
        validate_causal_campaign_manifest(attacked)


def test_campaign_manifest_accessor_returns_validated_copy(
    material: dict[str, Any],
) -> None:
    observed = material["ledger"].campaign_manifest()
    observed["campaign_id"] = "mutated-by-caller"

    assert material["ledger"].campaign_manifest() == material["manifest"]


def test_jit_groups_follow_actual_policy_lineage_and_reopen(
    material: dict[str, Any],
) -> None:
    policy_1, policy_2 = _complete(material)
    start_0, terminal_0 = material["ledger"].group_records_unclosed(sequence=0)
    assert start_0["policy_before_sha256"] == material["initial_policy"]
    assert terminal_0["policy_after_sha256"] == policy_1

    reopened = VerifiedTransitionCausalCampaignLedger.open(
        material["root"], policy=material["policy"]
    )
    start_1, terminal_1 = reopened.group_records_unclosed(sequence=1)
    assert start_1["policy_before_sha256"] == policy_1
    assert terminal_1["policy_after_sha256"] == policy_2


def test_open_group_exposes_validated_start_without_inventing_terminal(
    material: dict[str, Any],
) -> None:
    assert material["ledger"].group_start_if_exists(sequence=0) is None
    start, _group_material = _admit(material, sequence=0, policy_before=material["initial_policy"])

    assert material["ledger"].group_start(sequence=0) == start
    assert material["ledger"].group_start_if_exists(sequence=0) == start
    assert material["ledger"].group_terminal_if_exists(sequence=0) is None

    terminal = _finish_updated(material, sequence=0, policy_after=_sha("recovered-policy-after"))
    assert material["ledger"].group_terminal_if_exists(sequence=0) == terminal


def test_post_disclosure_group_plan_is_rejected(material: dict[str, Any]) -> None:
    boundary = BASE_SECOND + 160
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="causal_group_plan_post_disclosure",
    ):
        _admit(
            material,
            sequence=0,
            policy_before=material["initial_policy"],
            planned_second=boundary,
            admitted_second=boundary,
        )
    assert not (material["root"] / "group-00000000.started.json").exists()


def test_provider_accepted_early_signatures_are_ledger_compatible(
    material: dict[str, Any],
) -> None:
    start, _group_material = _admit(
        material,
        sequence=0,
        policy_before=material["initial_policy"],
        manifest_signed_second=BASE_SECOND + 130,
        lineage_signed_second=BASE_SECOND + 131,
    )
    assert start["sequence"] == 0
    assert start["policy_before_sha256"] == material["initial_policy"]


def test_policy_lineage_substitution_is_rejected(material: dict[str, Any]) -> None:
    actual_policy = _sha("actual-policy-after-0")
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    _finish_updated(material, sequence=0, policy_after=actual_policy)

    substituted = _sha("substituted-policy-after-0")
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="sequence_or_lineage_invalid",
    ):
        _admit(material, sequence=1, policy_before=substituted)
    assert not (material["root"] / "group-00000001.started.json").exists()


def test_updated_terminal_requires_actual_policy_after(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="terminal_policy_after_invalid",
    ):
        material["ledger"].finish_group(
            sequence=0,
            status="updated",
            reward_receipt_sha256=_sha("reward-0"),
            group_admission_sha256=_sha("admission-0"),
            update_receipt_sha256=_sha("update-0"),
            terminal_reason="optimizer_update_committed",
            finished_at_unix_ns=(BASE_SECOND + 162) * 1_000_000_000,
        )


def test_indeterminate_terminal_blocks_further_policy_lineage(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    terminal = material["ledger"].finish_group(
        sequence=0,
        status="indeterminate",
        reward_receipt_sha256=_sha("reward-observed-before-crash"),
        group_admission_sha256=_sha("admission-observed-before-crash"),
        update_receipt_sha256=None,
        terminal_reason="optimizer_state_requires_recovery",
        finished_at_unix_ns=(BASE_SECOND + 162) * 1_000_000_000,
    )
    assert terminal["policy_after_sha256"] is None
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="policy_lineage_indeterminate",
    ):
        _admit(
            material,
            sequence=1,
            policy_before=material["initial_policy"],
        )


@pytest.mark.parametrize(
    "field",
    ["reward_receipt_sha256", "update_receipt_sha256"],
)
def test_closed_campaign_rejects_terminal_evidence_substitution(
    material: dict[str, Any], field: str
) -> None:
    _complete(material)
    _close(material)
    path = material["root"] / "group-00000000.terminal.json"
    terminal = json.loads(path.read_bytes())
    terminal[field] = _sha(f"substituted-{field}")
    _rewrite_canonical(path, terminal)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="previous_terminal_mismatch|close_reconstruction_mismatch",
    ):
        material["ledger"].validate_closed(policy=material["policy"])


def test_unstarted_campaign_tail_closes_as_explicitly_aborted(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    policy_after = _sha("actual-policy-after-0")
    _finish_updated(material, sequence=0, policy_after=policy_after)
    evidence = _evidence_manifest(material, ["updated"])
    payload = material["ledger"].close_payload(
        completed_at_unix_ns=(BASE_SECOND + 181) * 1_000_000_000,
        policy=material["policy"],
        evidence_manifest=evidence,
        external_evidence_verification_receipt=(_verification_receipt(evidence)),
    )
    assert payload["group_statuses"] == ["updated", "aborted"]
    assert payload["group_start_sha256s"][1] is None
    assert payload["group_terminal_sha256s"][1] is None
    assert payload["final_policy_sha256"] == policy_after


def test_started_group_without_terminal_cannot_close(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    policy_after = _sha("actual-policy-after-0")
    _finish_updated(material, sequence=0, policy_after=policy_after)
    _admit(material, sequence=1, policy_before=policy_after)
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="causal_campaign_incomplete:sequence=1",
    ):
        evidence = _evidence_manifest(material, ["updated"])
        material["ledger"].close_payload(
            completed_at_unix_ns=(BASE_SECOND + 181) * 1_000_000_000,
            policy=material["policy"],
            evidence_manifest=evidence,
            external_evidence_verification_receipt=(_verification_receipt(evidence)),
        )


@pytest.mark.parametrize("policy_state_replay", (False, True))
def test_production_finalizer_closes_unstarted_tail_with_external_verifier(
    material: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    policy_state_replay: bool,
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    terminal = _finish_updated(
        material,
        sequence=0,
        policy_after=_sha("production-finalizer-policy-after"),
    )

    replay_contract = (
        _policy_state_replay_contract(
            material["root"].parent / "finalizer-replay-contract",
            initial_policy=material["initial_policy"],
        )
        if policy_state_replay
        else None
    )
    objective_receipt = {"schema": "fixture.objective.v1"}
    objective_sha256 = hashlib.sha256(canonical_json_bytes(objective_receipt)).hexdigest()
    state_source_sha256 = _sha("finalizer-state-source")
    transaction_stage_sha256 = _sha("finalizer-stage")

    class Broker:
        identity = "fixture-evidence-verifier"
        calls = 0
        verify_calls = 0
        replay_calls = 0

        @classmethod
        def replay_policy_states(cls, **kwargs):
            cls.replay_calls += 1
            request = kwargs["request"]
            assert replay_contract is not None
            row = request["evidence_manifest"]["group_packages"][0]
            expected = {
                "provider_contract_sha256": request["evidence_manifest"]["contract_sha256"],
                "campaign_schedule_root_sha256": request["evidence_manifest"][
                    "campaign_schedule_root_sha256"
                ],
                "campaign_manifest_sha256": material["manifest"]["manifest_sha256"],
                "sequence": 0,
                "group_manifest_sha256": row["group_manifest_sha256"],
                "group_admission_sha256": row["group_admission_sha256"],
                "update_receipt_sha256": row["update_receipt_sha256"],
                "pre_measurement_sha256": row["pre_measurement_sha256"],
                "state_source_sha256": state_source_sha256,
                "post_state_transaction_stage_sha256": (transaction_stage_sha256),
                "objective_receipt_sha256": objective_sha256,
                "policy_before_sha256": material["initial_policy"],
                "policy_after_sha256": terminal["policy_after_sha256"],
            }
            result = _external_policy_state_replay_result(
                replay_contract,
                expected,
                verifier_identity=cls.identity,
                verified_at_unix=request["verified_at_unix"],
            )
            result_root = hashlib.sha256(
                canonical_json_bytes(
                    [
                        {
                            "sequence": 0,
                            "receipt_sha256": result["receipt_sha256"],
                        }
                    ]
                )
            ).hexdigest()
            body = {
                "schema": ("aura.verified_transition.external_policy_state_replay_batch.v1"),
                "request_sha256": request["request_sha256"],
                "policy_state_replay_contract_sha256": (replay_contract["contract_sha256"]),
                "evidence_manifest_sha256": request["evidence_manifest"]["manifest_sha256"],
                "verifier_identity": cls.identity,
                "verified_at_unix": request["verified_at_unix"],
                "transition_results": [result],
                "transition_result_root_sha256": result_root,
                "completed_at_unix": request["verified_at_unix"],
            }
            return {
                **body,
                "result_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
            }

        @classmethod
        def verify_evidence_manifest(cls, _policy, **kwargs):
            cls.verify_calls += 1
            return _verification_receipt(
                kwargs["evidence_manifest"],
                verifier_identity=cls.identity,
                verified_at_unix=kwargs["verified_at_unix"],
            )

        @classmethod
        def attest(cls, policy, **kwargs):
            cls.calls += 1
            return build_role_attestation(
                policy,
                role=kwargs["role"],
                payload=kwargs["payload"],
                signed_at_unix=kwargs["signed_at_unix"],
                private_key=material["role_keys"][EVIDENCE_VERIFIER],
            )

    step = {
        "step_kind": "verified_optimizer_update",
        "campaign_sequence": 0,
        "group_manifest_sha256": terminal["group_manifest_sha256"],
        "reward_receipt_sha256": terminal["reward_receipt_sha256"],
        "group_admission_sha256": terminal["group_admission_sha256"],
        "update_receipt_sha256": terminal["update_receipt_sha256"],
        "receipt_sha256": _sha("trainer-step-0"),
    }
    package = {
        "sequence": 0,
        "contract_sha256": _sha("provider-contract"),
        "campaign_schedule_root_sha256": material["schedule_root"],
        "group_manifest": {"manifest_sha256": terminal["group_manifest_sha256"]},
        "reward_receipt_sha256": terminal["reward_receipt_sha256"],
        "group_admission_sha256": terminal["group_admission_sha256"],
        "receipt_sha256": _sha("package-0"),
        "sample_receipt_sha256s": [_sha("sample-0")],
        "evidence_receipt_sha256s": [_sha("evidence-0")],
    }
    monkeypatch.setattr(
        "core.learning.verified_recurrent_transition_repository._read_package",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(
        "core.learning.verified_recurrent_transition_repository._package_artifact_binding",
        lambda *_args, **_kwargs: {
            "path": "/private/replay/group-00000000.json",
            "sha256": _sha("package-bytes-0"),
            "size_bytes": 128,
        },
    )
    monkeypatch.setattr(
        "core.learning.verified_recurrent_transition_repository."
        "VerifiedTransitionTransactionStore.open",
        lambda *_args, **_kwargs: SimpleNamespace(
            load=lambda **_load_kwargs: SimpleNamespace(
                pending_step=(
                    {"pre_measurement_sha256": _sha("finalizer-pre-measurement")}
                    if policy_state_replay
                    else {}
                ),
                stage={"receipt_sha256": transaction_stage_sha256},
            )
        ),
    )
    if policy_state_replay:
        monkeypatch.setattr(
            "core.learning.verified_recurrent_transition_repository."
            "load_pre_measurement_for_transaction",
            lambda *_args, **_kwargs: {
                "state_source": {"state_source_sha256": state_source_sha256}
            },
        )
        monkeypatch.setattr(
            "core.learning.verified_recurrent_transition_repository."
            "VerifiedTransitionUpdateJournal.open",
            lambda *_args, **_kwargs: SimpleNamespace(
                read=lambda *_read_args, **_read_kwargs: {"objective_receipt": objective_receipt}
            ),
        )
    replay_group = SimpleNamespace(
        sequence=0,
        reward_receipt={"receipt_sha256": terminal["reward_receipt_sha256"]},
        group_admission_receipt={"receipt_sha256": terminal["group_admission_sha256"]},
        update_receipt={"receipt_sha256": terminal["update_receipt_sha256"]},
    )
    request = type(
        "Request",
        (),
        {
            "schema": "aura.verified_transition.finalize_request.v2",
            "contract_sha256": _sha("provider-contract"),
            "campaign_schedule_root_sha256": material["schedule_root"],
            "completed_groups": 1,
            "halt_reason": "max_steps",
            "step_receipts": (step,),
            "replay_artifact_root": str(material["ledger"].root.parent / "replay"),
            "campaign_ledger_root": str(material["ledger"].root),
            "transition_artifact_root": str(
                material["ledger"].root.parent / "transition-artifacts"
            ),
            "update_journal_root": str(material["ledger"].root.parent / "updates"),
            "transaction_root": str(material["ledger"].root.parent / "transactions"),
            "replay_groups": (replay_group,),
            "campaign_ledger": material["ledger"],
            "campaign_trust_policy": material["policy"],
            "evidence_verifier_signer": Broker(),
            "policy_state_replay_contract": replay_contract,
        },
    )()
    closure = finalize_verified_recurrent_transition_campaign(request)
    recovered = finalize_verified_recurrent_transition_campaign(request)
    assert (
        recovered.campaign_ledger is closure.campaign_ledger
        and Broker.calls == 1
        and Broker.verify_calls == 1
        and Broker.replay_calls == (1 if policy_state_replay else 0)
    )
    closed = closure.campaign_ledger.validate_closed(policy=material["policy"])
    assert closed["close_payload"]["group_statuses"] == ["updated", "aborted"]
    assert closed["close_payload"]["evidence_manifest"]["schema"] == (
        CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5
        if policy_state_replay
        else CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA
    )


def test_close_requires_external_evidence_verifier_signature(
    material: dict[str, Any],
) -> None:
    _policy_1, final_policy = _complete(material)
    completed_second = BASE_SECOND + 181
    evidence = _evidence_manifest(material, ["updated", "updated"])
    payload = material["ledger"].close_payload(
        completed_at_unix_ns=completed_second * 1_000_000_000,
        policy=material["policy"],
        evidence_manifest=evidence,
        external_evidence_verification_receipt=(_verification_receipt(evidence)),
    )
    wrong_role = build_role_attestation(
        material["policy"],
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][TASK_ISSUER],
    )
    with pytest.raises(ValueError, match="campaign_attestation_identity_mismatch"):
        material["ledger"].close(
            close_payload=payload,
            evidence_verifier_attestation=wrong_role,
            policy=material["policy"],
        )

    verifier = build_role_attestation(
        material["policy"],
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][EVIDENCE_VERIFIER],
    )
    receipt = material["ledger"].close(
        close_payload=payload,
        evidence_verifier_attestation=verifier,
        policy=material["policy"],
    )
    assert receipt["close_payload"]["final_policy_sha256"] == final_policy
    assert material["ledger"].validate_closed(policy=material["policy"]) == receipt
    start, terminal = material["ledger"].group_records(sequence=1, policy=material["policy"])
    assert start["receipt_sha256"] == payload["group_start_sha256s"][1]
    assert terminal["receipt_sha256"] == payload["group_terminal_sha256s"][1]


def test_append_validation_work_is_constant_in_campaign_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _make_material(tmp_path, group_count=32)
    policy_before = material["initial_policy"]
    for sequence in range(31):
        _admit(material, sequence=sequence, policy_before=policy_before)
        policy_after = _sha(f"scale-policy-{sequence}")
        _finish_updated(material, sequence=sequence, policy_after=policy_after)
        policy_before = policy_after

    observed_reads = 0
    original_read = VerifiedTransitionCausalCampaignLedger._read

    def counted_read(self: VerifiedTransitionCausalCampaignLedger, name: str) -> dict[str, Any]:
        nonlocal observed_reads
        if self is material["ledger"]:
            observed_reads += 1
        return original_read(self, name)

    monkeypatch.setattr(VerifiedTransitionCausalCampaignLedger, "_read", counted_read)
    _admit(material, sequence=31, policy_before=policy_before)
    assert observed_reads <= 8


def test_create_once_start_survives_two_ledger_instances_racing(
    material: dict[str, Any],
) -> None:
    group = _group(
        material,
        sequence=0,
        policy_before=material["initial_policy"],
    )
    ledgers = (
        material["ledger"],
        VerifiedTransitionCausalCampaignLedger.open(material["root"], policy=material["policy"]),
    )

    def admit(ledger: VerifiedTransitionCausalCampaignLedger) -> str:
        try:
            ledger.admit_group_plan(
                sequence=0,
                campaign_id=material["manifest"]["campaign_id"],
                campaign_schedule_root_sha256=material["schedule_root"],
                policy_before_sha256=material["initial_policy"],
                group_manifest=group["manifest"],
                group_manifest_attestation=group["manifest_attestation"],
                lineage_plan=group["lineage"],
                lineage_attestation=group["lineage_attestation"],
                policy=material["policy"],
                admitted_at_unix_ns=(group["planned_second"] + 1) * 1_000_000_000,
            )
        except VerifiedTransitionCausalCampaignError:
            return "rejected"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(admit, ledgers))
    assert sorted(outcomes) == ["created", "rejected"]
    assert (material["root"] / "group-00000000.started.json").is_file()
