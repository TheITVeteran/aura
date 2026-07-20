from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import CampaignPlan
from tools import verify_resident_recurrence_mechanics as verifier


def _sha(value: dict[str, object]) -> str:
    from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _evidence(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], CampaignPlan]:
    frozen = tmp_path.resolve()
    receipt = {
        "adapter_id": "resident-test",
        "adapter_sha256": "a" * 64,
        "base_checkpoint_fingerprint": "b" * 64,
        "composite_identity_sha256": "c" * 64,
    }
    model = {
        "fingerprint": "b" * 64,
        "files": 4,
        "model_behavior_bundle_sha256": "d" * 64,
        "runtime_bundle_sha256": "e" * 64,
        "runtime_environment_identity_sha256": "f" * 64,
        "personality_adapter_bundle_sha256": "",
        "effective_stack_sha256": "1" * 64,
    }
    freeze = {
        "adapter_id": "resident-test",
        "certificate_sha256": "2" * 64,
        "content_root_sha256": "3" * 64,
        "identity_receipt": receipt,
        "model_identity": model,
        "validator_identity": {
            "campaign_runner_sha256": "4" * 64,
            "freeze_contract_sha256": "5" * 64,
            "identity_validator_sha256": "6" * 64,
        },
    }
    promotion_material: dict[str, object] = {
        "schema": verifier.PROMOTION_SCHEMA,
        "claim_scope": "terminal_training_and_immutable_adapter_identity_only",
        "training_complete": True,
        "immutable_freeze_verified": True,
        "ready_for_mechanics_smoke": True,
        "pilot_eligible": False,
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "external_attestation_present": False,
        "external_trust_required_before_claim_campaign": True,
        "required_next_gate": "resident_32b_frozen_adapter_mechanics_smoke",
        "freeze_certificate_sha256": "2" * 64,
        "training": {
            "adapter_id": "resident-test",
            "adapter_sha256": "a" * 64,
            "base_checkpoint_sha256": "b" * 64,
            "content_root_sha256": "3" * 64,
            "frozen_adapter": str(frozen),
        },
    }
    promotion = {**promotion_material, "promotion_sha256": _sha(promotion_material)}
    metadata = {
        "claim_eligible": False,
        "arms": verifier.EXPECTED_ARMS,
        "adapter_identity": {
            "adapter_dir": str(frozen),
            "identity_receipt": receipt,
        },
        "model_identity": {
            "fingerprint": "b" * 64,
            "files": 4,
            "model_behavior_bundle": {"bundle_sha256": "d" * 64},
            "runtime_bundle": {
                "bundle_sha256": "e" * 64,
                "model_type": "qwen2",
                "logical_parameter_count_basis": "architecture_config_logical",
                "logical_parameter_count": 32_763_876_352,
            },
            "runtime_environment": {"identity_sha256": "f" * 64},
            "personality_adapter": {"bundle_sha256": ""},
            "effective_stack_sha256": "1" * 64,
        },
        "execution_config": {
            "effective_rlc_config": {"allow_vanilla_fallback": False},
            "adapter_execution_spec": {"adapter_scope": "latent_slots_only"},
        },
    }
    cells = [{"domain": "mathematics", "seed": 7, "task_sha256": "7" * 64}]
    plan = CampaignPlan.build("resident-mechanics", cells, metadata=metadata)
    return promotion, freeze, plan


def test_bindings_accept_exact_resident_generation(tmp_path: Path) -> None:
    promotion, freeze, plan = _evidence(tmp_path)

    identity = verifier._verify_bindings(
        promotion=promotion,
        freeze=freeze,
        plan=plan,
        frozen_adapter=tmp_path.resolve(),
    )

    assert identity["logical_parameter_count"] == 32_763_876_352
    assert identity["adapter_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        "promotion_freeze",
        "adapter_receipt",
        "model_identity",
        "adapter_path",
        "fallback",
        "claim_eligible",
    ],
)
def test_bindings_reject_substitution(tmp_path: Path, mutation: str) -> None:
    promotion, freeze, plan = _evidence(tmp_path)
    metadata = plan.to_dict()["metadata"]
    if mutation == "promotion_freeze":
        promotion["freeze_certificate_sha256"] = "9" * 64
    elif mutation == "adapter_receipt":
        metadata["adapter_identity"]["identity_receipt"]["adapter_sha256"] = "9" * 64
    elif mutation == "model_identity":
        metadata["model_identity"]["fingerprint"] = "9" * 64
    elif mutation == "adapter_path":
        metadata["adapter_identity"]["adapter_dir"] = str(tmp_path / "other")
    elif mutation == "fallback":
        metadata["execution_config"]["effective_rlc_config"][
            "allow_vanilla_fallback"
        ] = True
    else:
        metadata["claim_eligible"] = True
    tampered = CampaignPlan.build(
        "resident-mechanics",
        [{"domain": "mathematics", "seed": 7, "task_sha256": "7" * 64}],
        metadata=metadata,
    )

    with pytest.raises(
        verifier.ResidentMechanicsVerificationError,
        match="resident_mechanics_binding_mismatch",
    ):
        verifier._verify_bindings(
            promotion=promotion,
            freeze=freeze,
            plan=tampered,
            frozen_adapter=tmp_path.resolve(),
        )


def test_promotion_rejects_rehashed_capability_claim(tmp_path: Path) -> None:
    promotion, _, _ = _evidence(tmp_path)
    promotion["reasoning_gain_proven"] = True
    material = dict(promotion)
    material.pop("promotion_sha256")
    promotion["promotion_sha256"] = _sha(material)
    path = tmp_path / "promotion.json"
    from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

    path.write_bytes(canonical_json_bytes(promotion) + b"\n")

    with pytest.raises(
        verifier.ResidentMechanicsVerificationError,
        match="training_promotion_invalid",
    ):
        verifier._verified_promotion(path)


def test_verify_composes_mechanics_without_gain_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, freeze, plan = _evidence(tmp_path / "frozen")
    promotion_path = tmp_path / "promotion.json"
    frozen = tmp_path / "frozen"
    campaign = tmp_path / "campaign"
    frozen.mkdir()
    campaign.mkdir()
    promotion_path.touch()

    def read_document(path: Path, *, role: str) -> dict[str, object]:
        del role
        return promotion if path == promotion_path else plan.to_dict()

    monkeypatch.setattr(verifier, "read_canonical_json", read_document)
    monkeypatch.setattr(verifier, "verify_adapter_freeze", lambda path: freeze)
    monkeypatch.setattr(
        verifier,
        "_validator_identity",
        lambda: {
            "campaign_runner_sha256": "4" * 64,
            "freeze_contract_sha256": "5" * 64,
            "identity_validator_sha256": "6" * 64,
            "mechanics_verifier_sha256": "7" * 64,
            "smoke_replay_verifier_sha256": "8" * 64,
        },
    )
    monkeypatch.setattr(
        verifier,
        "_verify_campaign",
        lambda path: {
            "activation_totals": {"calls": 4},
            "causal_logit_digest_changes": 1,
            "committed_cells": 4,
            "ordinary_generation_exact_match": True,
            "task_count": 1,
        },
    )

    verdict = verifier.verify(
        promotion_path=promotion_path,
        frozen_adapter=frozen,
        campaign_dir=campaign,
    )

    assert verdict["passed"] is True
    assert verdict["ready_for_fresh_hidden_task_pilot"] is True
    assert verdict["reasoning_gain_proven"] is False
    assert verdict["frontier_gain_proven"] is False
