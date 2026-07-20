#!/usr/bin/env python3
"""Certify resident recurrence mechanics without making an intelligence claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    read_canonical_json,
    verify_adapter_freeze,
)
from tools.verify_recurrence_v2_smoke import (  # noqa: E402
    _atomic_create_or_verify,
    _verify_campaign,
)

SCHEMA = "aura.latent_cortex.resident_recurrence_mechanics.v1"
PROMOTION_SCHEMA = "aura.latent_cortex.recurrence_training_promotion.v1"
ADMISSION_SCHEMA = "aura.resident_v3_training_admission.v1"
RECOVERY_ADMISSION_SCHEMA = "aura.resident_v3_recovery_training_admission.v1"
ADMISSION_SCOPES = {
    ADMISSION_SCHEMA: "resident_v3_training_mechanics_admission_only",
    RECOVERY_ADMISSION_SCHEMA: "resident_v3_recovery_training_mechanics_admission_only",
}
EXPECTED_ARMS = ["base_vanilla", "base_rlc", "adapter_vanilla", "adapter_rlc"]


class ResidentMechanicsVerificationError(RuntimeError):
    """Stable fail-closed resident mechanics error."""


def _fail(reason: str) -> Never:
    raise ResidentMechanicsVerificationError(reason)


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validator_identity() -> dict[str, str]:
    paths = {
        "campaign_runner_sha256": REPO_ROOT
        / "tools/run_latent_cortex_paired_campaign.py",
        "freeze_contract_sha256": REPO_ROOT
        / "core/brain/llm/latent_cortex/campaign_launch_bundle.py",
        "identity_validator_sha256": REPO_ROOT
        / "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py",
        "mechanics_verifier_sha256": Path(__file__).resolve(),
        "smoke_replay_verifier_sha256": REPO_ROOT
        / "tools/verify_recurrence_v2_smoke.py",
    }
    return {role: _sha256_file(path) for role, path in paths.items()}


def _verified_promotion(path: Path) -> dict[str, Any]:
    promotion = read_canonical_json(path, role="training_promotion")
    claimed = promotion.get("promotion_sha256")
    material = dict(promotion)
    material.pop("promotion_sha256", None)
    if (
        promotion.get("schema") != PROMOTION_SCHEMA
        or claimed != _sha256(material)
        or promotion.get("claim_scope")
        != "terminal_training_and_immutable_adapter_identity_only"
        or promotion.get("training_complete") is not True
        or promotion.get("immutable_freeze_verified") is not True
        or promotion.get("ready_for_mechanics_smoke") is not True
        or promotion.get("pilot_eligible") is not False
        or promotion.get("reasoning_gain_proven") is not False
        or promotion.get("frontier_gain_proven") is not False
        or promotion.get("external_attestation_present") is not False
        or promotion.get("external_trust_required_before_claim_campaign") is not True
        or promotion.get("required_next_gate")
        != "resident_32b_frozen_adapter_mechanics_smoke"
    ):
        _fail("training_promotion_invalid")
    return promotion


def _verified_admission(path: Path) -> dict[str, Any]:
    admission = read_canonical_json(path, role="training_admission")
    claimed = admission.get("admission_sha256")
    material = dict(admission)
    material.pop("admission_sha256", None)
    state = admission.get("training_state")
    flags = admission.get("claim_flags")
    identity = admission.get("identity_receipt")
    if (
        admission.get("schema") not in ADMISSION_SCOPES
        or claimed != _sha256(material)
        or admission.get("decision") != "admit_to_freeze_and_mechanics"
        or admission.get("claim_scope") != ADMISSION_SCOPES.get(admission.get("schema"))
        or not isinstance(state, Mapping)
        or state.get("scope") != "complete_training"
        or state.get("complete") is not True
        or not isinstance(flags, Mapping)
        or flags.get("training_admitted") is not True
        or flags.get("adapter_freeze_eligible") is not True
        or any(
            flags.get(name) is not False
            for name in (
                "mechanics_proven",
                "reasoning_gain",
                "same_checkpoint_interaction",
                "frontier_level",
                "frontier_plus",
                "installed_desktop_gain",
            )
        )
        or not isinstance(identity, Mapping)
        or identity.get("complete") is not True
        or identity.get("load_eligible") is not True
        or identity.get("training_scope") != "complete_training"
    ):
        _fail("training_admission_invalid")
    return admission


def _verified_training_gate(path: Path) -> dict[str, Any]:
    document = read_canonical_json(path, role="training_gate")
    schema = document.get("schema")
    if schema == PROMOTION_SCHEMA:
        return _verified_promotion(path)
    if schema in ADMISSION_SCOPES:
        return _verified_admission(path)
    _fail("training_gate_schema_invalid")


def _model_identity_from_plan(model: Mapping[str, Any]) -> dict[str, Any]:
    behavior = model.get("model_behavior_bundle")
    runtime = model.get("runtime_bundle")
    environment = model.get("runtime_environment")
    personality = model.get("personality_adapter")
    if not all(
        isinstance(value, Mapping)
        for value in (behavior, runtime, environment, personality)
    ):
        _fail("campaign_model_identity_invalid")
    return {
        "fingerprint": model.get("fingerprint"),
        "files": model.get("files"),
        "model_behavior_bundle_sha256": behavior.get("bundle_sha256"),
        "runtime_bundle_sha256": runtime.get("bundle_sha256"),
        "runtime_environment_identity_sha256": environment.get("identity_sha256"),
        "personality_adapter_bundle_sha256": personality.get("bundle_sha256"),
        "effective_stack_sha256": model.get("effective_stack_sha256"),
    }


def _verify_bindings(
    *,
    promotion: Mapping[str, Any],
    freeze: Mapping[str, Any],
    plan: CampaignPlan,
    frozen_adapter: Path,
) -> dict[str, Any]:
    is_admission = promotion.get("schema") in ADMISSION_SCOPES
    training = promotion.get("identity_receipt") if is_admission else promotion.get("training")
    freeze_receipt = freeze.get("identity_receipt")
    freeze_model = freeze.get("model_identity")
    metadata = plan.to_dict().get("metadata")
    if not all(
        isinstance(value, Mapping)
        for value in (training, freeze_receipt, freeze_model, metadata)
    ):
        _fail("resident_mechanics_identity_missing")
    adapter_identity = metadata.get("adapter_identity")
    model_identity = metadata.get("model_identity")
    execution = metadata.get("execution_config")
    if not all(
        isinstance(value, Mapping)
        for value in (adapter_identity, model_identity, execution)
    ):
        _fail("campaign_identity_missing")
    plan_receipt = adapter_identity.get("identity_receipt")
    plan_adapter_dir = adapter_identity.get("adapter_dir")
    plan_arms = metadata.get("arms")
    effective = execution.get("effective_rlc_config")
    adapter_spec = execution.get("adapter_execution_spec")
    legacy_binding_invalid = not is_admission and (
        promotion.get("freeze_certificate_sha256")
        != freeze.get("certificate_sha256")
        or training.get("content_root_sha256") != freeze.get("content_root_sha256")
        or training.get("base_checkpoint_sha256") != freeze_model.get("fingerprint")
        or Path(str(training.get("frozen_adapter") or "")).expanduser().resolve(
            strict=False
        )
        != frozen_adapter
    )
    admission_binding_invalid = is_admission and (
        training != freeze_receipt
        or training.get("base_checkpoint_fingerprint")
        != freeze_model.get("fingerprint")
    )
    if (
        legacy_binding_invalid
        or admission_binding_invalid
        or training.get("adapter_id") != freeze.get("adapter_id")
        or training.get("adapter_sha256") != freeze_receipt.get("adapter_sha256")
        or plan_adapter_dir != str(frozen_adapter)
        or plan_receipt != freeze_receipt
        or _model_identity_from_plan(model_identity) != dict(freeze_model)
        or metadata.get("claim_eligible") is not False
        or plan_arms != EXPECTED_ARMS
        or not isinstance(effective, Mapping)
        or effective.get("allow_vanilla_fallback") is not False
        or not isinstance(adapter_spec, Mapping)
        or adapter_spec.get("adapter_scope") != "latent_slots_only"
    ):
        _fail("resident_mechanics_binding_mismatch")
    runtime = model_identity.get("runtime_bundle")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("model_type") != "qwen2"
        or runtime.get("logical_parameter_count_basis")
        != "architecture_config_logical"
        or type(runtime.get("logical_parameter_count")) is not int
        or int(runtime["logical_parameter_count"]) < 30_000_000_000
    ):
        _fail("resident_model_scale_invalid")
    return {
        "adapter_freeze_certificate_sha256": freeze["certificate_sha256"],
        "adapter_identity_sha256": freeze_receipt["composite_identity_sha256"],
        "adapter_sha256": freeze_receipt["adapter_sha256"],
        "base_checkpoint_sha256": freeze_model["fingerprint"],
        "content_root_sha256": freeze["content_root_sha256"],
        "logical_parameter_count": runtime["logical_parameter_count"],
        "training_gate_schema": promotion["schema"],
        "training_gate_sha256": promotion.get("admission_sha256")
        or promotion.get("promotion_sha256"),
    }


def verify(
    *,
    promotion_path: Path,
    frozen_adapter: Path,
    campaign_dir: Path,
) -> dict[str, Any]:
    promotion_path = promotion_path.expanduser().resolve(strict=True)
    frozen_adapter = frozen_adapter.expanduser().resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    promotion = _verified_training_gate(promotion_path)
    freeze = verify_adapter_freeze(frozen_adapter)
    validators = _validator_identity()
    frozen_validators = freeze.get("validator_identity")
    if not isinstance(frozen_validators, Mapping) or any(
        frozen_validators.get(role) != validators[role]
        for role in (
            "campaign_runner_sha256",
            "freeze_contract_sha256",
            "identity_validator_sha256",
        )
    ):
        _fail("freeze_validator_identity_mismatch")
    plan = CampaignPlan.from_dict(
        read_canonical_json(campaign_dir / "plan.json", role="campaign_plan")
    )
    identity = _verify_bindings(
        promotion=promotion,
        freeze=freeze,
        plan=plan,
        frozen_adapter=frozen_adapter,
    )
    campaign = _verify_campaign(campaign_dir)
    if (
        campaign.get("committed_cells") != 4 * campaign.get("task_count", 0)
        or campaign.get("causal_logit_digest_changes", 0) <= 0
        or campaign.get("ordinary_generation_exact_match") is not True
        or campaign.get("activation_totals", {}).get("calls", 0) <= 0
    ):
        _fail("resident_campaign_mechanics_invalid")
    material = {
        "schema": SCHEMA,
        "claim_scope": "resident_32b_frozen_adapter_mechanics_only",
        "passed": True,
        "training_complete": True,
        "immutable_freeze_verified": True,
        "resident_32b_loaded": True,
        "ordinary_generation_isolated": True,
        "causal_recurrence_adapter_effect_observed": True,
        "ready_for_fresh_hidden_task_pilot": True,
        "pilot_result_available": False,
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "external_attestation_present": False,
        "required_next_gate": "fresh_hidden_task_pilot",
        "validator_identity": validators,
        "identity": identity,
        "campaign": campaign,
    }
    return {**material, "verdict_sha256": _sha256(material)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", required=True)
    parser.add_argument("--frozen-adapter", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        verdict = verify(
            promotion_path=Path(args.promotion),
            frozen_adapter=Path(args.frozen_adapter),
            campaign_dir=Path(args.campaign_dir),
        )
        _atomic_create_or_verify(
            Path(args.output).expanduser().resolve(strict=False),
            canonical_json_bytes(verdict) + b"\n",
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            f"verify_resident_recurrence_mechanics: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
