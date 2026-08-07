#!/usr/bin/env python3
"""Materialize the exact powered-campaign handoff after a positive pilot.

This tool does not generate hidden tasks, credentials, signatures, or a launch
packet.  It binds a positive nonclaiming directional certificate to the exact
six-arm power floor and inherited execution contract that external campaign
custodians must use next.  The resulting artifact is therefore a deterministic
handoff, not authorization to run or activate an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
)
from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS  # noqa: E402
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_EQUAL_COMPUTE,
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_EQUAL_COMPUTE,
    BASE_RLC,
    BASE_VANILLA,
    exact_campaign_power_plan,
)
from tools.verify_latent_cortex_directional_gate import (  # noqa: E402
    EXPECTED_RULES,
)
from tools.verify_latent_cortex_directional_gate import (  # noqa: E402
    SCHEMA as DIRECTIONAL_GATE_SCHEMA,
)

SCHEMA = "aura.latent_cortex.powered_campaign_handoff.v1"
POWERED_ARMS = (
    BASE_VANILLA,
    BASE_RLC,
    ADAPTER_VANILLA,
    ADAPTER_RLC,
    BASE_EQUAL_COMPUTE,
    ADAPTER_EQUAL_COMPUTE,
)
PLANNED_OBSERVATIONS_PER_DOMAIN = 411
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class PoweredHandoffError(RuntimeError):
    """Stable fail-closed handoff error."""


def _fail(reason: str) -> Never:
    raise PoweredHandoffError(reason)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "role": role,
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _verified_directional_verdict(document: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(document)
    claimed = material.pop("verdict_sha256", None)
    rules = document.get("advance_rules")
    if (
        document.get("schema") != DIRECTIONAL_GATE_SCHEMA
        or claimed != _sha(material)
        or document.get("passed") is not True
        or document.get("evidence_valid") is not True
        or document.get("directional_gate_passed") is not True
        or document.get("decision") != "advance_to_powered_external_campaign"
        or document.get("required_next_gate") != "powered_external_campaign"
        or document.get("reasoning_gain_proven") is not False
        or document.get("frontier_gain_proven") is not False
        or document.get("production_activation_authorized") is not False
        or document.get("static_weight_fusion_authorized") is not False
        or not isinstance(rules, Mapping)
        or tuple(rules) != EXPECTED_RULES
        or any(rules.get(rule) is not True for rule in EXPECTED_RULES)
    ):
        _fail("positive_directional_certificate_required")
    return dict(document)


def _powered_design() -> dict[str, Any]:
    power = exact_campaign_power_plan(
        domain_count=len(FRONTIER_DOMAINS),
        comparison_count=6,
        arm_count=len(POWERED_ARMS),
        planned_observations_per_domain=PLANNED_OBSERVATIONS_PER_DOMAIN,
    )
    if (
        power.get("powered_for_zero_loss_noninferiority") is not True
        or power.get("minimum_observations") != PLANNED_OBSERVATIONS_PER_DOMAIN
        or power.get("planned_total_tasks")
        != PLANNED_OBSERVATIONS_PER_DOMAIN * len(FRONTIER_DOMAINS)
        or power.get("planned_total_cells")
        != PLANNED_OBSERVATIONS_PER_DOMAIN * len(FRONTIER_DOMAINS) * len(POWERED_ARMS)
    ):
        _fail("powered_design_contract_invalid")
    return power


def _inherited_execution_contract(execution: Mapping[str, Any]) -> dict[str, Any]:
    requested = execution.get("requested_rlc_shape")
    if not isinstance(requested, Mapping):
        _fail("directional_execution_shape_missing")
    fields = {
        "n_slots": execution.get("n_slots"),
        "branches": execution.get("branches"),
        "rlc_steps": execution.get("rlc_steps"),
        "rlc_profile": execution.get("rlc_profile"),
        "decode_max_tokens": execution.get("decode_max_tokens"),
        "difficulty": execution.get("difficulty"),
        "task_registry_version": execution.get("task_registry_version"),
        "equal_compute_max_samples": execution.get("equal_compute_max_samples"),
        "response_contract_policy": execution.get("response_contract_policy"),
        "effective_rlc_config_sha256": _sha(execution.get("effective_rlc_config")),
        "adapter_execution_spec_sha256": _sha(execution.get("adapter_execution_spec")),
    }
    if (
        type(fields["n_slots"]) is not int
        or type(fields["branches"]) is not int
        or type(fields["rlc_steps"]) is not int
        or fields["n_slots"] <= 0
        or fields["branches"] <= 0
        or fields["rlc_steps"] <= 0
        or requested.get("n_slots") != fields["n_slots"]
        or requested.get("branches") != fields["branches"]
        or requested.get("rlc_steps") != fields["rlc_steps"]
        or not isinstance(fields["rlc_profile"], str)
        or not isinstance(fields["task_registry_version"], str)
        or not isinstance(fields["response_contract_policy"], Mapping)
    ):
        _fail("directional_execution_contract_invalid")
    return fields


def build_handoff(
    *,
    directional: Mapping[str, Any],
    plan: CampaignPlan,
    directional_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    target_campaign_name: str,
) -> dict[str, Any]:
    directional = _verified_directional_verdict(directional)
    if not target_campaign_name or target_campaign_name != target_campaign_name.strip():
        _fail("target_campaign_name_invalid")
    plan_document = plan.to_dict()
    metadata = plan_document.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("directional_plan_metadata_invalid")
    if (
        directional.get("campaign_name") != plan.campaign_name
        or directional.get("plan_sha256") != plan.plan_sha256
        or directional.get("plan_file_sha256") != plan_binding.get("sha256")
        or metadata.get("claim_eligible") is not False
    ):
        _fail("directional_plan_binding_invalid")
    execution = metadata.get("execution_config")
    if not isinstance(execution, Mapping):
        _fail("directional_execution_contract_missing")
    inherited = _inherited_execution_contract(execution)
    power = _powered_design()
    material = {
        "schema": SCHEMA,
        "status": "awaiting_external_campaign_inputs",
        "source_directional_campaign": {
            "campaign_name": plan.campaign_name,
            "plan_sha256": plan.plan_sha256,
            "directional_verdict_sha256": directional["verdict_sha256"],
            "directional_verdict_artifact": dict(directional_binding),
            "plan_artifact": dict(plan_binding),
        },
        "target_campaign": {
            "campaign_name": target_campaign_name,
            "confirmatory": True,
            "profile": "full",
            "claim_eligible_only_after_external_admission": True,
            "domains": list(FRONTIER_DOMAINS),
            "arms": list(POWERED_ARMS),
            "planned_observations_per_domain": PLANNED_OBSERVATIONS_PER_DOMAIN,
            "planned_total_tasks": power["planned_total_tasks"],
            "planned_total_cells": power["planned_total_cells"],
            "exact_power": power,
            "exact_power_scope": "zero_loss_noninferiority_floor_only",
            "positive_interaction_power_simulation_required": True,
            "inherited_execution_contract": inherited,
            "model_identity_sha256": _sha(metadata.get("model_identity")),
            "adapter_identity_sha256": _sha(metadata.get("adapter_identity")),
        },
        "required_external_inputs": [
            "fresh_411_seed_manifest_with_at_least_60_bits_entropy_per_seed",
            "post_seed_hidden_task_commitment",
            "adapter_dataset_bound_zero_overlap_contamination_audit",
            "externally_signed_revisioned_campaign_policy",
            "distinct_task_issuer_and_campaign_runner_attestations",
            "post_seal_answer_reveal_attestation",
            "post_grade_runner_attestation",
            "post_evidence_independent_verifier_attestation",
            "preregistered_positive_interaction_power_simulation",
        ],
        "forbidden_claims": {
            "directional_result_proves_reasoning_gain": False,
            "directional_result_proves_frontier_gain": False,
            "directional_result_authorizes_production_activation": False,
            "directional_result_authorizes_static_weight_fusion": False,
        },
        "launch_authorized": False,
        "production_activation_authorized": False,
        "static_weight_fusion_authorized": False,
    }
    return {**material, "handoff_sha256": _sha(material)}


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            _fail("powered_handoff_output_conflict")
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("powered_handoff_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize(
    *,
    directional_verdict_path: Path,
    campaign_dir: Path,
    target_campaign_name: str,
    output: Path,
) -> dict[str, Any]:
    directional_path = directional_verdict_path.expanduser().resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    plan_path = campaign_dir / "plan.json"
    directional = read_canonical_json(directional_path, role="directional_gate")
    plan = CampaignPlan.from_dict(read_canonical_json(plan_path, role="directional_plan"))
    handoff = build_handoff(
        directional=directional,
        plan=plan,
        directional_binding=_file_binding(directional_path, role="directional_gate"),
        plan_binding=_file_binding(plan_path, role="directional_plan"),
        target_campaign_name=target_campaign_name,
    )
    _write_once(output, handoff)
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional-verdict", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--target-campaign-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = materialize(
            directional_verdict_path=args.directional_verdict,
            campaign_dir=args.campaign_dir,
            target_campaign_name=args.target_campaign_name,
            output=args.output,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            f"materialize_powered_latent_cortex_handoff: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
