#!/usr/bin/env python3
"""Verify the resident RLC pilot was fully preregistered before inference."""

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
)
from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CURRENT_REGISTRY_VERSION,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (  # noqa: E402
    IDENTITY_RECEIPT_SCHEMA as RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
)

SCHEMA = "aura.latent_cortex.resident_pilot_contract.v1"
SCHEMA_V2 = "aura.latent_cortex.resident_pilot_contract.v2"
SCHEMA_V3 = "aura.latent_cortex.resident_pilot_contract.v3"
PREFLIGHT_SCHEMA = "aura.latent_cortex.resident_pilot_preflight.v1"
DOMAINS = [
    "novel_algorithms",
    "mathematics",
    "coding",
    "scientific_inference",
    "long_horizon_planning",
    "calibration",
    "misleading_premise",
]
ARMS = ["base_vanilla", "base_rlc", "adapter_vanilla", "adapter_rlc"]
TERMINAL_ARTIFACTS = {
    "campaign.jsonl",
    "campaign_manifest.json",
    "detached_receipt.json",
    "grade.json",
    "runner.log",
}


class PilotPreflightError(RuntimeError):
    """Stable fail-closed pilot preflight error."""


def _fail(reason: str) -> Never:
    raise PilotPreflightError(reason)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    _fail(f"nonfinite_json_value:{value}")


def _read_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail("pilot_contract_storage_invalid")
    before = path.stat()
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PilotPreflightError("pilot_contract_json_invalid") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        _fail("pilot_contract_changed_while_reading")
    if not isinstance(value, dict):
        _fail("pilot_contract_not_object")
    return value


def _sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_v3_execution_binding(
    contract: Mapping[str, Any],
    execution: Mapping[str, Any],
    adapter_receipt: Mapping[str, Any],
) -> None:
    campaign = contract["campaign"]
    adapter = contract["adapter"]
    raw_spec = execution.get("adapter_execution_spec")
    effective = execution.get("effective_rlc_config")
    if not isinstance(raw_spec, Mapping) or not isinstance(effective, Mapping):
        _fail("pilot_v3_execution_spec_missing")
    try:
        spec = RLCExecutionSpec.from_dict(raw_spec)
    except (TypeError, ValueError) as exc:
        raise PilotPreflightError("pilot_v3_execution_spec_invalid") from exc
    workspace = effective.get("workspace")
    recurrence = effective.get("recurrence")
    branches = effective.get("branches")
    latent_opt = effective.get("latent_opt")
    fast_weights = effective.get("fast_weights")
    if not all(
        isinstance(value, Mapping)
        for value in (workspace, recurrence, branches, latent_opt, fast_weights)
    ):
        _fail("pilot_v3_effective_config_invalid")
    expected_bridge = (
        "assistant_answer_v3" if spec.decode_bridge_policy == "assistant_answer" else "none"
    )
    if (
        adapter_receipt.get("schema") != RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA
        or adapter_receipt.get("execution_spec_sha256") != spec.sha256
        or adapter.get("identity_receipt_schema") != RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA
        or adapter.get("execution_spec_sha256") != spec.sha256
        or campaign.get("adapter_execution_spec_sha256") != spec.sha256
        or campaign.get("n_slots") != spec.n_slots
        or campaign.get("branches") != len(spec.branch_roles)
        or campaign.get("rlc_steps") != spec.recurrent_steps
        or execution.get("n_slots") != spec.n_slots
        or execution.get("branches") != len(spec.branch_roles)
        or execution.get("rlc_steps") != spec.recurrent_steps
        or workspace.get("n_slots") != spec.n_slots
        or workspace.get("seed") != spec.slot_seed
        or workspace.get("roles") != list(spec.slot_roles)
        or workspace.get("anchor_scale") != spec.anchor_scale
        or recurrence.get("max_steps") != spec.recurrent_steps
        or recurrence.get("min_steps") != spec.recurrent_steps
        or recurrence.get("alpha") != spec.alpha
        or recurrence.get("alpha_schedule") != spec.alpha_schedule
        or recurrence.get("rms_clip_ratio") != spec.rms_clip_ratio
        or recurrence.get("fixed_depth") is not True
        or branches.get("n_branches") != len(spec.branch_roles)
        or branches.get("exchange_interval") != spec.exchange_interval
        or branches.get("exchange_gamma") != spec.exchange_gamma
        or branches.get("comm_slot") != spec.comm_slot
        or branches.get("collapse_cos_threshold") != spec.collapse_cos_threshold
        or branches.get("jitter_scale") != spec.jitter_scale
        or branches.get("roles") != list(spec.branch_roles)
        or effective.get("prelude_frac") != spec.prelude_frac
        or effective.get("coda_frac") != spec.coda_frac
        or effective.get("decode_bridge_policy") != expected_bridge
        or effective.get("allow_vanilla_fallback") is not False
        or latent_opt.get("enabled") is not False
        or fast_weights.get("enabled") is not False
    ):
        _fail("pilot_v3_execution_binding_mismatch")


def _verified_contract(path: Path) -> dict[str, Any]:
    contract = _read_contract(path)
    claimed = contract.get("contract_sha256")
    material = dict(contract)
    material.pop("contract_sha256", None)
    campaign = contract.get("campaign")
    decision = contract.get("decision")
    mechanics = contract.get("mechanics_gate")
    adapter = contract.get("adapter")
    if not all(isinstance(value, Mapping) for value in (campaign, decision, mechanics)):
        _fail("pilot_contract_sections_invalid")
    seeds = campaign.get("seeds")
    required_rules = decision.get("advance_only_if")
    if (
        contract.get("schema") not in {SCHEMA, SCHEMA_V2, SCHEMA_V3}
        or claimed != _sha(material)
        or contract.get("claim_scope") != "internal_directional_falsification_only"
        or contract.get("preregistered_before_model_output") is not True
        or contract.get("external_attestation_present") is not False
        or contract.get("reasoning_gain_proven_before_run") is not False
        or contract.get("frontier_gain_proven_before_run") is not False
        or mechanics.get("ready_for_fresh_hidden_task_pilot") is not True
        or not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(seed) is not int or seed.bit_length() != 63 for seed in seeds)
        or campaign.get("seed_entropy_bits") != 63
        or campaign.get("domains") != DOMAINS
        or campaign.get("arms") != ARMS
        or campaign.get("task_count") != 14
        or campaign.get("cell_count") != 56
        or campaign.get("claim_eligible") is not False
        or campaign.get("vanilla_fallback_allowed") is not False
        or not isinstance(required_rules, list)
        or len(required_rules) != 9
        or decision.get("post_hoc_task_selection_allowed") is not False
        or decision.get("pilot_can_prove_frontier_gain") is not False
        or decision.get("advance_target") != "powered_external_frontier_campaign"
    ):
        _fail("pilot_contract_invalid")
    if contract.get("schema") == SCHEMA_V2 and (
        campaign.get("task_registry_version") != CURRENT_REGISTRY_VERSION
        or campaign.get("n_slots") != 4
        or campaign.get("branches") != 2
        or campaign.get("rlc_steps") != 4
        or campaign.get("rlc_profile") != "recurrence_attribution"
        or campaign.get("decode_max_tokens") != 768
        or campaign.get("max_infra_attempts") != 3
    ):
        _fail("pilot_v2_execution_contract_invalid")
    if contract.get("schema") == SCHEMA_V3 and (
        not isinstance(adapter, Mapping)
        or campaign.get("task_registry_version") != CURRENT_REGISTRY_VERSION
        or campaign.get("profile") != "primary"
        or campaign.get("difficulty") != 2
        or campaign.get("rlc_profile") != "recurrence_attribution"
        or campaign.get("decode_max_tokens") != 768
        or campaign.get("max_infra_attempts") != 3
        or type(campaign.get("n_slots")) is not int
        or not 2 <= campaign["n_slots"] <= 128
        or type(campaign.get("branches")) is not int
        or not 1 <= campaign["branches"] <= 8
        or type(campaign.get("rlc_steps")) is not int
        or not 1 <= campaign["rlc_steps"] <= 64
        or not _is_sha256(campaign.get("adapter_execution_spec_sha256"))
        or adapter.get("identity_receipt_schema") != RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA
        or adapter.get("execution_spec_sha256") != campaign.get("adapter_execution_spec_sha256")
    ):
        _fail("pilot_v3_execution_contract_invalid")
    return contract


def _verify_plan(contract: Mapping[str, Any], plan_path: Path) -> CampaignPlan:
    campaign = contract["campaign"]
    plan_file_sha = _file_sha(plan_path)
    plan = CampaignPlan.from_dict(read_canonical_json(plan_path, role="pilot_plan"))
    document = plan.to_dict()
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("pilot_plan_metadata_invalid")
    task_manifest = metadata.get("task_manifest")
    execution = metadata.get("execution_config")
    model = metadata.get("model_identity")
    adapter = metadata.get("adapter_identity")
    if not all(isinstance(value, Mapping) for value in (task_manifest, execution, model, adapter)):
        _fail("pilot_plan_sections_invalid")
    adapter_receipt = adapter.get("identity_receipt")
    model_behavior = model.get("model_behavior_bundle")
    runtime = model.get("runtime_bundle")
    if not all(isinstance(value, Mapping) for value in (adapter_receipt, model_behavior, runtime)):
        _fail("pilot_plan_identity_invalid")
    requested = {
        "profile": campaign.get("profile"),
        "task_registry_version": campaign.get("task_registry_version"),
        "difficulty": campaign.get("difficulty"),
        "domains": campaign.get("domains"),
        "n_slots": campaign.get("n_slots"),
        "branches": campaign.get("branches"),
        "rlc_steps": campaign.get("rlc_steps"),
        "rlc_profile": campaign.get("rlc_profile"),
        "decode_max_tokens": campaign.get("decode_max_tokens"),
        "episode_timeout_s": campaign.get("episode_timeout_s"),
        "load_timeout_s": campaign.get("load_timeout_s"),
        "warmup_timeout_s": campaign.get("warmup_timeout_s"),
        "arm_timeout_s": campaign.get("arm_timeout_s"),
        "campaign_timeout_s": campaign.get("campaign_timeout_s"),
        "equal_compute_max_samples": campaign.get("equal_compute_max_samples"),
        "worker_origin_attempt_slots": campaign.get("max_infra_attempts"),
        "generation_seed_count": 2,
        "generation_seed_min_entropy_bits": 63,
    }
    if (
        plan.plan_sha256 != campaign.get("plan_sha256")
        or plan_file_sha != campaign.get("plan_file_sha256")
        or len(plan.cell_ids) != campaign.get("cell_count")
        or metadata.get("arms") != ARMS
        or metadata.get("claim_eligible") is not False
        or task_manifest.get("manifest_sha256") != campaign.get("task_manifest_sha256")
        or task_manifest.get("task_count") != campaign.get("task_count")
        or sorted(task_manifest.get("domains", [])) != sorted(DOMAINS)
        or any(execution.get(key) != value for key, value in requested.items())
        or execution.get("effective_rlc_config", {}).get("allow_vanilla_fallback") is not False
        or model.get("model_path") != contract["model"]["path"]
        or model.get("fingerprint") != contract["model"]["base_checkpoint_sha256"]
        or model_behavior.get("bundle_sha256") != contract["model"]["model_behavior_bundle_sha256"]
        or runtime.get("logical_parameter_count") != contract["model"]["logical_parameter_count"]
        or adapter.get("adapter_dir") != contract["adapter"]["path"]
        or adapter_receipt.get("adapter_id") != contract["adapter"]["adapter_id"]
        or adapter_receipt.get("adapter_sha256") != contract["adapter"]["adapter_sha256"]
        or adapter_receipt.get("composite_identity_sha256")
        != contract["adapter"]["identity_sha256"]
    ):
        _fail("pilot_plan_binding_mismatch")
    if contract.get("schema") == SCHEMA_V3:
        _verify_v3_execution_binding(contract, execution, adapter_receipt)
    return plan


def verify_preflight(
    *, contract_path: Path, mechanics_path: Path, campaign_dir: Path
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    mechanics_path = mechanics_path.expanduser().resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    contract = _verified_contract(contract_path)
    mechanics = read_canonical_json(mechanics_path, role="mechanics_verdict")
    mechanics_material = dict(mechanics)
    claimed_mechanics = mechanics_material.pop("verdict_sha256", None)
    gate = contract["mechanics_gate"]
    if (
        _file_sha(mechanics_path) != gate.get("file_sha256")
        or claimed_mechanics != gate.get("verdict_sha256")
        or claimed_mechanics != _sha(mechanics_material)
        or mechanics.get("passed") is not True
        or mechanics.get("ready_for_fresh_hidden_task_pilot") is not True
        or mechanics.get("reasoning_gain_proven") is not False
        or mechanics.get("frontier_gain_proven") is not False
    ):
        _fail("pilot_mechanics_gate_invalid")
    if campaign_dir != Path(str(contract["campaign"]["directory"])).resolve():
        _fail("pilot_campaign_directory_mismatch")
    plan = _verify_plan(contract, campaign_dir / "plan.json")
    observed_terminal = sorted(
        name for name in TERMINAL_ARTIFACTS if (campaign_dir / name).exists()
    )
    if observed_terminal:
        _fail(f"pilot_model_output_already_observed:{observed_terminal[0]}")
    material = {
        "schema": PREFLIGHT_SCHEMA,
        "passed": True,
        "model_output_observed": False,
        "preregistered_before_model_output": True,
        "claim_scope": contract["claim_scope"],
        "contract_sha256": contract["contract_sha256"],
        "mechanics_verdict_sha256": claimed_mechanics,
        "plan_sha256": plan.plan_sha256,
        "task_manifest_sha256": contract["campaign"]["task_manifest_sha256"],
        "task_count": contract["campaign"]["task_count"],
        "cell_count": contract["campaign"]["cell_count"],
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "required_next_gate": "execute_preregistered_resident_directional_pilot",
    }
    return {**material, "preflight_sha256": _sha(material)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mechanics-verdict", required=True)
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args()
    try:
        verdict = verify_preflight(
            contract_path=Path(args.contract),
            mechanics_path=Path(args.mechanics_verdict),
            campaign_dir=Path(args.campaign_dir),
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"verify_resident_pilot_preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
