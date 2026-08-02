#!/usr/bin/env python3
"""Build the resident-v3 directional-pilot contract before model output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
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
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (  # noqa: E402
    IDENTITY_RECEIPT_SCHEMA as RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
)
from tools.verify_recurrence_v2_smoke import _atomic_create_or_verify  # noqa: E402
from tools.verify_resident_pilot_preflight import (  # noqa: E402
    ARMS,
    DOMAINS,
    SCHEMA_V2,
    SCHEMA_V3,
)
from tools.verify_resident_pilot_result import EXPECTED_RULES  # noqa: E402

MECHANICS_SCHEMA = "aura.latent_cortex.resident_recurrence_mechanics.v1"
TERMINAL_ARTIFACTS = {
    "campaign.jsonl",
    "campaign_manifest.json",
    "detached_receipt.json",
    "grade.json",
    "runner.log",
}


class ResidentV3PilotContractError(RuntimeError):
    """Stable fail-closed pilot-contract construction error."""


def _fail(reason: str) -> Never:
    raise ResidentV3PilotContractError(reason)


def _sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_mechanics(path: Path) -> dict[str, Any]:
    mechanics = read_canonical_json(path, role="mechanics_verdict")
    material = dict(mechanics)
    claimed = material.pop("verdict_sha256", None)
    if (
        mechanics.get("schema") != MECHANICS_SCHEMA
        or claimed != _sha(material)
        or mechanics.get("passed") is not True
        or mechanics.get("ready_for_fresh_hidden_task_pilot") is not True
        or mechanics.get("reasoning_gain_proven") is not False
        or mechanics.get("frontier_gain_proven") is not False
    ):
        _fail("mechanics_gate_invalid")
    return mechanics


def _seed_values(values: Sequence[int]) -> list[int]:
    seeds = list(values)
    if (
        len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(seed) is not int or seed.bit_length() != 63 for seed in seeds)
    ):
        _fail("pilot_seed_contract_invalid")
    return seeds


def _resident_sft_execution_contract(
    adapter_receipt: Mapping[str, Any], execution: Mapping[str, Any]
) -> dict[str, Any] | None:
    if adapter_receipt.get("schema") != RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA:
        return None
    raw_spec = execution.get("adapter_execution_spec")
    effective = execution.get("effective_rlc_config")
    if not isinstance(raw_spec, Mapping) or not isinstance(effective, Mapping):
        _fail("pilot_resident_sft_execution_spec_missing")
    try:
        spec = RLCExecutionSpec.from_dict(raw_spec)
    except (TypeError, ValueError) as exc:
        raise ResidentV3PilotContractError("pilot_resident_sft_execution_spec_invalid") from exc
    workspace = effective.get("workspace")
    recurrence = effective.get("recurrence")
    branches = effective.get("branches")
    latent_opt = effective.get("latent_opt")
    fast_weights = effective.get("fast_weights")
    if not all(
        isinstance(value, Mapping)
        for value in (workspace, recurrence, branches, latent_opt, fast_weights)
    ):
        _fail("pilot_resident_sft_effective_config_invalid")
    expected_bridge = (
        "assistant_answer_v3" if spec.decode_bridge_policy == "assistant_answer" else "none"
    )
    if (
        adapter_receipt.get("execution_spec_sha256") != spec.sha256
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
        _fail("pilot_resident_sft_execution_binding_invalid")
    return {
        "n_slots": spec.n_slots,
        "branches": len(spec.branch_roles),
        "rlc_steps": spec.recurrent_steps,
        "execution_spec_sha256": spec.sha256,
    }


def build_contract(
    *,
    mechanics_path: Path,
    campaign_dir: Path,
    seeds: Sequence[int],
    contract_id: str,
    created_at: str,
    source_commit: str,
    personality_adapter: str,
) -> dict[str, Any]:
    mechanics_path = mechanics_path.expanduser().resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    mechanics = _verified_mechanics(mechanics_path)
    observed = sorted(name for name in TERMINAL_ARTIFACTS if (campaign_dir / name).exists())
    if observed:
        _fail(f"pilot_model_output_already_observed:{observed[0]}")
    plan_path = campaign_dir / "plan.json"
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
    model_behavior = model.get("model_behavior_bundle")
    runtime = model.get("runtime_bundle")
    adapter_receipt = adapter.get("identity_receipt")
    mechanics_identity = mechanics.get("identity")
    if not all(
        isinstance(value, Mapping)
        for value in (model_behavior, runtime, adapter_receipt, mechanics_identity)
    ):
        _fail("pilot_plan_identity_invalid")
    seed_list = _seed_values(seeds)
    resident_execution = _resident_sft_execution_contract(adapter_receipt, execution)
    contract_schema = SCHEMA_V3 if resident_execution is not None else SCHEMA_V2
    requested = {
        "profile": "primary",
        "task_registry_version": task_manifest.get("registry_version"),
        "difficulty": 2,
        "domains": DOMAINS,
        "n_slots": resident_execution["n_slots"] if resident_execution else 4,
        "branches": resident_execution["branches"] if resident_execution else 2,
        "rlc_steps": resident_execution["rlc_steps"] if resident_execution else 4,
        "rlc_profile": "recurrence_attribution",
        "decode_max_tokens": 768,
        "episode_timeout_s": 1200.0,
        "load_timeout_s": 1200.0,
        "warmup_timeout_s": 600.0,
        "arm_timeout_s": 10800.0,
        "campaign_timeout_s": 43200.0,
        "equal_compute_max_samples": 8,
        "max_infra_attempts": 3,
        "generation_seed_count": 2,
        "generation_seed_min_entropy_bits": 63,
    }
    plan_requested = {key: value for key, value in requested.items() if key != "max_infra_attempts"}
    # The campaign plan binds the number of pre-authorized, detached worker
    # slots. The CLI calls the same limit max_infra_attempts; translate the
    # public launch term to the persisted plan schema instead of requiring a
    # field the runner never emits.
    plan_requested["worker_origin_attempt_slots"] = requested["max_infra_attempts"]
    if (
        not contract_id
        or not created_at
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or personality_adapter != "none"
        or document.get("campaign_name") is None
        or metadata.get("arms") != ARMS
        or metadata.get("claim_eligible") is not False
        or task_manifest.get("task_count") != 14
        or sorted(task_manifest.get("domains", [])) != sorted(DOMAINS)
        or any(execution.get(key) != value for key, value in plan_requested.items())
        or execution.get("effective_rlc_config", {}).get("allow_vanilla_fallback") is not False
        or len(plan.cell_ids) != 56
        or adapter.get("adapter_dir") is None
        or adapter_receipt.get("adapter_id") is None
        or adapter_receipt.get("adapter_sha256") is None
        or adapter_receipt.get("composite_identity_sha256") is None
        or mechanics_identity.get("adapter_sha256") != adapter_receipt.get("adapter_sha256")
        or mechanics_identity.get("adapter_identity_sha256")
        != adapter_receipt.get("composite_identity_sha256")
        or mechanics_identity.get("base_checkpoint_sha256") != model.get("fingerprint")
    ):
        _fail("pilot_plan_binding_invalid")
    campaign = {
        "name": document["campaign_name"],
        "directory": str(campaign_dir),
        "task_registry_version": requested["task_registry_version"],
        "seeds": seed_list,
        "seed_entropy_bits": 63,
        "domains": DOMAINS,
        "arms": ARMS,
        "profile": requested["profile"],
        "difficulty": requested["difficulty"],
        "n_slots": requested["n_slots"],
        "branches": requested["branches"],
        "rlc_steps": requested["rlc_steps"],
        "rlc_profile": requested["rlc_profile"],
        "decode_max_tokens": requested["decode_max_tokens"],
        "episode_timeout_s": requested["episode_timeout_s"],
        "load_timeout_s": requested["load_timeout_s"],
        "warmup_timeout_s": requested["warmup_timeout_s"],
        "arm_timeout_s": requested["arm_timeout_s"],
        "campaign_timeout_s": requested["campaign_timeout_s"],
        "equal_compute_max_samples": requested["equal_compute_max_samples"],
        "max_infra_attempts": requested["max_infra_attempts"],
        "task_count": 14,
        "cell_count": 56,
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": _file_sha(plan_path),
        "task_manifest_sha256": task_manifest["manifest_sha256"],
        "claim_eligible": False,
        "vanilla_fallback_allowed": False,
    }
    if resident_execution is not None:
        campaign["adapter_execution_spec_sha256"] = resident_execution["execution_spec_sha256"]
    adapter_contract = {
        "path": adapter["adapter_dir"],
        "adapter_id": adapter_receipt["adapter_id"],
        "adapter_sha256": adapter_receipt["adapter_sha256"],
        "identity_sha256": adapter_receipt["composite_identity_sha256"],
        "freeze_certificate_sha256": mechanics_identity["adapter_freeze_certificate_sha256"],
        "content_root_sha256": mechanics_identity["content_root_sha256"],
    }
    if resident_execution is not None:
        adapter_contract.update(
            {
                "identity_receipt_schema": RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
                "execution_spec_sha256": resident_execution["execution_spec_sha256"],
            }
        )
    material = {
        "schema": contract_schema,
        "contract_id": contract_id,
        "created_at": created_at,
        "source_commit": source_commit,
        "claim_scope": "internal_directional_falsification_only",
        "preregistered_before_model_output": True,
        "mechanics_gate": {
            "verdict_sha256": mechanics["verdict_sha256"],
            "file_sha256": _file_sha(mechanics_path),
            "ready_for_fresh_hidden_task_pilot": True,
        },
        "model": {
            "path": model["model_path"],
            "base_checkpoint_sha256": model["fingerprint"],
            "model_behavior_bundle_sha256": model_behavior["bundle_sha256"],
            "logical_parameter_count": runtime["logical_parameter_count"],
            "personality_adapter": personality_adapter,
        },
        "adapter": adapter_contract,
        "campaign": campaign,
        "decision": {
            "advance_only_if": list(EXPECTED_RULES),
            "advance_target": "powered_external_frontier_campaign",
            "nonadvance_action": "diagnose_and_preregister_new_training_or_pilot_revision",
            "post_hoc_task_selection_allowed": False,
            "pilot_can_prove_frontier_gain": False,
        },
        "external_attestation_present": False,
        "reasoning_gain_proven_before_run": False,
        "frontier_gain_proven_before_run": False,
    }
    return {**material, "contract_sha256": _sha(material)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics-verdict", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--personality-adapter", default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        seeds = [int(value) for value in args.seeds.split(",") if value]
        contract = build_contract(
            mechanics_path=args.mechanics_verdict,
            campaign_dir=args.campaign_dir,
            seeds=seeds,
            contract_id=args.contract_id,
            created_at=args.created_at,
            source_commit=args.source_commit,
            personality_adapter=args.personality_adapter,
        )
        _atomic_create_or_verify(
            args.output.expanduser().resolve(strict=False),
            canonical_json_bytes(contract) + b"\n",
        )
    except (OSError, ValueError, ResidentV3PilotContractError) as exc:
        print(
            f"build_resident_v3_pilot_contract: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
