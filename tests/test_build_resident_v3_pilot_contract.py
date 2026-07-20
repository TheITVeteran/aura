from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
from tools import build_resident_v3_pilot_contract as builder
from tools import verify_resident_pilot_preflight as preflight


def _sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[int]]:
    campaign = tmp_path / "pilot"
    campaign.mkdir()
    mechanics_material: dict[str, object] = {
        "schema": builder.MECHANICS_SCHEMA,
        "passed": True,
        "ready_for_fresh_hidden_task_pilot": True,
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "identity": {
            "adapter_sha256": "a" * 64,
            "adapter_identity_sha256": "b" * 64,
            "base_checkpoint_sha256": "c" * 64,
            "adapter_freeze_certificate_sha256": "d" * 64,
            "content_root_sha256": "e" * 64,
        },
    }
    mechanics = {
        **mechanics_material,
        "verdict_sha256": _sha(mechanics_material),
    }
    mechanics_path = tmp_path / "mechanics.json"
    mechanics_path.write_bytes(canonical_json_bytes(mechanics) + b"\n")
    tasks = [
        {
            "task_id": f"task-{index}",
            "domain": domain,
            "task_payload_sha256": f"{index + 1:064x}",
        }
        for index, domain in enumerate(preflight.DOMAINS * 2)
    ]
    cells = [
        {
            "arm": arm,
            "domain": task["domain"],
            "task_id": task["task_id"],
            "task_payload_sha256": task["task_payload_sha256"],
        }
        for task in tasks
        for arm in preflight.ARMS
    ]
    execution = {
        "profile": "primary",
        "task_registry_version": preflight.CURRENT_REGISTRY_VERSION,
        "difficulty": 2,
        "domains": preflight.DOMAINS,
        "n_slots": 4,
        "branches": 2,
        "rlc_steps": 4,
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
        "effective_rlc_config": {"allow_vanilla_fallback": False},
    }
    metadata = {
        "arms": preflight.ARMS,
        "claim_eligible": False,
        "task_manifest": {
            "registry_version": preflight.CURRENT_REGISTRY_VERSION,
            "task_count": 14,
            "domains": preflight.DOMAINS,
            "manifest_sha256": "f" * 64,
        },
        "execution_config": execution,
        "model_identity": {
            "model_path": str(tmp_path / "model"),
            "fingerprint": "c" * 64,
            "model_behavior_bundle": {"bundle_sha256": "1" * 64},
            "runtime_bundle": {"logical_parameter_count": 32_763_876_352},
        },
        "adapter_identity": {
            "adapter_dir": str(tmp_path / "frozen"),
            "identity_receipt": {
                "adapter_id": "resident-v3",
                "adapter_sha256": "a" * 64,
                "composite_identity_sha256": "b" * 64,
            },
        },
    }
    plan = CampaignPlan.build("resident-v3-pilot", cells, metadata=metadata)
    (campaign / "plan.json").write_bytes(canonical_json_bytes(plan.to_dict()) + b"\n")
    return mechanics_path, campaign, [2**62 + 101, 2**62 + 303]


def test_builds_v2_contract_from_exact_preoutput_plan(tmp_path: Path) -> None:
    mechanics, campaign, seeds = _fixture(tmp_path)

    contract = builder.build_contract(
        mechanics_path=mechanics,
        campaign_dir=campaign,
        seeds=seeds,
        contract_id="cp190-resident-v3-pilot",
        created_at="2026-07-20T10:00:00Z",
        source_commit="1" * 40,
        personality_adapter="none",
    )

    assert contract["schema"] == preflight.SCHEMA_V2
    assert contract["campaign"]["decode_max_tokens"] == 768
    assert contract["campaign"]["n_slots"] == 4
    assert contract["campaign"]["seeds"] == seeds
    assert contract["decision"]["advance_only_if"] == list(builder.EXPECTED_RULES)


def test_contract_builder_rejects_any_observed_model_output(tmp_path: Path) -> None:
    mechanics, campaign, seeds = _fixture(tmp_path)
    (campaign / "runner.log").write_text("observed\n", encoding="utf-8")

    with pytest.raises(
        builder.ResidentV3PilotContractError,
        match="pilot_model_output_already_observed",
    ):
        builder.build_contract(
            mechanics_path=mechanics,
            campaign_dir=campaign,
            seeds=seeds,
            contract_id="cp190-resident-v3-pilot",
            created_at="2026-07-20T10:00:00Z",
            source_commit="1" * 40,
            personality_adapter="none",
        )


def test_contract_builder_rejects_non_63_bit_seeds(tmp_path: Path) -> None:
    mechanics, campaign, _seeds = _fixture(tmp_path)

    with pytest.raises(
        builder.ResidentV3PilotContractError,
        match="pilot_seed_contract_invalid",
    ):
        builder.build_contract(
            mechanics_path=mechanics,
            campaign_dir=campaign,
            seeds=[7, 11],
            contract_id="cp190-resident-v3-pilot",
            created_at="2026-07-20T10:00:00Z",
            source_commit="1" * 40,
            personality_adapter="none",
        )
