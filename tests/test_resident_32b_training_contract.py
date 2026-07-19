from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/latent_cortex/resident_32b_training_contract.json"
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate contract key: {key}")
        result[key] = value
    return result


def _contract() -> dict:
    return json.loads(
        CONTRACT_PATH.read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite contract value: {value}")
        ),
    )


def test_resident_32b_contract_is_self_hashed_and_internally_scoped() -> None:
    contract = _contract()
    material = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }

    assert contract["contract_sha256"] == hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    assert contract["claim_scope"] == "internal_mechanics_acceptance_only"
    assert contract["external_attestation_present"] is False
    assert "reasoning_and_frontier_gain_not_measured" in contract["evidence_limitations"]
    assert "powered_external_frontier_campaign" in contract["required_next_gates"]


def test_resident_32b_contract_fixes_full_factorial_workload() -> None:
    workload = _contract()["workload"]
    expected_examples = (
        len(workload["families"])
        * len(workload["task_depths"])
        * workload["per_cell"]
    )

    assert len(workload["families"]) == 12
    assert workload["task_depths"] == [2, 4, 8]
    assert expected_examples == 576
    assert workload["max_steps"] == expected_examples
    assert workload["log_every"] == 5
    assert workload["checkpoint_every"] == 5


def test_resident_32b_contract_binds_plan_model_sources_and_artifacts() -> None:
    contract = _contract()

    assert len(contract["accepted_plan"]["plan_sha256"]) == 64
    assert len(contract["accepted_plan"]["target_execution_manifest_sha256"]) == 64
    assert len(contract["model_identity"]["base_checkpoint_fingerprint"]) == 64
    assert len(contract["producer_sources"]) == 9
    assert all(
        len(record["sha256"]) == 64 and record["size_bytes"] > 0
        for record in contract["accepted_artifacts"].values()
    )
