from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import verify_resident_pilot_preflight as verifier

CONTRACT = (
    Path(__file__).resolve().parents[1] / "config/latent_cortex/resident_32b_pilot_contract.json"
)


def _rehash(document: dict[str, object]) -> None:
    material = dict(document)
    material.pop("contract_sha256", None)
    document["contract_sha256"] = hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def test_checked_in_pilot_contract_is_self_hashed_and_directional() -> None:
    contract = verifier._verified_contract(CONTRACT)
    assert contract["campaign"]["task_count"] == 14
    assert contract["campaign"]["cell_count"] == 56
    assert contract["decision"]["pilot_can_prove_frontier_gain"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("campaign", "seeds", [1, 2]),
        ("campaign", "claim_eligible", True),
        ("campaign", "vanilla_fallback_allowed", True),
        ("decision", "post_hoc_task_selection_allowed", True),
        ("decision", "pilot_can_prove_frontier_gain", True),
    ],
)
def test_rehashed_contract_cannot_weaken_protocol(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document[section][key] = value
    _rehash(document)
    path = tmp_path / "contract.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(verifier.PilotPreflightError, match="pilot_contract_invalid"):
        verifier._verified_contract(path)


def test_contract_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
    with pytest.raises(verifier.PilotPreflightError, match="duplicate_json_key"):
        verifier._verified_contract(path)


def test_rehashed_v3_contract_cannot_substitute_execution_spec_digest(
    tmp_path: Path,
) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["schema"] = verifier.SCHEMA_V3
    document["campaign"].update(
        {
            "profile": "primary",
            "difficulty": 2,
            "task_registry_version": verifier.CURRENT_REGISTRY_VERSION,
            "n_slots": 16,
            "branches": 2,
            "rlc_steps": 2,
            "rlc_profile": "recurrence_attribution",
            "decode_max_tokens": 768,
            "max_infra_attempts": 3,
            "adapter_execution_spec_sha256": "a" * 64,
        }
    )
    document["adapter"].update(
        {
            "identity_receipt_schema": verifier.RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
            "execution_spec_sha256": "b" * 64,
        }
    )
    _rehash(document)
    path = tmp_path / "contract.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(
        verifier.PilotPreflightError,
        match="pilot_v3_execution_contract_invalid",
    ):
        verifier._verified_contract(path)
