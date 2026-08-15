"""Independent replay contracts for the bounded recurrent decode evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.organism.model_validation import (  # noqa: E402
    _recurrent_memory_decode_certificate_holds,
    _resident_recurrent_memory_decode_certificate_holds,
)
from tools.verify_mathematics_memory_decode_canary import (
    CanaryVerificationError,
    _sha,
    verify_canary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp529_mathematics_memory_decode_canary.json"
)
RESIDENT_ARTIFACT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/"
    "cp533_resident_32b_mathematics_memory_decode_canary.json"
)


def _local_model(name: str) -> Path:
    checkout_model = REPO_ROOT / "models" / name
    if checkout_model.exists():
        return checkout_model
    return Path.home() / ".aura" / "live-source" / "models" / name


# Prefer checkout-local model fixtures; Aura's live-source store is the local
# fallback for worktrees, which do not replicate ignored model directories.
MODEL = _local_model("Qwen2.5-1.5B-Instruct-4bit")
RESIDENT_MODEL = _local_model("Qwen2.5-32B-Instruct-4bit")


@pytest.mark.skipif(not MODEL.exists(), reason="local frozen 1.5B unavailable")
def test_independent_verifier_reconstructs_the_checked_in_certificate() -> None:
    certificate = verify_canary(ARTIFACT, model_path=MODEL)

    assert certificate["independently_verified"] is True
    assert certificate["measurement_count"] == 240
    assert certificate["treatment_exact"] == 30
    assert certificate["ordinary_base_exact"] == 0
    assert certificate["matched_wire_base_exact"] == 0
    assert set(certificate["causal_control_exacts"].values()) == {0}


@pytest.mark.skipif(not MODEL.exists(), reason="local frozen 1.5B unavailable")
def test_independent_verifier_rejects_resigned_raw_output_mutation(
    tmp_path: Path,
) -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["raw_outputs"][0]["response"] += "mutated"
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = _sha(body)
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CanaryVerificationError, match="row_evidence_mismatch"):
        verify_canary(mutated, model_path=MODEL)


def test_model_validation_claim_is_bound_to_the_verified_certificate() -> None:
    assert _recurrent_memory_decode_certificate_holds() is True


@pytest.mark.skipif(
    not RESIDENT_MODEL.exists(), reason="local frozen resident 32B unavailable"
)
def test_independent_verifier_reconstructs_resident_certificate() -> None:
    certificate = verify_canary(RESIDENT_ARTIFACT, model_path=RESIDENT_MODEL)

    assert certificate["independently_verified"] is True
    assert certificate["measurement_count"] == 240
    assert certificate["treatment_exact"] == 30
    assert certificate["ordinary_base_exact"] == 0
    assert certificate["matched_wire_base_exact"] == 0
    assert set(certificate["causal_control_exacts"].values()) == {0}
    assert certificate["model_config_sha256"] == (
        "c027829d800805358d67ac87819a3754fd8240be973f7147840651310fd30ae3"
    )


def test_resident_model_validation_claim_is_bound_to_verified_certificate() -> None:
    assert _resident_recurrent_memory_decode_certificate_holds() is True
