"""Independent replay contracts for the bounded recurrent decode evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.organism.model_validation import (  # noqa: E402
    _recurrent_memory_decode_certificate_holds,
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
# Anchored to the checkout, not to one machine's home directory: the
# absolute literal made the skipif true on every host but this one, so a
# verifier contract could pass by never running. Matches the convention in
# tools/front_door_demo.py.
MODEL = REPO_ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"


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
