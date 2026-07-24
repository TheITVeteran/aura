from __future__ import annotations

import copy
import json

import pytest

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.atomic_decomposition import (
    MAX_ATOM_CHARS,
    atom_ids,
    build_atomic_decomposition,
    decomposition_check,
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier


def _recommit(value: dict) -> dict:
    from core.brain.llm.latent_cortex.atomic_decomposition import _canonical_sha256

    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = _canonical_sha256(payload)
    return value


def test_atomic_decomposition_covers_source_and_binds_dependencies_without_text() -> None:
    candidate = (
        "The cache is stale because the source version changed. "
        "Therefore the reader must invalidate it before reuse."
    )
    receipt = build_atomic_decomposition(
        candidate,
        objective="Explain why cache invalidation is required.",
    )

    assert receipt["grade_admissible"] is True
    assert receipt["coverage"]["coverage_ratio"] == 1.0
    assert receipt["dependencies"]["omitted_dependency_atom_ids"] == []
    assert {row["kind"] for row in receipt["transitions"]} == {
        "supports",
        "derives",
    }
    assert atom_ids(receipt) == ("a000", "a001", "a002")
    wire = json.dumps(receipt, sort_keys=True)
    assert candidate not in wire
    assert "source version changed" not in wire
    assert (
        validate_atomic_decomposition(
            receipt,
            candidate=candidate,
            objective="Explain why cache invalidation is required.",
        )
        == receipt
    )


def test_leading_dependency_cue_is_detected_as_omitted() -> None:
    candidate = "Therefore the deployment is safe."
    receipt = build_atomic_decomposition(candidate)

    assert receipt["grade_admissible"] is False
    assert receipt["dependencies"]["omitted_dependency_atom_ids"] == ["a000"]
    check = decomposition_check(candidate)
    assert check["valid"] is False
    assert check["score"] == 0.0
    assert check["failures"] == ["omitted_dependency:a000"]


def test_validator_rejects_tampered_span_even_with_outer_recommit() -> None:
    candidate = "A bounded request succeeds. Therefore the receipt is complete."
    receipt = build_atomic_decomposition(candidate)
    tampered = copy.deepcopy(receipt)
    tampered["atoms"][0]["end"] -= 1
    atom_payload = {
        key: value for key, value in tampered["atoms"][0].items() if key != "atom_sha256"
    }
    from core.brain.llm.latent_cortex.atomic_decomposition import _canonical_sha256

    tampered["atoms"][0]["atom_sha256"] = _canonical_sha256(atom_payload)
    _recommit(tampered)

    with pytest.raises(ValueError, match="source reconstruction|span|envelope"):
        validate_atomic_decomposition(tampered, candidate=candidate)


def test_validator_preserves_honest_omission_but_denies_grading_authority() -> None:
    candidate = "The checksum differs. Therefore the artifact must be rejected."
    receipt = build_atomic_decomposition(candidate)
    tampered = copy.deepcopy(receipt)
    tampered["transitions"] = []
    tampered["dependencies"] = {
        "cue_count": 1,
        "linked_cue_count": 0,
        "omitted_dependency_atom_ids": ["a001"],
    }
    tampered["grade_admissible"] = False
    _recommit(tampered)

    validated = validate_atomic_decomposition(tampered, candidate=candidate)
    assert validated["grade_admissible"] is False
    assert validated["dependencies"]["omitted_dependency_atom_ids"] == ["a001"]


def test_validator_rejects_dependency_cycle() -> None:
    candidate = "A is true. Therefore B is true."
    receipt = build_atomic_decomposition(candidate)
    tampered = copy.deepcopy(receipt)
    original = tampered["transitions"][0]
    from core.brain.llm.latent_cortex.atomic_decomposition import _canonical_sha256

    payload = {
        "transition_id": "t999.support",
        "kind": "supports",
        "premise_ids": [original["output_id"]],
        "output_id": original["premise_ids"][0],
        "cue": "support",
    }
    tampered["transitions"].append({**payload, "transition_sha256": _canonical_sha256(payload)})
    _recommit(tampered)

    with pytest.raises(ValueError, match="invalid claims|cycle"):
        validate_atomic_decomposition(tampered, candidate=candidate)


def test_code_fence_is_one_content_addressed_atom() -> None:
    candidate = "Use this implementation:\n```python\nprint(2 + 2)\n```\nIt terminates."
    receipt = build_atomic_decomposition(candidate)

    code_atoms = [row for row in receipt["atoms"] if row["kind"] == "code"]
    assert len(code_atoms) == 1
    assert receipt["coverage"]["coverage_ratio"] == 1.0
    assert receipt["grade_admissible"] is True


def test_large_code_fence_is_chunked_without_losing_code_kind_or_coverage() -> None:
    body = "\n".join(f"value_{index} = {index}" for index in range(100))
    candidate = f"```python\n{body}\n```"
    receipt = build_atomic_decomposition(candidate)

    assert len(receipt["atoms"]) > 1
    assert {row["kind"] for row in receipt["atoms"]} == {"code"}
    assert max(row["chars"] for row in receipt["atoms"]) <= MAX_ATOM_CHARS
    assert receipt["coverage"]["coverage_ratio"] == 1.0
    assert receipt["grade_admissible"] is True


def test_oversized_unpunctuated_candidate_is_split_without_gaps() -> None:
    candidate = " ".join(["bounded"] * 180)
    receipt = build_atomic_decomposition(candidate)

    assert len(receipt["atoms"]) >= 3
    assert max(row["chars"] for row in receipt["atoms"]) <= MAX_ATOM_CHARS
    assert receipt["coverage"]["coverage_ratio"] == 1.0
    assert receipt["grade_admissible"] is True


def test_task_verifier_receipts_atomic_graph_before_holistic_scores() -> None:
    verifier = EpisodeTaskVerifier("Explain why 2 + 2 = 4")
    valid = verifier.evaluate("2 + 2 = 4 because exact addition yields four.")
    invalid = verifier.evaluate("A control byte is invalid.\x00")

    assert valid["grade_admissible"] is True
    assert valid["checks"]["atomic_decomposition"]["valid"] is True
    assert invalid["grade_admissible"] is False
    assert invalid["score"] <= 0.25
    receipt = verifier.to_receipt()
    assert receipt["atomic_decomposition"]["schema"] == "aura.rlc.atomic_decomposition.v1"
    assert receipt["grade_admissible"] is True


def test_text_free_envelope_validator_rejects_false_grading_authority() -> None:
    receipt = build_atomic_decomposition("The receipt is bounded.")
    assert validate_atomic_decomposition_envelope(receipt) == receipt
    tampered = copy.deepcopy(receipt)
    tampered["grade_admissible"] = False
    _recommit(tampered)
    with pytest.raises(ValueError, match="grading authority"):
        validate_atomic_decomposition_envelope(tampered)


def test_service_independently_rejects_tampered_atomic_verifier_envelope() -> None:
    verifier = EpisodeTaskVerifier("Explain why the cache must be invalidated")
    verifier.evaluate("The version changed. Therefore the cache must be invalidated.")
    guidance = verifier.to_receipt()
    receipt = {"verifier_guidance": guidance}

    assert "atomic_decomposition_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, {})
    )
    tampered = copy.deepcopy(receipt)
    tampered["verifier_guidance"]["atomic_decomposition"]["atoms"][0]["text_sha256"] = "0" * 64
    assert "atomic_decomposition_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, {})
    )


def test_empty_candidate_cannot_acquire_grading_authority() -> None:
    check = decomposition_check("", objective="answer")
    assert check["applicable"] is False
    assert check["valid"] is False
    assert check["score"] is None
    assert check["receipt"]["atoms"] == []
    assert check["receipt"]["grade_admissible"] is False
