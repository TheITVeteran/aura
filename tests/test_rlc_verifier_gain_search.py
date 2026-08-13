from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.verifier_gain_search import (
    VERIFIER_GAIN_GRID,
    build_verifier_gain_search_receipt,
    validate_verifier_gain_search_receipt,
)


def _rows(
    arm: str,
    winning_index: int,
    *,
    unsafe_indices: frozenset[int] = frozenset(),
) -> list[dict]:
    return [
        {
            "arm": arm,
            "index": index,
            "gain": gain,
            "probe_tokens_sha256": hashlib.sha256(
                f"{arm}:{index}".encode()
            ).hexdigest(),
            "probe_token_count": 16,
            "score": 1.0 if index == winning_index else 0.25,
            "layer_apps": 512,
            "delta_finite": True,
            "max_effective_delta_rms": 0.06 if index in unsafe_indices else 0.01,
            "structurally_admissible": index not in unsafe_indices,
        }
        for index, gain in enumerate(VERIFIER_GAIN_GRID)
    ]


def test_gain_search_selects_each_arm_independently_and_reconstructs():
    receipt = build_verifier_gain_search_receipt(
        treatment_rows=_rows("treatment", 4),
        sham_rows=_rows("sham", 1),
        baseline_score=0.25,
        threshold_effective_delta_rms=0.05,
    )
    assert receipt["selected_treatment_gain"] == VERIFIER_GAIN_GRID[4]
    assert receipt["selected_sham_gain"] == VERIFIER_GAIN_GRID[1]
    assert receipt["compute_matched"] is True
    validate_verifier_gain_search_receipt(receipt)


def test_gain_search_rejects_rehashed_unequal_compute():
    receipt = build_verifier_gain_search_receipt(
        treatment_rows=_rows("treatment", 0),
        sham_rows=_rows("sham", 0),
        baseline_score=0.25,
        threshold_effective_delta_rms=0.05,
    )
    tampered = copy.deepcopy(receipt)
    tampered["sham"][0]["layer_apps"] += 1
    payload = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = hashlib.sha256(
        __import__("json").dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="verdict does not reconstruct"):
        validate_verifier_gain_search_receipt(tampered)


def test_gain_search_selects_best_structurally_admissible_point():
    receipt = build_verifier_gain_search_receipt(
        treatment_rows=_rows(
            "treatment",
            len(VERIFIER_GAIN_GRID) - 1,
            unsafe_indices=frozenset({len(VERIFIER_GAIN_GRID) - 1}),
        ),
        sham_rows=_rows("sham", 0),
        baseline_score=0.25,
        threshold_effective_delta_rms=0.05,
    )

    assert receipt["treatment"][-1]["score"] == 1.0
    assert receipt["treatment"][-1]["structurally_admissible"] is False
    assert receipt["selected_treatment_gain"] == VERIFIER_GAIN_GRID[0]
    assert receipt["selected_treatment_score"] == 0.25
    validate_verifier_gain_search_receipt(receipt)
