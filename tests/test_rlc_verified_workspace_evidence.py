from __future__ import annotations

import hashlib

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.verified_best import tensor_sha256  # noqa: E402
from core.brain.llm.latent_cortex.verified_workspace_evidence import (  # noqa: E402
    build_workspace_evidence_receipt,
    deterministic_semantic_sham,
    replace_workspace_slots,
    validate_workspace_evidence_receipt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_semantic_sham_preserves_shape_and_row_norm_but_destroys_coordinates():
    state = mx.arange(2 * 3 * 8).reshape(2, 3, 8).astype(mx.float32) + 1.0
    sham = deterministic_semantic_sham(state, salt="verified-evidence")

    assert sham.shape == state.shape
    assert not bool(mx.array_equal(sham, state))
    assert bool(
        mx.allclose(
            mx.linalg.norm(sham, axis=-1),
            mx.linalg.norm(state, axis=-1),
            rtol=1e-6,
            atol=1e-6,
        )
    )
    assert bool(
        mx.array_equal(
            sham,
            deterministic_semantic_sham(state, salt="verified-evidence"),
        )
    )


def test_workspace_replacement_changes_only_declared_slots():
    state = mx.arange(1 * 5 * 4).reshape(1, 5, 4).astype(mx.float32)
    evidence = mx.full((1, 2, 4), 99.0)
    changed = replace_workspace_slots(state, evidence, slot_indices=(2, 4))

    assert bool(mx.array_equal(changed[:, 0:2, :], state[:, 0:2, :]))
    assert bool(mx.array_equal(changed[:, 3:4, :], state[:, 3:4, :]))
    assert bool(mx.array_equal(changed[:, 2:3, :], evidence[:, 0:1, :]))
    assert bool(mx.array_equal(changed[:, 4:5, :], evidence[:, 1:2, :]))


def test_workspace_evidence_receipt_requires_treatment_to_beat_baseline_and_sham():
    kwargs = {
        "objective_sha256": _sha("objective"),
        "teaching_event_sha256": _sha("event"),
        "private_witness_sha256": _sha("witness"),
        "private_witness_token_count": 24,
        "target_slots": (3, 4),
        "assimilation_steps": 2,
        "source_state_sha256": _sha("source"),
        "treatment_seed_sha256": _sha("treatment-seed"),
        "sham_seed_sha256": _sha("sham-seed"),
        "treatment_state_sha256": _sha("treatment-state"),
        "sham_state_sha256": _sha("sham-state"),
        "baseline_score": 0.25,
        "treatment_score": 1.0,
        "sham_score": 0.25,
        "treatment_tokens_sha256": _sha("treatment-tokens"),
        "sham_tokens_sha256": _sha("sham-tokens"),
    }
    receipt = build_workspace_evidence_receipt(**kwargs)
    assert receipt["accepted"] is True
    assert receipt["answer_authority"] == "neural_decode_only"
    assert receipt["producer_answer_promoted"] is False
    validate_workspace_evidence_receipt(receipt)

    rejected = build_workspace_evidence_receipt(**{**kwargs, "treatment_score": 0.25})
    assert rejected["accepted"] is False
    assert rejected["disposition"] == "rejected_non_improvement"


def test_workspace_evidence_receipt_tampering_fails_closed():
    receipt = build_workspace_evidence_receipt(
        objective_sha256=_sha("objective"),
        teaching_event_sha256=_sha("event"),
        private_witness_sha256=_sha("witness"),
        private_witness_token_count=24,
        target_slots=(3,),
        assimilation_steps=1,
        source_state_sha256=_sha("source"),
        treatment_seed_sha256=_sha("treatment-seed"),
        sham_seed_sha256=_sha("sham-seed"),
        treatment_state_sha256=_sha("treatment-state"),
        sham_state_sha256=_sha("sham-state"),
        baseline_score=0.0,
        treatment_score=1.0,
        sham_score=0.0,
        treatment_tokens_sha256=_sha("treatment-tokens"),
        sham_tokens_sha256=_sha("sham-tokens"),
    )
    receipt["producer_answer_promoted"] = True
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_workspace_evidence_receipt(receipt)


def test_workspace_replacement_rejects_duplicate_or_misaligned_slots():
    state = mx.zeros((1, 4, 8))
    evidence = mx.zeros((1, 2, 8))
    with pytest.raises(ValueError, match="replacement is invalid"):
        replace_workspace_slots(state, evidence, slot_indices=(2, 2))
    with pytest.raises(ValueError, match="replacement is invalid"):
        replace_workspace_slots(state, evidence[:, :1, :], slot_indices=(1, 2))


def test_receipt_commits_private_values_without_containing_them():
    private = 'Start with 11 modulo 17. FINAL_ANSWER: {"residue":16}'
    receipt = build_workspace_evidence_receipt(
        objective_sha256=_sha("objective"),
        teaching_event_sha256=_sha("event"),
        private_witness_sha256=hashlib.sha256(private.encode()).hexdigest(),
        private_witness_token_count=24,
        target_slots=(2,),
        assimilation_steps=1,
        source_state_sha256=_sha("source"),
        treatment_seed_sha256=_sha("treatment-seed"),
        sham_seed_sha256=_sha("sham-seed"),
        treatment_state_sha256=_sha("treatment-state"),
        sham_state_sha256=_sha("sham-state"),
        baseline_score=0.0,
        treatment_score=1.0,
        sham_score=0.0,
        treatment_tokens_sha256=_sha("treatment-tokens"),
        sham_tokens_sha256=_sha("sham-tokens"),
    )
    assert private not in repr(receipt)
    assert "residue" not in repr(receipt)
    assert tensor_sha256(mx.ones((1, 1, 2))) not in repr(private)
