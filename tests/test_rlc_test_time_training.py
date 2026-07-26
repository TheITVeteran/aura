from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
)
from core.brain.llm.latent_cortex.fast_weight_learning import (
    build_fast_weight_admission,
    token_sequence_sha256,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
from core.brain.llm.latent_cortex.test_time_training import (
    MIN_PSEUDO_LABEL_CONFIDENCE,
    build_critic_recalibration_receipt,
    build_matched_compute_receipt,
    build_pseudo_label_admission,
    deterministic_sham_target,
    validate_critic_recalibration_receipt,
    validate_matched_compute_receipt,
)


class _ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))


def _structural() -> dict[str, object]:
    return {
        "certified": True,
        "receipt_sha256": hashlib.sha256(b"structural").hexdigest(),
    }


def _arm(
    name: str,
    *,
    target: list[int],
    probe: list[int],
    score: float,
    line_searches: int = 4,
) -> dict[str, object]:
    attempts = 2
    return {
        "arm": name,
        "target_tokens_sha256": token_sequence_sha256(target),
        "optimizer": "rms_normalized_sgd_backtracking_v1",
        "attempts": attempts,
        "forward_evaluations": attempts + line_searches,
        "backward_evaluations": attempts,
        "line_search_evaluations": line_searches,
        "layer_apps": 100,
        "probe_layer_apps": 32,
        "probe_tokens_sha256": token_sequence_sha256(probe),
        "probe_token_count": len(probe),
        "score": score,
    }


def test_held_out_exact_critic_recalibrates_above_admission_bound():
    receipt = build_critic_recalibration_receipt()

    validated = validate_critic_recalibration_receipt(receipt)
    assert validated["sample_count"] == 128
    assert validated["positives"] == validated["negatives"] == 64
    assert validated["false_accept_rate"] == 0.0
    assert validated["brier"] == 0.0
    assert validated["ece"] == 0.0
    assert validated["verified_precision_lower_95"] > MIN_PSEUDO_LABEL_CONFIDENCE

    tampered = copy.deepcopy(receipt)
    tampered["verified_successes"] -= 1
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_critic_recalibration_receipt(tampered)


def test_cached_critic_receipt_is_copy_isolated():
    mutated = build_critic_recalibration_receipt()
    mutated["case_receipt_sha256s"][0] = "0" * 64
    mutated["verified_successes"] = 0

    fresh = build_critic_recalibration_receipt()

    assert fresh["case_receipt_sha256s"][0] != "0" * 64
    assert fresh["verified_successes"] == 64
    validate_critic_recalibration_receipt(fresh)


def test_only_calibrated_correctness_verifier_can_authorize_pseudo_label():
    objective = "Check the answer."
    for candidate, expected in (
        ("19 + 23 = 42.", True),
        ("```python\nvalue = 42\n```", False),
        ('{"answer": 42}', False),
    ):
        verifier = EpisodeTaskVerifier(objective)
        evaluation = verifier.evaluate(candidate)
        admission, target = build_fast_weight_admission(
            evaluation,
            candidate=candidate,
            objective=objective,
            evaluation_index=0,
            tokenizer=_ByteTokenizer(),
            structural_diversity=_structural(),
        )
        assert admission["admitted"] is expected
        assert bool(target) is expected
        if not expected:
            assert admission["reason"] == "pseudo_label_verifier_not_calibrated"


def test_pseudo_label_requires_certified_structural_diversity():
    candidate = "7 * 8 = 56."
    objective = "Check exact arithmetic."
    atomic = build_atomic_decomposition(candidate, objective=objective)
    router = build_deterministic_router_receipt(
        candidate,
        objective=objective,
        atomic_receipt=atomic,
    )
    pseudo = build_pseudo_label_admission(
        router_receipt=router,
        atomic_receipt=atomic,
        source_sha256=atomic["source_sha256"],
        structural_diversity={
            "certified": False,
            "receipt_sha256": hashlib.sha256(b"uncertified").hexdigest(),
        },
    )
    assert pseudo["admitted"] is False
    assert pseudo["reason"] == "structural_diversity_unproven"


def test_pseudo_label_cannot_reuse_a_critic_calibration_case():
    candidate = "-498 + -1 = -499."
    objective = "Check the exact bounded integer arithmetic claim."
    atomic = build_atomic_decomposition(candidate, objective=objective)
    router = build_deterministic_router_receipt(
        candidate,
        objective=objective,
        atomic_receipt=atomic,
    )
    pseudo = build_pseudo_label_admission(
        router_receipt=router,
        atomic_receipt=atomic,
        source_sha256=atomic["source_sha256"],
        structural_diversity=_structural(),
    )
    assert pseudo["admitted"] is False
    assert pseudo["query_disjoint_from_calibration"] is False
    assert pseudo["reason"] == "query_overlaps_critic_calibration"


def test_equal_compute_treatment_must_beat_baseline_and_sham():
    critic = build_critic_recalibration_receipt()
    treatment = _arm(
        "treatment",
        target=[1, 2],
        probe=[8, 9],
        score=0.82,
    )
    sham = _arm(
        "sham",
        target=[3, 4],
        probe=[6, 7],
        score=0.55,
    )
    receipt = build_matched_compute_receipt(
        treatment=treatment,
        sham=sham,
        baseline_tokens_sha256=token_sequence_sha256([5, 5]),
        baseline_score=0.50,
        critic_before=critic,
        critic_after=critic,
    )
    assert receipt["accepted"] is True
    assert receipt["compute_matched"] is True
    assert receipt["incremental_gain_over_sham"] == pytest.approx(0.27)
    validate_matched_compute_receipt(
        receipt,
        critic_before=critic,
        critic_after=critic,
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"line_search_evaluations": 3, "forward_evaluations": 5}, "matched_compute_mismatch"),
        ({"probe_tokens_sha256": token_sequence_sha256([8, 9])}, "trajectory_diversity_collapse"),
        ({"score": 0.83}, "treatment_did_not_beat_sham"),
    ],
)
def test_matched_control_rejects_compute_mismatch_collapse_and_sham_gain(
    change: dict[str, object],
    reason: str,
):
    critic = build_critic_recalibration_receipt()
    treatment = _arm(
        "treatment",
        target=[1],
        probe=[8, 9],
        score=0.82,
    )
    sham = _arm(
        "sham",
        target=[2],
        probe=[6, 7],
        score=0.55,
    )
    sham.update(change)
    receipt = build_matched_compute_receipt(
        treatment=treatment,
        sham=sham,
        baseline_tokens_sha256=token_sequence_sha256([5, 5]),
        baseline_score=0.50,
        critic_before=critic,
        critic_after=critic,
    )
    assert receipt["accepted"] is False
    assert receipt["reason"] == reason


def test_sham_target_is_deterministic_distinct_and_shape_matched():
    first = deterministic_sham_target(
        [1, 2, 3, 4],
        vocab_size=128,
        episode_id="episode-1",
    )
    second = deterministic_sham_target(
        [1, 2, 3, 4],
        vocab_size=128,
        episode_id="episode-1",
    )
    assert first == second
    assert first != [1, 2, 3, 4]
    assert len(first) == 4
