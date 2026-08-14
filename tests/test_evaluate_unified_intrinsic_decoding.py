from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tools import evaluate_unified_intrinsic_decoding as decoding  # noqa: E402
from tools.evaluate_unified_intrinsic_decoding import (  # noqa: E402
    DECODE_CLAIM_BOUNDARY,
    _candidate_response,
    _decoded_arm_names,
    _force_next_token,
    _paired_training_effects,
    evaluate_decoding,
)


def test_decode_claim_boundary_defers_identity_to_campaign_binding() -> None:
    assert "Resident model identity" in DECODE_CLAIM_BOUNDARY
    assert "not a broad reasoning, frontier, fusion, or WOW result" in (DECODE_CLAIM_BOUNDARY)
    assert "not a preregistered broad reasoning, resident-32B" not in (DECODE_CLAIM_BOUNDARY)


def test_decoded_arm_identity_covers_controls_treatment_and_lesions() -> None:
    assert _decoded_arm_names((4, 8)) == (
        "base_t1",
        "untrained_t1",
        "trained_t1",
        "untrained_t4",
        "untrained_t8",
        "trained_t4",
        "trained_t8",
        "grammar_lesion_t8",
        "pointer_lesion_t8",
        "compiled_t4",
        "compiled_t8",
    )


def test_force_next_token_replaces_only_the_last_logit_row() -> None:
    logits = mx.zeros((1, 3, 8), dtype=mx.float32)
    forced = _force_next_token(logits, 5)
    assert bool(mx.array_equal(forced[:, :-1, :], logits[:, :-1, :]))
    assert int(mx.argmax(forced[0, -1]).item()) == 5


def test_candidate_response_reconstructs_exact_answer_envelope() -> None:
    assert (
        _candidate_response("\n\nFINAL_ANSWER: ", '{"residue":7}') == 'FINAL_ANSWER: {"residue":7}'
    )
    with pytest.raises(ValueError, match="exact answer bridge"):
        _candidate_response("Answer: ", "7")


def test_paired_training_effects_reports_improvements_and_regressions() -> None:
    candidates = [
        {"task_id": "a", "arm": "untrained_t4", "correct": False},
        {"task_id": "a", "arm": "trained_t4", "correct": True},
        {"task_id": "b", "arm": "untrained_t4", "correct": True},
        {"task_id": "b", "arm": "trained_t4", "correct": False},
        {"task_id": "a", "arm": "untrained_t1", "correct": False},
        {"task_id": "a", "arm": "trained_t1", "correct": True},
        {"task_id": "b", "arm": "untrained_t1", "correct": False},
        {"task_id": "b", "arm": "trained_t1", "correct": False},
    ]
    effects = _paired_training_effects(candidates, (4,))
    assert effects["1"] == {
        "tasks": 2,
        "control_arm": "untrained_t1",
        "trained_arm": "trained_t1",
        "untrained_correct": 0,
        "trained_correct": 1,
        "net_correct_gain": 1,
        "wrong_to_right": 1,
        "right_to_wrong": 0,
    }
    assert effects["4"]["net_correct_gain"] == 0
    assert effects["4"]["wrong_to_right"] == 1
    assert effects["4"]["right_to_wrong"] == 1


def test_paired_training_effects_refuses_an_incomplete_control() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        _paired_training_effects(
            [{"task_id": "a", "arm": "trained_t1", "correct": True}],
            (),
        )


def test_decode_task_depths_are_unique_positive_integers(tmp_path) -> None:
    with pytest.raises(ValueError, match="unique positive integers"):
        evaluate_decoding(
            tmp_path,
            stem="checkpoint",
            per_cell=1,
            evaluation_seed=3,
            max_tokens=8,
            task_depths=(2, 2),
        )
    with pytest.raises(ValueError, match="non-anchor campaign depths"):
        evaluate_decoding(
            tmp_path,
            stem="checkpoint",
            per_cell=1,
            evaluation_seed=3,
            max_tokens=8,
            recurrence_depths=(1,),
        )


def test_decode_forwards_explicit_bootstrap_transport(tmp_path, monkeypatch) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    observed: dict[str, object] = {}

    def stop_after_capture(*_args, **kwargs):
        observed.update(kwargs)
        raise RuntimeError("captured evaluation context")

    monkeypatch.setattr(decoding, "unified_evaluation_context", stop_after_capture)

    with pytest.raises(RuntimeError, match="captured evaluation context"):
        evaluate_decoding(
            tmp_path,
            stem="checkpoint",
            per_cell=1,
            evaluation_seed=3,
            max_tokens=8,
            bootstrap_output_dir=bootstrap,
        )

    assert observed["bootstrap_output_dir"] == bootstrap
