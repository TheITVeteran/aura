from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tools.evaluate_unified_intrinsic_decoding import (  # noqa: E402
    _candidate_response,
    _force_next_token,
    _paired_training_effects,
    evaluate_decoding,
)


def test_force_next_token_replaces_only_the_last_logit_row() -> None:
    logits = mx.zeros((1, 3, 8), dtype=mx.float32)
    forced = _force_next_token(logits, 5)
    assert bool(mx.array_equal(forced[:, :-1, :], logits[:, :-1, :]))
    assert int(mx.argmax(forced[0, -1]).item()) == 5


def test_candidate_response_reconstructs_exact_answer_envelope() -> None:
    assert (
        _candidate_response("\n\nFINAL_ANSWER: ", '{"residue":7}')
        == 'FINAL_ANSWER: {"residue":7}'
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
