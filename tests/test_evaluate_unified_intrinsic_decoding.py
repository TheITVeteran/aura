from __future__ import annotations

import pytest

from tools.evaluate_unified_intrinsic_decoding import (
    _candidate_response,
    evaluate_decoding,
)


def test_candidate_response_reconstructs_exact_answer_envelope() -> None:
    assert (
        _candidate_response("\n\nFINAL_ANSWER: ", '{"residue":7}')
        == 'FINAL_ANSWER: {"residue":7}'
    )
    with pytest.raises(ValueError, match="exact answer bridge"):
        _candidate_response("Answer: ", "7")


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
