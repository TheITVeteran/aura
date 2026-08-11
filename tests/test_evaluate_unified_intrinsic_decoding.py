from __future__ import annotations

import pytest

from tools.evaluate_unified_intrinsic_decoding import _candidate_response


def test_candidate_response_reconstructs_exact_answer_envelope() -> None:
    assert (
        _candidate_response("\n\nFINAL_ANSWER: ", '{"residue":7}')
        == 'FINAL_ANSWER: {"residue":7}'
    )
    with pytest.raises(ValueError, match="exact answer bridge"):
        _candidate_response("Answer: ", "7")
