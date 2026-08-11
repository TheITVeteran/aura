from __future__ import annotations

import pytest

from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract


def _contract() -> RecurrentAnswerEmissionContract:
    return RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=99,
        family_markers=(
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_trace", (3,)),
        ),
        syntax=(
            ("khop", (30,)),
            ("modular", (31,)),
            ("register_head", (32,)),
            ("register_mid_r1", (33,)),
            ("register_mid_r2", (34,)),
            ("close", (35,)),
        ),
    )


def test_terminal_public_state_compiles_to_canonical_answer_tokens() -> None:
    contract = _contract()
    assert contract.expected_tokens((1,), (4, 23, 0, 0, 1)) == (
        30,
        12,
        13,
        35,
        99,
    )
    assert contract.expected_tokens((2,), (4, 7, 0, 0, 1)) == (31, 17, 35, 99)
    assert contract.expected_tokens((3,), (4, 12, 3, 29, 1)) == (
        32,
        11,
        12,
        33,
        13,
        34,
        12,
        19,
        35,
        99,
    )


def test_answer_emission_refuses_incomplete_or_conflicting_public_state() -> None:
    contract = _contract()
    assert contract.expected_tokens((1,), (2, 7, 0, 0, 0)) is None
    assert contract.expected_tokens((8,), (2, 7, 0, 0, 1)) is None
    with pytest.raises(ValueError, match="conflicting"):
        contract.expected_tokens((1, 2), (2, 7, 0, 0, 1))


def test_answer_template_exposes_width_and_syntax_but_not_digit_identity() -> None:
    contract = _contract()

    assert contract.emission_template((3,), (4, 12, 3, 29, 1)) == (
        32,
        None,
        None,
        33,
        None,
        34,
        None,
        None,
        35,
        99,
    )
    assert contract.next_template_token((3,), (4, 12, 3, 29, 1), ()) == 32
    assert contract.next_template_token((3,), (4, 12, 3, 29, 1), (32,)) is None
    assert contract.next_template_token(
        (3,),
        (4, 12, 3, 29, 1),
        (32, 11, 12),
    ) == 33


def test_answer_template_rejects_invalid_history_instead_of_recovering() -> None:
    contract = _contract()

    with pytest.raises(ValueError, match="diverged"):
        contract.next_template_token((2,), (4, 7, 0, 0, 1), (30,))
    with pytest.raises(ValueError, match="digit slot"):
        contract.next_template_token((2,), (4, 7, 0, 0, 1), (31, 35))
