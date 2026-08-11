"""Contracts for tokenizer-bound numeric observations."""

from __future__ import annotations

import pytest

from core.learning.recurrent_literal_grounding import (
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [100 + int(text)] if len(text) == 1 and text.isdigit() else []

    def decode(
        self,
        token_ids: list[int],
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return str(token_ids[0] - 100)


def test_digit_contract_round_trips_and_reconstructs_multitoken_values() -> None:
    digit_ids = tokenizer_digit_token_ids(_Tokenizer())
    contract = LiteralObservationContract(digit_ids)
    values, masks = contract.observe(
        ((50, 101, 102, 51, 102, 109, 52, 103, 103),)
    )
    assert digit_ids == tuple(range(100, 110))
    assert values[0][2] == 12
    assert values[0][5] == 29
    assert values[0][8] == 0
    assert masks[0] == (
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert len(contract.contract_sha256) == 64


def test_literal_contract_rejects_ambiguous_tokenizers_and_noncanonical_spans() -> None:
    class Ambiguous(_Tokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [100]

    with pytest.raises(ValueError, match="round-trip|one-to-one"):
        tokenizer_digit_token_ids(Ambiguous())

    contract = LiteralObservationContract(tuple(range(100, 110)))
    values, masks = contract.observe(((100, 101, 50, 103, 103),))
    assert values == ((0, 0, 0, 0, 0),)
    assert masks == ((False, False, False, False, False),)
