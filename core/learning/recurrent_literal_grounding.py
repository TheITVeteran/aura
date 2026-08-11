"""Tokenizer-bound exact numeric observations for recurrent reasoning tissue."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

LITERAL_GROUNDING_SCHEMA: Final = "aura.recurrent_literal_grounding.v1"
LITERAL_MAX_VALUE: Final = 32


def tokenizer_digit_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the ten exact single-digit token ids or fail closed.

    Reconstructing integers from a model's token stream is only sound when the
    tokenizer proves a one-token representation for every ASCII digit.  The
    resulting ids are checkpoint identity, not an ambient tokenizer lookup.
    """

    ids: list[int] = []
    for digit in range(10):
        text = str(digit)
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        if (
            not isinstance(encoded, (list, tuple))
            or len(encoded) != 1
            or type(encoded[0]) is not int
            or encoded[0] < 0
        ):
            raise ValueError(f"tokenizer has no exact single-token digit {text}")
        token_id = int(encoded[0])
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            try:
                decoded = decode([token_id], clean_up_tokenization_spaces=False)
            except TypeError:
                decoded = decode([token_id])
            if decoded != text:
                raise ValueError(f"tokenizer digit {text} does not round-trip exactly")
        ids.append(token_id)
    if len(set(ids)) != 10:
        raise ValueError("tokenizer digit ids are not one-to-one")
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class LiteralObservationContract:
    """Immutable mapping from tokenizer digits to bounded integer observations."""

    digit_token_ids: tuple[int, ...]
    max_value: int = LITERAL_MAX_VALUE
    schema: str = LITERAL_GROUNDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LITERAL_GROUNDING_SCHEMA:
            raise ValueError("literal observation schema differs")
        if (
            len(self.digit_token_ids) != 10
            or len(set(self.digit_token_ids)) != 10
            or any(type(value) is not int or value < 0 for value in self.digit_token_ids)
        ):
            raise ValueError("literal digit token identity is invalid")
        if type(self.max_value) is not int or self.max_value < 9:
            raise ValueError("literal maximum value is invalid")

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "digit_token_ids": list(self.digit_token_ids),
            "max_value": self.max_value,
        }

    def observe(
        self,
        token_rows: Sequence[Sequence[int]],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
        """Mark the terminal token of every canonical bounded integer span.

        Values are observations only.  This deliberately does not infer a
        variable role, operation, or answer from surrounding text; learned
        recurrent attention remains responsible for that semantic assignment.
        """

        token_to_digit = {
            token_id: str(digit)
            for digit, token_id in enumerate(self.digit_token_ids)
        }
        observed_rows: list[tuple[int, ...]] = []
        mask_rows: list[tuple[bool, ...]] = []
        for row in token_rows:
            if not isinstance(row, (list, tuple)) or any(
                type(token_id) is not int or token_id < 0 for token_id in row
            ):
                raise ValueError("literal observation token row is invalid")
            values = [0] * len(row)
            masks = [False] * len(row)
            digits = ""
            end = -1

            def commit(
                pending: str,
                pending_end: int,
                value_row: list[int],
                mask_row: list[bool],
            ) -> None:
                if pending and (pending == "0" or not pending.startswith("0")):
                    value = int(pending)
                    if value <= self.max_value:
                        value_row[pending_end] = value
                        mask_row[pending_end] = True

            for index, token_id in enumerate(row):
                digit = token_to_digit.get(token_id)
                if digit is None:
                    commit(digits, end, values, masks)
                    digits = ""
                    end = -1
                    continue
                digits += digit
                end = index
            commit(digits, end, values, masks)
            observed_rows.append(tuple(values))
            mask_rows.append(tuple(masks))
        return tuple(observed_rows), tuple(mask_rows)


__all__ = [
    "LITERAL_GROUNDING_SCHEMA",
    "LITERAL_MAX_VALUE",
    "LiteralObservationContract",
    "tokenizer_digit_token_ids",
]
