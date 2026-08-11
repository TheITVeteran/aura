"""Tokenizer-bound emission contract for public recurrent machine states."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrent_literal_grounding import tokenizer_digit_token_ids
from core.learning.recurrent_opcode_grounding import OpcodeObservationContract

ANSWER_EMISSION_SCHEMA: Final = "aura.recurrent_answer_emission.v1"


def _encode_exact(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if not isinstance(encoded, (list, tuple)) or not encoded or any(
        type(token_id) is not int or token_id < 0 for token_id in encoded
    ):
        raise ValueError(f"answer emission syntax {text!r} is not tokenizable")
    token_ids = tuple(int(token_id) for token_id in encoded)
    try:
        decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = tokenizer.decode(token_ids)
    if decoded != text:
        raise ValueError(f"answer emission syntax {text!r} does not round-trip")
    return token_ids


@dataclass(frozen=True, slots=True)
class RecurrentAnswerEmissionContract:
    """Compile terminal public state into the admitted canonical JSON envelope."""

    digit_token_ids: tuple[int, ...]
    eos_token_id: int
    family_markers: tuple[tuple[str, tuple[int, ...]], ...]
    syntax: tuple[tuple[str, tuple[int, ...]], ...]
    schema: str = ANSWER_EMISSION_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != ANSWER_EMISSION_SCHEMA
            or len(self.digit_token_ids) != 10
            or len(set(self.digit_token_ids)) != 10
            or type(self.eos_token_id) is not int
            or self.eos_token_id < 0
            or {name for name, _pattern in self.family_markers}
            != {"khop", "modular", "register_trace"}
            or {name for name, _pattern in self.syntax}
            != {
                "close",
                "khop",
                "modular",
                "register_head",
                "register_mid_r1",
                "register_mid_r2",
            }
        ):
            raise ValueError("recurrent answer emission contract is incomplete")
        for _name, pattern in self.family_markers + self.syntax:
            if not pattern or any(type(token_id) is not int or token_id < 0 for token_id in pattern):
                raise ValueError("recurrent answer emission pattern is invalid")

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
            "eos_token_id": self.eos_token_id,
            "family_markers": [
                {"family": family, "token_ids": list(pattern)}
                for family, pattern in self.family_markers
            ],
            "syntax": [
                {"name": name, "token_ids": list(pattern)}
                for name, pattern in self.syntax
            ],
        }

    @staticmethod
    def _contains(row: Sequence[int], pattern: tuple[int, ...]) -> bool:
        width = len(pattern)
        return any(
            tuple(row[index : index + width]) == pattern
            for index in range(len(row) - width + 1)
        )

    def family(self, public_tokens: Sequence[int]) -> str | None:
        found = [
            family
            for family, marker in self.family_markers
            if self._contains(public_tokens, marker)
        ]
        if len(found) > 1:
            raise ValueError("public prompt declares conflicting answer grammars")
        return found[0] if found else None

    def number_tokens(self, value: int) -> tuple[int, ...]:
        if type(value) is not int or not 0 <= value <= 32:
            raise ValueError("recurrent answer value is outside the admitted vocabulary")
        return tuple(self.digit_token_ids[int(digit)] for digit in str(value))

    def emission_template(
        self,
        public_tokens: Sequence[int],
        state_values: Sequence[int],
    ) -> tuple[int | None, ...] | None:
        """Return canonical syntax while withholding every answer digit.

        ``None`` positions are value-bearing slots whose token identity must
        come from the neural state reader. Fixed positions carry only the
        tokenizer-bound JSON envelope and EOS. The typed state determines the
        number of digit slots, but this template never exposes their values.
        """

        if len(state_values) != 5 or any(type(value) is not int for value in state_values):
            raise ValueError("recurrent answer state width differs")
        if state_values[-1] != 1:
            return None
        family = self.family(public_tokens)
        if family is None:
            return None
        syntax = dict(self.syntax)

        def digit_slots(value: int) -> tuple[None, ...]:
            return (None,) * len(self.number_tokens(value))

        if family == "khop":
            body = syntax["khop"] + digit_slots(state_values[1]) + syntax["close"]
        elif family == "modular":
            body = syntax["modular"] + digit_slots(state_values[1]) + syntax["close"]
        else:
            body = (
                syntax["register_head"]
                + digit_slots(state_values[1])
                + syntax["register_mid_r1"]
                + digit_slots(state_values[2])
                + syntax["register_mid_r2"]
                + digit_slots(state_values[3])
                + syntax["close"]
            )
        return body + (self.eos_token_id,)

    def next_template_token(
        self,
        public_tokens: Sequence[int],
        state_values: Sequence[int],
        generated_tokens: Sequence[int],
    ) -> int | None:
        """Validate emitted history and return the next syntax constraint.

        A ``None`` result means the next position is a neural digit, not that
        the contract was absent. Callers should first require a non-``None``
        :meth:`emission_template` for the terminal state.
        """

        template = self.emission_template(public_tokens, state_values)
        if template is None:
            raise ValueError("terminal answer grammar is unavailable")
        generated = tuple(int(token_id) for token_id in generated_tokens)
        if len(generated) >= len(template):
            raise ValueError("generated answer exceeds the terminal grammar")
        digit_ids = frozenset(self.digit_token_ids)
        for actual, expected in zip(generated, template, strict=False):
            if expected is None:
                if actual not in digit_ids:
                    raise ValueError("generated answer put syntax in a digit slot")
            elif actual != expected:
                raise ValueError("generated answer diverged from terminal syntax")
        return template[len(generated)]

    def expected_tokens(
        self,
        public_tokens: Sequence[int],
        state_values: Sequence[int],
    ) -> tuple[int, ...] | None:
        if len(state_values) != 5 or any(type(value) is not int for value in state_values):
            raise ValueError("recurrent answer state width differs")
        if state_values[-1] != 1:
            return None
        family = self.family(public_tokens)
        if family is None:
            return None
        syntax = dict(self.syntax)
        if family == "khop":
            body = syntax["khop"] + self.number_tokens(state_values[1]) + syntax["close"]
        elif family == "modular":
            body = syntax["modular"] + self.number_tokens(state_values[1]) + syntax["close"]
        else:
            body = (
                syntax["register_head"]
                + self.number_tokens(state_values[1])
                + syntax["register_mid_r1"]
                + self.number_tokens(state_values[2])
                + syntax["register_mid_r2"]
                + self.number_tokens(state_values[3])
                + syntax["close"]
            )
        return body + (self.eos_token_id,)


def tokenizer_answer_emission_contract(
    tokenizer: Any,
    opcode_contract: OpcodeObservationContract,
) -> RecurrentAnswerEmissionContract:
    contexts = dict(opcode_contract.contexts)
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("answer emission requires one exact EOS token")
    return RecurrentAnswerEmissionContract(
        digit_token_ids=tokenizer_digit_token_ids(tokenizer),
        eos_token_id=eos,
        family_markers=(
            ("khop", contexts["graph"]),
            ("modular", contexts["modular_start"]),
            ("register_trace", contexts["register"]),
        ),
        syntax=(
            ("khop", _encode_exact(tokenizer, '{"node":')),
            ("modular", _encode_exact(tokenizer, '{"residue":')),
            ("register_head", _encode_exact(tokenizer, '{"r0":')),
            ("register_mid_r1", _encode_exact(tokenizer, ',"r1":')),
            ("register_mid_r2", _encode_exact(tokenizer, ',"r2":')),
            ("close", _encode_exact(tokenizer, '}')),
        ),
    )


__all__ = [
    "ANSWER_EMISSION_SCHEMA",
    "RecurrentAnswerEmissionContract",
    "tokenizer_answer_emission_contract",
]
