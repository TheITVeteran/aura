"""Tokenizer-bound operation observations for canonical recurrent microcode."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_COPY_VALUE,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_CALIBRATE,
    OP_FRONTIER_ENUMERATE,
    OP_FRONTIER_INFER,
    OP_FRONTIER_SCHEDULE,
    OP_FRONTIER_SIMULATE,
    OP_FRONTIER_TRAVERSE,
    OP_MUL_MOD,
    OP_REGISTER_AFFINE,
    OP_SUB_MOD,
)
from core.learning.recurrent_state_schema import state_slot_names

OPCODE_GROUNDING_SCHEMA: Final = "aura.recurrent_opcode_grounding.v1"
FRONTIER_FAMILY_GROUNDING_SCHEMA: Final = "aura.frontier_family_grounding.v1"
_OPCODE_TEXT: Final = (
    (OP_COPY_VALUE, "->"),
    (OP_ADD_MOD, " +"),
    (OP_MUL_MOD, " *"),
    (OP_SUB_MOD, " -"),
    (OP_BOOL_NOT, "not"),
    (OP_BOOL_AND, " and"),
    (OP_BOOL_OR, " or"),
    (OP_BOOL_XOR, " xor"),
)
_CONTEXT_TEXT: Final = (
    ("graph", " functional directed graph"),
    ("graph_edges_start", " edges:"),
    ("graph_edges_end", ". Start at"),
    ("modular_start", " Operations:"),
    ("modular_end", ". You may"),
    ("boolean_start", " is true:"),
    ("boolean_end", " Return"),
    ("register", "Trace three registers"),
    ("register_ops_start", " Apply in order:"),
    ("register_ops_end", ". End"),
)
_CONTEXT_KEYS: Final = frozenset(name for name, _text in _CONTEXT_TEXT)
_OBSERVED_OPCODES: Final = frozenset(opcode for opcode, _text in _OPCODE_TEXT)
_FRONTIER_FAMILY_TEXT: Final = (
    (OP_FRONTIER_TRAVERSE, "Fresh algorithm task."),
    (OP_FRONTIER_ENUMERATE, "Fresh combinatorics task."),
    (OP_FRONTIER_SIMULATE, "Fresh code-semantics task."),
    (OP_FRONTIER_INFER, "Fresh causal-inference task."),
    (OP_FRONTIER_SCHEDULE, "Fresh planning task."),
    (OP_FRONTIER_CALIBRATE, "Fresh calibration task."),
    (OP_FRONTIER_AUDIT, "Fresh premise-audit task."),
)
_FRONTIER_OPCODES: Final = frozenset(opcode for opcode, _text in _FRONTIER_FAMILY_TEXT)


def _public_state(values: Sequence[int], *, slots: int) -> tuple[int, ...]:
    """Project a public legacy register state into a registered topology."""

    state_slot_names(slots)
    if len(values) != 5 or slots < len(values):
        raise ValueError("public initial state topology is invalid")
    return (
        int(values[0]),
        *(int(value) for value in values[1:-1]),
        *(0 for _index in range(slots - len(values))),
        int(values[-1]),
    )


def _tokenizer_patterns(
    tokenizer: Any,
    rows: Sequence[tuple[Any, str]],
) -> tuple[tuple[Any, tuple[int, ...]], ...]:
    patterns = []
    for label, text in rows:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        if not isinstance(encoded, (list, tuple)) or not encoded or any(
            type(token_id) is not int or token_id < 0 for token_id in encoded
        ):
            raise ValueError(f"tokenizer cannot encode operation marker {text!r}")
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            try:
                decoded = decode(encoded, clean_up_tokenization_spaces=False)
            except TypeError:
                decoded = decode(encoded)
            if decoded != text:
                raise ValueError(f"operation marker {text!r} does not round-trip")
        patterns.append((label, tuple(int(token_id) for token_id in encoded)))
    if len({pattern for _label, pattern in patterns}) != len(patterns):
        raise ValueError("operation marker token patterns are not unique")
    return tuple(patterns)


def tokenizer_opcode_contract(tokenizer: Any) -> OpcodeObservationContract:
    """Bind canonical public grammar to exact tokenizer sequences."""

    return OpcodeObservationContract(
        _tokenizer_patterns(tokenizer, _OPCODE_TEXT),
        _tokenizer_patterns(tokenizer, _CONTEXT_TEXT),
    )


def tokenizer_frontier_family_contract(
    tokenizer: Any,
) -> FrontierFamilyObservationContract:
    """Bind public frontier task declarations to their canonical operation family."""

    return FrontierFamilyObservationContract(
        _tokenizer_patterns(tokenizer, _FRONTIER_FAMILY_TEXT)
    )


@dataclass(frozen=True, slots=True)
class FrontierFamilyObservationContract:
    """Recognize only the public task family, never its private transition trace."""

    patterns: tuple[tuple[int, tuple[int, ...]], ...]
    schema: str = FRONTIER_FAMILY_GROUNDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FRONTIER_FAMILY_GROUNDING_SCHEMA or not self.patterns:
            raise ValueError("frontier family observation schema differs")
        if (
            {opcode for opcode, _pattern in self.patterns} != _FRONTIER_OPCODES
            or any(
                type(opcode) is not int
                or not pattern
                or any(type(token_id) is not int or token_id < 0 for token_id in pattern)
                for opcode, pattern in self.patterns
            )
            or len({pattern for _opcode, pattern in self.patterns}) != len(self.patterns)
        ):
            raise ValueError("frontier family observation vocabulary is invalid")

    def observe(
        self,
        token_rows: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        values: list[int] = []
        recognized: list[bool] = []
        for row in token_rows:
            matches = [
                opcode
                for opcode, pattern in self.patterns
                if any(
                    tuple(row[index : index + len(pattern)]) == pattern
                    for index in range(len(row) - len(pattern) + 1)
                )
            ]
            if len(matches) > 1:
                raise ValueError("public prompt declares conflicting frontier families")
            values.append(matches[0] if matches else ACTION_NULL)
            recognized.append(bool(matches))
        return tuple(values), tuple(recognized)

    def public_initial_states(
        self,
        token_rows: Sequence[Sequence[int]],
        literal_contract: Any,
        *,
        slots: int = 5,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]:
        """Compile family-defined initial state from public evidence only.

        Frontier process families begin with cleared registers. Traversal also
        exposes the number of public nodes as its remaining-work counter. No
        private transition, answer, or holdout label enters this compiler.
        """

        _literal_values, literal_masks = literal_contract.observe(token_rows)
        opcodes, recognized = self.observe(token_rows)
        states: list[tuple[int, ...]] = []
        for opcode, known, masks in zip(
            opcodes, recognized, literal_masks, strict=True
        ):
            remaining = sum(bool(mask) for mask in masks)
            states.append(
                _public_state(
                    (
                    0,
                    0,
                    remaining if known and opcode == OP_FRONTIER_TRAVERSE else 0,
                    0,
                    0,
                    ),
                    slots=slots,
                )
            )
        return tuple(states), recognized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "patterns": [
                {"opcode": opcode, "token_ids": list(pattern)}
                for opcode, pattern in self.patterns
            ],
        }

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


@dataclass(frozen=True, slots=True)
class OpcodeObservationContract:
    patterns: tuple[tuple[int, tuple[int, ...]], ...]
    contexts: tuple[tuple[str, tuple[int, ...]], ...]
    schema: str = OPCODE_GROUNDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != OPCODE_GROUNDING_SCHEMA
            or not self.patterns
            or not self.contexts
        ):
            raise ValueError("opcode observation schema differs")
        if any(
            type(opcode) is not int
            or opcode < 0
            or not pattern
            or any(type(token_id) is not int or token_id < 0 for token_id in pattern)
            for opcode, pattern in self.patterns
        ):
            raise ValueError("opcode observation pattern is invalid")
        if len({pattern for _opcode, pattern in self.patterns}) != len(self.patterns):
            raise ValueError("opcode observation patterns are not unique")
        if {opcode for opcode, _pattern in self.patterns} != _OBSERVED_OPCODES:
            raise ValueError("opcode observation vocabulary is incomplete")
        if (
            {name for name, _pattern in self.contexts} != _CONTEXT_KEYS
            or any(
                not pattern
                or any(type(token_id) is not int or token_id < 0 for token_id in pattern)
                for _name, pattern in self.contexts
            )
            or len({pattern for _name, pattern in self.contexts}) != len(self.contexts)
        ):
            raise ValueError("opcode observation contexts are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "patterns": [
                {"opcode": opcode, "token_ids": list(pattern)}
                for opcode, pattern in self.patterns
            ],
            "contexts": [
                {"name": name, "token_ids": list(pattern)}
                for name, pattern in self.contexts
            ],
        }

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

    def observe(
        self,
        token_rows: Sequence[Sequence[int]],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
        values_rows = []
        masks_rows = []
        operations = dict(self.patterns)
        contexts = dict(self.contexts)

        def occurrences(
            row: Sequence[int],
            pattern: tuple[int, ...],
            *,
            start: int = 0,
            stop: int | None = None,
        ) -> tuple[int, ...]:
            boundary = len(row) if stop is None else stop
            width = len(pattern)
            return tuple(
                index
                for index in range(start, boundary - width + 1)
                if tuple(row[index : index + width]) == pattern
            )

        def first_after(
            row: Sequence[int], pattern: tuple[int, ...], start: int
        ) -> int | None:
            found = occurrences(row, pattern, start=start)
            return found[0] if found else None

        for row in token_rows:
            values = [0] * len(row)
            masks = [False] * len(row)
            register = occurrences(row, contexts["register"])
            graph = occurrences(row, contexts["graph"])
            modular = occurrences(row, contexts["modular_start"])
            boolean = occurrences(row, contexts["boolean_start"])
            detected = sum(bool(found) for found in (register, graph, modular, boolean))
            if detected > 1:
                raise ValueError("public prompt declares conflicting operation grammars")
            candidates: list[tuple[int, tuple[int, ...], int, int]] = []
            if register:
                candidates.append(
                    (
                        OP_REGISTER_AFFINE,
                        contexts["register"],
                        register[0],
                        register[0] + len(contexts["register"]),
                    )
                )
            elif graph:
                candidates.append((OP_COPY_VALUE, operations[OP_COPY_VALUE], 0, len(row)))
            elif modular:
                region_start = modular[0] + len(contexts["modular_start"])
                region_stop = first_after(row, contexts["modular_end"], region_start)
                if region_stop is None:
                    raise ValueError("modular operation region is unterminated")
                for opcode in (OP_ADD_MOD, OP_MUL_MOD, OP_SUB_MOD):
                    candidates.append((opcode, operations[opcode], region_start, region_stop))
            elif boolean:
                region_start = boolean[0] + len(contexts["boolean_start"])
                region_stop = first_after(row, contexts["boolean_end"], region_start)
                if region_stop is None:
                    raise ValueError("boolean operation region is unterminated")
                for opcode in (OP_BOOL_NOT, OP_BOOL_AND, OP_BOOL_OR, OP_BOOL_XOR):
                    candidates.append((opcode, operations[opcode], region_start, region_stop))
            for opcode, pattern, start, stop in candidates:
                for match in occurrences(row, pattern, start=start, stop=stop):
                    end = match + len(pattern) - 1
                    if masks[end] and values[end] != opcode:
                        raise ValueError("operation observations conflict at one token")
                    values[end] = opcode
                    masks[end] = True
            values_rows.append(tuple(values))
            masks_rows.append(tuple(masks))
        return tuple(values_rows), tuple(masks_rows)

    def public_initial_states(
        self,
        token_rows: Sequence[Sequence[int]],
        literal_contract: Any,
        *,
        slots: int = 5,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]:
        """Decode public initial registers for grammars with exact contracts."""

        literal_values, literal_masks = literal_contract.observe(token_rows)
        contexts = dict(self.contexts)
        states: list[tuple[int, ...]] = []
        recognized: list[bool] = []
        for row, values, masks in zip(
            token_rows, literal_values, literal_masks, strict=True
        ):
            state = (0, 0, 0, 0, 0)
            graph = self._occurrences(row, contexts["graph"])
            modular = self._occurrences(row, contexts["modular_start"])
            register = self._occurrences(row, contexts["register"])
            if graph:
                boundary = self._required_first(
                    row,
                    contexts["graph_edges_end"],
                    graph[0],
                    "graph edge region is unterminated",
                )
                trailing = self._literal_values(values, masks, boundary, len(row))
                if len(trailing) < 2:
                    raise ValueError("graph start and depth are absent")
                state = (0, trailing[0], 0, 0, 0)
                known = True
            elif modular:
                leading = self._literal_values(values, masks, 0, modular[0])
                if len(leading) < 2:
                    raise ValueError("modular initial state is absent")
                state = (0, leading[1], 0, 0, 0)
                known = True
            elif register:
                start = self._required_first(
                    row,
                    contexts["register_ops_start"],
                    register[0],
                    "register operation region is absent",
                )
                leading = self._literal_values(values, masks, 0, start)
                if len(leading) < 6:
                    raise ValueError("register initial state is absent")
                state = (0, leading[1], leading[3], leading[5], 0)
                known = True
            else:
                known = False
            states.append(_public_state(state, slots=slots))
            recognized.append(known)
        return tuple(states), tuple(recognized)

    def public_instructions(
        self,
        token_rows: Sequence[Sequence[int]],
        literal_contract: Any,
        state_rows: Sequence[Sequence[int]],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]:
        """Compile public syntax into one state-selected micro-instruction."""

        if len(token_rows) != len(state_rows):
            raise ValueError("public instruction state batch differs")
        literal_values, literal_masks = literal_contract.observe(token_rows)
        opcode_values, opcode_masks = self.observe(token_rows)
        contexts = dict(self.contexts)
        instructions: list[tuple[int, ...]] = []
        recognized: list[bool] = []
        for row, values, masks, opcodes, op_masks, state in zip(
            token_rows,
            literal_values,
            literal_masks,
            opcode_values,
            opcode_masks,
            state_rows,
            strict=True,
        ):
            if len(state) < 2:
                raise ValueError("public instruction state is incomplete")
            pc = int(state[0])
            instruction = (ACTION_NULL,) * 8
            known = False
            graph = self._occurrences(row, contexts["graph"])
            modular = self._occurrences(row, contexts["modular_start"])
            register = self._occurrences(row, contexts["register"])
            if graph:
                edge_start = self._required_first(
                    row,
                    contexts["graph_edges_start"],
                    graph[0],
                    "graph edge region is absent",
                ) + len(contexts["graph_edges_start"])
                edge_stop = self._required_first(
                    row,
                    contexts["graph_edges_end"],
                    edge_start,
                    "graph edge region is unterminated",
                )
                edge_values = self._literal_values(values, masks, edge_start, edge_stop)
                trailing = self._literal_values(values, masks, edge_stop, len(row))
                if len(edge_values) % 2 or len(trailing) < 2:
                    raise ValueError("graph public program is malformed")
                mapping = dict(zip(edge_values[::2], edge_values[1::2], strict=True))
                current = trailing[0] if pc == 0 else int(state[1])
                depth = trailing[1]
                if pc < depth and current in mapping:
                    instruction = (
                        OP_COPY_VALUE,
                        mapping[current],
                        ACTION_NULL,
                        ACTION_NULL,
                        ACTION_NULL,
                        ACTION_NULL,
                        ACTION_NULL,
                        int(pc + 1 == depth),
                    )
                    known = True
            elif modular:
                region_start = modular[0] + len(contexts["modular_start"])
                region_stop = self._required_first(
                    row,
                    contexts["modular_end"],
                    region_start,
                    "modular operation region is unterminated",
                )
                operands = self._literal_values(values, masks, region_start, region_stop)
                operations = self._literal_values(
                    opcodes, op_masks, region_start, region_stop
                )
                leading = self._literal_values(values, masks, 0, modular[0])
                if len(leading) < 2 or len(operands) != len(operations):
                    raise ValueError("modular public program is malformed")
                if pc < len(operations):
                    instruction = (
                        operations[pc],
                        operands[pc],
                        leading[0],
                        ACTION_NULL,
                        ACTION_NULL,
                        ACTION_NULL,
                        ACTION_NULL,
                        int(pc + 1 == len(operations)),
                    )
                    known = True
            elif register:
                region_start = self._required_first(
                    row,
                    contexts["register_ops_start"],
                    register[0],
                    "register operation region is absent",
                ) + len(contexts["register_ops_start"])
                region_stop = self._required_first(
                    row,
                    contexts["register_ops_end"],
                    region_start,
                    "register operation region is unterminated",
                )
                fields = self._literal_values(values, masks, region_start, region_stop)
                if len(fields) % 6:
                    raise ValueError("register public program is malformed")
                operations = tuple(
                    fields[index : index + 6] for index in range(0, len(fields), 6)
                )
                if pc < len(operations):
                    destination, left, multiplier, right, offset, modulus = operations[pc]
                    instruction = (
                        OP_REGISTER_AFFINE,
                        destination,
                        left,
                        right,
                        multiplier,
                        offset,
                        modulus,
                        int(pc + 1 == len(operations)),
                    )
                    known = True
            instructions.append(instruction)
            recognized.append(known)
        return tuple(instructions), tuple(recognized)

    @staticmethod
    def _occurrences(
        row: Sequence[int], pattern: tuple[int, ...], *, start: int = 0
    ) -> tuple[int, ...]:
        width = len(pattern)
        return tuple(
            index
            for index in range(start, len(row) - width + 1)
            if tuple(row[index : index + width]) == pattern
        )

    @classmethod
    def _required_first(
        cls,
        row: Sequence[int],
        pattern: tuple[int, ...],
        start: int,
        message: str,
    ) -> int:
        matches = cls._occurrences(row, pattern, start=start)
        if not matches:
            raise ValueError(message)
        return matches[0]

    @staticmethod
    def _literal_values(
        values: Sequence[int],
        masks: Sequence[bool],
        start: int,
        stop: int,
    ) -> tuple[int, ...]:
        return tuple(
            value
            for index, (value, mask) in enumerate(zip(values, masks, strict=True))
            if start <= index < stop and mask
        )


__all__ = [
    "FRONTIER_FAMILY_GROUNDING_SCHEMA",
    "OPCODE_GROUNDING_SCHEMA",
    "FrontierFamilyObservationContract",
    "OpcodeObservationContract",
    "tokenizer_frontier_family_contract",
    "tokenizer_opcode_contract",
]
