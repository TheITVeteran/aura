"""Compile public task evidence into verified typed recurrence actions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Final

TYPED_ACTION_PROGRAM_SCHEMA: Final = "aura.typed_action_program.v1"
_BOOLEAN_PROMPT = re.compile(
    r"\AEvaluate this (?P<declared_depth>[1-9][0-9]*)-operation expression "
    r"with 1=true, 0=false, and xor meaning exactly one operand is true: "
    r"(?P<expression>.+?)\. Return a value of 1 or 0\. You may reason before "
    r"answering\. Finish with exactly one final line using the envelope "
    r"FINAL_ANSWER: <JSON object>\. FINAL_ANSWER is the envelope, not a JSON key\. "
    r"The JSON object must contain exactly these keys: value\.\Z"
)
_MODULAR_PROMPT = re.compile(
    r"\AStart at the given value and apply each operation modulo "
    r"(?P<modulus>[1-9][0-9]*): start=(?P<initial>[0-9]+)\. Operations: "
    r"(?P<operations>.+?)\. You may reason before answering\. Finish with exactly "
    r"one final line using the envelope FINAL_ANSWER: <JSON object>\. FINAL_ANSWER "
    r"is the envelope, not a JSON key\. The JSON object must contain exactly these "
    r"keys: residue\.\Z"
)
_BOOLEAN_TOKEN = re.compile(r"\s*(?:(not|and|or|xor)|([01])|([()]))")
_MODULAR_ACTION = re.compile(r"([+*\-])([0-9]+)")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TypedActionProgram:
    schema: str
    family: str
    depth: int
    field_names: tuple[str, ...]
    initial_state: tuple[int, ...]
    action_field_names: tuple[str, ...]
    actions: tuple[tuple[int, ...], ...]
    compiler_id: str
    public_source_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != TYPED_ACTION_PROGRAM_SCHEMA
            or self.family not in {"boolean", "modular"}
            or type(self.depth) is not int
            or self.depth < 1
            or not isinstance(self.field_names, tuple)
            or len(self.field_names) < 3
            or self.field_names[0] != "pc"
            or self.field_names[-1] != "done"
            or len(set(self.field_names)) != len(self.field_names)
            or any(
                not isinstance(name, str)
                or not name
                or not name.replace("_", "").isalnum()
                for name in self.field_names
            )
            or not isinstance(self.initial_state, tuple)
            or len(self.actions) != self.depth
            or len(self.initial_state) != len(self.field_names)
            or any(type(value) is not int or value < 0 for value in self.initial_state)
            or self.initial_state[0] != 0
            or self.initial_state[-1] != 0
            or not isinstance(self.action_field_names, tuple)
            or not self.action_field_names
            or len(set(self.action_field_names)) != len(self.action_field_names)
            or any(
                not isinstance(name, str)
                or not name
                or not name.replace("_", "").isalnum()
                for name in self.action_field_names
            )
            or not isinstance(self.actions, tuple)
            or any(
                not isinstance(action, tuple)
                or len(action) != len(self.action_field_names)
                or any(type(value) is not int or value < 0 for value in action)
                for action in self.actions
            )
            or not isinstance(self.compiler_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", self.compiler_id)
            or not isinstance(self.public_source_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.public_source_sha256)
        ):
            raise ValueError("typed action program is invalid")

    @property
    def program_sha256(self) -> str:
        return _canonical_sha256(self.private_payload())

    def private_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["field_names"] = list(self.field_names)
        payload["initial_state"] = list(self.initial_state)
        payload["action_field_names"] = list(self.action_field_names)
        payload["actions"] = [list(action) for action in self.actions]
        return payload

    def public_receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "family": self.family,
            "depth": self.depth,
            "field_names": list(self.field_names),
            "action_field_names": list(self.action_field_names),
            "action_count": len(self.actions),
            "compiler_id": self.compiler_id,
            "public_source_sha256": self.public_source_sha256,
            "program_sha256": self.program_sha256,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def _tokenize_boolean(expression: str) -> tuple[str, ...]:
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = _BOOLEAN_TOKEN.match(expression, position)
        if match is None or match.end() == position:
            raise ValueError("Boolean expression contains unsupported syntax")
        tokens.append(next(value for value in match.groups() if value is not None))
        position = match.end()
    if not tokens:
        raise ValueError("Boolean expression is empty")
    return tuple(tokens)


def compile_boolean_expression(
    expression: str,
    *,
    public_source_sha256: str,
) -> TypedActionProgram:
    """Compile the declared left-fold Boolean grammar without an answer label."""

    if not isinstance(expression, str) or not expression:
        raise ValueError("Boolean expression is invalid")
    tokens = _tokenize_boolean(expression)
    opcodes = {"not": 0, "and": 1, "or": 2, "xor": 3}

    def parse(index: int) -> tuple[int, list[tuple[int, int, int]], int]:
        if index >= len(tokens):
            raise ValueError("Boolean expression ended early")
        token = tokens[index]
        if token in ("0", "1"):
            return int(token), [], index + 1
        if token != "(":
            raise ValueError("Boolean expression is not in canonical grammar")
        if index + 1 < len(tokens) and tokens[index + 1] == "not":
            initial, actions, cursor = parse(index + 2)
            if cursor >= len(tokens) or tokens[cursor] != ")":
                raise ValueError("Boolean not expression is unbalanced")
            return initial, [*actions, (opcodes["not"], 0, 0)], cursor + 1
        initial, actions, cursor = parse(index + 1)
        if cursor >= len(tokens) or tokens[cursor] not in ("and", "or", "xor"):
            raise ValueError("Boolean binary operator is invalid")
        operation = tokens[cursor]
        if cursor + 1 >= len(tokens) or tokens[cursor + 1] not in ("0", "1"):
            raise ValueError("Boolean right operand must be a literal")
        operand = int(tokens[cursor + 1])
        cursor += 2
        if cursor >= len(tokens) or tokens[cursor] != ")":
            raise ValueError("Boolean binary expression is unbalanced")
        return initial, [*actions, (opcodes[operation], operand, 1)], cursor + 1

    initial, actions, cursor = parse(0)
    if cursor != len(tokens) or not actions:
        raise ValueError("Boolean expression has trailing or missing operations")
    return TypedActionProgram(
        schema=TYPED_ACTION_PROGRAM_SCHEMA,
        family="boolean",
        depth=len(actions),
        field_names=("pc", "value", "done"),
        initial_state=(0, initial, 0),
        action_field_names=("opcode", "operand", "has_operand"),
        actions=tuple(actions),
        compiler_id="boolean_expression_recursive_descent.v1",
        public_source_sha256=public_source_sha256,
    )


def compile_modular_operations(
    *,
    initial: int,
    modulus: int,
    operations: tuple[str, ...],
    public_source_sha256: str,
) -> TypedActionProgram:
    """Compile bounded integer operations without evaluating their answer."""

    if (
        type(initial) is not int
        or type(modulus) is not int
        or not 2 <= modulus <= 256
        or not 0 <= initial < modulus
        or not isinstance(operations, tuple)
        or not operations
    ):
        raise ValueError("modular program metadata is invalid")
    opcodes = {"+": 0, "*": 1, "-": 2}
    actions = []
    for text in operations:
        if not isinstance(text, str):
            raise ValueError("modular action syntax is invalid")
        match = _MODULAR_ACTION.fullmatch(text)
        if match is None:
            raise ValueError("modular action syntax is invalid")
        operation, operand_text = match.groups()
        operand = int(operand_text)
        if not 0 <= operand < modulus:
            raise ValueError("modular action operand is outside its modulus")
        actions.append((opcodes[operation], operand, modulus))
    return TypedActionProgram(
        schema=TYPED_ACTION_PROGRAM_SCHEMA,
        family="modular",
        depth=len(actions),
        field_names=("pc", "residue", "done"),
        initial_state=(0, initial, 0),
        action_field_names=("opcode", "operand", "modulus"),
        actions=tuple(actions),
        compiler_id="modular_operation_list.v1",
        public_source_sha256=public_source_sha256,
    )


def compile_public_transition_program(prompt: str) -> TypedActionProgram:
    """Compile a supported public prompt; ambiguity and unknown syntax refuse."""

    if (
        not isinstance(prompt, str)
        or not prompt
        or prompt != prompt.strip()
        or "\x00" in prompt
    ):
        raise ValueError("public transition prompt is invalid")
    source_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    boolean_match = _BOOLEAN_PROMPT.fullmatch(prompt)
    modular_match = _MODULAR_PROMPT.fullmatch(prompt)
    if (boolean_match is None) == (modular_match is None):
        raise ValueError("public transition prompt is unsupported or ambiguous")
    if boolean_match is not None:
        program = compile_boolean_expression(
            boolean_match.group("expression"),
            public_source_sha256=source_sha256,
        )
        if program.depth != int(boolean_match.group("declared_depth")):
            raise ValueError("declared Boolean depth differs from compiled actions")
        return program
    assert modular_match is not None
    operations = tuple(
        item.strip() for item in modular_match.group("operations").split(",")
    )
    return compile_modular_operations(
        initial=int(modular_match.group("initial")),
        modulus=int(modular_match.group("modulus")),
        operations=operations,
        public_source_sha256=source_sha256,
    )


__all__ = [
    "TYPED_ACTION_PROGRAM_SCHEMA",
    "TypedActionProgram",
    "compile_boolean_expression",
    "compile_modular_operations",
    "compile_public_transition_program",
]
