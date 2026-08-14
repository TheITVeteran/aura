"""Canonical typed action targets for recurrent program execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

RECURRENT_ACTION_SCHEMA: Final = "aura.recurrent_action_target.v2"
ACTION_SLOT_NAMES: Final = (
    "opcode",
    "arg0",
    "arg1",
    "arg2",
    "arg3",
    "arg4",
    "arg5",
    "terminal",
)
ACTION_CARDINALITY: Final = 33
ACTION_NULL: Final = ACTION_CARDINALITY - 1
OP_COPY_VALUE: Final = 0
OP_ADD_MOD: Final = 1
OP_MUL_MOD: Final = 2
OP_SUB_MOD: Final = 3
OP_BOOL_NOT: Final = 4
OP_BOOL_AND: Final = 5
OP_BOOL_OR: Final = 6
OP_BOOL_XOR: Final = 7
OP_REGISTER_AFFINE: Final = 8
OP_FRONTIER_TRAVERSE: Final = 9
OP_FRONTIER_ENUMERATE: Final = 10
OP_FRONTIER_SIMULATE: Final = 11
OP_FRONTIER_INFER: Final = 12
OP_FRONTIER_SCHEDULE: Final = 13
OP_FRONTIER_CALIBRATE: Final = 14
OP_FRONTIER_AUDIT: Final = 15

_OPCODE_LABELS: Final = {
    OP_COPY_VALUE: "copy value",
    OP_ADD_MOD: "add modulo",
    OP_MUL_MOD: "multiply modulo",
    OP_SUB_MOD: "subtract modulo",
    OP_BOOL_NOT: "boolean not",
    OP_BOOL_AND: "boolean and",
    OP_BOOL_OR: "boolean or",
    OP_BOOL_XOR: "boolean xor",
    OP_REGISTER_AFFINE: "affine register update",
    OP_FRONTIER_TRAVERSE: "select the next stable traversal item",
    OP_FRONTIER_ENUMERATE: "advance exact constrained enumeration",
    OP_FRONTIER_SIMULATE: "apply one stateful program event",
    OP_FRONTIER_INFER: "advance one causal inference stage",
    OP_FRONTIER_SCHEDULE: "commit one feasible schedule action",
    OP_FRONTIER_CALIBRATE: "advance one exact calibration stage",
    OP_FRONTIER_AUDIT: "audit one premise-bearing evidence row",
}


def action_value_semantic_label(slot_name: str, value: int) -> str:
    """Give the frozen prelude a meaningful label for each instruction category."""

    if slot_name not in ACTION_SLOT_NAMES or type(value) is not int or not 0 <= value < ACTION_CARDINALITY:
        raise ValueError("action semantic label coordinate is invalid")
    if value == ACTION_NULL:
        return f"Canonical instruction field {slot_name} is unused"
    if slot_name == "opcode":
        operation = _OPCODE_LABELS.get(value, f"reserved operation {value}")
        return f"Canonical instruction operation is {operation}"
    if slot_name == "terminal":
        return f"Canonical instruction terminal flag is {value}"
    return f"Canonical instruction field {slot_name} has numeric value {value}"


def _canonical_instruction(
    family: str,
    field_names: tuple[str, ...],
    action: tuple[int, ...],
    *,
    terminal: int,
) -> tuple[int, ...]:
    """Compile family encodings into one executable micro-instruction set."""

    arguments = [ACTION_NULL] * 6
    if family == "khop" and field_names == ("next_node",):
        opcode = OP_COPY_VALUE
        arguments[0] = action[0]
    elif family == "modular" and field_names == ("opcode", "operand", "modulus"):
        opcode = {0: OP_ADD_MOD, 1: OP_MUL_MOD, 2: OP_SUB_MOD}.get(action[0], -1)
        arguments[:2] = action[1:]
    elif family == "boolean" and field_names == ("opcode", "operand", "has_operand"):
        opcode = {
            0: OP_BOOL_NOT,
            1: OP_BOOL_AND,
            2: OP_BOOL_OR,
            3: OP_BOOL_XOR,
        }.get(action[0], -1)
        arguments[:2] = action[1:]
    elif family == "register_trace" and field_names == (
        "destination",
        "left",
        "right",
        "multiplier",
        "offset",
        "modulus",
    ):
        opcode = OP_REGISTER_AFFINE
        arguments[:] = action
    elif family == "frontier_novel_algorithms" and field_names == (
        "selected_index",
        "value_lo",
        "value_hi",
        "checksum_lo",
        "checksum_hi",
    ):
        opcode = OP_FRONTIER_TRAVERSE
        arguments[:5] = action
    elif family == "frontier_mathematics" and field_names == (
        "input_index",
        "value_lo",
        "value_hi",
        "valid_added_lo",
        "valid_added_hi",
        "witness_member",
    ):
        opcode = OP_FRONTIER_ENUMERATE
        arguments[:] = action
    elif family == "frontier_coding" and field_names == (
        "case_index",
        "name_index",
        "signed_delta",
        "pressure",
        "active_count",
        "case_terminal",
    ):
        opcode = OP_FRONTIER_SIMULATE
        arguments[:] = action
    elif family == "frontier_scientific_inference" and field_names == (
        "stage",
        "arg0",
        "arg1",
        "arg2",
        "arg3",
    ):
        opcode = OP_FRONTIER_INFER
        arguments[:5] = action
    elif family == "frontier_long_horizon_planning" and field_names == (
        "task_index",
        "duration",
        "deadline",
        "reward",
        "dependency_code",
    ):
        opcode = OP_FRONTIER_SCHEDULE
        arguments[:5] = action
    elif family == "frontier_calibration" and field_names == (
        "stage",
        "numerator_lo",
        "numerator_hi",
        "denominator_lo",
        "denominator_hi",
        "decision_code",
    ):
        opcode = OP_FRONTIER_CALIBRATE
        arguments[:] = action
    elif family == "frontier_misleading_premise" and field_names == (
        "project_index",
        "impact",
        "reliability",
        "cost",
        "score_lo",
        "score_hi",
    ):
        opcode = OP_FRONTIER_AUDIT
        arguments[:] = action
    else:
        raise ValueError("structured action has no canonical micro-instruction")
    instruction = (opcode, *arguments, terminal)
    if opcode == ACTION_NULL or any(
        not 0 <= value <= ACTION_NULL for value in instruction
    ):
        raise ValueError("canonical micro-instruction is outside the vocabulary")
    return instruction


@dataclass(frozen=True, slots=True)
class RecurrentActionTargets:
    """Fixed-width categorical actions for one verified transition program."""

    family: str
    field_names: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]
    masks: tuple[tuple[bool, ...], ...]
    program_sha256: str

    def __post_init__(self) -> None:
        width = len(ACTION_SLOT_NAMES)
        if (
            not self.family
            or not self.values
            or len(self.values) != len(self.masks)
            or not 1 <= len(self.field_names) <= width
            or any(len(row) != width for row in self.values + self.masks)
        ):
            raise ValueError("recurrent action target metadata is invalid")
        if any(
            type(value) is not int or not 0 <= value < ACTION_CARDINALITY
            for row in self.values
            for value in row
        ):
            raise ValueError("recurrent action target value is outside the vocabulary")
        if any(type(value) is not bool for row in self.masks for value in row):
            raise ValueError("recurrent action target mask is invalid")
        if len(self.program_sha256) != 64:
            raise ValueError("recurrent action program commitment is invalid")

    @property
    def target_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": RECURRENT_ACTION_SCHEMA,
                    "family": self.family,
                    "field_names": self.field_names,
                    "values": self.values,
                    "masks": self.masks,
                    "program_sha256": self.program_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    def commitment(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_ACTION_SCHEMA,
            "family": self.family,
            "source_field_names": list(self.field_names),
            "action_slot_names": list(ACTION_SLOT_NAMES),
            "steps": len(self.values),
            "program_sha256": self.program_sha256,
            "target_sha256": self.target_sha256,
            "private_values_exposed": False,
        }


def action_targets_from_program(program: Any, iterations: int) -> RecurrentActionTargets:
    """Project a verified program into six categorical action registers.

    Missing fields and post-completion steps use an explicit null category. The
    active mask keeps scientific accuracy honest while still training the null
    behavior needed for stable stuttering after program completion.
    """

    if type(iterations) is not int or iterations < 1:
        raise ValueError("recurrent action iterations must be positive")
    trace = getattr(program, "state_trace", None)
    family = getattr(trace, "family", None)
    depth = getattr(trace, "depth", None)
    field_names = getattr(program, "action_field_names", None)
    actions = getattr(program, "actions", None)
    program_sha256 = getattr(program, "program_sha256", None)
    if (
        not isinstance(family, str)
        or type(depth) is not int
        or not isinstance(field_names, tuple)
        or not isinstance(actions, tuple)
        or not isinstance(program_sha256, str)
        or not 1 <= len(field_names) <= len(ACTION_SLOT_NAMES)
        or len(actions) != depth
    ):
        raise ValueError("structured transition program cannot populate actions")
    values: list[tuple[int, ...]] = []
    masks: list[tuple[bool, ...]] = []
    for step in range(iterations):
        row = [ACTION_NULL] * len(ACTION_SLOT_NAMES)
        active = [False] * len(ACTION_SLOT_NAMES)
        if step < depth:
            action = actions[step]
            if any(type(value) is not int or not 0 <= value < ACTION_NULL for value in action):
                raise ValueError("verified action is outside the categorical vocabulary")
            instruction = _canonical_instruction(
                family,
                field_names,
                action,
                terminal=int(step + 1 == depth),
            )
            row[:] = instruction
            active[:] = [value != ACTION_NULL for value in instruction]
        values.append(tuple(row))
        masks.append(tuple(active))
    return RecurrentActionTargets(
        family=family,
        field_names=field_names,
        values=tuple(values),
        masks=tuple(masks),
        program_sha256=program_sha256,
    )


__all__ = [
    "ACTION_CARDINALITY",
    "ACTION_NULL",
    "ACTION_SLOT_NAMES",
    "OP_ADD_MOD",
    "OP_BOOL_AND",
    "OP_BOOL_NOT",
    "OP_BOOL_OR",
    "OP_BOOL_XOR",
    "OP_COPY_VALUE",
    "OP_MUL_MOD",
    "OP_REGISTER_AFFINE",
    "OP_FRONTIER_AUDIT",
    "OP_FRONTIER_CALIBRATE",
    "OP_FRONTIER_ENUMERATE",
    "OP_FRONTIER_INFER",
    "OP_FRONTIER_SCHEDULE",
    "OP_FRONTIER_SIMULATE",
    "OP_FRONTIER_TRAVERSE",
    "OP_SUB_MOD",
    "RECURRENT_ACTION_SCHEMA",
    "RecurrentActionTargets",
    "action_targets_from_program",
    "action_value_semantic_label",
]
