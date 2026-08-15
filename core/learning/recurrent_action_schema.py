"""Canonical typed action targets for recurrent program execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

RECURRENT_ACTION_SCHEMA: Final = "aura.recurrent_action_target.v4"
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
OP_PAIR_SET: Final = 16
OP_PAIR_ADD: Final = 17
OP_PAIR_MUL_IMMEDIATE: Final = 18
OP_PAIR_SUB_IMMEDIATE: Final = 19
OP_PAIR_SIGNED_SUB_IMMEDIATE: Final = 20
OP_PAIR_DIV: Final = 21
OP_RATIO_CHOICE: Final = 22
OP_RATIO_BAND: Final = 23
OP_SIGNED_PAIR_ADD_IMMEDIATE: Final = 24
OP_SIGNED_RANKED_GREATER: Final = 25
OP_RANKED_COMMIT: Final = 26
OP_SET_SCALAR: Final = 27
OP_PAIR_COPY: Final = 28
OP_PAIR_EUCLID_STEP: Final = 29
OP_PAIR_PRODUCT: Final = 30
MAX_RECURRENT_OPCODE: Final = OP_PAIR_PRODUCT
SEMANTIC_MICRO_OPCODES: Final = frozenset(
    range(OP_PAIR_SET, MAX_RECURRENT_OPCODE + 1)
)
SEMANTIC_MICRO_ACTION_FIELD_NAMES: Final = (
    "micro_opcode",
    "arg0",
    "arg1",
    "arg2",
    "arg3",
    "arg4",
    "arg5",
)
_SEMANTIC_MICRO_ARGUMENT_COUNTS: Final = {
    OP_PAIR_SET: 3,
    OP_PAIR_ADD: 3,
    OP_PAIR_MUL_IMMEDIATE: 2,
    OP_PAIR_SUB_IMMEDIATE: 2,
    OP_PAIR_SIGNED_SUB_IMMEDIATE: 2,
    OP_PAIR_DIV: 3,
    OP_RATIO_CHOICE: 3,
    OP_RATIO_BAND: 3,
    OP_SIGNED_PAIR_ADD_IMMEDIATE: 2,
    OP_SIGNED_RANKED_GREATER: 6,
    OP_RANKED_COMMIT: 4,
    OP_SET_SCALAR: 2,
    OP_PAIR_COPY: 2,
    OP_PAIR_EUCLID_STEP: 2,
    OP_PAIR_PRODUCT: 3,
}

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
    OP_PAIR_SET: "write one radix pair",
    OP_PAIR_ADD: "add two radix pairs",
    OP_PAIR_MUL_IMMEDIATE: "multiply one radix pair by an immediate",
    OP_PAIR_SUB_IMMEDIATE: "subtract an immediate from a radix pair",
    OP_PAIR_SIGNED_SUB_IMMEDIATE: "subtract and encode a signed radix pair",
    OP_PAIR_DIV: "divide one radix pair by an exact radix-pair divisor",
    OP_RATIO_CHOICE: "compare an exact ratio with one half",
    OP_RATIO_BAND: "classify an exact ratio confidence band",
    OP_SIGNED_PAIR_ADD_IMMEDIATE: "add an immediate to a signed radix pair",
    OP_SIGNED_RANKED_GREATER: "compare signed scores with a stable tie break",
    OP_RANKED_COMMIT: "commit a conditionally selected ranked candidate",
    OP_SET_SCALAR: "write one scalar register",
    OP_PAIR_COPY: "copy one radix pair",
    OP_PAIR_EUCLID_STEP: "advance one radix-pair Euclidean reduction step",
    OP_PAIR_PRODUCT: "multiply two immediate values into a radix pair",
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
    step: int,
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
        # The recurrent machine computes the running checksum from the public
        # selected value.  Copying the teacher's future checksum here made the
        # nominal "action" an answer-bearing label rather than an instruction.
        arguments[:3] = action[:3]
    elif family == "frontier_mathematics" and field_names == (
        "arg0",
        "arg1",
        "arg2",
        "arg3",
        "arg4",
        "arg5",
    ):
        opcode = OP_FRONTIER_ENUMERATE
        if step == 0:
            arguments[:] = action
        else:
            arguments[:3] = action[:3]
    elif family == "frontier_coding" and field_names == (
        "case_index",
        "name_index",
        "signed_delta",
        "pressure",
        "active_count",
        "case_terminal",
    ):
        opcode = OP_FRONTIER_SIMULATE
        # Stateful execution reconstructs balances from the causal action tape.
        # Pressure, active-count and case-terminal are derived future facts, not
        # operands the neural action reader should be asked to guess.
        arguments[:3] = action[:3]
    elif family == "frontier_scientific_inference" and field_names == (
        "stage",
        "arg0",
        "arg1",
        "arg2",
        "arg3",
        "arg4",
    ):
        opcode = OP_FRONTIER_INFER
        arguments[:] = action
    elif family == "frontier_long_horizon_planning" and field_names == (
        "task_index",
        "duration",
        "deadline",
        "reward",
        "dependency_code",
    ):
        opcode = OP_FRONTIER_SCHEDULE
        # Deadline and dependency are admission predicates used by the private
        # compiler.  The executable state update only consumes task identity,
        # duration and reward, so do not train two decorative action fields.
        arguments[0] = action[0]
        arguments[1] = action[1]
        arguments[3] = action[3]
    elif family == "frontier_calibration" and field_names == (
        "prior_numerator",
        "prior_denominator",
        "likelihood_h_numerator",
        "likelihood_h_denominator",
        "likelihood_not_h_numerator",
        "likelihood_not_h_denominator",
    ):
        opcode = OP_FRONTIER_CALIBRATE
        if step == 0:
            arguments[:] = action
    elif family == "frontier_misleading_premise" and field_names == (
        "row_index",
        "impact",
        "reliability",
        "cost",
        "name_rank",
        "reserved",
    ):
        opcode = OP_FRONTIER_AUDIT
        # Execute the public score formula in recurrent state.  The source
        # program contains the running winner and score as verification facts;
        # exposing those values as targets taught the decoder to predict the
        # answer instead of selecting an evidence row and its operands.
        arguments[:5] = action[:5]
    elif (
        family
        in {
            "frontier_coding",
            "frontier_calibration",
            "frontier_misleading_premise",
        }
        and field_names == SEMANTIC_MICRO_ACTION_FIELD_NAMES
    ):
        opcode = action[0]
        argument_count = _SEMANTIC_MICRO_ARGUMENT_COUNTS.get(opcode)
        if argument_count is None:
            raise ValueError("semantic micro-instruction opcode is unsupported")
        arguments[:argument_count] = action[1 : 1 + argument_count]
    else:
        raise ValueError("structured action has no canonical micro-instruction")
    instruction = (opcode, *arguments, terminal)
    if opcode == ACTION_NULL or any(
        not 0 <= value <= ACTION_NULL for value in instruction
    ):
        raise ValueError("canonical micro-instruction is outside the vocabulary")
    return instruction


def canonical_instruction_from_public_fields(
    family: str,
    field_names: tuple[str, ...],
    action: tuple[int, ...],
    *,
    step: int,
    terminal: int,
) -> tuple[int, ...]:
    """Compile public operands into the canonical recurrent instruction.

    The public process compiler deliberately shares this projection with the
    private supervision path.  Keeping one projection prevents an inference-
    only instruction dialect while the compiler's API excludes state traces,
    verifier answers, and other teaching evidence.
    """

    return _canonical_instruction(
        family,
        field_names,
        action,
        step=step,
        terminal=terminal,
    )


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
                step=step,
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
    "MAX_RECURRENT_OPCODE",
    "OP_PAIR_ADD",
    "OP_PAIR_DIV",
    "OP_PAIR_COPY",
    "OP_PAIR_EUCLID_STEP",
    "OP_PAIR_PRODUCT",
    "SEMANTIC_MICRO_OPCODES",
    "OP_PAIR_MUL_IMMEDIATE",
    "OP_PAIR_SET",
    "OP_PAIR_SUB_IMMEDIATE",
    "OP_PAIR_SIGNED_SUB_IMMEDIATE",
    "OP_RANKED_COMMIT",
    "OP_RATIO_BAND",
    "OP_RATIO_CHOICE",
    "OP_SET_SCALAR",
    "OP_SIGNED_PAIR_ADD_IMMEDIATE",
    "OP_SIGNED_RANKED_GREATER",
    "OP_SUB_MOD",
    "RECURRENT_ACTION_SCHEMA",
    "SEMANTIC_MICRO_ACTION_FIELD_NAMES",
    "RecurrentActionTargets",
    "action_targets_from_program",
    "action_value_semantic_label",
    "canonical_instruction_from_public_fields",
]
