"""Canonical evaluator-only state targets for intrinsic recurrence training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

RECURRENT_STATE_SCHEMA: Final = "aura.recurrent_state_target.v2"
STATE_SLOT_NAMES: Final = ("pc", "value0", "value1", "value2", "done")
# Nine value registers are the smallest fixed bank that can retain the four
# signed coding balances at the generator's declared bounds plus the active
# case identity.  This is a versioned extension so existing five-slot
# controller artifacts remain loadable and bit-identical.
SEMANTIC_STATE_SLOT_NAMES: Final = (
    "pc",
    "value0",
    "value1",
    "value2",
    "value3",
    "value4",
    "value5",
    "value6",
    "value7",
    "value8",
    "done",
)
# Keep control and value gradients comparable.  Family-specific semantic
# diagnostics, rather than positional loss weights, identify weak registers.
STATE_SLOT_LOSS_WEIGHTS: Final = (1.0, 4.0, 4.0, 4.0, 1.0)
STATE_CARDINALITY: Final = 33
STATE_INVALID: Final = STATE_CARDINALITY - 1


def state_slot_names(width: int) -> tuple[str, ...]:
    if width == len(STATE_SLOT_NAMES):
        return STATE_SLOT_NAMES
    if width == len(SEMANTIC_STATE_SLOT_NAMES):
        return SEMANTIC_STATE_SLOT_NAMES
    raise ValueError("recurrent state width has no registered schema")


def state_slot_loss_weights(width: int) -> tuple[float, ...]:
    if width == len(STATE_SLOT_NAMES):
        return STATE_SLOT_LOSS_WEIGHTS
    if width == len(SEMANTIC_STATE_SLOT_NAMES):
        return (1.0,) * width
    raise ValueError("recurrent state width has no registered loss contract")


@dataclass(frozen=True, slots=True)
class RecurrentStateTargets:
    """Fixed-width categorical targets and masks for one recurrent rollout."""

    family: str
    field_names: tuple[str, ...]
    initial_values: tuple[int, ...]
    initial_masks: tuple[bool, ...]
    values: tuple[tuple[int, ...], ...]
    masks: tuple[tuple[bool, ...], ...]
    trace_sha256: str

    def __post_init__(self) -> None:
        if not self.family or not self.values or len(self.values) != len(self.masks):
            raise ValueError("recurrent state target metadata is invalid")
        width = len(self.initial_values)
        state_slot_names(width)
        if len(self.initial_values) != width or len(self.initial_masks) != width:
            raise ValueError("recurrent initial state target width differs")
        if any(len(row) != width for row in self.values + self.masks):
            raise ValueError("recurrent state target width differs")
        if any(
            type(value) is not int or not 0 <= value < STATE_CARDINALITY
            for row in (self.initial_values,) + self.values
            for value in row
        ):
            raise ValueError("recurrent state target value is outside the vocabulary")
        if any(
            type(value) is not bool
            for row in (self.initial_masks,) + self.masks
            for value in row
        ):
            raise ValueError("recurrent state target mask is invalid")
        if any(
            not row[0] or not row[-1]
            for row in (self.initial_masks,) + self.masks
        ):
            raise ValueError("program counter and completion targets must be observed")
        if len(self.trace_sha256) != 64:
            raise ValueError("recurrent state trace commitment is invalid")

    @property
    def target_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": RECURRENT_STATE_SCHEMA,
                    "family": self.family,
                    "field_names": self.field_names,
                    "initial_values": self.initial_values,
                    "initial_masks": self.initial_masks,
                    "values": self.values,
                    "masks": self.masks,
                    "trace_sha256": self.trace_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    def commitment(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_STATE_SCHEMA,
            "family": self.family,
            "source_field_names": list(self.field_names),
            "state_slot_names": list(state_slot_names(len(self.initial_values))),
            "steps": len(self.values),
            "trace_sha256": self.trace_sha256,
            "target_sha256": self.target_sha256,
            "private_values_exposed": False,
        }


def state_targets_from_trace(
    trace: Any,
    iterations: int,
    *,
    state_slots: int | None = None,
) -> RecurrentStateTargets:
    """Project a private exact trace into the common categorical register bank.

    The trace is training authority only.  It is never serialized into the model
    prompt.  Runs longer than the program repeat its terminal state, providing an
    explicit stutter-stability target instead of rewarding arbitrary motion.
    """

    if type(iterations) is not int or iterations < 1:
        raise ValueError("recurrent target iterations must be positive")
    family = getattr(trace, "family", None)
    depth = getattr(trace, "depth", None)
    field_names = getattr(trace, "field_names", None)
    states = getattr(trace, "states", None)
    trace_sha256 = getattr(trace, "trace_sha256", None)
    if state_slots is None and isinstance(field_names, tuple):
        state_slots = (
            len(SEMANTIC_STATE_SLOT_NAMES)
            if len(field_names) > len(STATE_SLOT_NAMES)
            else len(STATE_SLOT_NAMES)
        )
    if (
        not isinstance(family, str)
        or type(depth) is not int
        or not isinstance(field_names, tuple)
        or not isinstance(states, tuple)
        or not isinstance(trace_sha256, str)
        or len(field_names) < 3
        or type(state_slots) is not int
        or len(field_names) > state_slots
        or field_names[0] != "pc"
        or field_names[-1] != "done"
        or len(states) != depth + 1
    ):
        raise ValueError("structured transition trace cannot populate state targets")
    slot_names = state_slot_names(state_slots)
    interior_width = len(field_names) - 2

    def project(state: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        row = [0] * len(slot_names)
        row[0] = state[0]
        row[-1] = state[-1]
        row[1 : 1 + interior_width] = state[1:-1]
        mask = tuple(
            index == 0 or index == len(slot_names) - 1 or index <= interior_width
            for index in range(len(slot_names))
        )
        return tuple(row), mask

    initial_values, initial_masks = project(states[0])
    values: list[tuple[int, ...]] = []
    masks: list[tuple[bool, ...]] = []
    for step in range(1, iterations + 1):
        row, mask = project(states[min(step, depth)])
        values.append(row)
        masks.append(mask)
    return RecurrentStateTargets(
        family=family,
        field_names=field_names,
        initial_values=initial_values,
        initial_masks=initial_masks,
        values=tuple(values),
        masks=tuple(masks),
        trace_sha256=trace_sha256,
    )


__all__ = [
    "RECURRENT_STATE_SCHEMA",
    "STATE_CARDINALITY",
    "STATE_INVALID",
    "SEMANTIC_STATE_SLOT_NAMES",
    "STATE_SLOT_NAMES",
    "STATE_SLOT_LOSS_WEIGHTS",
    "RecurrentStateTargets",
    "state_targets_from_trace",
    "state_slot_loss_weights",
    "state_slot_names",
]
