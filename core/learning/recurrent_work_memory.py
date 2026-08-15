"""Bounded addressable work memory for recurrent mathematics programs."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Final

MATHEMATICS_WORK_MEMORY_SCHEMA: Final = "aura.recurrent_mathematics_work_memory.v1"
MATHEMATICS_WORK_MEMORY_TRACE_SCHEMA: Final = (
    "aura.recurrent_mathematics_work_memory_trace.v1"
)
MATHEMATICS_MAX_VALUES: Final = 10
MATHEMATICS_MAX_CHOOSE: Final = 4
MATHEMATICS_MAX_VALUE: Final = 33
MATHEMATICS_MAX_SUM: Final = MATHEMATICS_MAX_CHOOSE * MATHEMATICS_MAX_VALUE
MATHEMATICS_WORK_MEMORY_CAPACITY: Final = sum(
    math.comb(MATHEMATICS_MAX_VALUES, width)
    for width in range(MATHEMATICS_MAX_CHOOSE + 1)
)


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


@dataclass(frozen=True, order=True, slots=True)
class MathematicsWorkMemoryAddress:
    """One semantic address in the sparse dynamic-programming table."""

    selected_count: int
    last_value: int
    total_sum: int

    def __post_init__(self) -> None:
        if (
            type(self.selected_count) is not int
            or not 0 <= self.selected_count <= MATHEMATICS_MAX_CHOOSE
            or type(self.last_value) is not int
            or not 0 <= self.last_value <= MATHEMATICS_MAX_VALUE
            or type(self.total_sum) is not int
            or not 0 <= self.total_sum <= MATHEMATICS_MAX_SUM
            or self.selected_count == 0
            and (self.last_value != 0 or self.total_sum != 0)
            or self.selected_count > 0
            and (self.last_value == 0 or self.total_sum < self.last_value)
        ):
            raise ValueError("mathematics work-memory address is invalid")


@dataclass(frozen=True, slots=True)
class MathematicsWorkMemoryCell:
    """An exact multiplicity and witness stored at one content address."""

    address: MathematicsWorkMemoryAddress
    multiplicity: int
    witness: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.address, MathematicsWorkMemoryAddress)
            or type(self.multiplicity) is not int
            or self.multiplicity < 1
            or not isinstance(self.witness, tuple)
            or len(self.witness) != self.address.selected_count
            or any(
                type(value) is not int
                or not 1 <= value <= MATHEMATICS_MAX_VALUE
                for value in self.witness
            )
            or tuple(sorted(self.witness)) != self.witness
            or len(set(self.witness)) != len(self.witness)
            or bool(self.witness)
            and (
                self.witness[-1] != self.address.last_value
                or sum(self.witness) != self.address.total_sum
            )
        ):
            raise ValueError("mathematics work-memory cell is invalid")


@dataclass(frozen=True, slots=True)
class MathematicsWorkMemory:
    """A finite sparse state sufficient for the bounded subset-count task."""

    choose: int
    gap: int
    low: int
    high: int
    processed_values: tuple[int, ...]
    cells: tuple[MathematicsWorkMemoryCell, ...]
    capacity: int = MATHEMATICS_WORK_MEMORY_CAPACITY
    schema: str = MATHEMATICS_WORK_MEMORY_SCHEMA

    def __post_init__(self) -> None:
        addresses = tuple(cell.address for cell in self.cells)
        if (
            self.schema != MATHEMATICS_WORK_MEMORY_SCHEMA
            or type(self.choose) is not int
            or not 1 <= self.choose <= MATHEMATICS_MAX_CHOOSE
            or type(self.gap) is not int
            or self.gap < 1
            or type(self.low) is not int
            or type(self.high) is not int
            or not 0 <= self.low <= self.high <= MATHEMATICS_MAX_SUM
            or not isinstance(self.processed_values, tuple)
            or len(self.processed_values) > MATHEMATICS_MAX_VALUES
            or any(
                type(value) is not int
                or not 1 <= value <= MATHEMATICS_MAX_VALUE
                for value in self.processed_values
            )
            or tuple(sorted(self.processed_values)) != self.processed_values
            or len(set(self.processed_values)) != len(self.processed_values)
            or type(self.capacity) is not int
            or not 1 <= self.capacity <= MATHEMATICS_WORK_MEMORY_CAPACITY
            or not self.cells
            or len(self.cells) > self.capacity
            or addresses != tuple(sorted(addresses))
            or len(set(addresses)) != len(addresses)
            or self.cells[0]
            != MathematicsWorkMemoryCell(
                MathematicsWorkMemoryAddress(0, 0, 0),
                1,
                (),
            )
        ):
            raise ValueError("mathematics work-memory state is invalid")
        seen = set(self.processed_values)
        for cell in self.cells[1:]:
            if (
                cell.address.selected_count > self.choose
                or any(value not in seen for value in cell.witness)
                or any(
                    right - left < self.gap
                    for left, right in zip(
                        cell.witness,
                        cell.witness[1:],
                        strict=False,
                    )
                )
            ):
                raise ValueError("mathematics work-memory semantics differ")

    @classmethod
    def empty(
        cls,
        *,
        choose: int,
        gap: int,
        low: int,
        high: int,
        capacity: int = MATHEMATICS_WORK_MEMORY_CAPACITY,
    ) -> MathematicsWorkMemory:
        return cls(
            choose=choose,
            gap=gap,
            low=low,
            high=high,
            processed_values=(),
            cells=(
                MathematicsWorkMemoryCell(
                    MathematicsWorkMemoryAddress(0, 0, 0),
                    1,
                    (),
                ),
            ),
            capacity=capacity,
        )

    def apply_value(self, value: int) -> MathematicsWorkMemory:
        """Apply one public sorted value without reading a verifier answer."""

        if (
            type(value) is not int
            or not 1 <= value <= MATHEMATICS_MAX_VALUE
            or len(self.processed_values) >= MATHEMATICS_MAX_VALUES
            or self.processed_values
            and value <= self.processed_values[-1]
        ):
            raise ValueError("mathematics work-memory input order is invalid")
        merged = {cell.address: cell for cell in self.cells}
        for cell in self.cells:
            selected = cell.address.selected_count
            if selected >= self.choose:
                continue
            if selected and value - cell.address.last_value < self.gap:
                continue
            witness = (*cell.witness, value)
            address = MathematicsWorkMemoryAddress(
                selected + 1,
                value,
                cell.address.total_sum + value,
            )
            existing = merged.get(address)
            merged[address] = MathematicsWorkMemoryCell(
                address=address,
                multiplicity=cell.multiplicity + (
                    existing.multiplicity if existing is not None else 0
                ),
                witness=min(
                    witness,
                    existing.witness if existing is not None else witness,
                ),
            )
        cells = tuple(merged[address] for address in sorted(merged))
        if len(cells) > self.capacity:
            raise OverflowError("mathematics work-memory capacity exceeded")
        return MathematicsWorkMemory(
            choose=self.choose,
            gap=self.gap,
            low=self.low,
            high=self.high,
            processed_values=(*self.processed_values, value),
            cells=cells,
            capacity=self.capacity,
        )

    def result(self) -> tuple[int, tuple[int, ...]]:
        admitted = tuple(
            cell
            for cell in self.cells
            if cell.address.selected_count == self.choose
            and self.low <= cell.address.total_sum <= self.high
        )
        return (
            sum(cell.multiplicity for cell in admitted),
            min((cell.witness for cell in admitted), default=()),
        )

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "choose": self.choose,
            "gap": self.gap,
            "low": self.low,
            "high": self.high,
            "processed_values": list(self.processed_values),
            "capacity": self.capacity,
            "cells": [
                {
                    "address": [
                        cell.address.selected_count,
                        cell.address.last_value,
                        cell.address.total_sum,
                    ],
                    "multiplicity": cell.multiplicity,
                    "witness": list(cell.witness),
                }
                for cell in self.cells
            ],
        }

    @property
    def state_sha256(self) -> str:
        return _canonical_sha256(self.private_payload())


@dataclass(frozen=True, slots=True)
class MathematicsWorkMemoryTrace:
    """Training-only exact memory states with a value-free public commitment."""

    states: tuple[MathematicsWorkMemory, ...]
    configuration_steps: int = 1
    schema: str = MATHEMATICS_WORK_MEMORY_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != MATHEMATICS_WORK_MEMORY_TRACE_SCHEMA
            or len(self.states) < 2
            or any(not isinstance(state, MathematicsWorkMemory) for state in self.states)
            or self.states[0].processed_values
            or self.configuration_steps != 1
        ):
            raise ValueError("mathematics work-memory trace is invalid")
        first = self.states[0]
        for index, state in enumerate(self.states):
            if (
                (state.choose, state.gap, state.low, state.high, state.capacity)
                != (first.choose, first.gap, first.low, first.high, first.capacity)
                or len(state.processed_values)
                != max(0, index - self.configuration_steps)
                or index > self.configuration_steps
                and state.processed_values[:-1]
                != self.states[index - 1].processed_values
                or 0 < index <= self.configuration_steps
                and state != first
            ):
                raise ValueError("mathematics work-memory trace continuity differs")

    @property
    def trace_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": self.schema,
                "configuration_steps": self.configuration_steps,
                "states": [state.private_payload() for state in self.states],
            }
        )

    def public_commitment(self) -> dict[str, Any]:
        final = self.states[-1]
        body = {
            "schema": self.schema,
            "steps": len(self.states) - 1,
            "configuration_steps": self.configuration_steps,
            "capacity": final.capacity,
            "maximum_live_cells": max(len(state.cells) for state in self.states),
            "trace_sha256": self.trace_sha256,
            "private_addresses_exposed": False,
            "private_multiplicities_exposed": False,
            "private_witnesses_exposed": False,
            "runtime_teacher_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def compile_mathematics_work_memory(
    *,
    choose: int,
    gap: int,
    low: int,
    high: int,
    values: tuple[int, ...],
    capacity: int = MATHEMATICS_WORK_MEMORY_CAPACITY,
) -> MathematicsWorkMemoryTrace:
    """Compile exact bounded supervision from public objective literals only."""

    state = MathematicsWorkMemory.empty(
        choose=choose,
        gap=gap,
        low=low,
        high=high,
        capacity=capacity,
    )
    # The first canonical action configures choose/gap/bounds but does not
    # consume an input value. Keep that no-op in the program-aligned trace.
    states = [state, state]
    for value in values:
        state = state.apply_value(value)
        states.append(state)
    return MathematicsWorkMemoryTrace(tuple(states))


def brute_force_mathematics_result(
    *,
    choose: int,
    gap: int,
    low: int,
    high: int,
    values: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Independent test oracle; never import this into a measured candidate."""

    valid = tuple(
        combination
        for combination in itertools.combinations(values, choose)
        if all(
            right - left >= gap
            for left, right in zip(combination, combination[1:], strict=False)
        )
        and low <= sum(combination) <= high
    )
    return len(valid), min(valid, default=())


__all__ = [
    "MATHEMATICS_MAX_CHOOSE",
    "MATHEMATICS_MAX_SUM",
    "MATHEMATICS_MAX_VALUE",
    "MATHEMATICS_MAX_VALUES",
    "MATHEMATICS_WORK_MEMORY_CAPACITY",
    "MATHEMATICS_WORK_MEMORY_SCHEMA",
    "MATHEMATICS_WORK_MEMORY_TRACE_SCHEMA",
    "MathematicsWorkMemory",
    "MathematicsWorkMemoryAddress",
    "MathematicsWorkMemoryCell",
    "MathematicsWorkMemoryTrace",
    "brute_force_mathematics_result",
    "compile_mathematics_work_memory",
]
