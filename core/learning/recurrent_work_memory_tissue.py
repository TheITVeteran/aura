"""Teacher-removed neural predicates over a generic addressable memory bus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from core.learning.recurrent_work_memory import (
    MATHEMATICS_MAX_CHOOSE,
    MATHEMATICS_MAX_SUM,
    MATHEMATICS_MAX_VALUE,
    MATHEMATICS_MAX_VALUES,
    MATHEMATICS_WORK_MEMORY_CAPACITY,
    MathematicsWorkMemoryAddress,
    MathematicsWorkMemoryCell,
)

MATHEMATICS_MEMORY_TISSUE_SCHEMA: Final = "aura.mathematics_memory_tissue.v1"
MATHEMATICS_MEMORY_EXECUTION_SCHEMA: Final = (
    "aura.mathematics_memory_execution.v1"
)
WRITE_MODES: Final = ("learned", "always", "never")
READ_MODES: Final = ("learned", "always", "never")
ROUTING_MODES: Final = ("identity", "rotated")
MEMORY_MODES: Final = ("active", "reset_each_step")


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


def _validate_public_objective(
    *,
    choose: int,
    gap: int,
    low: int,
    high: int,
    values: tuple[int, ...],
    allow_empty: bool = False,
) -> None:
    if (
        type(choose) is not int
        or not 1 <= choose <= MATHEMATICS_MAX_CHOOSE
        or type(gap) is not int
        or not 1 <= gap <= MATHEMATICS_MAX_VALUE
        or type(low) is not int
        or type(high) is not int
        or not 0 <= low <= high <= MATHEMATICS_MAX_SUM
        or not isinstance(values, tuple)
        or not int(not allow_empty) <= len(values) <= MATHEMATICS_MAX_VALUES
        or any(
            type(value) is not int or not 1 <= value <= MATHEMATICS_MAX_VALUE
            for value in values
        )
        or tuple(sorted(values)) != values
        or len(set(values)) != len(values)
    ):
        raise ValueError("mathematics memory objective is outside its registry")


@dataclass(frozen=True, slots=True)
class NeuralMathematicsMemoryState:
    """Structurally valid student memory; incorrect semantic writes remain visible."""

    choose: int
    gap: int
    low: int
    high: int
    processed_values: tuple[int, ...]
    cells: tuple[MathematicsWorkMemoryCell, ...]
    capacity: int = MATHEMATICS_WORK_MEMORY_CAPACITY

    def __post_init__(self) -> None:
        addresses = tuple(cell.address for cell in self.cells)
        if (
            type(self.capacity) is not int
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
            raise ValueError("neural mathematics memory structure is invalid")
        _validate_public_objective(
            choose=self.choose,
            gap=self.gap,
            low=self.low,
            high=self.high,
            values=self.processed_values,
            allow_empty=True,
        )
        seen = set(self.processed_values)
        for cell in self.cells[1:]:
            if (
                cell.address.selected_count > self.choose
                or any(value not in seen for value in cell.witness)
            ):
                raise ValueError("neural mathematics memory payload is invalid")

    @classmethod
    def empty(
        cls,
        *,
        choose: int,
        gap: int,
        low: int,
        high: int,
        capacity: int = MATHEMATICS_WORK_MEMORY_CAPACITY,
    ) -> NeuralMathematicsMemoryState:
        _validate_public_objective(
            choose=choose,
            gap=gap,
            low=low,
            high=high,
            values=(),
            allow_empty=True,
        )
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

    def payload(self) -> dict[str, Any]:
        return {
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
        return _canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MathematicsMemoryExecution:
    count: int
    witness: tuple[int, ...]
    state_sha256s: tuple[str, ...]
    tissue_sha256: str
    write_mode: str
    read_mode: str
    routing_mode: str
    memory_mode: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": MATHEMATICS_MEMORY_EXECUTION_SCHEMA,
            "state_sha256s": list(self.state_sha256s),
            "transition_count": len(self.state_sha256s) - 1,
            "tissue_sha256": self.tissue_sha256,
            "write_mode": self.write_mode,
            "read_mode": self.read_mode,
            "routing_mode": self.routing_mode,
            "memory_mode": self.memory_mode,
            "result_sha256": _canonical_sha256(
                {"count": self.count, "witness": list(self.witness)}
            ),
            "teacher_available": False,
            "verifier_available": False,
            "student_memory_rollin": True,
            "generic_address_bus": True,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class MathematicsMemoryTissue(nn.Module):
    """Learn write and result predicates; storage routing stays task-agnostic."""

    write_feature_count: Final = 10
    read_feature_count: Final = 8

    def __init__(self, *, hidden_size: int = 32, seed: int = 2026081507) -> None:
        super().__init__()
        if type(hidden_size) is not int or not 4 <= hidden_size <= 256:
            raise ValueError("mathematics memory tissue width is invalid")
        if type(seed) is not int or seed < 0:
            raise ValueError("mathematics memory tissue seed is invalid")
        self.hidden_size = hidden_size
        self.seed = seed
        mx.random.seed(seed)
        self.write_hidden = nn.Linear(self.write_feature_count, hidden_size)
        self.write_output = nn.Linear(hidden_size, 1)
        # Result admission is a conjunction of one equality and two ordered
        # bounds. Preserve that public algebraic shape and learn its calibrated
        # margins; an unconstrained classifier made rare false positives that
        # catastrophically corrupted an otherwise exact aggregate.
        self.read_scale_raw = mx.zeros((3,), dtype=mx.float32)
        self.read_bias = mx.full((3,), -1.0, dtype=mx.float32)

    @staticmethod
    def write_features(
        cell: MathematicsWorkMemoryCell,
        *,
        value: int,
        choose: int,
        gap: int,
        processed_count: int,
    ) -> tuple[float, ...]:
        selected = cell.address.selected_count
        last = cell.address.last_value
        return (
            selected / MATHEMATICS_MAX_CHOOSE,
            choose / MATHEMATICS_MAX_CHOOSE,
            (choose - selected) / MATHEMATICS_MAX_CHOOSE,
            last / MATHEMATICS_MAX_VALUE,
            value / MATHEMATICS_MAX_VALUE,
            gap / MATHEMATICS_MAX_VALUE,
            (value - last - gap) / MATHEMATICS_MAX_VALUE,
            float(selected == 0),
            cell.address.total_sum / MATHEMATICS_MAX_SUM,
            processed_count / MATHEMATICS_MAX_VALUES,
        )

    @staticmethod
    def read_features(
        cell: MathematicsWorkMemoryCell,
        *,
        choose: int,
        low: int,
        high: int,
    ) -> tuple[float, ...]:
        selected = cell.address.selected_count
        total = cell.address.total_sum
        return (
            selected / MATHEMATICS_MAX_CHOOSE,
            choose / MATHEMATICS_MAX_CHOOSE,
            0.5 - abs(selected - choose),
            total / MATHEMATICS_MAX_SUM,
            low / MATHEMATICS_MAX_SUM,
            high / MATHEMATICS_MAX_SUM,
            total - low + 0.5,
            high - total + 0.5,
        )

    def write_logits(self, features: Any) -> Any:
        return self.write_output(mx.tanh(self.write_hidden(features)))[..., 0]

    def read_logits(self, features: Any) -> Any:
        if int(features.shape[-1]) != self.read_feature_count:
            raise ValueError("mathematics memory read feature width differs")
        margins = mx.take(features, mx.array((2, 6, 7)), axis=-1)
        positive_scale = mx.logaddexp(
            self.read_scale_raw,
            mx.zeros_like(self.read_scale_raw),
        )
        calibrated = margins * positive_scale + self.read_bias
        return mx.min(calibrated, axis=-1)

    def parameter_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(tree_flatten(self.parameters())):
            mx.eval(value)
            digest.update(name.encode("ascii"))
            digest.update(bytes(memoryview(value.astype(mx.float32))))
        return digest.hexdigest()


def _hard_decisions(logits: Any) -> tuple[bool, ...]:
    mx.eval(logits)
    return tuple(bool(value > 0.0) for value in logits.tolist())


def _apply_writes(
    state: NeuralMathematicsMemoryState,
    value: int,
    decisions: tuple[bool, ...],
) -> NeuralMathematicsMemoryState:
    if len(decisions) != len(state.cells):
        raise ValueError("neural mathematics write decision count differs")
    merged = {cell.address: cell for cell in state.cells}
    for cell, accepted in zip(state.cells, decisions, strict=True):
        if not accepted or cell.address.selected_count >= state.choose:
            continue
        witness = (*cell.witness, value)
        address = MathematicsWorkMemoryAddress(
            cell.address.selected_count + 1,
            value,
            cell.address.total_sum + value,
        )
        existing = merged.get(address)
        merged[address] = MathematicsWorkMemoryCell(
            address=address,
            multiplicity=cell.multiplicity
            + (existing.multiplicity if existing is not None else 0),
            witness=min(
                witness,
                existing.witness if existing is not None else witness,
            ),
        )
    cells = tuple(merged[address] for address in sorted(merged))
    if len(cells) > state.capacity:
        raise OverflowError("neural mathematics memory capacity exceeded")
    return NeuralMathematicsMemoryState(
        choose=state.choose,
        gap=state.gap,
        low=state.low,
        high=state.high,
        processed_values=(*state.processed_values, value),
        cells=cells,
        capacity=state.capacity,
    )


def execute_mathematics_memory(
    tissue: MathematicsMemoryTissue,
    *,
    choose: int,
    gap: int,
    low: int,
    high: int,
    values: tuple[int, ...],
    write_mode: Literal["learned", "always", "never"] = "learned",
    read_mode: Literal["learned", "always", "never"] = "learned",
    routing_mode: Literal["identity", "rotated"] = "identity",
    memory_mode: Literal["active", "reset_each_step"] = "active",
) -> MathematicsMemoryExecution:
    """Execute from public inputs with no compiler, verifier, or answer access."""

    if not isinstance(tissue, MathematicsMemoryTissue):
        raise TypeError("mathematics memory execution requires neural tissue")
    if (
        write_mode not in WRITE_MODES
        or read_mode not in READ_MODES
        or routing_mode not in ROUTING_MODES
        or memory_mode not in MEMORY_MODES
    ):
        raise ValueError("mathematics memory lesion mode differs")
    _validate_public_objective(
        choose=choose,
        gap=gap,
        low=low,
        high=high,
        values=values,
    )
    state = NeuralMathematicsMemoryState.empty(
        choose=choose,
        gap=gap,
        low=low,
        high=high,
    )
    state_hashes = [state.state_sha256]
    for value in values:
        if memory_mode == "reset_each_step" and state.processed_values:
            state = NeuralMathematicsMemoryState.empty(
                choose=choose,
                gap=gap,
                low=low,
                high=high,
            )
        if write_mode == "learned":
            features = mx.array(
                [
                    tissue.write_features(
                        cell,
                        value=value,
                        choose=choose,
                        gap=gap,
                        processed_count=len(state.processed_values),
                    )
                    for cell in state.cells
                ],
                dtype=mx.float32,
            )
            decisions = _hard_decisions(tissue.write_logits(features))
        else:
            decisions = (write_mode == "always",) * len(state.cells)
        if routing_mode == "rotated" and len(decisions) > 1:
            decisions = (*decisions[1:], decisions[0])
        state = _apply_writes(state, value, decisions)
        state_hashes.append(state.state_sha256)
    if read_mode == "learned":
        features = mx.array(
            [
                tissue.read_features(
                    cell,
                    choose=choose,
                    low=low,
                    high=high,
                )
                for cell in state.cells
            ],
            dtype=mx.float32,
        )
        admitted = _hard_decisions(tissue.read_logits(features))
    else:
        admitted = (read_mode == "always",) * len(state.cells)
    selected = tuple(
        cell
        for cell, accept in zip(state.cells, admitted, strict=True)
        if accept
    )
    return MathematicsMemoryExecution(
        count=sum(cell.multiplicity for cell in selected),
        witness=min((cell.witness for cell in selected), default=()),
        state_sha256s=tuple(state_hashes),
        tissue_sha256=tissue.parameter_sha256(),
        write_mode=write_mode,
        read_mode=read_mode,
        routing_mode=routing_mode,
        memory_mode=memory_mode,
    )


__all__ = [
    "MATHEMATICS_MEMORY_EXECUTION_SCHEMA",
    "MATHEMATICS_MEMORY_TISSUE_SCHEMA",
    "MathematicsMemoryExecution",
    "MathematicsMemoryTissue",
    "NeuralMathematicsMemoryState",
    "MEMORY_MODES",
    "READ_MODES",
    "ROUTING_MODES",
    "WRITE_MODES",
    "execute_mathematics_memory",
]
