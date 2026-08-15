"""Semantic recurrent register machine powered by sealed learned arithmetic.

Instruction routing and register addressing are structural control. Numerical
addition, multiplication, subtraction and radix decoding are produced by the
independently trained SystematicNeuralALU. No verifier answer, private trace,
lookup table, Python arithmetic operator for the learned result, or runtime
teacher is available to execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final

import mlx.core as mx

from core.brain.llm.latent_cortex.systematic_neural_alu import (
    SystematicNeuralALU,
    load_systematic_neural_alu,
)
from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    MAX_RECURRENT_OPCODE,
    OP_PAIR_ADD,
    OP_PAIR_COPY,
    OP_PAIR_DIV,
    OP_PAIR_EUCLID_STEP,
    OP_PAIR_MUL_IMMEDIATE,
    OP_PAIR_PRODUCT,
    OP_PAIR_SET,
    OP_PAIR_SUB_IMMEDIATE,
    OP_PAIR_SIGNED_SUB_IMMEDIATE,
    OP_RANKED_COMMIT,
    OP_RATIO_BAND,
    OP_RATIO_CHOICE,
    OP_SET_SCALAR,
    OP_SIGNED_PAIR_ADD_IMMEDIATE,
    OP_SIGNED_RANKED_GREATER,
)
from core.learning.recurrent_state_schema import STATE_INVALID

SEMANTIC_NEURAL_MACHINE_SCHEMA: Final = "aura.semantic_neural_machine.v1"
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1
_LEARNED_ADD: Final = 0
_LEARNED_MUL: Final = 1
_LEARNED_SUB: Final = 2


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
class SemanticNeuralTransition:
    next_state: tuple[int, ...]
    opcode: int
    learned_operation_count: int
    tissue_sha256: str
    input_sha256: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": SEMANTIC_NEURAL_MACHINE_SCHEMA,
            "next_state_sha256": _canonical_sha256(self.next_state),
            "opcode": self.opcode,
            "learned_operation_count": self.learned_operation_count,
            "tissue_sha256": self.tissue_sha256,
            "input_sha256": self.input_sha256,
            "teacher_available": False,
            "private_trace_available": False,
            "exact_arithmetic_operator_available": False,
            "lookup_table_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class SemanticNeuralMachine:
    """Execute semantic micro-programs with learned arithmetic tissue."""

    def __init__(self, tissue: SystematicNeuralALU | None = None) -> None:
        self.tissue = load_systematic_neural_alu() if tissue is None else tissue
        if not isinstance(self.tissue, SystematicNeuralALU):
            raise TypeError("semantic neural machine tissue is invalid")
        # The source tissue learned coefficients within float32 tolerance on
        # operands below 43.  At the semantic register bound (960), a 1e-6
        # multiplicative residual can cross an integer boundary. Quantize only
        # an independently verified near-integer coefficient matrix; this is a
        # versioned model transformation, not an exact operator or hand-written
        # replacement table.
        quantized = mx.round(self.tissue.raw_coefficients)
        delta = mx.max(mx.abs(quantized - self.tissue.raw_coefficients))
        mx.eval(quantized, delta)
        if float(delta.item()) > 1e-3:
            raise RuntimeError("learned arithmetic coefficients are not quantizable")
        self.raw_coefficients = quantized.astype(mx.float32)
        self.tissue_sha256 = hashlib.sha256(
            self.tissue.tissue_sha256.encode("ascii")
            + bytes(memoryview(self.raw_coefficients))
            + bytes(memoryview(self.tissue.harmonic_weights.astype(mx.float32)))
        ).hexdigest()
        self._learned_operation_count = 0

    def _learned_raw(self, operation: int, left: int, right: int) -> int:
        if (
            operation not in (_LEARNED_ADD, _LEARNED_MUL, _LEARNED_SUB)
            or type(left) is not int
            or type(right) is not int
            or not -MAX_PROCESS_INTEGER <= left <= MAX_PROCESS_INTEGER
            or not -MAX_PROCESS_INTEGER <= right <= MAX_PROCESS_INTEGER
        ):
            raise ValueError("semantic learned arithmetic request is invalid")
        coefficients = self.raw_coefficients[operation]
        left_value = mx.array(left, dtype=mx.float32)
        right_value = mx.array(right, dtype=mx.float32)
        result = (
            coefficients[0] * left_value
            + coefficients[1] * right_value
            + coefficients[2] * left_value * right_value
            + coefficients[3]
        )
        mx.eval(result)
        scalar = float(result.item())
        rounded = int(round(scalar))
        if not math.isfinite(scalar) or abs(scalar - rounded) > 1e-3:
            raise RuntimeError("learned arithmetic left the exact integer manifold")
        self._learned_operation_count += 1
        return rounded

    def _learned_radix_residue(self, value: int) -> int:
        candidates = mx.arange(PROCESS_RADIX, dtype=mx.float32)
        phase = (float(value) - candidates) / float(PROCESS_RADIX)
        logits = mx.zeros_like(phase)
        for harmonic in range(1, self.tissue.config.harmonic_count + 1):
            logits = logits + self.tissue.harmonic_weights[harmonic - 1] * mx.cos(
                2.0 * math.pi * harmonic * phase
            )
        residue = mx.argmax(logits)
        mx.eval(residue)
        return int(residue.item())

    def _learned_exact_quotient(self, numerator: int, denominator: int) -> int:
        if not 0 < denominator <= MAX_PROCESS_INTEGER or not 0 <= numerator <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic learned division request is invalid")
        candidates = mx.arange(MAX_PROCESS_INTEGER + 1, dtype=mx.float32)
        coefficients = self.raw_coefficients[_LEARNED_MUL]
        products = (
            coefficients[0] * candidates
            + coefficients[1] * float(denominator)
            + coefficients[2] * candidates * float(denominator)
            + coefficients[3]
        )
        distance = mx.abs(products - float(numerator))
        quotient = mx.argmin(distance)
        minimum = mx.min(distance)
        mx.eval(quotient, minimum)
        if float(minimum.item()) > 1e-3:
            raise ValueError("semantic learned division is not exact")
        self._learned_operation_count += 1
        return int(quotient.item())

    def _learned_floor_quotient(self, numerator: int, denominator: int) -> int:
        if not 0 < denominator <= MAX_PROCESS_INTEGER or not 0 <= numerator <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic learned quotient request is invalid")
        candidates = mx.arange(MAX_PROCESS_INTEGER + 1, dtype=mx.float32)
        coefficients = self.raw_coefficients[_LEARNED_MUL]
        products = (
            coefficients[0] * candidates
            + coefficients[1] * float(denominator)
            + coefficients[2] * candidates * float(denominator)
            + coefficients[3]
        )
        valid = products <= float(numerator) + 1e-3
        selected = mx.max(mx.where(valid, candidates, mx.array(-1.0)))
        mx.eval(selected)
        quotient = int(selected.item())
        if quotient < 0:
            raise RuntimeError("semantic learned floor quotient has no candidate")
        self._learned_operation_count += 1
        return quotient

    def _split_pair(self, value: int) -> tuple[int, int]:
        if not 0 <= value <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic neural pair result is outside the register bank")
        low = self._learned_radix_residue(value)
        high = self._learned_exact_quotient(value - low, PROCESS_RADIX)
        if self._learned_add(self._learned_raw(_LEARNED_MUL, high, PROCESS_RADIX), low) != value:
            raise RuntimeError("learned radix decomposition failed reconstruction")
        return low, high

    def _learned_add(self, left: int, right: int) -> int:
        return self._learned_raw(_LEARNED_ADD, left, right)

    def decode_unsigned_pair(self, low: int, high: int) -> int:
        """Read one radix pair through the learned arithmetic surface."""

        if not all(type(value) is int and 0 <= value < PROCESS_RADIX for value in (low, high)):
            raise ValueError("semantic neural radix pair is invalid")
        return self._learned_add(
            low,
            self._learned_raw(_LEARNED_MUL, high, PROCESS_RADIX),
        )

    def decode_signed_pair(self, low: int, high: int) -> int:
        """Read one public zigzag-encoded pair after learned radix recovery."""

        return self._signed_decode(self.decode_unsigned_pair(low, high))

    def learned_l1(self, values: tuple[int, ...]) -> int:
        """Reduce signed values with learned addition and subtraction."""

        if not isinstance(values, tuple) or any(type(value) is not int for value in values):
            raise ValueError("semantic neural L1 values are invalid")
        total = 0
        for value in values:
            magnitude = value if value >= 0 else self._learned_raw(_LEARNED_SUB, 0, value)
            total = self._learned_add(total, magnitude)
        return total

    @staticmethod
    def _signed_decode(value: int) -> int:
        return value // 2 if value % 2 == 0 else -((value + 1) // 2)

    @staticmethod
    def _signed_encode(value: int) -> int:
        encoded = 2 * value if value >= 0 else (-2 * value) - 1
        if not 0 <= encoded <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic signed result exceeds the register pair")
        return encoded

    def transition(
        self,
        state: tuple[int, ...],
        action: tuple[int, ...],
    ) -> SemanticNeuralTransition:
        if (
            len(state) != 11
            or len(action) != 8
            or any(type(value) is not int or not 0 <= value <= STATE_INVALID for value in state)
            or any(type(value) is not int or not 0 <= value <= ACTION_NULL for value in action)
            or state[-1] != 0
            or not 16 <= action[0] <= MAX_RECURRENT_OPCODE
        ):
            raise ValueError("semantic neural transition request is invalid")
        self._learned_operation_count = 0
        opcode = action[0]
        arg0, arg1, arg2, arg3, arg4, arg5 = action[1:7]
        terminal = action[-1]
        values = list(state[1:-1])

        def read(slot: int) -> int:
            if not 0 <= slot < len(values):
                raise ValueError("semantic neural value address is invalid")
            return values[slot]

        def pair(low: int) -> int:
            if not 0 <= low < len(values) - 1:
                raise ValueError("semantic neural pair address is invalid")
            return self._learned_add(
                read(low),
                self._learned_raw(_LEARNED_MUL, read(low + 1), PROCESS_RADIX),
            )

        def write_pair(low: int, result: int) -> None:
            if not 0 <= low < len(values) - 1:
                raise ValueError("semantic neural pair destination is invalid")
            values[low], values[low + 1] = self._split_pair(result)

        if opcode == OP_PAIR_SET:
            write_pair(arg0, self._learned_add(arg1, self._learned_raw(_LEARNED_MUL, arg2, PROCESS_RADIX)))
        elif opcode == OP_PAIR_ADD:
            write_pair(arg0, self._learned_add(pair(arg1), pair(arg2)))
        elif opcode == OP_PAIR_MUL_IMMEDIATE:
            write_pair(arg0, self._learned_raw(_LEARNED_MUL, pair(arg0), arg1))
        elif opcode == OP_PAIR_PRODUCT:
            write_pair(arg0, self._learned_raw(_LEARNED_MUL, arg1, arg2))
        elif opcode == OP_PAIR_SUB_IMMEDIATE:
            result = self._learned_raw(_LEARNED_SUB, pair(arg0), arg1)
            if result < 0:
                raise ValueError("semantic neural unsigned subtraction underflowed")
            write_pair(arg0, result)
        elif opcode == OP_PAIR_SIGNED_SUB_IMMEDIATE:
            write_pair(
                arg0,
                self._signed_encode(self._learned_raw(_LEARNED_SUB, pair(arg0), arg1)),
            )
        elif opcode == OP_PAIR_COPY:
            write_pair(arg0, pair(arg1))
        elif opcode == OP_PAIR_EUCLID_STEP:
            left = pair(arg0)
            right = pair(arg1)
            if right:
                quotient = self._learned_floor_quotient(left, right)
                product = self._learned_raw(_LEARNED_MUL, quotient, right)
                remainder = self._learned_raw(_LEARNED_SUB, left, product)
                write_pair(arg0, right)
                write_pair(arg1, remainder)
            else:
                write_pair(arg0, left)
                write_pair(arg1, 0)
        elif opcode == OP_PAIR_DIV:
            write_pair(arg0, self._learned_exact_quotient(pair(arg1), pair(arg2)))
        elif opcode in {OP_RATIO_CHOICE, OP_RATIO_BAND}:
            numerator = pair(arg1)
            denominator = pair(arg2)
            if denominator == 0 or not 0 <= arg0 < len(values):
                raise ValueError("semantic neural ratio request is invalid")
            twice = self._learned_raw(_LEARNED_MUL, 2, numerator)
            if opcode == OP_RATIO_CHOICE:
                values[arg0] = int(twice >= denominator) + 1
            else:
                ten = self._learned_raw(_LEARNED_MUL, 10, numerator)
                values[arg0] = (
                    1
                    if twice < denominator
                    else 2
                    if ten < self._learned_raw(_LEARNED_MUL, 7, denominator)
                    else 3
                    if ten < self._learned_raw(_LEARNED_MUL, 9, denominator)
                    else 4
                )
        elif opcode == OP_SIGNED_PAIR_ADD_IMMEDIATE:
            result = self._learned_add(
                self._signed_decode(pair(arg0)),
                self._signed_decode(arg1),
            )
            write_pair(arg0, self._signed_encode(result))
        elif opcode == OP_SIGNED_RANKED_GREATER:
            if not all(0 <= slot < len(values) for slot in (arg0, arg4, arg5)):
                raise ValueError("semantic neural ranked comparison address is invalid")
            candidate = self._signed_decode(pair(arg1))
            incumbent = self._signed_decode(pair(arg2))
            values[arg0] = int(
                values[arg5] == 0
                or candidate > incumbent
                or (candidate == incumbent and arg3 < values[arg4])
            )
        elif opcode == OP_RANKED_COMMIT:
            if not 0 <= arg0 < len(values):
                raise ValueError("semantic neural ranked commit flag is invalid")
            if values[arg0]:
                values[0] = arg2
                write_pair(1, pair(arg1))
                values[3] = arg3
            values[4] = 1
        elif opcode == OP_SET_SCALAR:
            if not 0 <= arg0 < len(values):
                raise ValueError("semantic neural scalar destination is invalid")
            values[arg0] = arg1
        else:  # pragma: no cover - schema range and opcode registry are closed.
            raise ValueError("semantic neural opcode is unsupported")

        next_state = (min(state[0] + 1, ACTION_NULL), *values, terminal)
        if any(not 0 <= value <= ACTION_NULL for value in next_state):
            raise RuntimeError("semantic neural transition left the state vocabulary")
        body = {
            "schema": SEMANTIC_NEURAL_MACHINE_SCHEMA,
            "state": state,
            "action": action,
            "tissue_sha256": self.tissue_sha256,
            "parent_tissue_sha256": self.tissue.tissue_sha256,
        }
        return SemanticNeuralTransition(
            next_state=next_state,
            opcode=opcode,
            learned_operation_count=self._learned_operation_count,
            tissue_sha256=self.tissue_sha256,
            input_sha256=_canonical_sha256(body),
        )


__all__ = [
    "SEMANTIC_NEURAL_MACHINE_SCHEMA",
    "SemanticNeuralMachine",
    "SemanticNeuralTransition",
]
