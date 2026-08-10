"""Certified typed transitions for recurrence that must be exactly correct."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

CERTIFIED_TRANSITION_SCHEMA: Final = "aura.certified_typed_transition.v1"


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
class TypedTransitionInput:
    family: str
    depth: int
    field_names: tuple[str, ...]
    state: tuple[int, ...]
    action_field_names: tuple[str, ...]
    action: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.family
            or not self.family.replace("_", "").isalnum()
            or type(self.depth) is not int
            or not 1 <= self.depth <= 1_024
            or len(self.field_names) < 3
            or len(set(self.field_names)) != len(self.field_names)
            or len(self.state) != len(self.field_names)
            or not self.action_field_names
            or len(set(self.action_field_names)) != len(self.action_field_names)
            or len(self.action) != len(self.action_field_names)
            or any(type(value) is not int or value < 0 for value in self.state)
            or any(type(value) is not int or value < 0 for value in self.action)
        ):
            raise ValueError("typed transition input is invalid")
        pc = self.state[0]
        done = self.state[-1]
        if pc >= self.depth or done != 0:
            raise ValueError("typed transition input is already terminal")

    @property
    def input_sha256(self) -> str:
        return _canonical_sha256(self.private_payload())

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema": CERTIFIED_TRANSITION_SCHEMA,
            "family": self.family,
            "depth": self.depth,
            "field_names": list(self.field_names),
            "state": list(self.state),
            "action_field_names": list(self.action_field_names),
            "action": list(self.action),
        }


@dataclass(frozen=True, slots=True)
class TransitionFamily:
    family: str
    field_names: tuple[str, ...]
    action_field_names: tuple[str, ...]
    implementation: Callable[[TypedTransitionInput], tuple[int, ...]]
    implementation_id: str

    def __post_init__(self) -> None:
        if (
            not self.family
            or not callable(self.implementation)
            or not self.implementation_id
            or len(self.field_names) < 3
            or not self.action_field_names
        ):
            raise ValueError("transition family contract is invalid")


@dataclass(frozen=True, slots=True)
class CertifiedTransitionResult:
    family: str
    next_state: tuple[int, ...]
    input_sha256: str
    output_sha256: str
    implementation_id: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": CERTIFIED_TRANSITION_SCHEMA,
            "family": self.family,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "implementation_id": self.implementation_id,
            "exact": True,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def _boolean_transition(request: TypedTransitionInput) -> tuple[int, ...]:
    pc, value, _done = request.state
    opcode, operand, has_operand = request.action
    if value not in (0, 1) or operand not in (0, 1) or has_operand not in (0, 1):
        raise ValueError("Boolean transition value is outside {0,1}")
    if opcode == 0:
        if operand != 0 or has_operand != 0:
            raise ValueError("Boolean not action has an operand")
        next_value = 1 - value
    elif opcode in (1, 2, 3):
        if has_operand != 1:
            raise ValueError("binary Boolean action has no operand")
        if opcode == 1:
            next_value = value & operand
        elif opcode == 2:
            next_value = value | operand
        else:
            next_value = value ^ operand
    else:
        raise ValueError("Boolean opcode is unsupported")
    next_pc = pc + 1
    return (next_pc, next_value, int(next_pc == request.depth))


def _modular_transition(request: TypedTransitionInput) -> tuple[int, ...]:
    pc, residue, _done = request.state
    opcode, operand, modulus = request.action
    if not 2 <= modulus <= 256:
        raise ValueError("modulus is outside [2,256]")
    if not 0 <= residue < modulus or not 0 <= operand < modulus:
        raise ValueError("modular operand is outside its residue class")
    if opcode == 0:
        next_value = (residue + operand) % modulus
    elif opcode == 1:
        next_value = (residue * operand) % modulus
    elif opcode == 2:
        next_value = (residue - operand) % modulus
    else:
        raise ValueError("modular opcode is unsupported")
    next_pc = pc + 1
    return (next_pc, next_value, int(next_pc == request.depth))


BUILTIN_TRANSITION_FAMILIES: Final = (
    TransitionFamily(
        family="boolean",
        field_names=("pc", "value", "done"),
        action_field_names=("opcode", "operand", "has_operand"),
        implementation=_boolean_transition,
        implementation_id="boolean_truth_table.v1",
    ),
    TransitionFamily(
        family="modular",
        field_names=("pc", "residue", "done"),
        action_field_names=("opcode", "operand", "modulus"),
        implementation=_modular_transition,
        implementation_id="bounded_integer_modular_arithmetic.v1",
    ),
)


class CertifiedTransitionExecutor:
    """Closed-by-default registry for exact, extensible transition families."""

    def __init__(self, extensions: Sequence[TransitionFamily] = ()) -> None:
        if isinstance(extensions, (str, bytes)):
            raise TypeError("transition extensions must be structured families")
        registry: dict[str, TransitionFamily] = {}
        for contract in (*BUILTIN_TRANSITION_FAMILIES, *tuple(extensions)):
            if not isinstance(contract, TransitionFamily):
                raise TypeError("transition family has the wrong type")
            if contract.family in registry:
                raise ValueError(f"duplicate transition family: {contract.family}")
            registry[contract.family] = contract
        self._registry: Mapping[str, TransitionFamily] = MappingProxyType(registry)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry))

    def execute(self, request: TypedTransitionInput) -> CertifiedTransitionResult:
        if not isinstance(request, TypedTransitionInput):
            raise TypeError("typed transition request has the wrong type")
        contract = self._registry.get(request.family)
        if contract is None:
            raise ValueError(f"unsupported transition family: {request.family}")
        if (
            request.field_names != contract.field_names
            or request.action_field_names != contract.action_field_names
        ):
            raise ValueError("typed transition schema differs from its family")
        next_state = contract.implementation(request)
        if (
            not isinstance(next_state, tuple)
            or len(next_state) != len(request.state)
            or any(type(value) is not int or value < 0 for value in next_state)
            or next_state[0] != request.state[0] + 1
            or next_state[-1] != int(next_state[0] == request.depth)
        ):
            raise RuntimeError("transition implementation violated state invariants")
        output_sha256 = _canonical_sha256(
            {
                "schema": CERTIFIED_TRANSITION_SCHEMA,
                "family": request.family,
                "next_state": list(next_state),
            }
        )
        return CertifiedTransitionResult(
            family=request.family,
            next_state=next_state,
            input_sha256=request.input_sha256,
            output_sha256=output_sha256,
            implementation_id=contract.implementation_id,
        )


__all__ = [
    "BUILTIN_TRANSITION_FAMILIES",
    "CERTIFIED_TRANSITION_SCHEMA",
    "CertifiedTransitionExecutor",
    "CertifiedTransitionResult",
    "TransitionFamily",
    "TypedTransitionInput",
]
