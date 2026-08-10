"""Compose certified typed transitions from student-produced state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.typed_action_compiler import TypedActionProgram
from core.brain.llm.latent_cortex.typed_transition_executor import (
    CertifiedTransitionExecutor,
    TypedTransitionInput,
)

CERTIFIED_PROGRAM_EXECUTION_SCHEMA = "aura.certified_program_execution.v1"


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
class CertifiedProgramExecution:
    family: str
    depth: int
    states: tuple[tuple[int, ...], ...]
    transition_receipts: tuple[Mapping[str, Any], ...]
    chain_sha256: str

    @property
    def terminal_state(self) -> tuple[int, ...]:
        return self.states[-1]

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": CERTIFIED_PROGRAM_EXECUTION_SCHEMA,
            "family": self.family,
            "depth": self.depth,
            "state_count": len(self.states),
            "transition_count": len(self.transition_receipts),
            "initial_state_sha256": _canonical_sha256(list(self.states[0])),
            "terminal_state_sha256": _canonical_sha256(list(self.states[-1])),
            "transition_receipt_sha256s": [
                receipt["receipt_sha256"] for receipt in self.transition_receipts
            ],
            "chain_sha256": self.chain_sha256,
            "student_rollin": True,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def execute_action_sequence(
    *,
    family: str,
    depth: int,
    field_names: tuple[str, ...],
    initial_state: tuple[int, ...],
    action_field_names: tuple[str, ...],
    actions: Sequence[tuple[int, ...]],
    executor: CertifiedTransitionExecutor | None = None,
) -> CertifiedProgramExecution:
    """Execute every step from the prior predicted state, never a teacher state."""

    if (
        isinstance(actions, (str, bytes))
        or type(depth) is not int
        or depth < 1
        or len(actions) != depth
    ):
        raise ValueError("certified action sequence is invalid")
    machine = executor or CertifiedTransitionExecutor()
    states = [initial_state]
    receipts: list[Mapping[str, Any]] = []
    chain = _canonical_sha256(
        {
            "schema": CERTIFIED_PROGRAM_EXECUTION_SCHEMA,
            "family": family,
            "depth": depth,
            "initial_state": list(initial_state),
        }
    )
    for action in actions:
        result = machine.execute(
            TypedTransitionInput(
                family=family,
                depth=depth,
                field_names=field_names,
                state=states[-1],
                action_field_names=action_field_names,
                action=action,
            )
        )
        receipt = result.receipt()
        states.append(result.next_state)
        receipts.append(receipt)
        chain = _canonical_sha256(
            {
                "prior_chain_sha256": chain,
                "transition_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    return CertifiedProgramExecution(
        family=family,
        depth=depth,
        states=tuple(states),
        transition_receipts=tuple(receipts),
        chain_sha256=chain,
    )


def execute_compiled_action_program(
    program: TypedActionProgram,
    *,
    executor: CertifiedTransitionExecutor | None = None,
) -> CertifiedProgramExecution:
    if not isinstance(program, TypedActionProgram):
        raise TypeError("compiled action program has the wrong type")
    return execute_action_sequence(
        family=program.family,
        depth=program.depth,
        field_names=program.field_names,
        initial_state=program.initial_state,
        action_field_names=program.action_field_names,
        actions=program.actions,
        executor=executor,
    )


__all__ = [
    "CERTIFIED_PROGRAM_EXECUTION_SCHEMA",
    "CertifiedProgramExecution",
    "execute_action_sequence",
    "execute_compiled_action_program",
]
