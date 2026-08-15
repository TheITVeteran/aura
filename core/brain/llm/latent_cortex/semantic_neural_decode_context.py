"""Teacher-free semantic ingress from neural register execution to language."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.frontier_tasks import (
    FrontierTaskError,
    parse_final_answer,
)
from core.learning.public_frontier_action_compiler import (
    compile_public_frontier_actions,
    public_frontier_operands,
)
from core.learning.recurrent_action_schema import OP_SIGNED_PAIR_ADD_IMMEDIATE
from core.learning.semantic_neural_machine import SemanticNeuralMachine

SEMANTIC_NEURAL_DECODE_STATE_SCHEMA: Final = "aura.semantic_neural_decode_state.v1"
SEMANTIC_NEURAL_DECODE_CONTEXT_SCHEMA: Final = "aura.semantic_neural_decode_context.v1"
SUPPORTED_FAMILIES: Final = frozenset(
    {"frontier_coding", "frontier_calibration", "frontier_misleading_premise"}
)


def _sha(value: Any) -> str:
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
class SemanticNeuralDecodeState:
    objective_sha256: str
    family: str
    states: tuple[tuple[int, ...], ...]
    action_program_receipt: dict[str, Any]
    transition_receipts: tuple[dict[str, Any], ...]
    semantic_result: dict[str, Any]
    tissue_sha256: str
    schema: str = SEMANTIC_NEURAL_DECODE_STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SEMANTIC_NEURAL_DECODE_STATE_SCHEMA
            or self.family not in SUPPORTED_FAMILIES
            or len(self.objective_sha256) != 64
            or len(self.states) != len(self.transition_receipts) + 1
            or not self.transition_receipts
            or self.states[0] != (0,) * 11
            or self.states[-1][-1] != 1
            or self.action_program_receipt.get("verifier_answer_available") is not False
            or self.action_program_receipt.get("private_state_trace_available") is not False
            or self.action_program_receipt.get("receipt_sha256") != _sha(
                {
                    key: value
                    for key, value in self.action_program_receipt.items()
                    if key != "receipt_sha256"
                }
            )
            or any(
                receipt.get("teacher_available") is not False
                or receipt.get("private_trace_available") is not False
                or receipt.get("tissue_sha256") != self.tissue_sha256
                for receipt in self.transition_receipts
            )
        ):
            raise ValueError("semantic neural decode state is invalid")

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "objective_sha256": self.objective_sha256,
            "family": self.family,
            "state_trajectory_sha256": _sha(self.states),
            "semantic_result_sha256": _sha(self.semantic_result),
            "action_program_receipt_sha256": self.action_program_receipt[
                "receipt_sha256"
            ],
            "transition_receipt_sha256s": [
                receipt["receipt_sha256"] for receipt in self.transition_receipts
            ],
            "tissue_sha256": self.tissue_sha256,
            "student_rollin": True,
            "teacher_available": False,
            "private_trace_available": False,
            "verifier_available": False,
            "answer_key_available": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def _coding_result(
    machine: SemanticNeuralMachine,
    operands: dict[str, Any],
    actions: tuple[tuple[int, ...], ...],
    states: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    names = operands["names"]
    cases = operands["cases"]
    pressures = [[] for _case in cases]
    final_balances: list[tuple[int, ...] | None] = [None for _case in cases]
    for action, state in zip(actions, states[1:], strict=True):
        if action[0] != OP_SIGNED_PAIR_ADD_IMMEDIATE:
            continue
        case_index = state[1]
        balances = tuple(
            machine.decode_signed_pair(state[2 + 2 * index], state[3 + 2 * index])
            for index in range(len(names))
        )
        pressures[case_index].append(machine.learned_l1(balances))
        final_balances[case_index] = balances
    if any(value is None for value in final_balances) or any(
        len(pressure) != len(case) for pressure, case in zip(pressures, cases, strict=True)
    ):
        raise RuntimeError("semantic coding trajectory omitted a public event")
    returns = []
    for balances, pressure in zip(final_balances, pressures, strict=True):
        assert balances is not None
        returns.append(
            {
                "state": [
                    [name, value]
                    for name, value in zip(names, balances, strict=True)
                    if value != 0
                ],
                "pressure": pressure,
            }
        )
    return {"returns": returns, "time_complexity": "O(n^2)"}


def _calibration_result(
    machine: SemanticNeuralMachine,
    terminal: tuple[int, ...],
) -> dict[str, Any]:
    numerator = machine.decode_unsigned_pair(terminal[1], terminal[2])
    denominator = machine.decode_unsigned_pair(terminal[3], terminal[4])
    choices = ("not_H", "H")
    bands = ("below_50", "50_to_69", "70_to_89", "90_to_100")
    if not 1 <= terminal[5] <= len(choices) or not 1 <= terminal[6] <= len(bands):
        raise RuntimeError("semantic calibration state is outside its public codebook")
    return {
        "choice": choices[terminal[5] - 1],
        "posterior": f"{numerator}/{denominator}",
        "confidence_band": bands[terminal[6] - 1],
    }


def _premise_result(
    machine: SemanticNeuralMachine,
    operands: dict[str, Any],
    terminal: tuple[int, ...],
) -> dict[str, Any]:
    rows = operands["rows"]
    winner_index = terminal[1]
    if not 0 <= winner_index < len(rows) or terminal[5] != 1:
        raise RuntimeError("semantic premise state has no admitted winner")
    winner = rows[winner_index]["name"]
    score = machine.decode_signed_pair(terminal[2], terminal[3])
    return {
        "premise_valid": operands["claim"] == winner,
        "actual_winner": winner,
        "actual_score": score,
    }


def execute_semantic_neural_decode_state(
    objective: str,
    family: str,
    *,
    machine: SemanticNeuralMachine | None = None,
) -> SemanticNeuralDecodeState:
    """Execute one public semantic objective without a trace or answer key."""

    if not isinstance(objective, str) or family not in SUPPORTED_FAMILIES:
        raise ValueError("semantic neural decode objective is invalid")
    active = SemanticNeuralMachine() if machine is None else machine
    program = compile_public_frontier_actions(objective, family)
    operands = public_frontier_operands(objective, family)
    states = [(0,) * 11]
    receipts = []
    for action in program.values:
        result = active.transition(states[-1], action)
        states.append(result.next_state)
        receipts.append(result.receipt())
    trajectory = tuple(states)
    semantic_result = (
        _coding_result(active, operands, program.values, trajectory)
        if family == "frontier_coding"
        else _calibration_result(active, trajectory[-1])
        if family == "frontier_calibration"
        else _premise_result(active, operands, trajectory[-1])
    )
    return SemanticNeuralDecodeState(
        objective_sha256=hashlib.sha256(objective.encode("utf-8")).hexdigest(),
        family=family,
        states=trajectory,
        action_program_receipt=program.receipt(),
        transition_receipts=tuple(receipts),
        semantic_result=semantic_result,
        tissue_sha256=active.tissue_sha256,
    )


def render_semantic_neural_decode_context(
    state: SemanticNeuralDecodeState,
) -> str:
    """Render authenticated neural state while leaving wording to the model."""

    if not isinstance(state, SemanticNeuralDecodeState):
        raise TypeError("semantic neural decode context requires a typed state")
    return (
        "Internal recurrent semantic state, computed from the public objective "
        "without an answer key, private trace, or verifier. Treat this typed "
        "state as your own completed computation. The semantic_result object is "
        "the canonical result of that computation: emit that object exactly, "
        "byte for byte, after the supplied FINAL_ANSWER syntax prefix. Do not "
        "describe it, rename fields, emit a scalar shortcut, or add prose.\n"
        "semantic_result="
        + json.dumps(
            state.semantic_result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def semantic_result_matches_response(
    state: SemanticNeuralDecodeState,
    response: str,
) -> bool:
    """Check serialization against authenticated state, never an answer key."""

    if not isinstance(state, SemanticNeuralDecodeState) or not isinstance(response, str):
        return False
    try:
        return parse_final_answer(response) == state.semantic_result
    except FrontierTaskError:
        return False


def render_semantic_neural_decode_correction(
    state: SemanticNeuralDecodeState,
) -> str:
    """Request one clean reserialization without anchoring on failed text."""

    return (
        render_semantic_neural_decode_context(state)
        + "\nThe preceding serialization did not equal this authenticated object. "
        "Discard that serialization completely and copy semantic_result again."
    )


__all__ = [
    "SEMANTIC_NEURAL_DECODE_CONTEXT_SCHEMA",
    "SEMANTIC_NEURAL_DECODE_STATE_SCHEMA",
    "SemanticNeuralDecodeState",
    "execute_semantic_neural_decode_state",
    "render_semantic_neural_decode_context",
    "render_semantic_neural_decode_correction",
    "semantic_result_matches_response",
]
