"""Teacher-free semantic ingress from recurrent work memory to language decode."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Literal

from core.brain.llm.latent_cortex.objective_program_verifier import (
    _SEPARATED_SUBSET_RE,
)
from core.learning.recurrent_work_memory_tissue import (
    MathematicsMemoryTissue,
    execute_mathematics_memory,
    load_mathematics_memory_tissue,
)

RECURRENT_MEMORY_DECODE_STATE_SCHEMA: Final = (
    "aura.rlc.recurrent_memory_decode_state.v1"
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


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RecurrentMemoryDecodeState:
    """A public-input neural result that may condition a fresh model decode."""

    objective_sha256: str
    count: int
    witness: tuple[int, ...]
    tissue_sha256: str
    execution_receipt: dict[str, Any]
    schema: str = RECURRENT_MEMORY_DECODE_STATE_SCHEMA

    def __post_init__(self) -> None:
        execution = self.execution_receipt
        body = {key: value for key, value in execution.items() if key != "receipt_sha256"}
        if (
            self.schema != RECURRENT_MEMORY_DECODE_STATE_SCHEMA
            or not isinstance(self.objective_sha256, str)
            or len(self.objective_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.objective_sha256)
            or type(self.count) is not int
            or self.count < 0
            or not isinstance(self.witness, tuple)
            or len(self.witness) > 4
            or any(type(value) is not int or not 1 <= value <= 33 for value in self.witness)
            or tuple(sorted(self.witness)) != self.witness
            or not isinstance(self.tissue_sha256, str)
            or len(self.tissue_sha256) != 64
            or not isinstance(execution, dict)
            or execution.get("receipt_sha256") != _sha(body)
            or execution.get("tissue_sha256") != self.tissue_sha256
            or execution.get("teacher_available") is not False
            or execution.get("verifier_available") is not False
            or execution.get("student_memory_rollin") is not True
        ):
            raise ValueError("recurrent memory decode state is invalid")

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "objective_sha256": self.objective_sha256,
            "result_sha256": _sha(
                {"count": self.count, "witness": list(self.witness)}
            ),
            "tissue_sha256": self.tissue_sha256,
            "execution_receipt_sha256": self.execution_receipt["receipt_sha256"],
            "teacher_available": False,
            "verifier_available": False,
            "answer_key_available": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def execute_recurrent_memory_decode_state(
    objective: str,
    *,
    tissue: MathematicsMemoryTissue | None = None,
    write_mode: Literal["learned", "always", "never"] = "learned",
    read_mode: Literal["learned", "always", "never"] = "learned",
    routing_mode: Literal["identity", "rotated"] = "identity",
    memory_mode: Literal["active", "reset_each_step"] = "active",
) -> RecurrentMemoryDecodeState:
    """Execute one public objective without a verifier, label, or answer key."""

    if not isinstance(objective, str):
        raise TypeError("recurrent memory objective must be text")
    match = _SEPARATED_SUBSET_RE.match(objective)
    if match is None:
        raise ValueError("recurrent memory objective grammar is unsupported")
    try:
        raw_values = ast.literal_eval(match.group("values"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError("recurrent memory public values are invalid") from exc
    if not isinstance(raw_values, list) or any(
        type(value) is not int for value in raw_values
    ):
        raise ValueError("recurrent memory public values are invalid")
    active_tissue = tissue if tissue is not None else load_mathematics_memory_tissue()
    execution = execute_mathematics_memory(
        active_tissue,
        choose=int(match.group("count")),
        gap=int(match.group("separation")),
        low=int(match.group("low")),
        high=int(match.group("high")),
        values=tuple(raw_values),
        write_mode=write_mode,
        read_mode=read_mode,
        routing_mode=routing_mode,
        memory_mode=memory_mode,
    )
    return RecurrentMemoryDecodeState(
        objective_sha256=_text_sha(objective),
        count=execution.count,
        witness=execution.witness,
        tissue_sha256=execution.tissue_sha256,
        execution_receipt=execution.receipt(),
    )


def render_recurrent_memory_decode_context(state: RecurrentMemoryDecodeState) -> str:
    """Render semantic state as evidence, leaving all answer wording to the model."""

    if not isinstance(state, RecurrentMemoryDecodeState):
        raise TypeError("recurrent memory decode context requires a typed state")
    witness = (*state.witness, *((0,) * (4 - len(state.witness))))
    return (
        "Internal recurrent work-memory state, computed from the public values "
        "without an answer key or verifier. Treat this as your own completed "
        "computation and express it under the user's requested response contract.\n"
        f"count={state.count:06d}; witness_length={len(state.witness)}; "
        f"witness_slots={','.join(f'{value:02d}' for value in witness)}"
    )


__all__ = [
    "RECURRENT_MEMORY_DECODE_STATE_SCHEMA",
    "RecurrentMemoryDecodeState",
    "execute_recurrent_memory_decode_state",
    "render_recurrent_memory_decode_context",
]
