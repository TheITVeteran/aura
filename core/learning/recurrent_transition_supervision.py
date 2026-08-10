"""Exact state supervision for one shared RLC recurrent transition.

The decoder is a fixed, hash-derived codebook rather than a learned head. A
jointly trained head could reduce its own loss while leaving recurrence
unchanged; this boundary forces the recurrent operator itself to place each
declared state field into a stable, independently decodable region.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_curriculum import StructuredTransitionTrace
from core.learning.recurrence_native_objective_v2 import (
    PreparedRecurrentTransitionInput,
    execute_prepared_recurrent_transition,
)

RECURRENT_TRANSITION_SUPERVISION_SCHEMA = (
    "aura.recurrent_transition_supervision.v1"
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


@dataclass(frozen=True, slots=True)
class StateCodebookSpec:
    """Public deterministic encoding for private exact machine-state labels."""

    schema: str = RECURRENT_TRANSITION_SUPERVISION_SCHEMA
    seed: int = 20260810179
    max_program_depth: int = 4
    residue_classes: int = 23
    field_slot_indices: tuple[int, int, int] = (-3, -2, -1)
    temperature: float = 0.125

    def __post_init__(self) -> None:
        if (
            self.schema != RECURRENT_TRANSITION_SUPERVISION_SCHEMA
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.max_program_depth) is not int
            or not 1 <= self.max_program_depth <= 64
            or type(self.residue_classes) is not int
            or not 2 <= self.residue_classes <= 256
            or len(self.field_slot_indices) != 3
            or len(set(self.field_slot_indices)) != 3
            or any(type(index) is not int or index >= 0 for index in self.field_slot_indices)
            or isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0.01 <= float(self.temperature) <= 2.0
        ):
            raise ValueError("state codebook specification is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["field_slot_indices"] = list(self.field_slot_indices)
        return payload


@dataclass(frozen=True, slots=True)
class StateTransitionEvaluation:
    loss: float
    exact_fields: int
    field_count: int
    predicted: tuple[int, ...]
    expected: tuple[int, ...]
    transition_index: int
    codebook_sha256: str
    execution_spec_sha256: str

    @property
    def exact(self) -> bool:
        return self.exact_fields == self.field_count

    def receipt(self) -> dict[str, Any]:
        """Publish aggregate correctness without exposing private state values."""

        body = {
            "schema": RECURRENT_TRANSITION_SUPERVISION_SCHEMA,
            "loss": self.loss,
            "exact_fields": self.exact_fields,
            "field_count": self.field_count,
            "transition_index": self.transition_index,
            "codebook_sha256": self.codebook_sha256,
            "execution_spec_sha256": self.execution_spec_sha256,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class StateTransitionGradient:
    value: float
    gradients: Any
    transition_index: int
    codebook_sha256: str
    execution_spec_sha256: str


def _field_class_count(
    *,
    family: str,
    field_name: str,
    codebook: StateCodebookSpec,
) -> int:
    if field_name == "pc":
        return codebook.max_program_depth + 1
    if field_name == "done":
        return 2
    if family == "boolean" and field_name == "value":
        return 2
    if family == "modular" and field_name == "residue":
        return codebook.residue_classes
    raise ValueError(f"unsupported structured state field: {family}.{field_name}")


def _codebook_matrix(
    *,
    family: str,
    field_name: str,
    classes: int,
    hidden_size: int,
    seed: int,
) -> Any:
    import mlx.core as mx

    material = f"{RECURRENT_TRANSITION_SUPERVISION_SCHEMA}:{seed}:{family}:{field_name}"
    key_seed = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:4], "big")
    matrix = mx.random.normal((classes, hidden_size), key=mx.random.key(key_seed)).astype(
        mx.float32
    )
    return matrix / mx.maximum(mx.linalg.norm(matrix, axis=-1, keepdims=True), 1e-6)


def _field_logits(
    state: Any,
    *,
    family: str,
    field_name: str,
    slot_index: int,
    codebook: StateCodebookSpec,
) -> Any:
    import mlx.core as mx

    if state.ndim != 3 or int(state.shape[0]) != 1:
        raise ValueError("recurrent state must have shape (1, slots, hidden)")
    resolved_slot = int(state.shape[1]) + slot_index
    if not 0 <= resolved_slot < int(state.shape[1]):
        raise ValueError("state codebook slot is outside the recurrent workspace")
    classes = _field_class_count(
        family=family,
        field_name=field_name,
        codebook=codebook,
    )
    matrix = _codebook_matrix(
        family=family,
        field_name=field_name,
        classes=classes,
        hidden_size=int(state.shape[-1]),
        seed=codebook.seed,
    )
    vector = state[0, resolved_slot, :].astype(mx.float32)
    vector = vector / mx.maximum(mx.linalg.norm(vector), 1e-6)
    return (matrix @ vector) / float(codebook.temperature)


def _validate_target(
    trace: StructuredTransitionTrace,
    *,
    transition_index: int,
    codebook: StateCodebookSpec,
) -> tuple[int, ...]:
    if not isinstance(trace, StructuredTransitionTrace):
        raise TypeError("transition trace has the wrong type")
    if (
        type(transition_index) is not int
        or transition_index < 0
        or transition_index >= trace.depth
        or trace.depth > codebook.max_program_depth
        or len(trace.field_names) != len(codebook.field_slot_indices)
    ):
        raise ValueError("transition target is outside the codebook contract")
    target = trace.states[transition_index + 1]
    for field_name, value in zip(trace.field_names, target, strict=True):
        classes = _field_class_count(
            family=trace.family,
            field_name=field_name,
            codebook=codebook,
        )
        if not 0 <= value < classes:
            raise ValueError("transition target value exceeds its codebook")
    return target


def structured_state_loss(
    state: Any,
    trace: StructuredTransitionTrace,
    *,
    transition_index: int,
    codebook: StateCodebookSpec,
) -> Any:
    """Cross-entropy against a fixed codebook for the exact next state."""

    import mlx.core as mx
    import mlx.nn as nn

    target = _validate_target(
        trace,
        transition_index=transition_index,
        codebook=codebook,
    )
    losses = []
    for field_name, slot_index, value in zip(
        trace.field_names,
        codebook.field_slot_indices,
        target,
        strict=True,
    ):
        logits = _field_logits(
            state,
            family=trace.family,
            field_name=field_name,
            slot_index=slot_index,
            codebook=codebook,
        )
        losses.append(
            nn.losses.cross_entropy(
                logits[None, :],
                mx.array([value], dtype=mx.int32),
                reduction="mean",
            )
        )
    return sum(losses) / len(losses)


def encode_structured_state(
    state: Any,
    trace: StructuredTransitionTrace,
    *,
    transition_index: int,
    codebook: StateCodebookSpec,
) -> Any:
    """Replace only declared control slots with exact codebook prototypes."""

    _validate_target(
        trace,
        transition_index=transition_index,
        codebook=codebook,
    )
    return encode_trace_state(
        state,
        trace,
        state_index=transition_index + 1,
        codebook=codebook,
    )


def encode_trace_state(
    state: Any,
    trace: StructuredTransitionTrace,
    *,
    state_index: int,
    codebook: StateCodebookSpec,
) -> Any:
    """Place any exact trace state into the protected codebook slots."""

    import mlx.core as mx

    if (
        not isinstance(trace, StructuredTransitionTrace)
        or type(state_index) is not int
        or not 0 <= state_index <= trace.depth
        or trace.depth > codebook.max_program_depth
        or len(trace.field_names) != len(codebook.field_slot_indices)
    ):
        raise ValueError("trace state is outside the codebook contract")
    target = trace.states[state_index]
    rows = [state[:, index : index + 1, :] for index in range(int(state.shape[1]))]
    for field_name, slot_index, value in zip(
        trace.field_names,
        codebook.field_slot_indices,
        target,
        strict=True,
    ):
        resolved_slot = int(state.shape[1]) + slot_index
        classes = _field_class_count(
            family=trace.family,
            field_name=field_name,
            codebook=codebook,
        )
        if not 0 <= value < classes:
            raise ValueError("trace state value exceeds its codebook")
        matrix = _codebook_matrix(
            family=trace.family,
            field_name=field_name,
            classes=classes,
            hidden_size=int(state.shape[-1]),
            seed=codebook.seed,
        )
        prior = state[:, resolved_slot : resolved_slot + 1, :].astype(mx.float32)
        norm = mx.maximum(mx.linalg.norm(prior, axis=-1, keepdims=True), 1e-6)
        rows[resolved_slot] = (matrix[value][None, None, :] * norm).astype(state.dtype)
    return mx.concatenate(rows, axis=1)


def encode_trace_state_operand(
    trace: StructuredTransitionTrace,
    *,
    state_index: int,
    width: int,
    codebook: StateCodebookSpec,
) -> Any:
    """Encode exact current state in the native core's typed coordinates."""

    import mlx.core as mx

    if (
        not isinstance(trace, StructuredTransitionTrace)
        or type(state_index) is not int
        or not 0 <= state_index <= trace.depth
        or trace.depth > codebook.max_program_depth
        or len(trace.field_names) != len(codebook.field_slot_indices)
        or type(width) is not int
        or width < 8
    ):
        raise ValueError("typed trace state is outside the codebook contract")
    rows = []
    for field_name, value in zip(
        trace.field_names,
        trace.states[state_index],
        strict=True,
    ):
        classes = _field_class_count(
            family=trace.family,
            field_name=field_name,
            codebook=codebook,
        )
        if not 0 <= value < classes:
            raise ValueError("typed trace state value exceeds its codebook")
        matrix = _codebook_matrix(
            family=trace.family,
            field_name=field_name,
            classes=classes,
            hidden_size=width,
            seed=codebook.seed,
        )
        rows.append(matrix[value])
    encoded = mx.stack(rows, axis=0)[None, :, :]
    mx.eval(encoded)
    return encoded


def decode_structured_state(
    state: Any,
    trace: StructuredTransitionTrace,
    *,
    transition_index: int,
    codebook: StateCodebookSpec,
) -> tuple[int, ...]:
    """Read every declared field through the immutable codebook decoder."""

    import mlx.core as mx

    _validate_target(
        trace,
        transition_index=transition_index,
        codebook=codebook,
    )
    return tuple(
        int(
            mx.argmax(
                _field_logits(
                    state,
                    family=trace.family,
                    field_name=field_name,
                    slot_index=slot_index,
                    codebook=codebook,
                )
            ).item()
        )
        for field_name, slot_index in zip(
            trace.field_names,
            codebook.field_slot_indices,
            strict=True,
        )
    )


def decode_trace_state(
    state: Any,
    trace: StructuredTransitionTrace,
    *,
    state_index: int,
    codebook: StateCodebookSpec,
) -> tuple[int, ...]:
    """Decode a trace state without implying that a transition produced it."""

    import mlx.core as mx

    if (
        not isinstance(trace, StructuredTransitionTrace)
        or type(state_index) is not int
        or not 0 <= state_index <= trace.depth
        or trace.depth > codebook.max_program_depth
        or len(trace.field_names) != len(codebook.field_slot_indices)
    ):
        raise ValueError("trace state is outside the codebook contract")
    return tuple(
        int(
            mx.argmax(
                _field_logits(
                    state,
                    family=trace.family,
                    field_name=field_name,
                    slot_index=slot_index,
                    codebook=codebook,
                )
            ).item()
        )
        for field_name, slot_index in zip(
            trace.field_names,
            codebook.field_slot_indices,
            strict=True,
        )
    )


def state_supervised_transition_loss(
    model: Any,
    prepared: PreparedRecurrentTransitionInput,
    trace: StructuredTransitionTrace,
    *,
    spec: RLCExecutionSpec,
    codebook: StateCodebookSpec,
) -> Any:
    """Run one exact live update and score only its declared machine state."""

    if len(spec.branch_roles) != 1:
        raise ValueError("state-supervised discriminator requires exactly one branch")
    if prepared.transition_index >= trace.depth:
        raise ValueError("prepared transition exceeds the exact trace")
    children = execute_prepared_recurrent_transition(model, prepared, spec=spec)
    if len(children) != 1:
        raise RuntimeError("state-supervised transition produced multiple branches")
    return structured_state_loss(
        children[0],
        trace,
        transition_index=prepared.transition_index,
        codebook=codebook,
    )


def state_supervised_transition_value_and_grad(
    model: Any,
    prepared: PreparedRecurrentTransitionInput,
    trace: StructuredTransitionTrace,
    *,
    spec: RLCExecutionSpec,
    codebook: StateCodebookSpec,
) -> StateTransitionGradient:
    """Differentiate exact next-state loss only into the recurrent operator."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    def objective(current_model: Any) -> Any:
        return state_supervised_transition_loss(
            current_model,
            prepared,
            trace,
            spec=spec,
            codebook=codebook,
        )

    value, gradients = nn.value_and_grad(model, objective)(model)
    finite = [mx.all(mx.isfinite(item)) for _path, item in tree_flatten(gradients)]
    mx.eval(value, gradients, finite)
    scalar = float(value)
    if (
        not math.isfinite(scalar)
        or not finite
        or not all(bool(flag) for flag in finite)
    ):
        raise FloatingPointError("state-supervised transition gradient is non-finite")
    return StateTransitionGradient(
        value=scalar,
        gradients=gradients,
        transition_index=prepared.transition_index,
        codebook_sha256=codebook.sha256,
        execution_spec_sha256=spec.sha256,
    )


def evaluate_state_supervised_transition(
    model: Any,
    prepared: PreparedRecurrentTransitionInput,
    trace: StructuredTransitionTrace,
    *,
    spec: RLCExecutionSpec,
    codebook: StateCodebookSpec,
) -> StateTransitionEvaluation:
    """Decode one child with the fixed codebook and report exact field count."""

    import mlx.core as mx

    target = _validate_target(
        trace,
        transition_index=prepared.transition_index,
        codebook=codebook,
    )
    children = execute_prepared_recurrent_transition(model, prepared, spec=spec)
    state = children[0]
    loss = structured_state_loss(
        state,
        trace,
        transition_index=prepared.transition_index,
        codebook=codebook,
    )
    predictions = decode_structured_state(
        state,
        trace,
        transition_index=prepared.transition_index,
        codebook=codebook,
    )
    mx.eval(loss)
    predicted = tuple(predictions)
    return StateTransitionEvaluation(
        loss=float(loss),
        exact_fields=sum(left == right for left, right in zip(predicted, target, strict=True)),
        field_count=len(target),
        predicted=predicted,
        expected=target,
        transition_index=prepared.transition_index,
        codebook_sha256=codebook.sha256,
        execution_spec_sha256=spec.sha256,
    )


__all__ = [
    "RECURRENT_TRANSITION_SUPERVISION_SCHEMA",
    "StateCodebookSpec",
    "StateTransitionEvaluation",
    "StateTransitionGradient",
    "decode_trace_state",
    "decode_structured_state",
    "encode_trace_state",
    "encode_trace_state_operand",
    "encode_structured_state",
    "evaluate_state_supervised_transition",
    "state_supervised_transition_loss",
    "state_supervised_transition_value_and_grad",
    "structured_state_loss",
]
