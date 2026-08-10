"""State/action supervision for Aura's native recurrent transition core."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from core.brain.llm.latent_cortex.recurrent_transition_core import (
    RecurrentTransitionCore,
)
from core.learning.recurrence_curriculum import StructuredTransitionProgram
from core.learning.recurrent_transition_supervision import (
    StateCodebookSpec,
    decode_trace_state,
    encode_trace_state,
    structured_state_loss,
)

NATIVE_TRANSITION_OBJECTIVE_SCHEMA = "aura.native_recurrent_transition_objective.v1"


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
class ActionCodebookSpec:
    schema: str = NATIVE_TRANSITION_OBJECTIVE_SCHEMA
    seed: int = 20260810184
    temperature: float = 0.125

    def __post_init__(self) -> None:
        if (
            self.schema != NATIVE_TRANSITION_OBJECTIVE_SCHEMA
            or type(self.seed) is not int
            or self.seed < 0
            or isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0.01 <= float(self.temperature) <= 2.0
        ):
            raise ValueError("action codebook specification is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NativeTransitionEvaluation:
    loss: float
    state_exact_fields: int
    state_field_count: int
    action_exact_fields: int
    action_field_count: int
    predicted_state: tuple[int, ...]
    expected_state: tuple[int, ...]
    predicted_action: tuple[int, ...]
    expected_action: tuple[int, ...]
    transition_index: int
    state_codebook_sha256: str
    action_codebook_sha256: str

    @property
    def state_exact(self) -> bool:
        return self.state_exact_fields == self.state_field_count

    @property
    def action_exact(self) -> bool:
        return self.action_exact_fields == self.action_field_count

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": NATIVE_TRANSITION_OBJECTIVE_SCHEMA,
            "loss": self.loss,
            "state_exact_fields": self.state_exact_fields,
            "state_field_count": self.state_field_count,
            "action_exact_fields": self.action_exact_fields,
            "action_field_count": self.action_field_count,
            "transition_index": self.transition_index,
            "state_codebook_sha256": self.state_codebook_sha256,
            "action_codebook_sha256": self.action_codebook_sha256,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class NativeTransitionGradient:
    value: float
    gradients: Any
    transition_index: int
    state_codebook_sha256: str
    action_codebook_sha256: str


def _action_class_count(*, family: str, field_name: str) -> int:
    classes = {
        ("boolean", "opcode"): 4,
        ("boolean", "operand"): 2,
        ("boolean", "has_operand"): 2,
        ("modular", "opcode"): 3,
        ("modular", "operand"): 23,
        ("modular", "modulus"): 24,
    }.get((family, field_name))
    if classes is None:
        raise ValueError(f"unsupported transition action field: {family}.{field_name}")
    return classes


def _action_codebook_matrix(
    *,
    family: str,
    field_name: str,
    classes: int,
    width: int,
    seed: int,
) -> Any:
    import mlx.core as mx

    material = f"{NATIVE_TRANSITION_OBJECTIVE_SCHEMA}:{seed}:{family}:{field_name}"
    key_seed = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:4], "big")
    matrix = mx.random.normal((classes, width), key=mx.random.key(key_seed)).astype(mx.float32)
    return matrix / mx.maximum(mx.linalg.norm(matrix, axis=-1, keepdims=True), 1e-6)


def _action_logits(
    action_features: Any,
    *,
    family: str,
    field_name: str,
    field_index: int,
    codebook: ActionCodebookSpec,
) -> Any:
    import mlx.core as mx

    if (
        action_features.ndim != 3
        or int(action_features.shape[0]) != 1
        or not 0 <= field_index < int(action_features.shape[1])
    ):
        raise ValueError("native action feature shape is invalid")
    classes = _action_class_count(family=family, field_name=field_name)
    matrix = _action_codebook_matrix(
        family=family,
        field_name=field_name,
        classes=classes,
        width=int(action_features.shape[-1]),
        seed=codebook.seed,
    )
    vector = action_features[0, field_index, :].astype(mx.float32)
    vector = vector / mx.maximum(mx.linalg.norm(vector), 1e-6)
    return (matrix @ vector) / float(codebook.temperature)


def encode_transition_action(
    program: StructuredTransitionProgram,
    *,
    transition_index: int,
    width: int,
    codebook: ActionCodebookSpec,
) -> Any:
    """Encode the typed causal action as an immutable core operand."""

    import mlx.core as mx

    action = _validate_program_target(program, transition_index)
    if type(width) is not int or width < 8:
        raise ValueError("native action width is invalid")
    rows = []
    for field_name, value in zip(program.action_field_names, action, strict=True):
        classes = _action_class_count(
            family=program.state_trace.family,
            field_name=field_name,
        )
        matrix = _action_codebook_matrix(
            family=program.state_trace.family,
            field_name=field_name,
            classes=classes,
            width=width,
            seed=codebook.seed,
        )
        rows.append(matrix[value])
    encoded = mx.stack(rows, axis=0)[None, :, :]
    mx.eval(encoded)
    return encoded


def _validate_program_target(
    program: StructuredTransitionProgram,
    transition_index: int,
) -> tuple[int, ...]:
    if (
        not isinstance(program, StructuredTransitionProgram)
        or type(transition_index) is not int
        or not 0 <= transition_index < program.state_trace.depth
        or len(program.action_field_names) != 3
    ):
        raise ValueError("native transition target is invalid")
    action = program.actions[transition_index]
    for field_name, value in zip(program.action_field_names, action, strict=True):
        if not 0 <= value < _action_class_count(
            family=program.state_trace.family,
            field_name=field_name,
        ):
            raise ValueError("native transition action exceeds its codebook")
    return action


def native_transition_loss(
    core: RecurrentTransitionCore,
    base_state: Any,
    context: Any,
    program: StructuredTransitionProgram,
    *,
    transition_index: int,
    state_codebook: StateCodebookSpec,
    action_codebook: ActionCodebookSpec,
    action_weight: float = 0.5,
) -> Any:
    """Score one shared native transition against exact action and next state."""

    import mlx.core as mx
    import mlx.nn as nn

    if (
        isinstance(action_weight, bool)
        or not isinstance(action_weight, (int, float))
        or not math.isfinite(float(action_weight))
        or not 0.0 <= float(action_weight) <= 10.0
    ):
        raise ValueError("native transition action weight is invalid")
    action = _validate_program_target(program, transition_index)
    trace = program.state_trace
    current = encode_trace_state(
        base_state,
        trace,
        state_index=transition_index,
        codebook=state_codebook,
    )
    encoded_action = encode_transition_action(
        program,
        transition_index=transition_index,
        width=core.config.bottleneck_size,
        codebook=action_codebook,
    )
    output = core(current, context, encoded_action)
    state_loss = structured_state_loss(
        output.state,
        trace,
        transition_index=transition_index,
        codebook=state_codebook,
    )
    action_losses = []
    for field_index, (field_name, value) in enumerate(
        zip(program.action_field_names, action, strict=True)
    ):
        logits = _action_logits(
            output.action_features,
            family=trace.family,
            field_name=field_name,
            field_index=field_index,
            codebook=action_codebook,
        )
        action_losses.append(
            nn.losses.cross_entropy(
                logits[None, :],
                mx.array([value], dtype=mx.int32),
                reduction="mean",
            )
        )
    return state_loss + float(action_weight) * (sum(action_losses) / len(action_losses))


def native_transition_value_and_grad(
    core: RecurrentTransitionCore,
    base_state: Any,
    context: Any,
    program: StructuredTransitionProgram,
    *,
    transition_index: int,
    state_codebook: StateCodebookSpec,
    action_codebook: ActionCodebookSpec,
    action_weight: float = 0.5,
) -> NativeTransitionGradient:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    def objective(candidate: RecurrentTransitionCore) -> Any:
        return native_transition_loss(
            candidate,
            base_state,
            context,
            program,
            transition_index=transition_index,
            state_codebook=state_codebook,
            action_codebook=action_codebook,
            action_weight=action_weight,
        )

    value, gradients = nn.value_and_grad(core, objective)(core)
    finite = [mx.all(mx.isfinite(item)) for _path, item in tree_flatten(gradients)]
    mx.eval(value, gradients, finite)
    scalar = float(value)
    if (
        not math.isfinite(scalar)
        or not finite
        or not all(bool(flag) for flag in finite)
    ):
        raise FloatingPointError("native transition gradient is non-finite")
    return NativeTransitionGradient(
        value=scalar,
        gradients=gradients,
        transition_index=transition_index,
        state_codebook_sha256=state_codebook.sha256,
        action_codebook_sha256=action_codebook.sha256,
    )


def evaluate_native_transition(
    core: RecurrentTransitionCore,
    base_state: Any,
    context: Any,
    program: StructuredTransitionProgram,
    *,
    transition_index: int,
    state_codebook: StateCodebookSpec,
    action_codebook: ActionCodebookSpec,
    action_weight: float = 0.5,
) -> NativeTransitionEvaluation:
    import mlx.core as mx

    action = _validate_program_target(program, transition_index)
    trace = program.state_trace
    current = encode_trace_state(
        base_state,
        trace,
        state_index=transition_index,
        codebook=state_codebook,
    )
    encoded_action = encode_transition_action(
        program,
        transition_index=transition_index,
        width=core.config.bottleneck_size,
        codebook=action_codebook,
    )
    output = core(current, context, encoded_action)
    loss = native_transition_loss(
        core,
        base_state,
        context,
        program,
        transition_index=transition_index,
        state_codebook=state_codebook,
        action_codebook=action_codebook,
        action_weight=action_weight,
    )
    predicted_state = decode_trace_state(
        output.state,
        trace,
        state_index=transition_index + 1,
        codebook=state_codebook,
    )
    predicted_action = tuple(
        int(
            mx.argmax(
                _action_logits(
                    output.action_features,
                    family=trace.family,
                    field_name=field_name,
                    field_index=field_index,
                    codebook=action_codebook,
                )
            ).item()
        )
        for field_index, field_name in enumerate(program.action_field_names)
    )
    mx.eval(loss)
    expected_state = trace.states[transition_index + 1]
    return NativeTransitionEvaluation(
        loss=float(loss),
        state_exact_fields=sum(
            left == right
            for left, right in zip(predicted_state, expected_state, strict=True)
        ),
        state_field_count=len(expected_state),
        action_exact_fields=sum(
            left == right for left, right in zip(predicted_action, action, strict=True)
        ),
        action_field_count=len(action),
        predicted_state=predicted_state,
        expected_state=expected_state,
        predicted_action=predicted_action,
        expected_action=action,
        transition_index=transition_index,
        state_codebook_sha256=state_codebook.sha256,
        action_codebook_sha256=action_codebook.sha256,
    )


__all__ = [
    "ActionCodebookSpec",
    "NATIVE_TRANSITION_OBJECTIVE_SCHEMA",
    "NativeTransitionEvaluation",
    "NativeTransitionGradient",
    "encode_transition_action",
    "evaluate_native_transition",
    "native_transition_loss",
    "native_transition_value_and_grad",
]
