"""One intrinsic recurrent forward with depth, memory, correction, and halting.

The repository previously held these mechanisms in separate execution
architectures. This module makes them act on the same resident-transformer
trajectory. It is additive and identity-initialized: one iteration remains the
base forward until learned controller parameters are admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn

from core.learning.intrinsic_recurrence import (
    RecurrentDepthPlan,
    _rms,
    _run,
    recurrent_iteration,
)
from core.learning.protected_memory import (
    MemoryLayout,
    apply_protected_transition,
    memory_retention,
    semantic_convergence,
)
from core.learning.recurrent_action_schema import (
    ACTION_CARDINALITY,
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_COPY_VALUE,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_CALIBRATE,
    OP_FRONTIER_ENUMERATE,
    OP_FRONTIER_INFER,
    OP_FRONTIER_SCHEDULE,
    OP_FRONTIER_SIMULATE,
    OP_FRONTIER_TRAVERSE,
    OP_MUL_MOD,
    OP_REGISTER_AFFINE,
    OP_SUB_MOD,
)
from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract
from core.learning.recurrent_literal_grounding import (
    LITERAL_MAX_VALUE,
    LiteralObservationContract,
)
from core.learning.recurrent_opcode_grounding import OpcodeObservationContract
from core.learning.recurrent_state_schema import (
    STATE_CARDINALITY,
    STATE_SLOT_NAMES,
)

UNIFIED_INTRINSIC_RECURRENCE_SCHEMA: Final = "aura.unified_intrinsic_recurrence.v1"
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1


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
class UnifiedRecurrenceConfig:
    hidden_size: int
    correction_rank: int = 8
    depth_basis_size: int = 4
    memory_write_threshold: float = 0.5
    halt_threshold: float = 0.9
    minimum_iterations: int = 2
    state_slots: int = len(STATE_SLOT_NAMES)
    state_cardinality: int = STATE_CARDINALITY
    action_slots: int = len(ACTION_SLOT_NAMES)
    action_cardinality: int = ACTION_CARDINALITY
    literal_digit_token_ids: tuple[int, ...] = ()
    opcode_token_patterns: tuple[tuple[int, tuple[int, ...]], ...] = ()
    opcode_context_patterns: tuple[tuple[str, tuple[int, ...]], ...] = ()
    initialization_seed: int = 20260810198
    schema: str = UNIFIED_INTRINSIC_RECURRENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNIFIED_INTRINSIC_RECURRENCE_SCHEMA:
            raise ValueError("unified recurrence schema differs")
        for name in (
            "hidden_size",
            "correction_rank",
            "depth_basis_size",
            "state_slots",
            "state_cardinality",
            "action_slots",
            "action_cardinality",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.correction_rank > self.hidden_size:
            raise ValueError("correction rank exceeds hidden size")
        if self.state_slots != len(STATE_SLOT_NAMES):
            raise ValueError("state slot count differs from the canonical schema")
        if self.state_cardinality != STATE_CARDINALITY:
            raise ValueError("state cardinality differs from the canonical schema")
        if self.action_slots != len(ACTION_SLOT_NAMES):
            raise ValueError("action slot count differs from the canonical schema")
        if self.action_cardinality != ACTION_CARDINALITY:
            raise ValueError("action cardinality differs from the canonical schema")
        if self.literal_digit_token_ids and (
            len(self.literal_digit_token_ids) != 10
            or len(set(self.literal_digit_token_ids)) != 10
            or any(
                type(value) is not int or value < 0
                for value in self.literal_digit_token_ids
            )
        ):
            raise ValueError("literal digit token identity is invalid")
        if bool(self.opcode_token_patterns) != bool(self.opcode_context_patterns):
            raise ValueError("opcode pattern and context contracts differ")
        if self.opcode_token_patterns:
            OpcodeObservationContract(
                self.opcode_token_patterns,
                self.opcode_context_patterns,
            )
        if type(self.minimum_iterations) is not int or self.minimum_iterations < 1:
            raise ValueError("minimum_iterations must be positive")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative")
        for name in ("memory_write_threshold", "halt_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must be inside (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnifiedRecurrenceTelemetry:
    configured_iterations: int
    executed_iterations: int
    halt_probabilities: tuple[float, ...]
    memory_write_means: tuple[float, ...]
    transport_gates: tuple[float, ...]
    halted: bool
    halt_reason: str
    memory_retention: dict[str, float] | None
    semantic_residuals: tuple[float, ...]
    controller_sha256: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": UNIFIED_INTRINSIC_RECURRENCE_SCHEMA,
            "configured_iterations": self.configured_iterations,
            "executed_iterations": self.executed_iterations,
            "halt_probabilities": list(self.halt_probabilities),
            "memory_write_means": list(self.memory_write_means),
            "transport_gates": list(self.transport_gates),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "memory_retention": self.memory_retention,
            "semantic_residuals": list(self.semantic_residuals),
            "controller_sha256": self.controller_sha256,
            "teacher_available": False,
            "solver_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class UnifiedRecurrentController(nn.Module):
    """Small trainable control tissue acting on the real recurrent stream.

    A bounded rational depth basis ``u=step/(step+1)`` is used instead of a
    per-depth lookup bank. The resulting operator is defined at every unseen
    depth and converges rather than growing without bound.
    """

    def __init__(self, config: UnifiedRecurrenceConfig) -> None:
        super().__init__()
        self.config = config
        (
            key_a,
            key_b,
            key_depth,
            key_memory,
            key_halt,
            key_state,
            key_state_slots,
            key_state_values,
            key_transition_query,
            key_transition_key,
            key_transition_value,
            key_transition_self,
            key_transition_output,
            key_transition_depth,
            key_action_slots,
            key_action_values,
            key_action_query,
            key_action_key,
            key_action_value,
            key_action_output,
            key_action_depth,
            key_state_action,
            key_literal_values,
            key_answer_query,
            key_answer_key,
            key_answer_value,
            key_answer_output,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed),
            num=27,
        )
        scale = 1.0 / math.sqrt(config.hidden_size)
        self.correction_a = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_a,
            ).astype(mx.float32)
            * scale
        )
        self.correction_b = mx.zeros(
            (config.correction_rank, config.hidden_size),
            dtype=mx.float32,
        )
        self.depth_scale = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.memory_write_weight = (
            mx.random.normal((config.hidden_size,), key=key_memory).astype(mx.float32)
            * scale
        )
        self.memory_write_bias = mx.array(-6.0, dtype=mx.float32)
        self.transport_depth_weight = mx.zeros(
            (config.depth_basis_size,),
            dtype=mx.float32,
        )
        # Zero initialization preserves the CP203 depth-only operator exactly.
        # Training can then learn to accept a useful recurrent proposal for one
        # token/task while rejecting destructive motion for another.
        self.transport_state_weight = mx.zeros(
            (config.hidden_size,),
            dtype=mx.float32,
        )
        self.transport_motion_weight = mx.zeros(
            (config.hidden_size,),
            dtype=mx.float32,
        )
        # Re-entry starts conservative. Step zero bypasses this gate exactly,
        # retaining base-forward parity; later steps learn how much of the new
        # window state can be admitted without leaving the coda's manifold.
        self.transport_bias = mx.array(-0.5, dtype=mx.float32)
        # p stays inside (0.5, 1.0). The cumulative displacement of unseen
        # passes therefore grows sublinearly instead of linearly with depth,
        # while gradients can still relax the schedule toward sqrt decay.
        self.transport_decay_logit = mx.array(4.0, dtype=mx.float32)
        self.halt_state_weight = (
            mx.random.normal((config.hidden_size,), key=key_halt).astype(mx.float32)
            * scale
        )
        self.halt_motion_weight = mx.array(0.0, dtype=mx.float32)
        self.halt_bias = mx.array(-6.0, dtype=mx.float32)
        self.state_readout_weight = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.state_cardinality,
                ),
                key=key_state,
            ).astype(mx.float32)
            * scale
        )
        self.state_readout_bias = mx.zeros(
            (config.state_slots, config.state_cardinality),
            dtype=mx.float32,
        )
        self.state_slot_embeddings = (
            mx.random.normal(
                (config.state_slots, config.hidden_size),
                key=key_state_slots,
            ).astype(mx.float32)
            * 0.02
        )
        self.state_value_embeddings = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.state_cardinality,
                    config.hidden_size,
                ),
                key=key_state_values,
            ).astype(mx.float32)
            * 0.02
        )
        self.state_transition_query = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.correction_rank,
                ),
                key=key_transition_query,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_transition_key,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_transition_value,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_self = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.correction_rank,
                ),
                key=key_transition_self,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_output = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.correction_rank,
                    config.state_cardinality,
                ),
                key=key_transition_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.state_transition_depth = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.state_transition_bias = mx.zeros(
            (config.state_slots, config.state_cardinality),
            dtype=mx.float32,
        )
        self.action_slot_embeddings = (
            mx.random.normal(
                (config.action_slots, config.hidden_size), key=key_action_slots
            ).astype(mx.float32)
            * 0.02
        )
        self.action_value_embeddings = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.hidden_size,
                ),
                key=key_action_values,
            ).astype(mx.float32)
            * 0.02
        )
        self.action_query = (
            mx.random.normal(
                (config.action_slots, config.hidden_size, config.correction_rank),
                key=key_action_query,
            ).astype(mx.float32)
            * scale
        )
        self.action_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank), key=key_action_key
            ).astype(mx.float32)
            * scale
        )
        self.action_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank), key=key_action_value
            ).astype(mx.float32)
            * scale
        )
        self.action_output = (
            mx.random.normal(
                (config.action_slots, config.correction_rank, config.action_cardinality),
                key=key_action_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.action_depth = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank), key=key_action_depth
            ).astype(mx.float32)
            * 0.01
        )
        self.action_bias = mx.zeros(
            (config.action_slots, config.action_cardinality), dtype=mx.float32
        )
        self.state_action_projection = (
            mx.random.normal(
                (config.action_slots, config.hidden_size, config.correction_rank),
                key=key_state_action,
            ).astype(mx.float32)
            * scale
        )
        self.literal_value_embeddings = (
            mx.random.normal(
                (LITERAL_MAX_VALUE + 1, config.hidden_size), key=key_literal_values
            ).astype(mx.float32)
            * 0.02
        )
        self.answer_query = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_query,
            ).astype(mx.float32)
            * scale
        )
        self.answer_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_key,
            ).astype(mx.float32)
            * scale
        )
        self.answer_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_value,
            ).astype(mx.float32)
            * scale
        )
        self.answer_output = (
            mx.random.normal(
                (config.correction_rank, config.hidden_size),
                key=key_answer_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.answer_gate_query = mx.zeros(
            (config.hidden_size, 1), dtype=mx.float32
        )
        self.answer_gate_logit = mx.array(-2.0, dtype=mx.float32)
        self.answer_role_projection = mx.zeros(
            (config.hidden_size, config.state_slots + 1), dtype=mx.float32
        )
        self.answer_role_bias = mx.concatenate(
            [mx.array([6.0], dtype=mx.float32), mx.zeros((config.state_slots,))]
        )
        self.answer_place_projection = mx.zeros(
            (config.hidden_size, 3), dtype=mx.float32
        )
        self.answer_place_state_projection = mx.zeros(
            (config.hidden_size, 3), dtype=mx.float32
        )
        self.answer_place_width_projection = mx.zeros(
            (2, 3), dtype=mx.float32
        )
        self.answer_place_bias = mx.array((6.0, 0.0, 0.0), dtype=mx.float32)
        # A high-confidence learned role/place decision must be authoritative.
        # The no-op prior is carried by the role/place ``none`` classes, whose
        # mass makes pointer confidence structurally zero on syntax positions.
        self.answer_digit_gate_logit = mx.array(4.0, dtype=mx.float32)
        self.literal_grounding_logit = mx.array(-1.1, dtype=mx.float32)
        self.state_literal_copy_logit = mx.array(
            (-4.0, 0.5, 0.5, 0.5, -4.0), dtype=mx.float32
        )
        self.action_literal_copy_logit = mx.array(
            (-4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -4.0),
            dtype=mx.float32,
        )
        self.opcode_copy_logit = mx.array(1.5, dtype=mx.float32)

    def depth_features(self, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("recurrent step must be a non-negative integer")
        u = float(step) / float(step + 1) if step else 0.0
        return mx.array(
            [u ** (index + 1) for index in range(self.config.depth_basis_size)],
            dtype=mx.float32,
        )

    def correction(self, hidden: Any, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("correction step must be a non-negative integer")
        # T=1 is the semantic anchor.  Recurrent control tissue must earn an
        # improvement through additional passes, never by silently rewriting
        # the baseline against which those passes are compared.
        if step == 0:
            return hidden
        features = self.depth_features(step)
        scale = 1.0 + features @ self.depth_scale
        low_rank = (hidden.astype(mx.float32) @ self.correction_a) * scale
        delta = low_rank @ self.correction_b
        return hidden + delta.astype(hidden.dtype)

    def memory_write_probabilities(self, previous: Any, candidate: Any) -> Any:
        disagreement = (candidate - previous).astype(mx.float32)
        logits = disagreement @ self.memory_write_weight + self.memory_write_bias
        return mx.sigmoid(logits)[..., None]

    def transport_gate(
        self,
        step: int,
        previous: Any | None = None,
        candidate: Any | None = None,
    ) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("transport step must be a non-negative integer")
        if (previous is None) != (candidate is None):
            raise ValueError("transport state and candidate must be supplied together")
        if step == 0:
            return mx.array(1.0, dtype=mx.float32)
        logit = (
            self.transport_bias
            + self.depth_features(step) @ self.transport_depth_weight
        )
        if previous is not None and candidate is not None:
            if previous.shape != candidate.shape:
                raise ValueError("transport state and candidate shapes differ")
            previous_wide = previous.astype(mx.float32)
            motion = candidate.astype(mx.float32) - previous_wide
            previous_features = previous_wide / _rms(previous_wide)
            motion_features = motion / _rms(motion)
            feature_scale = 1.0 / math.sqrt(self.config.hidden_size)
            logit = logit + feature_scale * (
                previous_features @ self.transport_state_weight
                + motion_features @ self.transport_motion_weight
            )
        base = mx.sigmoid(logit)
        exponent = self.transport_decay_exponent()
        gate = base / mx.power(
            mx.array(float(step), dtype=mx.float32),
            exponent,
        )
        return gate[..., None] if previous is not None else gate

    def transport_decay_exponent(self) -> Any:
        return 0.5 + 0.5 * mx.sigmoid(self.transport_decay_logit)

    def transport(self, previous: Any, candidate: Any, step: int) -> tuple[Any, Any]:
        if step == 0:
            return candidate, self.transport_gate(step)
        matched = candidate * (_rms(previous) / _rms(candidate)).astype(
            candidate.dtype
        )
        gate = self.transport_gate(step, previous, matched)
        transported = previous + gate.astype(previous.dtype) * (matched - previous)
        return transported, gate

    def halt_probability(self, previous: Any, candidate: Any) -> Any:
        current = candidate.astype(mx.float32)
        previous_wide = previous.astype(mx.float32)
        pooled = mx.mean(current, axis=(0, 1))
        motion = mx.mean(mx.abs(current - previous_wide)) / mx.maximum(
            mx.mean(mx.abs(previous_wide)),
            1e-6,
        )
        logit = (
            pooled @ self.halt_state_weight
            + self.halt_motion_weight * (1.0 - mx.minimum(motion, 1.0))
            + self.halt_bias
        )
        return mx.sigmoid(logit)

    def initial_state_slots(self, batch_size: int, dtype: Any) -> Any:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("state slot batch size must be positive")
        return mx.broadcast_to(
            self.state_slot_embeddings.astype(dtype)[None, :, :],
            (batch_size, self.config.state_slots, self.config.hidden_size),
        )

    def attend_answer_to_state(
        self,
        candidate: Any,
        committed_state: Any,
        *,
        state_slot_start: int,
        state_probabilities: Any | None = None,
        role_logit_trajectory: list[Any] | None = None,
        place_logit_trajectory: list[Any] | None = None,
        binding_feature_trajectory: list[tuple[Any, Any, Any]] | None = None,
    ) -> Any:
        """Give answer positions a direct neural read path to typed registers."""

        stop = state_slot_start + self.config.state_slots
        answer_start = stop - 1
        if (
            len(candidate.shape) != 3
            or candidate.shape != committed_state.shape
            or int(candidate.shape[-1]) != self.config.hidden_size
            or not 0 <= state_slot_start < stop <= int(candidate.shape[1])
        ):
            raise ValueError("state-to-answer bridge layout differs")
        answer = candidate[:, answer_start:, :].astype(mx.float32)
        state = committed_state[:, state_slot_start:stop, :].astype(mx.float32)
        query = answer @ self.answer_query
        key = state @ self.answer_key
        value = state @ self.answer_value
        attention = mx.softmax(
            mx.einsum("bar,bsr->bas", query, key)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("bas,bsr->bar", attention, value)
        delta = context @ self.answer_output
        # State is useful at value-bearing positions and harmful when a frozen
        # language manifold is already emitting syntax or EOS. A global gate
        # applied the same correction everywhere and eventually produced
        # repeated braces under generated-history training. Keep the write
        # token-conditioned and no larger than the local residual scale.
        answer_rms = mx.sqrt(mx.mean(answer**2, axis=-1, keepdims=True) + 1e-6)
        delta_rms = mx.sqrt(mx.mean(delta**2, axis=-1, keepdims=True) + 1e-6)
        delta = delta * mx.minimum(1.0, answer_rms / delta_rms)
        gate = mx.sigmoid(answer @ self.answer_gate_query + self.answer_gate_logit)
        bridged = answer + gate * delta

        role_logits, place_logits = self.answer_binding_logits(
            answer,
            state,
            state_probabilities,
        )
        if binding_feature_trajectory is not None:
            if state_probabilities is None:
                raise ValueError("cached answer binding requires typed probabilities")
            binding_feature_trajectory.append(
                (
                    mx.stop_gradient(answer),
                    mx.stop_gradient(state),
                    mx.stop_gradient(state_probabilities.astype(mx.float32)),
                )
            )
        if role_logit_trajectory is not None:
            role_logit_trajectory.append(role_logits)
        if place_logit_trajectory is not None:
            place_logit_trajectory.append(place_logits)
        return mx.concatenate(
            [candidate[:, :answer_start, :], bridged.astype(candidate.dtype)],
            axis=1,
        )

    def answer_binding_logits(
        self,
        answer: Any,
        state: Any,
        state_probabilities: Any | None,
    ) -> tuple[Any, Any]:
        """Decode answer role and decimal place from reusable causal features."""

        if (
            len(answer.shape) != 3
            or len(state.shape) != 3
            or int(answer.shape[0]) != int(state.shape[0])
            or int(answer.shape[-1]) != self.config.hidden_size
            or state.shape[1:] != (
                self.config.state_slots,
                self.config.hidden_size,
            )
        ):
            raise ValueError("answer binding feature layout differs")
        role_logits = answer @ self.answer_role_projection + self.answer_role_bias
        role_probabilities = mx.softmax(role_logits.astype(mx.float32), axis=-1)
        selected_state = mx.einsum(
            "bar,brh->bah",
            role_probabilities[..., 1:],
            state.astype(mx.float32),
        )
        width_logits = mx.zeros((*answer.shape[:2], 3), dtype=mx.float32)
        if state_probabilities is not None:
            if state_probabilities.shape != (
                int(answer.shape[0]),
                self.config.state_slots,
                self.config.state_cardinality,
            ):
                raise ValueError("answer bridge state probabilities differ")
            selected_values = mx.einsum(
                "bar,brv->bav",
                role_probabilities[..., 1:],
                state_probabilities.astype(mx.float32),
            )
            width_features = mx.stack(
                (
                    mx.sum(selected_values[..., :10], axis=-1),
                    mx.sum(selected_values[..., 10:], axis=-1),
                ),
                axis=-1,
            )
            width_logits = width_features @ self.answer_place_width_projection
        # Prefix position identifies the output field, but it cannot reveal
        # whether that field's value has one or two digits. The selected typed
        # register makes digit width causally observable to the neural head.
        place_logits = (
            answer @ self.answer_place_projection
            + selected_state @ self.answer_place_state_projection
            + width_logits
            + self.answer_place_bias
        )
        return role_logits, place_logits

    def apply_answer_digit_pointer(
        self,
        logits: Any,
        role_logits: Any,
        place_logits: Any,
        state_probabilities: Any,
    ) -> Any:
        """Mix a learned terminal-register digit pointer into frozen logits."""

        if not self.config.literal_digit_token_ids:
            raise ValueError("answer digit pointer requires bound tokenizer digits")
        if (
            len(logits.shape) != 3
            or len(role_logits.shape) != 3
            or len(place_logits.shape) != 3
            or len(state_probabilities.shape) != 3
            or logits.shape[:2] != role_logits.shape[:2]
            or logits.shape[:2] != place_logits.shape[:2]
            or int(role_logits.shape[-1]) != self.config.state_slots + 1
            or int(place_logits.shape[-1]) != 3
            or state_probabilities.shape[1:] != (
                self.config.state_slots,
                self.config.state_cardinality,
            )
        ):
            raise ValueError("answer digit pointer tensor layout differs")
        vocabulary_size = int(logits.shape[-1])
        if max(self.config.literal_digit_token_ids) >= vocabulary_size:
            raise ValueError("answer digit pointer token is outside the vocabulary")

        raw_roles = mx.softmax(role_logits.astype(mx.float32), axis=-1)
        raw_places = mx.softmax(place_logits.astype(mx.float32), axis=-1)
        value_indices = mx.arange(self.config.state_cardinality)
        digit_indices = mx.arange(10)
        ones = ((value_indices[:, None] % 10) == digit_indices[None, :]).astype(
            mx.float32
        )
        tens = (
            ((value_indices[:, None] // 10) == digit_indices[None, :])
            * (value_indices[:, None] >= 10)
        ).astype(mx.float32)
        digit_probabilities_by_position: list[Any] = []
        confidences: list[Any] = []
        previous_role = mx.concatenate(
            (
                mx.ones((*raw_roles.shape[:1], 1), dtype=mx.float32),
                mx.zeros(
                    (*raw_roles.shape[:1], self.config.state_slots),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        )
        previous_two_digit_start = mx.zeros(
            (*raw_roles.shape[:1], 1), dtype=mx.float32
        )
        previous_value_complete = mx.zeros(
            (*raw_roles.shape[:1], 1), dtype=mx.float32
        )
        for position in range(int(raw_roles.shape[1])):
            raw_role = raw_roles[:, position, :]
            raw_place = raw_places[:, position, :]
            learned_digit_mass = (
                mx.sum(raw_place[:, 1:], axis=-1, keepdims=True)
                * (1.0 - previous_value_complete)
            )
            # Once exact typed state establishes a two-digit value, its ones
            # token is mandatory. The frozen language prior often calls that
            # position syntax after seeing the tens token; allowing that local
            # guess to disengage the pointer produced plausible but wrong 19s.
            digit_mass = mx.maximum(learned_digit_mass, previous_two_digit_start)
            previous_role_mass = mx.sum(
                previous_role[:, 1:], axis=-1, keepdims=True
            )
            # A second digit is still part of the field selected at the first
            # digit.  Letting an independently classified role replace it made
            # multi-register answers read the ones digit from another slot.
            continuation = (
                previous_role_mass * previous_two_digit_start
            )
            role = (1.0 - continuation) * raw_role + continuation * previous_role
            role_mass = mx.sum(role[:, 1:], axis=-1, keepdims=True)
            selected_values = mx.einsum(
                "br,brv->bv",
                role[:, 1:],
                state_probabilities.astype(mx.float32),
            ) / mx.maximum(role_mass, 1e-8)
            one_digit_mass = mx.sum(selected_values[:, :10], axis=-1, keepdims=True)
            two_digit_mass = mx.sum(selected_values[:, 10:], axis=-1, keepdims=True)
            start = 1.0 - previous_role_mass
            tens_mass = digit_mass * start * two_digit_mass
            ones_mass = digit_mass * (
                previous_role_mass + start * one_digit_mass
            )
            digit_probabilities_by_position.append(
                tens_mass
                * ((selected_values @ tens) / mx.maximum(two_digit_mass, 1e-8))
                + ones_mass * (selected_values @ ones)
            )
            confidences.append(
                role_mass
                * digit_mass
                * mx.sigmoid(self.answer_digit_gate_logit)
            )
            previous_value_complete = (
                digit_mass * start * one_digit_mass + continuation
            )
            previous_two_digit_start = digit_mass * start * two_digit_mass
            previous_role = role
        digit_probabilities = mx.stack(digit_probabilities_by_position, axis=1)
        vocabulary = mx.arange(vocabulary_size)
        digit_tokens = mx.array(self.config.literal_digit_token_ids)
        digit_to_vocabulary = (digit_tokens[:, None] == vocabulary[None, :]).astype(
            mx.float32
        )
        pointer_probabilities = digit_probabilities @ digit_to_vocabulary
        confidence = mx.stack(confidences, axis=1)
        base_probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        mixed = (
            (1.0 - confidence) * base_probabilities
            + confidence * pointer_probabilities
        )
        return mx.log(mx.maximum(mixed, 1e-12)).astype(logits.dtype)

    def state_logits(
        self,
        hidden: Any,
        *,
        public_token_count: int | None = None,
        state_slot_start: int | None = None,
    ) -> Any:
        """Decode typed state from causal slots, or a public-only diagnostic."""

        if len(hidden.shape) != 3 or int(hidden.shape[-1]) != self.config.hidden_size:
            raise ValueError("recurrent state readout hidden shape differs")
        if (public_token_count is None) == (state_slot_start is None):
            raise ValueError("exactly one recurrent state readout source is required")
        if state_slot_start is not None:
            if (
                type(state_slot_start) is not int
                or state_slot_start < 0
                or state_slot_start + self.config.state_slots > int(hidden.shape[1])
            ):
                raise ValueError("state slot range is outside the recurrent state")
            selected = hidden[
                :,
                state_slot_start : state_slot_start + self.config.state_slots,
                :,
            ].astype(mx.float32)
            return mx.einsum(
                "bsh,shc->bsc",
                selected,
                self.state_readout_weight,
            ) + self.state_readout_bias
        if (
            type(public_token_count) is not int
            or not 1 <= public_token_count <= int(hidden.shape[1])
        ):
            raise ValueError("public token count is outside the recurrent state")
        public_summary = hidden[:, public_token_count - 1, :].astype(mx.float32)
        return mx.einsum(
            "bh,shc->bsc",
            public_summary,
            self.state_readout_weight,
        ) + self.state_readout_bias

    def initial_state_logits(
        self,
        problem_evidence: Any,
        token_ids: Any | None = None,
    ) -> Any:
        """Decode slot-specific initial state from the complete public prefix."""

        if (
            len(problem_evidence.shape) != 3
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
            or int(problem_evidence.shape[1]) < 1
        ):
            raise ValueError("initial state problem evidence shape differs")
        slot_queries = mx.einsum(
            "sh,shr->sr",
            self.state_slot_embeddings.astype(mx.float32),
            self.state_transition_query,
        )
        keys = problem_evidence.astype(mx.float32) @ self.state_transition_key
        values = problem_evidence.astype(mx.float32) @ self.state_transition_value
        attention = mx.softmax(
            mx.einsum("sr,bnr->bsn", slot_queries, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("bsn,bnr->bsr", attention, values)
        logits = mx.einsum(
            "bsr,src->bsc", context, self.state_transition_output
        ) + self.state_transition_bias
        if token_ids is not None and self.config.literal_digit_token_ids:
            pointer = self._literal_pointer_logits(
                problem_evidence,
                token_ids,
                mx.broadcast_to(
                    slot_queries[None, :, :],
                    (int(problem_evidence.shape[0]),) + tuple(slot_queries.shape),
                ),
                self.state_transition_key,
                self.config.state_cardinality,
            )
            copy_strength = mx.logaddexp(
                self.state_literal_copy_logit,
                mx.zeros_like(self.state_literal_copy_logit),
            )
            logits = logits + copy_strength[None, :, None] * pointer
        if (
            token_ids is not None
            and self.config.literal_digit_token_ids
            and self.config.opcode_token_patterns
        ):
            opcode_contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            literal_contract = LiteralObservationContract(
                self.config.literal_digit_token_ids
            )
            public_values, recognized = opcode_contract.public_initial_states(
                token_ids.tolist(),
                literal_contract,
            )
            exact = self._exact_categorical_logits(
                public_values,
                slots=self.config.state_slots,
                cardinality=self.config.state_cardinality,
            )
            known = mx.array(recognized, dtype=mx.bool_)
            logits = mx.where(known[:, None, None], exact, logits)
        return logits

    def ground_literal_evidence(self, problem_evidence: Any, token_ids: Any) -> Any:
        """Add exact integer observations without assigning them semantic roles."""

        if not self.config.literal_digit_token_ids:
            return problem_evidence
        if (
            len(problem_evidence.shape) != 3
            or len(token_ids.shape) != 2
            or tuple(problem_evidence.shape[:2]) != tuple(token_ids.shape)
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
        ):
            raise ValueError("literal grounding evidence and tokens differ")
        contract = LiteralObservationContract(self.config.literal_digit_token_ids)
        values, masks = contract.observe(token_ids.tolist())
        value_ids = mx.array(values, dtype=mx.int32)
        observed = mx.array(masks, dtype=mx.float32)[..., None]
        literal = self.literal_value_embeddings[value_ids]
        scale = mx.sigmoid(self.literal_grounding_logit)
        return problem_evidence + (
            observed * scale * literal.astype(problem_evidence.dtype)
        )

    def _literal_pointer_logits(
        self,
        problem_evidence: Any,
        token_ids: Any,
        query: Any,
        key_projection: Any,
        cardinality: int,
    ) -> Any:
        """Map learned role attention onto exact observed numeric categories."""

        contract = LiteralObservationContract(self.config.literal_digit_token_ids)
        values, masks = contract.observe(token_ids.tolist())
        return self._categorical_pointer_logits(
            problem_evidence,
            values,
            masks,
            query,
            key_projection,
            cardinality,
        )

    def _categorical_pointer_logits(
        self,
        problem_evidence: Any,
        values: Sequence[Sequence[int]],
        masks: Sequence[Sequence[bool]],
        query: Any,
        key_projection: Any,
        cardinality: int,
    ) -> Any:
        value_ids = mx.array(values, dtype=mx.int32)
        observed = mx.array(masks, dtype=mx.float32)
        keys = problem_evidence.astype(mx.float32) @ key_projection
        attention_logits = mx.einsum("bsr,bnr->bsn", query, keys) / math.sqrt(
            self.config.correction_rank
        )
        attention = mx.softmax(attention_logits, axis=-1) * observed[:, None, :]
        attention = attention / mx.maximum(
            mx.sum(attention, axis=-1, keepdims=True),
            1e-12,
        )
        categories = mx.arange(cardinality)[None, None, :]
        one_hot = (value_ids[..., None] == categories).astype(mx.float32)
        probabilities = mx.einsum("bsn,bnc->bsc", attention, one_hot)
        pointer = mx.log(mx.maximum(probabilities, 1e-6))
        has_observation = mx.sum(observed, axis=-1) > 0.0
        return mx.where(has_observation[:, None, None], pointer, mx.zeros_like(pointer))

    def typed_state_transition(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
    ) -> tuple[Any, Any]:
        """Commit one explicit categorical state for the next recurrent step.

        The straight-through categorical bottleneck makes the forward path
        discrete while retaining gradients through the decision probabilities.
        This prevents the auxiliary state head from merely describing a drifting
        residual: the state it predicts is the state the next pass receives.
        """

        logits = self.state_logits(hidden, state_slot_start=state_slot_start)
        return self.commit_state_logits(
            hidden,
            state_slot_start=state_slot_start,
            logits=logits,
        ), logits

    def state_transition_logits(
        self,
        problem_evidence: Any,
        hidden: Any,
        *,
        state_slot_start: int,
        step: int,
        action_state: Any | None = None,
        state_probabilities: Any | None = None,
        action_probabilities: Any | None = None,
    ) -> Any:
        """Predict one shared typed transition from state plus immutable evidence."""

        if (
            len(problem_evidence.shape) != 3
            or int(problem_evidence.shape[0]) != int(hidden.shape[0])
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
            or int(problem_evidence.shape[1]) < 1
        ):
            raise ValueError("state transition problem evidence shape differs")
        stop = state_slot_start + self.config.state_slots
        if not 0 <= state_slot_start < stop <= int(hidden.shape[1]):
            raise ValueError("state transition slots are outside the recurrent state")
        state = hidden[:, state_slot_start:stop, :].astype(mx.float32)
        evidence = problem_evidence.astype(mx.float32)
        query = mx.einsum("bsh,shr->bsr", state, self.state_transition_query)
        keys = evidence @ self.state_transition_key
        values = evidence @ self.state_transition_value
        attention = mx.softmax(
            mx.einsum("bsr,bnr->bsn", query, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("bsn,bnr->bsr", attention, values)
        self_state = mx.einsum(
            "bsh,shr->bsr",
            state,
            self.state_transition_self,
        )
        depth = self.depth_features(step) @ self.state_transition_depth
        features = mx.tanh(context + self_state + depth[None, None, :])
        if action_state is not None:
            if action_state.shape != (
                int(hidden.shape[0]),
                self.config.action_slots,
                self.config.hidden_size,
            ):
                raise ValueError("typed action state shape differs")
            action = mx.mean(
                mx.einsum(
                    "bah,ahr->bar",
                    action_state.astype(mx.float32),
                    self.state_action_projection,
                ),
                axis=1,
            )
            features = mx.tanh(features + action[:, None, :])
        learned_logits = mx.einsum(
            "bsr,src->bsc",
            features,
            self.state_transition_output,
        ) + self.state_transition_bias
        if state_probabilities is None or action_probabilities is None:
            return learned_logits
        microcode_logits, recognized = self.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
        )
        state_values = mx.argmax(state_probabilities, axis=-1).astype(mx.int32)
        terminal = state_values[:, -1] == 1
        state_categories = mx.arange(self.config.state_cardinality)[None, None, :]
        terminal_exact = (state_values[..., None] == state_categories).astype(mx.float32)
        terminal_logits = mx.log(mx.maximum(terminal_exact, 1e-6))
        return mx.where(
            terminal[:, None, None],
            terminal_logits,
            mx.where(recognized[:, None, None], microcode_logits, learned_logits),
        )

    def microcode_transition_logits(
        self,
        state_probabilities: Any,
        action_probabilities: Any,
    ) -> tuple[Any, Any]:
        """Execute recognized canonical instructions on categorical registers."""

        if state_probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("microcode categorical state differs from its schema")
        state = mx.argmax(state_probabilities, axis=-1).astype(mx.int32)
        action = mx.argmax(action_probabilities, axis=-1).astype(mx.int32)
        opcode = action[:, 0]
        arguments = action[:, 1:7]
        terminal = action[:, 7]
        recognized = (opcode >= OP_COPY_VALUE) & (opcode <= OP_FRONTIER_AUDIT)
        recognized = recognized & (opcode != ACTION_NULL)

        pc = mx.minimum(state[:, 0] + 1, self.config.state_cardinality - 1)
        value0 = state[:, 1]
        value1 = state[:, 2]
        value2 = state[:, 3]
        arg0, arg1, arg2, arg3, arg4, arg5 = (
            arguments[:, index] for index in range(6)
        )
        modulus = mx.maximum(arg1, 1)
        value0 = mx.where(opcode == OP_COPY_VALUE, arg0, value0)
        value0 = mx.where(opcode == OP_ADD_MOD, (value0 + arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_MUL_MOD, (value0 * arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_SUB_MOD, (value0 - arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_BOOL_NOT, 1 - mx.minimum(value0, 1), value0)
        value0 = mx.where(
            opcode == OP_BOOL_AND,
            mx.minimum(value0, 1) & mx.minimum(arg0, 1),
            value0,
        )
        value0 = mx.where(
            opcode == OP_BOOL_OR,
            mx.minimum(value0, 1) | mx.minimum(arg0, 1),
            value0,
        )
        value0 = mx.where(
            opcode == OP_BOOL_XOR,
            mx.minimum(value0, 1) ^ mx.minimum(arg0, 1),
            value0,
        )

        registers = mx.stack((state[:, 1], state[:, 2], state[:, 3]), axis=1)
        left = mx.take_along_axis(
            registers, mx.minimum(arg1, 2)[:, None], axis=1
        )[:, 0]
        right = mx.take_along_axis(
            registers, mx.minimum(arg2, 2)[:, None], axis=1
        )[:, 0]
        register_modulus = mx.maximum(arg5, 1)
        register_result = (left + arg3 * right + arg4) % register_modulus
        is_register = opcode == OP_REGISTER_AFFINE
        value0 = mx.where(is_register & (arg0 == 0), register_result, value0)
        value1 = mx.where(is_register & (arg0 == 1), register_result, value1)
        value2 = mx.where(is_register & (arg0 == 2), register_result, value2)

        # Broad process instructions use the same finite categorical machine as
        # the original executable curriculum.  Their private training traces
        # teach action selection; once selected, these transitions are exact and
        # require neither a runtime teacher nor a task-specific answer producer.
        is_traverse = opcode == OP_FRONTIER_TRAVERSE
        value0 = mx.where(is_traverse, arg0, value0)
        value1 = mx.where(is_traverse, mx.maximum(value1 - 1, 0), value1)
        value2 = mx.where(is_traverse, arg3, value2)

        is_enumerate = opcode == OP_FRONTIER_ENUMERATE
        count = state[:, 1] + PROCESS_RADIX * state[:, 2]
        added = arg3 + PROCESS_RADIX * arg4
        next_count = mx.minimum(count + added, MAX_PROCESS_INTEGER)
        value0 = mx.where(is_enumerate, next_count % PROCESS_RADIX, value0)
        value1 = mx.where(is_enumerate, next_count // PROCESS_RADIX, value1)
        value2 = mx.where(is_enumerate, arg5, value2)

        is_simulate = opcode == OP_FRONTIER_SIMULATE
        value0 = mx.where(is_simulate, arg0, value0)
        value1 = mx.where(is_simulate, arg4, value1)
        value2 = mx.where(is_simulate, arg3, value2)

        is_infer = opcode == OP_FRONTIER_INFER
        inferred_role = mx.where(
            arg0 == 0,
            arg1 + 1,
            mx.where(
                arg0 == 1,
                arg1 * 3 + arg2 + 1,
                value0,
            ),
        )
        value0 = mx.where(is_infer, inferred_role, value0)
        value1 = mx.where(is_infer & (arg0 == 3), arg1, value1)
        value2 = mx.where(is_infer & (arg0 == 3), arg2, value2)

        is_schedule = opcode == OP_FRONTIER_SCHEDULE
        reward = state[:, 2] + PROCESS_RADIX * state[:, 3]
        next_reward = mx.minimum(reward + arg3, MAX_PROCESS_INTEGER)
        value0 = mx.where(
            is_schedule,
            mx.minimum(value0 + arg1, self.config.state_cardinality - 1),
            value0,
        )
        value1 = mx.where(is_schedule, next_reward % PROCESS_RADIX, value1)
        value2 = mx.where(is_schedule, next_reward // PROCESS_RADIX, value2)

        is_calibrate = opcode == OP_FRONTIER_CALIBRATE
        value0 = mx.where(is_calibrate, arg1, value0)
        value1 = mx.where(is_calibrate, arg3, value1)
        value2 = mx.where(
            is_calibrate,
            mx.where(arg0 == 0, 0, arg5 + 1),
            value2,
        )

        is_audit = opcode == OP_FRONTIER_AUDIT
        value0 = mx.where(is_audit, arg0, value0)
        value1 = mx.where(is_audit, arg4, value1)
        value2 = mx.where(is_audit, arg5, value2)
        next_values = mx.stack((pc, value0, value1, value2, terminal), axis=1)
        categories = mx.arange(self.config.state_cardinality)[None, None, :]
        exact = (next_values[..., None] == categories).astype(mx.float32)
        return mx.log(mx.maximum(exact, 1e-6)), recognized

    def action_logits(
        self,
        problem_evidence: Any,
        hidden: Any,
        *,
        state_slot_start: int,
        step: int,
        token_ids: Any | None = None,
        state_probabilities: Any | None = None,
    ) -> Any:
        """Decode the next typed operation from public evidence and current state."""

        stop = state_slot_start + self.config.state_slots
        if not 0 <= state_slot_start < stop <= int(hidden.shape[1]):
            raise ValueError("action decoder state slots are outside the recurrent state")
        state = mx.mean(
            hidden[:, state_slot_start:stop, :].astype(mx.float32), axis=1
        )
        query_input = state[:, None, :] + self.action_slot_embeddings[None, :, :]
        query = mx.einsum("bah,ahr->bar", query_input, self.action_query)
        keys = problem_evidence.astype(mx.float32) @ self.action_key
        values = problem_evidence.astype(mx.float32) @ self.action_value
        attention = mx.softmax(
            mx.einsum("bar,bnr->ban", query, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("ban,bnr->bar", attention, values)
        depth = self.depth_features(step) @ self.action_depth
        features = mx.tanh(context + depth[None, None, :])
        logits = (
            mx.einsum("bar,arc->bac", features, self.action_output)
            + self.action_bias
        )
        if token_ids is not None and self.config.literal_digit_token_ids:
            pointer = self._literal_pointer_logits(
                problem_evidence,
                token_ids,
                query,
                self.action_key,
                self.config.action_cardinality,
            )
            copy_strength = mx.logaddexp(
                self.action_literal_copy_logit,
                mx.zeros_like(self.action_literal_copy_logit),
            )
            logits = logits + copy_strength[None, :, None] * pointer
        if token_ids is not None and self.config.opcode_token_patterns:
            contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            opcode_values, opcode_masks = contract.observe(token_ids.tolist())
            opcode_pointer = self._categorical_pointer_logits(
                problem_evidence,
                opcode_values,
                opcode_masks,
                query[:, :1, :],
                self.action_key,
                self.config.action_cardinality,
            )
            opcode_strength = mx.logaddexp(
                self.opcode_copy_logit,
                mx.zeros_like(self.opcode_copy_logit),
            )
            logits = mx.concatenate(
                [
                    logits[:, :1, :] + opcode_strength * opcode_pointer,
                    logits[:, 1:, :],
                ],
                axis=1,
            )
        if (
            token_ids is not None
            and state_probabilities is not None
            and self.config.literal_digit_token_ids
            and self.config.opcode_token_patterns
        ):
            contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            literal_contract = LiteralObservationContract(
                self.config.literal_digit_token_ids
            )
            state_rows = mx.argmax(state_probabilities, axis=-1).tolist()
            public_values, recognized = contract.public_instructions(
                token_ids.tolist(),
                literal_contract,
                state_rows,
            )
            exact = self._exact_categorical_logits(
                public_values,
                slots=self.config.action_slots,
                cardinality=self.config.action_cardinality,
            )
            known = mx.array(recognized, dtype=mx.bool_)
            logits = mx.where(known[:, None, None], exact, logits)
        return logits

    @staticmethod
    def _exact_categorical_logits(
        values: Sequence[Sequence[int]],
        *,
        slots: int,
        cardinality: int,
    ) -> Any:
        if any(
            len(row) != slots
            or any(type(value) is not int or not 0 <= value < cardinality for value in row)
            for row in values
        ):
            raise ValueError("exact categorical values differ from their schema")
        labels = mx.array(values, dtype=mx.int32)[..., None]
        categories = mx.arange(cardinality)[None, None, :]
        exact = (labels == categories).astype(mx.float32)
        return mx.log(mx.maximum(exact, 1e-6))

    def commit_action_logits(self, logits: Any) -> Any:
        """Convert predicted categorical operations into causal action tissue."""

        if (
            len(logits.shape) != 3
            or int(logits.shape[1]) != self.config.action_slots
            or int(logits.shape[2]) != self.config.action_cardinality
        ):
            raise ValueError("typed action logits differ from the canonical schema")
        probabilities = self.straight_through_probabilities(logits)
        return self.commit_action_probabilities(probabilities)

    @staticmethod
    def straight_through_probabilities(logits: Any) -> Any:
        """Use hard categories in the forward pass and soft gradients backward."""

        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        selected = mx.argmax(probabilities, axis=-1)
        categories = mx.arange(int(logits.shape[-1]))[None, None, :]
        hard = (categories == selected[..., None]).astype(mx.float32)
        return probabilities + mx.stop_gradient(hard - probabilities)

    @staticmethod
    def exact_probabilities(
        values: Sequence[int],
        *,
        slots: int,
        cardinality: int,
    ) -> Any:
        if len(values) != slots or any(
            type(value) is not int or not 0 <= value < cardinality
            for value in values
        ):
            raise ValueError("exact categorical values differ from their schema")
        labels = mx.array(values, dtype=mx.int32)[None, :, None]
        categories = mx.arange(cardinality)[None, None, :]
        return (categories == labels).astype(mx.float32)

    def commit_action_probabilities(self, probabilities: Any) -> Any:
        if probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("typed action probabilities differ from the schema")
        return mx.einsum(
            "bac,ach->bah", probabilities, self.action_value_embeddings
        )

    def teacher_action_state(self, values: Sequence[int]) -> Any:
        if len(values) != self.config.action_slots or any(
            type(value) is not int or not 0 <= value < self.config.action_cardinality
            for value in values
        ):
            raise ValueError("teacher action values differ from the canonical schema")
        exact = self.exact_probabilities(
            values,
            slots=self.config.action_slots,
            cardinality=self.config.action_cardinality,
        )
        return mx.einsum("bac,ach->bah", exact, self.action_value_embeddings)

    def commit_state_logits(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        logits: Any,
    ) -> Any:
        """Commit supplied transition logits through the categorical codebook."""

        if logits.shape != (
            int(hidden.shape[0]),
            self.config.state_slots,
            self.config.state_cardinality,
        ):
            raise ValueError("state transition logits differ from the canonical schema")
        probabilities = self.straight_through_probabilities(logits)
        return self.commit_state_probabilities(
            hidden,
            state_slot_start=state_slot_start,
            probabilities=probabilities,
        )

    def commit_state_probabilities(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        probabilities: Any,
    ) -> Any:
        if probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ):
            raise ValueError("typed state probabilities differ from the schema")
        replacement = mx.einsum(
            "bsc,sch->bsh",
            probabilities,
            self.state_value_embeddings,
        )
        return self._replace_state_slots(hidden, state_slot_start, replacement)

    def teacher_state_transition(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        values: Sequence[int],
        probability: float,
    ) -> Any:
        """Blend an exact prior state into training roll-in without prompt leakage."""

        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("state teacher-forcing probability must be inside [0, 1]")
        if len(values) != self.config.state_slots or any(
            type(value) is not int or not 0 <= value < self.config.state_cardinality
            for value in values
        ):
            raise ValueError("teacher state values differ from the canonical schema")
        labels = mx.array(values, dtype=mx.int32)[None, :, None]
        categories = mx.arange(self.config.state_cardinality)[None, None, :]
        exact = (categories == labels).astype(mx.float32)
        replacement = mx.einsum(
            "bsc,sch->bsh",
            exact,
            self.state_value_embeddings,
        )
        teacher = self._replace_state_slots(hidden, state_slot_start, replacement)
        if float(probability) == 1.0:
            return teacher
        start = state_slot_start
        stop = start + self.config.state_slots
        mixed = (
            (1.0 - float(probability)) * hidden[:, start:stop, :]
            + float(probability) * teacher[:, start:stop, :]
        )
        return mx.concatenate(
            [hidden[:, :start, :], mixed, hidden[:, stop:, :]],
            axis=1,
        )

    def _replace_state_slots(
        self,
        hidden: Any,
        state_slot_start: int,
        replacement: Any,
    ) -> Any:
        """Replace typed slots at matched residual scale without scale gradients."""

        previous = hidden[
            :,
            state_slot_start : state_slot_start + self.config.state_slots,
            :,
        ]
        scale = mx.stop_gradient(
            _rms(previous.astype(mx.float32))
            / _rms(replacement.astype(mx.float32))
        )
        replacement = replacement * scale.astype(replacement.dtype)
        committed = mx.concatenate(
            [
                hidden[:, :state_slot_start, :],
                replacement.astype(hidden.dtype),
                hidden[:, state_slot_start + self.config.state_slots :, :],
            ],
            axis=1,
        )
        return committed

    def identity_initialized(self) -> bool:
        return bool(mx.all(self.correction_b == 0))

    def parameter_sha256(self) -> str:
        digest = hashlib.sha256()
        for name in (
            "correction_a",
            "correction_b",
            "depth_scale",
            "memory_write_weight",
            "memory_write_bias",
            "transport_depth_weight",
            "transport_state_weight",
            "transport_motion_weight",
            "transport_bias",
            "transport_decay_logit",
            "halt_state_weight",
            "halt_motion_weight",
            "halt_bias",
            "answer_query",
            "answer_key",
            "answer_value",
            "answer_output",
            "answer_gate_query",
            "answer_gate_logit",
            "answer_role_projection",
            "answer_role_bias",
            "answer_place_projection",
            "answer_place_state_projection",
            "answer_place_width_projection",
            "answer_place_bias",
            "answer_digit_gate_logit",
            "state_readout_weight",
            "state_readout_bias",
            "state_slot_embeddings",
            "state_value_embeddings",
            "state_transition_query",
            "state_transition_key",
            "state_transition_value",
            "state_transition_self",
            "state_transition_output",
            "state_transition_depth",
            "state_transition_bias",
            "action_slot_embeddings",
            "action_value_embeddings",
            "action_query",
            "action_key",
            "action_value",
            "action_output",
            "action_depth",
            "action_bias",
            "state_action_projection",
            "literal_value_embeddings",
            "literal_grounding_logit",
            "state_literal_copy_logit",
            "action_literal_copy_logit",
            "opcode_copy_logit",
        ):
            value = getattr(self, name)
            mx.eval(value)
            digest.update(name.encode("ascii"))
            digest.update(bytes(memoryview(value.astype(mx.float32))))
        return digest.hexdigest()

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": UNIFIED_INTRINSIC_RECURRENCE_SCHEMA,
            "config": self.config.to_dict(),
            "identity_initialized": self.identity_initialized(),
            "continuous_depth_basis": "bounded_rational_polynomial",
            "depth_extrapolation_defined": True,
            "typed_state_bottleneck": "straight_through_categorical",
            "predicted_state_is_next_step_input": True,
            "state_processor": "shared_evidence_attention_transition",
            "action_processor": "public_evidence_typed_action_transition",
            "predicted_action_is_state_transition_input": True,
            "terminal_state_semantics": "exact_idempotent_stutter",
            "terminal_decode_semantics": "first_terminal_state_preserved",
            "state_problem_evidence": "frozen_deep_prefix_no_decoder_suffix",
            "literal_grounding": (
                "tokenizer_bound_bounded_integer_observations"
                if self.config.literal_digit_token_ids
                else "disabled"
            ),
            "literal_role_binding": (
                "learned_attention_exact_category_pointer"
                if self.config.literal_digit_token_ids
                else "disabled"
            ),
            "opcode_grounding": (
                "tokenizer_bound_operation_observations"
                if self.config.opcode_token_patterns
                else "disabled"
            ),
            "public_instruction_decoder": (
                "tokenizer_bound_state_selected_exact_microcode"
                if self.config.opcode_token_patterns
                else "disabled"
            ),
            "transformer_answer_passes_per_state": 1,
            "state_to_answer_bridge": (
                "role_and_digit_place_conditioned_pointer_over_frozen_readout"
            ),
            "parameter_sha256": self.parameter_sha256(),
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def unified_recurrent_hidden_states(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    *,
    memory_layout: MemoryLayout | None = None,
    adaptive_halt: bool = False,
    soft_memory_writes: bool = False,
    state_slot_start: int | None = None,
    state_logit_trajectory: list[Any] | None = None,
    action_logit_trajectory: list[Any] | None = None,
    initial_state_logit_trajectory: list[Any] | None = None,
    decode_state_trajectory: list[Any] | None = None,
    recurrent_input_trajectory: list[Any] | None = None,
    state_probability_trajectory: list[Any] | None = None,
    answer_role_logit_trajectory: list[Any] | None = None,
    answer_place_logit_trajectory: list[Any] | None = None,
    answer_binding_feature_trajectory: list[tuple[Any, Any, Any]] | None = None,
    state_teacher_values: Sequence[Sequence[int]] | None = None,
    action_teacher_values: Sequence[Sequence[int]] | None = None,
    initial_state_teacher_values: Sequence[int] | None = None,
    state_teacher_forcing_probability: float = 0.0,
    typed_action_lesion: bool = False,
    caches: dict[str, Any] | None = None,
) -> tuple[Any, list[Any], UnifiedRecurrenceTelemetry]:
    """Run all Level-3 control mechanisms on one transformer trajectory."""

    if not isinstance(controller, UnifiedRecurrentController):
        raise TypeError("unified recurrence controller is invalid")
    if (
        type(adaptive_halt) is not bool
        or type(soft_memory_writes) is not bool
        or type(typed_action_lesion) is not bool
    ):
        raise TypeError("unified recurrence mode flags must be bools")
    if adaptive_halt and controller.config.minimum_iterations > plan.iterations:
        raise ValueError("minimum iterations exceed the recurrence plan")
    if caches is not None:
        if set(caches) != {"prelude", "window", "coda"}:
            raise ValueError("caches must come from make_recurrent_caches")
        if len(caches["window"]) != plan.iterations:
            raise ValueError("cache iteration count does not match the plan")
        expected_window_layers = plan.coda_start - plan.prelude_end
        if (
            len(caches["prelude"]) != plan.prelude_end
            or any(
                len(iteration_caches) != expected_window_layers
                for iteration_caches in caches["window"]
            )
        ):
            raise ValueError("cache layer topology does not match the plan")
        if any(
            value is not None
            for value in (
                memory_layout,
                state_slot_start,
                state_teacher_values,
                action_teacher_values,
                initial_state_teacher_values,
            )
        ) or adaptive_halt or soft_memory_writes:
            raise ValueError(
                "incremental recurrent caches require the untyped fixed-depth path"
            )
    if state_teacher_values is not None and state_slot_start is None:
        raise ValueError("state teacher roll-in requires typed state slots")
    if initial_state_teacher_values is not None and state_slot_start is None:
        raise ValueError("initial state teacher requires typed state slots")
    if action_teacher_values is not None and state_slot_start is None:
        raise ValueError("action teacher requires typed state slots")
    if typed_action_lesion and action_teacher_values is not None:
        raise ValueError("typed action lesion cannot accompany an action teacher")
    if typed_action_lesion and state_slot_start is None:
        raise ValueError("typed action lesion requires typed state slots")
    if state_teacher_values is not None and len(state_teacher_values) < plan.iterations:
        raise ValueError("state teacher roll-in is shorter than the recurrence plan")
    if action_teacher_values is not None and len(action_teacher_values) < plan.iterations:
        raise ValueError("action teacher is shorter than the recurrence plan")
    if (
        isinstance(state_teacher_forcing_probability, bool)
        or not isinstance(state_teacher_forcing_probability, (int, float))
        or not 0.0 <= float(state_teacher_forcing_probability) <= 1.0
    ):
        raise ValueError("state teacher-forcing probability must be inside [0, 1]")
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers or plan.coda_start > len(layers):
        raise ValueError("model layers do not satisfy the recurrence plan")
    if caches is not None and len(caches["coda"]) != len(layers) - plan.coda_start:
        raise ValueError("cache layer topology does not match the plan")

    hidden = inner.embed_tokens(tokens)
    hidden = _run(
        layers[: plan.prelude_end],
        hidden,
        caches["prelude"] if caches else None,
    )
    if state_slot_start is not None:
        if (
            type(state_slot_start) is not int
            or not 0 <= state_slot_start <= int(hidden.shape[1])
        ):
            raise ValueError("state slot insertion is outside the token sequence")
        slots = controller.initial_state_slots(int(hidden.shape[0]), hidden.dtype)
        hidden = mx.concatenate(
            [hidden[:, :state_slot_start, :], slots, hidden[:, state_slot_start:, :]],
            axis=1,
        )
    anchor = hidden
    anchor_rms = _rms(anchor) if plan.renormalize else None
    if memory_layout is not None and memory_layout.n_slots != int(hidden.shape[1]):
        raise ValueError("protected memory layout differs from token positions")

    trajectory: list[Any] = []
    halt_probabilities: list[float] = []
    memory_write_means: list[float] = []
    transport_gates: list[float] = []
    window = layers[plan.prelude_end : plan.coda_start]
    problem_evidence: Any | None = None
    state_probabilities: Any | None = None
    if state_slot_start is not None:
        # Prefix-only execution is causally identical to the same positions in
        # the complete sequence, but cannot expose teacher-forced answer tokens.
        # Detaching gives the state machine a stable deep evidence surface.
        with recurrent_iteration(0):
            problem_evidence = mx.stop_gradient(
                _run(window, anchor[:, :state_slot_start, :])
            )
        problem_evidence = controller.ground_literal_evidence(
            problem_evidence,
            tokens[:, :state_slot_start],
        )
        # A recurrent machine must start from the problem's state, not from one
        # task-independent learned vector.  The prediction uses only the public
        # prefix and remains the sole source at inference; exact initial values
        # are an annealed training authority only.
        initial_state_logits = controller.initial_state_logits(
            problem_evidence,
            tokens[:, :state_slot_start],
        )
        state_probabilities = controller.straight_through_probabilities(
            initial_state_logits
        )
        if initial_state_logit_trajectory is not None:
            initial_state_logit_trajectory.append(initial_state_logits)
        if (
            initial_state_teacher_values is not None
            and state_teacher_forcing_probability > 0.0
        ):
            teacher_initial = controller.exact_probabilities(
                initial_state_teacher_values,
                slots=controller.config.state_slots,
                cardinality=controller.config.state_cardinality,
            )
            state_probabilities = (
                (1.0 - state_teacher_forcing_probability) * state_probabilities
                + state_teacher_forcing_probability * teacher_initial
            )
        hidden = controller.commit_state_probabilities(
            hidden,
            state_slot_start=state_slot_start,
            probabilities=state_probabilities,
        )
    halted = False
    halt_reason = "configured_depth_exhausted"
    last_decode_state: Any | None = None
    for iteration in range(plan.iterations):
        if state_slot_start is not None:
            state_stop = state_slot_start + controller.config.state_slots
            prior_terminal_mask: Any | None = None
            if iteration > 0:
                hidden = mx.concatenate(
                    [
                        anchor[:, :state_slot_start, :],
                        hidden[:, state_slot_start:state_stop, :],
                        anchor[:, state_stop:, :],
                    ],
                    axis=1,
                )
            if last_decode_state is not None and state_probabilities is not None:
                prior_terminal_mask = (
                    mx.argmax(state_probabilities[:, -1, :], axis=-1) == 1
                )
            prior_state = hidden
            recurrent_input = hidden
            if recurrent_input_trajectory is not None:
                recurrent_input_trajectory.append(recurrent_input)
            with recurrent_iteration(iteration):
                action_logits = controller.action_logits(
                    problem_evidence,
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    step=iteration,
                    token_ids=tokens[:, :state_slot_start],
                    state_probabilities=state_probabilities,
                )
                action_probabilities = controller.straight_through_probabilities(
                    action_logits
                )
                if typed_action_lesion:
                    action_probabilities = controller.exact_probabilities(
                        (ACTION_NULL,) * (controller.config.action_slots - 1) + (0,),
                        slots=controller.config.action_slots,
                        cardinality=controller.config.action_cardinality,
                    )
                if (
                    action_teacher_values is not None
                    and state_teacher_forcing_probability > 0.0
                ):
                    teacher_action = controller.exact_probabilities(
                        action_teacher_values[iteration],
                        slots=controller.config.action_slots,
                        cardinality=controller.config.action_cardinality,
                    )
                    action_probabilities = (
                        (1.0 - state_teacher_forcing_probability)
                        * action_probabilities
                        + state_teacher_forcing_probability * teacher_action
                    )
                action_state = controller.commit_action_probabilities(
                    action_probabilities
                )
                state_logits = controller.state_transition_logits(
                    problem_evidence,
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    step=iteration,
                    action_state=action_state,
                    state_probabilities=state_probabilities,
                    action_probabilities=action_probabilities,
                )
                next_state_probabilities = (
                    controller.straight_through_probabilities(state_logits)
                )
                if (
                    state_teacher_values is not None
                    and state_teacher_forcing_probability > 0.0
                ):
                    teacher_state = controller.exact_probabilities(
                        state_teacher_values[iteration],
                        slots=controller.config.state_slots,
                        cardinality=controller.config.state_cardinality,
                    )
                    next_state_probabilities = (
                        (1.0 - state_teacher_forcing_probability)
                        * next_state_probabilities
                        + state_teacher_forcing_probability * teacher_state
                    )
                hidden = controller.commit_state_probabilities(
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    probabilities=next_state_probabilities,
                )
                state_probabilities = next_state_probabilities
                if state_probability_trajectory is not None:
                    state_probability_trajectory.append(state_probabilities)
                # State recurrence is cheap and explicit.  The resident
                # transformer is used once per candidate state as the shared
                # semantic answer bridge, never as the state transition itself.
                candidate = _run(window, hidden)
                candidate = controller.correction(candidate, iteration)
                candidate = controller.attend_answer_to_state(
                    candidate,
                    hidden,
                    state_slot_start=state_slot_start,
                    state_probabilities=state_probabilities,
                    role_logit_trajectory=answer_role_logit_trajectory,
                    place_logit_trajectory=answer_place_logit_trajectory,
                    binding_feature_trajectory=answer_binding_feature_trajectory,
                )
                if prior_terminal_mask is not None:
                    # A completed program stutters semantically as well as in
                    # its typed registers. Re-running the transformer window
                    # after termination used to move an already-correct answer
                    # representation even though the machine state was fixed.
                    # Preserve the associated neural emission policy too;
                    # keeping only the hidden state let later no-op iterations
                    # overwrite role/place logits and corrupt deep decoding.
                    candidate = mx.where(
                        prior_terminal_mask[:, None, None],
                        last_decode_state,
                        candidate,
                    )
                    for binding_trajectory in (
                        answer_role_logit_trajectory,
                        answer_place_logit_trajectory,
                    ):
                        if (
                            binding_trajectory is not None
                            and len(binding_trajectory) >= 2
                        ):
                            binding_trajectory[-1] = mx.where(
                                prior_terminal_mask[:, None, None],
                                binding_trajectory[-2],
                                binding_trajectory[-1],
                            )
            if state_logit_trajectory is not None:
                state_logit_trajectory.append(state_logits)
            if action_logit_trajectory is not None:
                action_logit_trajectory.append(action_logits)
            last_decode_state = candidate
            if decode_state_trajectory is not None:
                decode_state_trajectory.append(candidate)
            probability = controller.halt_probability(prior_state, hidden)
            transport_gate = mx.array(1.0, dtype=mx.float32)
            mx.eval(hidden, candidate, probability, transport_gate)
            halt_probabilities.append(float(probability.item()))
            memory_write_means.append(0.0)
            transport_gates.append(1.0)
            trajectory.append(hidden)
            if (
                adaptive_halt
                and iteration + 1 >= controller.config.minimum_iterations
                and halt_probabilities[-1] >= controller.config.halt_threshold
            ):
                halted = True
                halt_reason = "learned_threshold"
                break
            continue

        prior_state = hidden
        if iteration > 0:
            if plan.anchor_injection > 0.0:
                hidden = hidden + plan.anchor_injection * anchor
            if plan.interpass_noise > 0.0:
                key = mx.random.key(plan.noise_seed * 1_000_003 + iteration)
                kick = mx.random.normal(hidden.shape, key=key).astype(mx.float32)
                hidden = hidden + (
                    kick * (plan.interpass_noise * _rms(hidden))
                ).astype(hidden.dtype)
            if plan.renormalize:
                hidden = hidden * (anchor_rms / _rms(hidden)).astype(hidden.dtype)
        recurrent_input = hidden
        if recurrent_input_trajectory is not None:
            recurrent_input_trajectory.append(recurrent_input)
        with recurrent_iteration(iteration):
            candidate = _run(
                window,
                recurrent_input,
                caches["window"][iteration] if caches else None,
            )
            candidate = controller.correction(candidate, iteration)
            candidate, transport_gate = controller.transport(
                prior_state,
                candidate,
                iteration,
            )

        if memory_layout is not None and iteration > 0:
            probabilities = controller.memory_write_probabilities(
                prior_state,
                candidate,
            )
            if soft_memory_writes:
                gates = probabilities
            else:
                gates = mx.where(
                    probabilities >= controller.config.memory_write_threshold,
                    probabilities,
                    mx.zeros_like(probabilities),
                )
            hidden, applied = apply_protected_transition(
                prior_state,
                candidate,
                memory_layout,
                write_gate=gates,
            )
            memory_write_means.append(float(mx.mean(applied)))
        else:
            hidden = candidate
            memory_write_means.append(0.0)

        probability = controller.halt_probability(prior_state, hidden)
        mx.eval(hidden, probability, transport_gate)
        halt_probabilities.append(float(probability.item()))
        transport_gates.append(float(mx.mean(transport_gate).item()))
        trajectory.append(hidden)
        if (
            adaptive_halt
            and iteration + 1 >= controller.config.minimum_iterations
            and halt_probabilities[-1] >= controller.config.halt_threshold
        ):
            halted = True
            halt_reason = "learned_threshold"
            break

    final_source = last_decode_state if last_decode_state is not None else hidden
    final = _run(
        layers[plan.coda_start :],
        final_source,
        caches["coda"] if caches else None,
    )
    final = inner.norm(final)
    retention = None
    residuals: tuple[float, ...] = ()
    if memory_layout is not None and trajectory:
        retention = memory_retention(trajectory[0], trajectory[-1], memory_layout)
        residuals = tuple(semantic_convergence(trajectory, memory_layout))
    telemetry = UnifiedRecurrenceTelemetry(
        configured_iterations=plan.iterations,
        executed_iterations=len(trajectory),
        halt_probabilities=tuple(halt_probabilities),
        memory_write_means=tuple(memory_write_means),
        transport_gates=tuple(transport_gates),
        halted=halted,
        halt_reason=halt_reason,
        memory_retention=retention,
        semantic_residuals=residuals,
        controller_sha256=controller.parameter_sha256(),
    )
    return final, trajectory, telemetry


def apply_terminal_answer_grammar(
    logits: Any,
    tokens: Any,
    *,
    state_slot_start: int,
    state_probabilities: Any,
    contract: RecurrentAnswerEmissionContract,
) -> Any:
    """Constrain only canonical syntax around neurally emitted answer digits."""

    if (
        len(logits.shape) != 3
        or len(tokens.shape) != 2
        or len(state_probabilities.shape) != 3
        or int(logits.shape[0]) != int(tokens.shape[0])
        or int(logits.shape[0]) != int(state_probabilities.shape[0])
        or state_probabilities.shape[1:] != (
            len(STATE_SLOT_NAMES),
            STATE_CARDINALITY,
        )
        or type(state_slot_start) is not int
        or not 0 <= state_slot_start <= int(tokens.shape[1])
    ):
        raise ValueError("terminal answer grammar tensor layout differs")
    rows: list[Any] = []
    vocabulary = mx.arange(int(logits.shape[-1]))
    state_rows = mx.argmax(state_probabilities, axis=-1).tolist()
    token_rows = tokens.tolist()
    for batch_index, state_values in enumerate(state_rows):
        public_tokens = token_rows[batch_index][:state_slot_start]
        generated_tokens = token_rows[batch_index][state_slot_start:]
        template = contract.emission_template(public_tokens, state_values)
        row = logits[batch_index, -1, :]
        if template is not None:
            forced_token = contract.next_template_token(
                public_tokens,
                state_values,
                generated_tokens,
            )
            if forced_token is None:
                digit_tokens = mx.array(contract.digit_token_ids)
                allowed = mx.any(
                    digit_tokens[:, None] == vocabulary[None, :],
                    axis=0,
                )
                row = mx.where(allowed, row, -1e9).astype(logits.dtype)
            else:
                if forced_token >= int(logits.shape[-1]):
                    raise ValueError("answer grammar token is outside the vocabulary")
                row = mx.where(vocabulary == forced_token, 0.0, -1e9).astype(
                    logits.dtype
                )
        rows.append(row)
    constrained = mx.stack(rows, axis=0)
    return mx.concatenate([logits[:, :-1, :], constrained[:, None, :]], axis=1)


def unified_recurrent_logits(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    **kwargs: Any,
) -> tuple[Any, UnifiedRecurrenceTelemetry]:
    answer_emission_contract = kwargs.pop("answer_emission_contract", None)
    answer_digit_pointer_enabled = kwargs.pop("answer_digit_pointer_enabled", True)
    if answer_emission_contract is not None and not isinstance(
        answer_emission_contract, RecurrentAnswerEmissionContract
    ):
        raise TypeError("answer emission contract is invalid")
    if type(answer_digit_pointer_enabled) is not bool:
        raise TypeError("answer digit pointer flag must be bool")
    state_slot_start = kwargs.get("state_slot_start")
    pointer_enabled = (
        answer_digit_pointer_enabled
        and state_slot_start is not None
        and bool(controller.config.literal_digit_token_ids)
    )
    state_probabilities: list[Any] = kwargs.setdefault(
        "state_probability_trajectory", []
    )
    role_logits: list[Any] = kwargs.setdefault("answer_role_logit_trajectory", [])
    place_logits: list[Any] = kwargs.setdefault("answer_place_logit_trajectory", [])
    hidden, _trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        tokens,
        plan,
        controller,
        **kwargs,
    )
    if getattr(model, "lm_head", None) is not None:
        logits = model.lm_head(hidden)
    else:
        logits = model.model.embed_tokens.as_linear(hidden)
    if pointer_enabled:
        if not state_probabilities or not role_logits or not place_logits:
            raise RuntimeError("answer digit pointer emitted no recurrent trajectory")
        answer_start = int(state_slot_start) + controller.config.state_slots - 1
        pointed = controller.apply_answer_digit_pointer(
            logits[:, answer_start:, :],
            role_logits[-1],
            place_logits[-1],
            state_probabilities[-1],
        )
        logits = mx.concatenate([logits[:, :answer_start, :], pointed], axis=1)
    if answer_emission_contract is not None:
        if state_slot_start is None or not state_probabilities:
            raise RuntimeError("answer grammar emitted no recurrent typed state")
        if (
            tuple(answer_emission_contract.digit_token_ids)
            != tuple(controller.config.literal_digit_token_ids)
        ):
            raise RuntimeError("answer grammar and neural digit contracts differ")
        logits = apply_terminal_answer_grammar(
            logits,
            tokens,
            state_slot_start=int(state_slot_start),
            state_probabilities=state_probabilities[-1],
            contract=answer_emission_contract,
        )
    return logits, telemetry


__all__ = [
    "UNIFIED_INTRINSIC_RECURRENCE_SCHEMA",
    "UnifiedRecurrenceConfig",
    "UnifiedRecurrenceTelemetry",
    "UnifiedRecurrentController",
    "apply_terminal_answer_grammar",
    "unified_recurrent_hidden_states",
    "unified_recurrent_logits",
]
