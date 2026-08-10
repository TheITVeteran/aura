"""A small recurrence-native state machine over Aura's latent workspace."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

NATIVE_RECURRENT_CORE_SCHEMA = "aura.native_recurrent_transition_core.v1"


@dataclass(frozen=True, slots=True)
class RecurrentTransitionCoreConfig:
    """Frozen topology for the shared identity-initialized transition."""

    schema: str = NATIVE_RECURRENT_CORE_SCHEMA
    hidden_size: int = 1536
    bottleneck_size: int = 128
    attention_heads: int = 4
    control_slots: int = 3
    gate_bias: float = -2.0

    def __post_init__(self) -> None:
        if (
            self.schema != NATIVE_RECURRENT_CORE_SCHEMA
            or type(self.hidden_size) is not int
            or self.hidden_size < 8
            or type(self.bottleneck_size) is not int
            or self.bottleneck_size < 8
            or type(self.attention_heads) is not int
            or self.attention_heads < 1
            or self.bottleneck_size % self.attention_heads != 0
            or type(self.control_slots) is not int
            or self.control_slots != 3
            or isinstance(self.gate_bias, bool)
            or not isinstance(self.gate_bias, (int, float))
            or not math.isfinite(float(self.gate_bias))
            or not -12.0 <= float(self.gate_bias) <= 0.0
        ):
            raise ValueError("native recurrent core configuration is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecurrentTransitionCoreOutput:
    state: Any
    state_features: Any
    action_features: Any
    write_gate: Any
    delta: Any


class RecurrentTransitionCore(nn.Module):
    """One shared transition with read-only context and protected state lanes.

    The semantic prefix is copied byte-for-byte. Only the final three control
    slots can change. The output projection starts at exact zero, so attaching
    an untrained core is an identity operation rather than an uncalibrated
    intervention.
    """

    def __init__(self, config: RecurrentTransitionCoreConfig):
        super().__init__()
        self.config = config
        width = config.bottleneck_size
        hidden = config.hidden_size
        self.state_down = nn.Linear(hidden, width, bias=False)
        self.context_down = nn.Linear(hidden, width, bias=False)
        self.state_norm = nn.RMSNorm(width)
        self.context_norm = nn.RMSNorm(width)
        self.typed_state_norm = nn.RMSNorm(width)
        self.action_norm = nn.RMSNorm(width)
        self.cross_attention = nn.MultiHeadAttention(width, config.attention_heads)
        self.typed_state_attention = nn.MultiHeadAttention(
            width,
            config.attention_heads,
        )
        self.action_attention = nn.MultiHeadAttention(width, config.attention_heads)
        self.self_attention = nn.MultiHeadAttention(width, config.attention_heads)
        self.mixed_norm = nn.RMSNorm(width)
        self.ff_up = nn.Linear(width, width * 4, bias=False)
        self.ff_down = nn.Linear(width * 4, width, bias=False)
        self.output_norm = nn.RMSNorm(width)
        self.delta_up = nn.Linear(width, hidden, bias=False)
        self.write_gate = nn.Linear(width, 1, bias=True)

        self.delta_up.weight = mx.zeros_like(self.delta_up.weight)
        self.write_gate.weight = mx.zeros_like(self.write_gate.weight)
        self.write_gate.bias = mx.full_like(
            self.write_gate.bias,
            float(config.gate_bias),
        )

    def __call__(
        self,
        state: Any,
        context: Any,
        typed_state: Any,
        action: Any,
    ) -> RecurrentTransitionCoreOutput:
        if (
            state.ndim != 3
            or context.ndim != 3
            or typed_state.ndim != 3
            or action.ndim != 3
            or int(state.shape[0]) != int(context.shape[0])
            or int(state.shape[0]) != int(typed_state.shape[0])
            or int(state.shape[0]) != int(action.shape[0])
            or int(state.shape[-1]) != self.config.hidden_size
            or int(context.shape[-1]) != self.config.hidden_size
            or int(typed_state.shape[1]) != self.config.control_slots
            or int(typed_state.shape[-1]) != self.config.bottleneck_size
            or int(action.shape[1]) != self.config.control_slots
            or int(action.shape[-1]) != self.config.bottleneck_size
            or int(state.shape[1]) <= self.config.control_slots
            or int(context.shape[1]) < 1
        ):
            raise ValueError("native recurrent core tensor shape is invalid")
        split = int(state.shape[1]) - self.config.control_slots
        semantic = state[:, :split, :]
        control = state[:, split:, :]
        control_hidden = self.state_norm(self.state_down(control.astype(mx.float32)))
        context_hidden = self.context_norm(self.context_down(context.astype(mx.float32)))
        typed_state_hidden = self.typed_state_norm(typed_state.astype(mx.float32))
        action_hidden = self.action_norm(action.astype(mx.float32))
        attended = self.cross_attention(
            control_hidden,
            context_hidden,
            context_hidden,
        )
        action_attended = self.action_attention(
            control_hidden,
            action_hidden,
            action_hidden,
        )
        typed_state_attended = self.typed_state_attention(
            control_hidden,
            typed_state_hidden,
            typed_state_hidden,
        )
        reflected = self.self_attention(
            control_hidden,
            control_hidden,
            control_hidden,
        )
        mixed = self.mixed_norm(
            control_hidden
            + attended
            + reflected
            + typed_state_hidden
            + typed_state_attended
            + action_hidden
            + action_attended
        )
        transformed = self.ff_down(nn.gelu(self.ff_up(mixed)))
        state_features = self.output_norm(mixed + transformed)
        # The typed action is already an admitted causal operand. Preserve its
        # exact codebook direction rather than asking the core to reconstruct
        # an input it has just received.
        action_features = action_hidden
        delta = self.delta_up(state_features)
        write_gate = mx.sigmoid(self.write_gate(state_features))
        updated_control = control + (write_gate * delta).astype(control.dtype)
        updated = mx.concatenate([semantic, updated_control], axis=1)
        return RecurrentTransitionCoreOutput(
            state=updated,
            state_features=state_features,
            action_features=action_features,
            write_gate=write_gate,
            delta=delta,
        )


__all__ = [
    "NATIVE_RECURRENT_CORE_SCHEMA",
    "RecurrentTransitionCore",
    "RecurrentTransitionCoreConfig",
    "RecurrentTransitionCoreOutput",
]
