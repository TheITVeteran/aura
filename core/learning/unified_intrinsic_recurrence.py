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

UNIFIED_INTRINSIC_RECURRENCE_SCHEMA: Final = "aura.unified_intrinsic_recurrence.v1"


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
    schema: str = UNIFIED_INTRINSIC_RECURRENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNIFIED_INTRINSIC_RECURRENCE_SCHEMA:
            raise ValueError("unified recurrence schema differs")
        for name in ("hidden_size", "correction_rank", "depth_basis_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.correction_rank > self.hidden_size:
            raise ValueError("correction rank exceeds hidden size")
        if type(self.minimum_iterations) is not int or self.minimum_iterations < 1:
            raise ValueError("minimum_iterations must be positive")
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
        key_a, key_b, key_depth, key_memory, key_halt = mx.random.split(
            mx.random.key(20260810198),
            num=5,
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
        # Re-entry starts conservative. Step zero bypasses this gate exactly,
        # retaining base-forward parity; later steps learn how much of the new
        # window state can be admitted without leaving the coda's manifold.
        self.transport_bias = mx.array(-0.5, dtype=mx.float32)
        self.halt_state_weight = (
            mx.random.normal((config.hidden_size,), key=key_halt).astype(mx.float32)
            * scale
        )
        self.halt_motion_weight = mx.array(0.0, dtype=mx.float32)
        self.halt_bias = mx.array(-6.0, dtype=mx.float32)

    def depth_features(self, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("recurrent step must be a non-negative integer")
        u = float(step) / float(step + 1) if step else 0.0
        return mx.array(
            [u ** (index + 1) for index in range(self.config.depth_basis_size)],
            dtype=mx.float32,
        )

    def correction(self, hidden: Any, step: int) -> Any:
        features = self.depth_features(step)
        scale = 1.0 + features @ self.depth_scale
        low_rank = (hidden.astype(mx.float32) @ self.correction_a) * scale
        delta = low_rank @ self.correction_b
        return hidden + delta.astype(hidden.dtype)

    def memory_write_probabilities(self, previous: Any, candidate: Any) -> Any:
        disagreement = (candidate - previous).astype(mx.float32)
        logits = disagreement @ self.memory_write_weight + self.memory_write_bias
        return mx.sigmoid(logits)[..., None]

    def transport_gate(self, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("transport step must be a non-negative integer")
        if step == 0:
            return mx.array(1.0, dtype=mx.float32)
        return mx.sigmoid(
            self.transport_bias
            + self.depth_features(step) @ self.transport_depth_weight
        )

    def transport(self, previous: Any, candidate: Any, step: int) -> tuple[Any, Any]:
        gate = self.transport_gate(step)
        if step == 0:
            return candidate, gate
        matched = candidate * (_rms(previous) / _rms(candidate)).astype(
            candidate.dtype
        )
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
            "transport_bias",
            "halt_state_weight",
            "halt_motion_weight",
            "halt_bias",
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
) -> tuple[Any, list[Any], UnifiedRecurrenceTelemetry]:
    """Run all Level-3 control mechanisms on one transformer trajectory."""

    if not isinstance(controller, UnifiedRecurrentController):
        raise TypeError("unified recurrence controller is invalid")
    if type(adaptive_halt) is not bool or type(soft_memory_writes) is not bool:
        raise TypeError("unified recurrence mode flags must be bools")
    if adaptive_halt and controller.config.minimum_iterations > plan.iterations:
        raise ValueError("minimum iterations exceed the recurrence plan")
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers or plan.coda_start > len(layers):
        raise ValueError("model layers do not satisfy the recurrence plan")

    hidden = inner.embed_tokens(tokens)
    hidden = _run(layers[: plan.prelude_end], hidden)
    anchor = hidden
    anchor_rms = _rms(anchor) if plan.renormalize else None
    if memory_layout is not None and memory_layout.n_slots != int(hidden.shape[1]):
        raise ValueError("protected memory layout differs from token positions")

    trajectory: list[Any] = []
    halt_probabilities: list[float] = []
    memory_write_means: list[float] = []
    transport_gates: list[float] = []
    window = layers[plan.prelude_end : plan.coda_start]
    halted = False
    halt_reason = "configured_depth_exhausted"
    for iteration in range(plan.iterations):
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
        with recurrent_iteration(iteration):
            candidate = _run(window, recurrent_input)
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
        transport_gates.append(float(transport_gate.item()))
        trajectory.append(hidden)
        if (
            adaptive_halt
            and iteration + 1 >= controller.config.minimum_iterations
            and halt_probabilities[-1] >= controller.config.halt_threshold
        ):
            halted = True
            halt_reason = "learned_threshold"
            break

    final = _run(layers[plan.coda_start :], hidden)
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


def unified_recurrent_logits(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    **kwargs: Any,
) -> tuple[Any, UnifiedRecurrenceTelemetry]:
    hidden, _trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        tokens,
        plan,
        controller,
        **kwargs,
    )
    if getattr(model, "lm_head", None) is not None:
        return model.lm_head(hidden), telemetry
    return model.model.embed_tokens.as_linear(hidden), telemetry


__all__ = [
    "UNIFIED_INTRINSIC_RECURRENCE_SCHEMA",
    "UnifiedRecurrenceConfig",
    "UnifiedRecurrenceTelemetry",
    "UnifiedRecurrentController",
    "unified_recurrent_hidden_states",
    "unified_recurrent_logits",
]
