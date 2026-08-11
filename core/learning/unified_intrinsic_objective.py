"""Semantic training objective for Aura's unified intrinsic recurrence."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from core.learning.intrinsic_recurrence import RecurrentDepthPlan, _run
from core.learning.protected_memory import MemoryLayout
from core.learning.unified_intrinsic_recurrence import (
    UnifiedRecurrentController,
    unified_recurrent_hidden_states,
)

UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA: Final = "aura.unified_intrinsic_objective.v1"


@dataclass(frozen=True, slots=True)
class UnifiedIntrinsicTrainingSpec:
    prelude_end: int
    coda_start: int
    train_depths: tuple[int, ...] = (1, 2, 4)
    heldout_depths: tuple[int, ...] = (8, 16)
    anchor_weight: float = 1.0
    trajectory_weight: float = 0.25
    progression_margin: float = 0.01
    halt_weight: float = 0.1
    anchor_injection: float = 0.0
    renormalize: bool = True
    schema: str = UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA:
            raise ValueError("unified objective schema differs")
        if self.prelude_end >= self.coda_start:
            raise ValueError("prelude_end must precede coda_start")
        if not self.train_depths or 1 not in self.train_depths:
            raise ValueError("train depths must include the T=1 anchor")
        if not self.heldout_depths:
            raise ValueError("heldout depths must not be empty")
        if any(type(depth) is not int or depth < 1 for depth in self.depths):
            raise ValueError("all recurrence depths must be positive integers")
        if set(self.train_depths) & set(self.heldout_depths):
            raise ValueError("train and heldout depths must be disjoint")
        if min(self.heldout_depths) <= max(self.train_depths):
            raise ValueError("heldout depths must extrapolate beyond training")
        for name in (
            "anchor_weight",
            "trajectory_weight",
            "progression_margin",
            "halt_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 10.0
            ):
                raise ValueError(f"{name} must be finite and inside [0, 10]")

    @property
    def depths(self) -> tuple[int, ...]:
        return self.train_depths + self.heldout_depths

    def plan_at(self, depth: int) -> RecurrentDepthPlan:
        if depth not in self.depths:
            raise ValueError("requested depth is outside the frozen ladder")
        return RecurrentDepthPlan(
            prelude_end=self.prelude_end,
            coda_start=self.coda_start,
            iterations=depth,
            anchor_injection=self.anchor_injection,
            renormalize=self.renormalize,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def readout_fingerprint(model: Any, coda_start: int) -> str:
    """Hash the coda, final norm, and LM head that must remain frozen."""

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers or type(coda_start) is not int or not 0 <= coda_start < len(layers):
        raise ValueError("readout fingerprint coda boundary is invalid")
    inventory: list[tuple[str, Any]] = []
    for layer_index in range(coda_start, len(layers)):
        inventory.extend(
            (f"model.layers.{layer_index}.{name}", value)
            for name, value in tree_flatten(layers[layer_index].parameters())
        )
    inventory.extend(
        (f"model.norm.{name}", value)
        for name, value in tree_flatten(inner.norm.parameters())
    )
    head = getattr(model, "lm_head", None)
    if head is not None:
        inventory.extend(
            (f"lm_head.{name}", value)
            for name, value in tree_flatten(head.parameters())
        )
    else:
        inventory.extend(
            (f"tied_readout.{name}", value)
            for name, value in tree_flatten(inner.embed_tokens.parameters())
        )
    digest = hashlib.sha256()
    for name, value in sorted(inventory, key=lambda row: row[0]):
        mx.eval(value)
        digest.update(name.encode("utf-8"))
        digest.update(bytes(memoryview(value)))
    return digest.hexdigest()


def _answer_ce_from_hidden(
    model: Any,
    hidden: Any,
    answer_tokens: Any,
    answer_start: int,
) -> Any:
    if getattr(model, "lm_head", None) is not None:
        logits = model.lm_head(hidden)
    else:
        logits = model.model.embed_tokens.as_linear(hidden)
    answer_count = int(answer_tokens.shape[-1])
    predicted = logits[:, answer_start : answer_start + answer_count, :]
    return mx.mean(
        nn.losses.cross_entropy(
            predicted.astype(mx.float32),
            answer_tokens,
            reduction="none",
        )
    )


def unified_answer_trajectory(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    *,
    memory_layout: MemoryLayout | None = None,
) -> tuple[list[Any], list[Any]]:
    """Decode every recurrent state through one frozen coda and readout."""

    if int(answer_tokens.shape[-1]) < 1:
        raise ValueError("answer tokens must not be empty")
    full = mx.concatenate([tokens, answer_tokens], axis=1)
    _final, trajectory, _telemetry = unified_recurrent_hidden_states(
        model,
        full,
        plan,
        controller,
        memory_layout=memory_layout,
        soft_memory_writes=True,
    )
    answer_start = int(full.shape[1]) - int(answer_tokens.shape[-1]) - 1
    hidden_states: list[Any] = []
    losses: list[Any] = []
    for state in trajectory:
        hidden = _run(model.model.layers[plan.coda_start :], state)
        hidden = model.model.norm(hidden)
        hidden_states.append(hidden)
        losses.append(
            _answer_ce_from_hidden(model, hidden, answer_tokens, answer_start)
        )
    return hidden_states, losses


def _progression_loss(losses: Sequence[Any], margin: float) -> Any:
    if len(losses) < 2:
        return mx.zeros(())
    return mx.mean(
        mx.stack(
            [
                mx.maximum(current - previous + margin, 0.0)
                for previous, current in zip(losses, losses[1:], strict=False)
            ]
        )
    )


def _halt_loss(
    controller: UnifiedRecurrentController,
    states: Sequence[Any],
    losses: Sequence[Any],
) -> Any:
    if len(states) < 2:
        return mx.zeros(())
    detached = [float(value.item()) for value in losses]
    best_index = min(range(len(detached)), key=detached.__getitem__)
    terms = []
    for index in range(1, len(states)):
        probability = controller.halt_probability(states[index - 1], states[index])
        target = mx.array(float(index >= best_index), dtype=mx.float32)
        terms.append(
            -(target * mx.log(probability + 1e-6))
            - (1.0 - target) * mx.log(1.0 - probability + 1e-6)
        )
    return mx.mean(mx.stack(terms))


def unified_intrinsic_training_loss(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    controller: UnifiedRecurrentController,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    memory_layout: MemoryLayout | None = None,
    readout_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train semantics at shallow depths while keeping readout immutable."""

    if readout_sha256 is None:
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
    elif re.fullmatch(r"[0-9a-f]{64}", readout_sha256) is None:
        raise ValueError("readout commitment is invalid")
    final_losses: list[Any] = []
    progression_terms: list[Any] = []
    halt_terms: list[Any] = []
    per_depth: dict[str, dict[str, Any]] = {}
    for depth in spec.train_depths:
        states, losses = unified_answer_trajectory(
            model,
            tokens,
            answer_tokens,
            spec.plan_at(depth),
            controller,
            memory_layout=memory_layout,
        )
        final_losses.append(losses[-1])
        progression = _progression_loss(losses, spec.progression_margin)
        halting = _halt_loss(controller, states, losses)
        progression_terms.append(progression)
        halt_terms.append(halting)
        per_depth[f"T{depth}"] = {
            "step_ce": [float(value.item()) for value in losses],
            "final_ce": float(losses[-1].item()),
            "progression_loss": float(progression.item()),
            "halt_loss": float(halting.item()),
        }
    anchor = final_losses[spec.train_depths.index(1)]
    final_mean = mx.mean(mx.stack(final_losses))
    progression_mean = mx.mean(mx.stack(progression_terms))
    halt_mean = mx.mean(mx.stack(halt_terms))
    total = (
        final_mean
        + spec.anchor_weight * anchor
        + spec.trajectory_weight * progression_mean
        + spec.halt_weight * halt_mean
    )
    return total, {
        "schema": UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA,
        "spec": spec.to_dict(),
        "per_depth": per_depth,
        "anchor_ce": float(anchor.item()),
        "final_mean_ce": float(final_mean.item()),
        "progression_loss": float(progression_mean.item()),
        "halt_loss": float(halt_mean.item()),
        "total": float(total.item()),
        "readout_sha256": readout_sha256,
        "readout_frozen_by_training_contract": True,
        "heldout_depths_unopened": list(spec.heldout_depths),
    }


__all__ = [
    "UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA",
    "UnifiedIntrinsicTrainingSpec",
    "readout_fingerprint",
    "unified_answer_trajectory",
    "unified_intrinsic_training_loss",
]
