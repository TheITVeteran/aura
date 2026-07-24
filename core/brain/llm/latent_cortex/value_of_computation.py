"""Typed value-of-computation policy for recurrent cognitive actions.

The execution controller chooses an episode configuration.  This policy owns
the finer question asked *inside* that episode: given the public state signals
available now, which cognitive operator is expected to buy the most verified
progress per unit of compute?

No private reasoning text enters the policy or its receipts.  Decisions use
bounded scalar signals, independently checked transition outcomes, and an
explicit executor inventory.  An action without a real executor is in the
vocabulary but is never selectable.  Sparse bootstrap exploration is named as
such; it is never presented as learned evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.epistemic_state import OperationKind

VALUE_OF_COMPUTATION_SCHEMA = "aura.rlc.value_of_computation.v1"
ACTION_EVIDENCE_SCHEMA = "aura.rlc.value_of_computation.evidence.v1"
ACTION_TRANSITION_SCHEMA = "aura.rlc.value_of_computation.transition.v1"

MIN_ACTION_TRIALS = 8
MAX_ACTION_TRIALS = 100_000
_Z95 = 1.959963984540054
_EPSILON_COST = 1e-6

ACTION_VOCABULARY: tuple[OperationKind, ...] = tuple(OperationKind)


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"value-of-computation payload is not canonical: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return parsed


def _bounded_text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > limit
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"{name} must be bounded printable text")
    return normalized


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    """Sufficient statistics for independently measured action transitions."""

    n: int = 0
    gain_sum: float = 0.0
    gain_sq_sum: float = 0.0
    cost_sum: float = 0.0
    cost_sq_sum: float = 0.0

    def __post_init__(self) -> None:
        if type(self.n) is not int or not 0 <= self.n <= MAX_ACTION_TRIALS:
            raise ValueError("action evidence count is out of bounds")
        for name, value, maximum in (
            ("gain_sum", self.gain_sum, 4.0 * max(1, self.n)),
            ("gain_sq_sum", self.gain_sq_sum, 16.0 * max(1, self.n)),
            ("cost_sum", self.cost_sum, 1.0 * max(1, self.n)),
            ("cost_sq_sum", self.cost_sq_sum, 1.0 * max(1, self.n)),
        ):
            parsed = _finite(value, name=name, minimum=-maximum, maximum=maximum)
            if (name.endswith("sq_sum") or name.startswith("cost_")) and parsed < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, parsed)
        if self.n == 0 and any(
            abs(value) > 1e-12
            for value in (
                self.gain_sum,
                self.gain_sq_sum,
                self.cost_sum,
                self.cost_sq_sum,
            )
        ):
            raise ValueError("empty action evidence cannot carry measurements")
        if self.n > 0:
            minimum_gain_square = self.gain_sum * self.gain_sum / self.n
            minimum_cost_square = self.cost_sum * self.cost_sum / self.n
            moment_tolerance = 1e-9 * max(
                1.0,
                self.gain_sq_sum,
                minimum_gain_square,
                self.cost_sum,
                self.cost_sq_sum,
                minimum_cost_square,
            )
            if self.gain_sq_sum + moment_tolerance < minimum_gain_square:
                raise ValueError("action gain moments are mathematically inconsistent")
            if self.cost_sq_sum + moment_tolerance < minimum_cost_square:
                raise ValueError("action cost moments are mathematically inconsistent")
            if self.cost_sq_sum > self.cost_sum + moment_tolerance:
                raise ValueError("action cost moments exceed bounded observations")

    @classmethod
    def from_dict(cls, value: Any) -> ActionEvidence:
        fields = {"n", "gain_sum", "gain_sq_sum", "cost_sum", "cost_sq_sum"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("action evidence fields differ")
        return cls(**{name: value[name] for name in fields})

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n": self.n,
            "gain_sum": self.gain_sum,
            "gain_sq_sum": self.gain_sq_sum,
            "cost_sum": self.cost_sum,
            "cost_sq_sum": self.cost_sq_sum,
        }

    def append(self, *, gain: float, cost: float) -> ActionEvidence:
        if self.n >= MAX_ACTION_TRIALS:
            raise ValueError("action evidence trial cap is exhausted")
        measured_gain = _finite(gain, name="gain", minimum=-4.0, maximum=4.0)
        measured_cost = _finite(cost, name="cost", minimum=0.0, maximum=1.0)
        return ActionEvidence(
            n=self.n + 1,
            gain_sum=self.gain_sum + measured_gain,
            gain_sq_sum=self.gain_sq_sum + measured_gain * measured_gain,
            cost_sum=self.cost_sum + measured_cost,
            cost_sq_sum=self.cost_sq_sum + measured_cost * measured_cost,
        )

    @staticmethod
    def _bound(*, total: float, square_total: float, n: int, lower: bool) -> float:
        mean = total / n
        if n <= 1:
            return mean
        variance = max(0.0, (square_total - n * mean * mean) / (n - 1))
        radius = _Z95 * math.sqrt(variance / n)
        return mean - radius if lower else mean + radius

    def estimate(self) -> dict[str, int | float | bool]:
        if self.n == 0:
            return {
                "n": 0,
                "measured": False,
                "gain_mean": 0.0,
                "gain_lcb": -4.0,
                "cost_mean": 1.0,
                "cost_ucb": 1.0,
            }
        gain_mean = self.gain_sum / self.n
        cost_mean = self.cost_sum / self.n
        return {
            "n": self.n,
            "measured": self.n >= MIN_ACTION_TRIALS,
            "gain_mean": round(gain_mean, 8),
            "gain_lcb": round(
                max(
                    -4.0,
                    self._bound(
                        total=self.gain_sum,
                        square_total=self.gain_sq_sum,
                        n=self.n,
                        lower=True,
                    ),
                ),
                8,
            ),
            "cost_mean": round(cost_mean, 8),
            "cost_ucb": round(
                min(
                    1.0,
                    max(
                        _EPSILON_COST,
                        self._bound(
                            total=self.cost_sum,
                            square_total=self.cost_sq_sum,
                            n=self.n,
                            lower=False,
                        ),
                    ),
                ),
                8,
            ),
        }


def build_evidence_snapshot(
    *,
    bucket: str,
    cells: Mapping[OperationKind | str, ActionEvidence | Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the exact bounded evidence object sent to the worker."""

    normalized_bucket = _bounded_text(bucket, name="bucket", limit=160)
    normalized: dict[str, dict[str, int | float]] = {}
    for raw_action, raw_cell in cells.items():
        try:
            action = (
                raw_action if isinstance(raw_action, OperationKind) else OperationKind(raw_action)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown cognitive action: {raw_action!r}") from exc
        cell = (
            raw_cell if isinstance(raw_cell, ActionEvidence) else ActionEvidence.from_dict(raw_cell)
        )
        normalized[action.value] = cell.to_dict()
    payload = {
        "schema": ACTION_EVIDENCE_SCHEMA,
        "bucket": normalized_bucket,
        "cells": {name: normalized[name] for name in sorted(normalized)},
    }
    return {**payload, "snapshot_sha256": _canonical_sha256(payload)}


def validate_evidence_snapshot(value: Any) -> dict[str, Any]:
    fields = {"schema", "bucket", "cells", "snapshot_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action evidence snapshot fields differ")
    if value.get("schema") != ACTION_EVIDENCE_SCHEMA:
        raise ValueError("action evidence snapshot schema is invalid")
    bucket = _bounded_text(value.get("bucket"), name="bucket", limit=160)
    raw_cells = value.get("cells")
    if not isinstance(raw_cells, Mapping) or len(raw_cells) > len(ACTION_VOCABULARY):
        raise ValueError("action evidence snapshot cells are invalid")
    cells: dict[str, dict[str, int | float]] = {}
    for raw_action, raw_cell in raw_cells.items():
        try:
            action = OperationKind(raw_action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown action evidence cell: {raw_action!r}") from exc
        cells[action.value] = ActionEvidence.from_dict(raw_cell).to_dict()
    payload = {
        "schema": ACTION_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "cells": {name: cells[name] for name in sorted(cells)},
    }
    expected = _canonical_sha256(payload)
    if value.get("snapshot_sha256") != expected:
        raise ValueError("action evidence snapshot digest does not match")
    return {**payload, "snapshot_sha256": expected}


@dataclass(frozen=True, slots=True)
class CognitiveStateSignal:
    """Public, bounded features available at one recurrent decision point."""

    step_index: int
    max_steps: int
    neural_steps: int
    min_neural_steps: int
    active_branches: int
    total_branches: int
    residual: float
    residual_delta: float
    verifier_score: float | None
    verifier_delta: float | None
    disagreement: float
    uncertainty: float
    budget_remaining_fraction: float
    has_memory: bool
    has_evidence: bool
    has_verifier: bool
    has_savepoint: bool
    can_execute: bool
    answer_verified: bool
    irreducible_uncertainty: bool
    previously_selected: tuple[OperationKind, ...] = ()

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("step_index", self.step_index, 0),
            ("max_steps", self.max_steps, 1),
            ("neural_steps", self.neural_steps, 0),
            ("min_neural_steps", self.min_neural_steps, 1),
            ("active_branches", self.active_branches, 0),
            ("total_branches", self.total_branches, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.step_index > self.max_steps:
            raise ValueError("step_index exceeds max_steps")
        if self.neural_steps > self.max_steps or self.min_neural_steps > self.max_steps:
            raise ValueError("neural step bounds exceed max_steps")
        if self.active_branches > self.total_branches:
            raise ValueError("active branch count exceeds total")
        for name, value in (
            ("residual", self.residual),
            ("residual_delta", self.residual_delta),
            ("disagreement", self.disagreement),
            ("uncertainty", self.uncertainty),
            ("budget_remaining_fraction", self.budget_remaining_fraction),
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    value,
                    name=name,
                    minimum=-1.0 if name == "residual_delta" else 0.0,
                    maximum=1.0,
                ),
            )
        for name in (
            "has_memory",
            "has_evidence",
            "has_verifier",
            "has_savepoint",
            "can_execute",
            "answer_verified",
            "irreducible_uncertainty",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("verifier_score", "verifier_delta"):
            value = getattr(self, name)
            if value is not None:
                minimum = -1.0 if name == "verifier_delta" else 0.0
                object.__setattr__(
                    self,
                    name,
                    _finite(value, name=name, minimum=minimum, maximum=1.0),
                )
        if any(not isinstance(action, OperationKind) for action in self.previously_selected):
            raise ValueError("previously_selected contains an invalid action")
        if len(self.previously_selected) != self.step_index:
            raise ValueError("previously_selected must cover every prior action step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "max_steps": self.max_steps,
            "neural_steps": self.neural_steps,
            "min_neural_steps": self.min_neural_steps,
            "active_branches": self.active_branches,
            "total_branches": self.total_branches,
            "residual": self.residual,
            "residual_delta": self.residual_delta,
            "verifier_score": self.verifier_score,
            "verifier_delta": self.verifier_delta,
            "disagreement": self.disagreement,
            "uncertainty": self.uncertainty,
            "budget_remaining_fraction": self.budget_remaining_fraction,
            "has_memory": self.has_memory,
            "has_evidence": self.has_evidence,
            "has_verifier": self.has_verifier,
            "has_savepoint": self.has_savepoint,
            "can_execute": self.can_execute,
            "answer_verified": self.answer_verified,
            "irreducible_uncertainty": self.irreducible_uncertainty,
            "previously_selected": [action.value for action in self.previously_selected],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CognitiveStateSignal:
        fields = {
            "step_index",
            "max_steps",
            "neural_steps",
            "min_neural_steps",
            "active_branches",
            "total_branches",
            "residual",
            "residual_delta",
            "verifier_score",
            "verifier_delta",
            "disagreement",
            "uncertainty",
            "budget_remaining_fraction",
            "has_memory",
            "has_evidence",
            "has_verifier",
            "has_savepoint",
            "can_execute",
            "answer_verified",
            "irreducible_uncertainty",
            "previously_selected",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("cognitive state signal fields differ")
        previous = value.get("previously_selected")
        if not isinstance(previous, list) or len(previous) > int(value.get("max_steps") or 0):
            raise ValueError("cognitive state previous actions are invalid")
        try:
            previous_actions = tuple(OperationKind(item) for item in previous)
        except (TypeError, ValueError) as exc:
            raise ValueError("cognitive state previous actions are invalid") from exc
        return cls(
            **{name: value[name] for name in fields - {"previously_selected"}},
            previously_selected=previous_actions,
        )


_BASE_COST: dict[OperationKind, float] = {
    OperationKind.DECOMPOSE: 0.10,
    OperationKind.BLIND_RESOLVE: 0.14,
    OperationKind.BRANCH: 0.22,
    OperationKind.SEARCH_MEMORY: 0.05,
    OperationKind.RETRIEVE_EVIDENCE: 0.08,
    OperationKind.EXECUTE: 0.18,
    OperationKind.SIMULATE: 0.16,
    OperationKind.FALSIFY: 0.13,
    OperationKind.CHECK_ASSUMPTION: 0.12,
    OperationKind.REGENERATE_FROM_PREFIX: 0.17,
    OperationKind.FORMALIZE: 0.15,
    OperationKind.COMPARE: 0.10,
    OperationKind.BACKTRACK: 0.03,
    OperationKind.COMPRESS_STATE: 0.04,
    OperationKind.ANSWER: 0.01,
    OperationKind.ABSTAIN: 0.01,
}


def _bootstrap_gain(action: OperationKind, state: CognitiveStateSignal) -> float:
    """Conservative structural prior used only while measurements are sparse."""

    progress = state.step_index / max(1, state.max_steps)
    regression = max(0.0, -float(state.verifier_delta or 0.0))
    stagnation = max(0.0, state.residual - max(0.0, state.residual_delta))
    values = {
        OperationKind.DECOMPOSE: 0.32 if state.step_index == 0 else 0.08,
        OperationKind.BLIND_RESOLVE: 0.20 + 0.12 * state.uncertainty,
        OperationKind.BRANCH: 0.18 + 0.35 * state.disagreement,
        OperationKind.SEARCH_MEMORY: 0.28 if state.has_memory else -1.0,
        OperationKind.RETRIEVE_EVIDENCE: 0.30 if state.has_evidence else -1.0,
        OperationKind.EXECUTE: 0.34 if state.can_execute else -1.0,
        OperationKind.SIMULATE: 0.18 + 0.20 * state.uncertainty,
        OperationKind.FALSIFY: 0.16 + 0.34 * state.disagreement,
        OperationKind.CHECK_ASSUMPTION: 0.18 + 0.30 * state.uncertainty,
        OperationKind.REGENERATE_FROM_PREFIX: 0.08 + 0.55 * regression,
        OperationKind.FORMALIZE: 0.16 + 0.16 * state.uncertainty,
        OperationKind.COMPARE: 0.12 + 0.40 * state.disagreement,
        OperationKind.BACKTRACK: 0.06 + 0.65 * max(regression, stagnation),
        OperationKind.COMPRESS_STATE: 0.04 + 0.22 * progress,
        OperationKind.ANSWER: 0.75 if state.answer_verified else 0.04 * progress,
        OperationKind.ABSTAIN: (
            0.70 if state.irreducible_uncertainty else 0.25 * state.uncertainty * progress
        ),
    }
    return values[action]


def feasible_actions(
    state: CognitiveStateSignal,
    *,
    executors: tuple[OperationKind, ...],
) -> tuple[OperationKind, ...]:
    available = set(executors)
    neural_actions = {
        OperationKind.DECOMPOSE,
        OperationKind.BLIND_RESOLVE,
        OperationKind.BRANCH,
        OperationKind.SEARCH_MEMORY,
        OperationKind.RETRIEVE_EVIDENCE,
        OperationKind.SIMULATE,
        OperationKind.FALSIFY,
        OperationKind.CHECK_ASSUMPTION,
        OperationKind.REGENERATE_FROM_PREFIX,
        OperationKind.FORMALIZE,
    }
    neural_floor_pending = state.neural_steps < state.min_neural_steps
    feasible: list[OperationKind] = []
    for action in ACTION_VOCABULARY:
        if action not in available:
            continue
        if neural_floor_pending and action not in neural_actions:
            continue
        allowed = True
        if action is OperationKind.DECOMPOSE:
            allowed = state.step_index <= 1 and action not in state.previously_selected
        elif action is OperationKind.BRANCH:
            allowed = state.active_branches >= 2
        elif action is OperationKind.SEARCH_MEMORY:
            allowed = state.has_memory and action not in state.previously_selected
        elif action is OperationKind.RETRIEVE_EVIDENCE:
            allowed = state.has_evidence and action not in state.previously_selected
        elif action is OperationKind.EXECUTE:
            allowed = state.can_execute
        elif action in {OperationKind.FALSIFY, OperationKind.CHECK_ASSUMPTION}:
            allowed = state.has_verifier
        elif action in {
            OperationKind.REGENERATE_FROM_PREFIX,
            OperationKind.BACKTRACK,
        }:
            allowed = state.has_savepoint and (
                float(state.verifier_delta or 0.0) < -1e-9 or state.residual_delta < -1e-9
            )
        elif action is OperationKind.COMPARE:
            allowed = state.active_branches >= 2
        elif action is OperationKind.COMPRESS_STATE:
            allowed = state.step_index >= 2
        elif action is OperationKind.ANSWER:
            allowed = state.step_index >= 2
        elif action is OperationKind.ABSTAIN:
            allowed = state.step_index >= 2
        elif action in {
            OperationKind.SIMULATE,
            OperationKind.FORMALIZE,
        }:
            allowed = state.active_branches >= 1
        elif action is OperationKind.BLIND_RESOLVE:
            allowed = state.active_branches == 1
        if allowed:
            feasible.append(action)
    return tuple(feasible)


class ValueOfComputationPolicy:
    """Select one executable action from measured gain-per-cost evidence."""

    def __init__(self, evidence_snapshot: Mapping[str, Any]) -> None:
        snapshot = validate_evidence_snapshot(evidence_snapshot)
        self.snapshot = snapshot
        self.bucket = snapshot["bucket"]
        self.cells = {
            OperationKind(name): ActionEvidence.from_dict(cell)
            for name, cell in snapshot["cells"].items()
        }

    def choose(
        self,
        state: CognitiveStateSignal,
        *,
        executors: tuple[OperationKind, ...],
    ) -> dict[str, Any]:
        feasible = feasible_actions(state, executors=executors)
        if not feasible:
            raise ValueError("no executable cognitive action is feasible")

        # Terminal rules are explicit, preregistered, and dominate exploration.
        if OperationKind.ANSWER in feasible and state.answer_verified:
            chosen = OperationKind.ANSWER
            mode = "verified_stop"
        elif state.budget_remaining_fraction <= 0.03:
            if OperationKind.ANSWER in feasible and state.uncertainty <= 0.35:
                chosen = OperationKind.ANSWER
                mode = "budget_stop"
            elif OperationKind.ABSTAIN in feasible:
                chosen = OperationKind.ABSTAIN
                mode = "budget_abstain"
            else:
                chosen = feasible[0]
                mode = "budget_last_action"
        elif state.irreducible_uncertainty and OperationKind.ABSTAIN in feasible:
            chosen = OperationKind.ABSTAIN
            mode = "irreducible_abstain"
        else:
            scored: list[tuple[float, str, OperationKind, dict[str, Any]]] = []
            for action in feasible:
                cell = self.cells.get(action, ActionEvidence())
                estimate = cell.estimate()
                measured = bool(estimate["measured"])
                gain = float(estimate["gain_lcb"]) if measured else _bootstrap_gain(action, state)
                cost = float(estimate["cost_ucb"]) if measured else _BASE_COST[action]
                value = gain / max(_EPSILON_COST, cost)
                scored.append(
                    (
                        value,
                        action.value,
                        action,
                        {
                            **estimate,
                            "basis": "measured_lcb_per_cost_ucb" if measured else "bootstrap_prior",
                            "gain_used": round(gain, 8),
                            "cost_used": round(cost, 8),
                            "value": round(value, 8),
                        },
                    )
                )

            # Sparse deterministic exploration only among non-terminal actions.
            under_sampled = sorted(
                (
                    action
                    for action in feasible
                    if action not in {OperationKind.ANSWER, OperationKind.ABSTAIN}
                    and self.cells.get(action, ActionEvidence()).n < MIN_ACTION_TRIALS
                ),
                key=lambda action: (
                    self.cells.get(action, ActionEvidence()).n,
                    action.value,
                ),
            )
            if under_sampled and (state.step_index + 1) % 4 == 0:
                chosen = under_sampled[0]
                mode = "bounded_explore"
            else:
                _, _, chosen, _ = max(scored, key=lambda row: (row[0], row[1]))
                chosen_cell = self.cells.get(chosen, ActionEvidence())
                mode = "measured" if chosen_cell.n >= MIN_ACTION_TRIALS else "bootstrap"

        selected_cell = self.cells.get(chosen, ActionEvidence())
        selected_estimate = selected_cell.estimate()
        gain_used = (
            float(selected_estimate["gain_lcb"])
            if selected_estimate["measured"]
            else _bootstrap_gain(chosen, state)
        )
        cost_used = (
            float(selected_estimate["cost_ucb"])
            if selected_estimate["measured"]
            else _BASE_COST[chosen]
        )
        decision = {
            "schema": VALUE_OF_COMPUTATION_SCHEMA,
            "bucket": self.bucket,
            "snapshot_sha256": self.snapshot["snapshot_sha256"],
            "step_index": state.step_index,
            "action": chosen.value,
            "mode": mode,
            "feasible_actions": [action.value for action in feasible],
            "evidence": {
                **selected_estimate,
                "basis": (
                    "measured_lcb_per_cost_ucb"
                    if selected_estimate["measured"]
                    else "bootstrap_prior"
                ),
                "gain_used": round(gain_used, 8),
                "cost_used": round(cost_used, 8),
                "value": round(gain_used / max(_EPSILON_COST, cost_used), 8),
            },
        }
        decision["decision_sha256"] = _canonical_sha256(decision)
        return decision


def transition_reward(
    *,
    verified_delta: float,
    information_gain: float,
    diversity_gain: float,
    unsupported_confidence: float,
    cost: float,
) -> dict[str, Any]:
    """Measure Spark's progress reward from public before/after signals."""

    verified = _finite(
        verified_delta,
        name="verified_delta",
        minimum=-1.0,
        maximum=1.0,
    )
    information = _finite(
        information_gain,
        name="information_gain",
        minimum=-1.0,
        maximum=1.0,
    )
    diversity = _finite(
        diversity_gain,
        name="diversity_gain",
        minimum=-1.0,
        maximum=1.0,
    )
    gaming = _finite(
        unsupported_confidence,
        name="unsupported_confidence",
        minimum=0.0,
        maximum=1.0,
    )
    measured_cost = _finite(cost, name="cost", minimum=0.0, maximum=1.0)
    gain = verified + 0.35 * information + 0.20 * diversity - 0.55 * gaming
    reward = gain - 0.20 * measured_cost
    return {
        "verified_delta": round(verified, 8),
        "information_gain": round(information, 8),
        "diversity_gain": round(diversity, 8),
        "unsupported_confidence": round(gaming, 8),
        "cost": round(measured_cost, 8),
        "gain": round(gain, 8),
        "reward": round(reward, 8),
    }


def validate_action_decision(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "bucket",
        "snapshot_sha256",
        "step_index",
        "action",
        "mode",
        "feasible_actions",
        "evidence",
        "decision_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action decision fields differ")
    normalized = dict(value)
    if normalized.get("schema") != VALUE_OF_COMPUTATION_SCHEMA:
        raise ValueError("action decision schema is invalid")
    normalized["bucket"] = _bounded_text(normalized.get("bucket"), name="bucket", limit=160)
    for name in ("snapshot_sha256", "decision_sha256"):
        digest = normalized.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"action decision {name} is invalid")
    if type(normalized.get("step_index")) is not int or normalized["step_index"] < 0:
        raise ValueError("action decision step_index is invalid")
    try:
        selected = OperationKind(normalized.get("action"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action decision selected action is invalid") from exc
    feasible_raw = normalized.get("feasible_actions")
    if (
        not isinstance(feasible_raw, list)
        or not feasible_raw
        or len(feasible_raw) > len(ACTION_VOCABULARY)
    ):
        raise ValueError("action decision feasible actions are invalid")
    try:
        feasible = tuple(OperationKind(item) for item in feasible_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("action decision feasible actions are invalid") from exc
    if len(set(feasible)) != len(feasible) or selected not in feasible:
        raise ValueError("action decision selected an infeasible or duplicate action")
    normalized["action"] = selected.value
    normalized["feasible_actions"] = [action.value for action in feasible]
    normalized["mode"] = _bounded_text(normalized.get("mode"), name="mode", limit=32)
    if normalized["mode"] not in {
        "verified_stop",
        "budget_stop",
        "budget_abstain",
        "budget_last_action",
        "irreducible_abstain",
        "bounded_explore",
        "measured",
        "bootstrap",
    }:
        raise ValueError("action decision mode is unsupported")
    evidence_fields = {
        "n",
        "measured",
        "gain_mean",
        "gain_lcb",
        "cost_mean",
        "cost_ucb",
        "basis",
        "gain_used",
        "cost_used",
        "value",
    }
    evidence = normalized.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise ValueError("action decision evidence fields differ")
    evidence = dict(evidence)
    if type(evidence.get("n")) is not int or not 0 <= evidence["n"] <= MAX_ACTION_TRIALS:
        raise ValueError("action decision evidence count is invalid")
    if type(evidence.get("measured")) is not bool:
        raise ValueError("action decision measured flag is invalid")
    if evidence["measured"] is not (evidence["n"] >= MIN_ACTION_TRIALS):
        raise ValueError("action decision measured flag contradicts trial count")
    expected_basis = "measured_lcb_per_cost_ucb" if evidence["measured"] else "bootstrap_prior"
    if evidence.get("basis") != expected_basis:
        raise ValueError("action decision evidence basis is contradictory")
    bounds = {
        "gain_mean": (-4.0, 4.0),
        "gain_lcb": (-4.0, 4.0),
        "cost_mean": (0.0, 1.0),
        "cost_ucb": (_EPSILON_COST, 1.0),
        "gain_used": (-4.0, 4.0),
        "cost_used": (_EPSILON_COST, 1.0),
        "value": (-4.0 / _EPSILON_COST, 4.0 / _EPSILON_COST),
    }
    for name, (minimum, maximum) in bounds.items():
        evidence[name] = _finite(
            evidence.get(name),
            name=f"evidence.{name}",
            minimum=minimum,
            maximum=maximum,
        )
    expected_value = evidence["gain_used"] / evidence["cost_used"]
    if abs(evidence["value"] - expected_value) > 1e-6:
        raise ValueError("action decision value does not equal gain per cost")
    normalized["evidence"] = evidence
    payload = {name: normalized[name] for name in fields - {"decision_sha256"}}
    if normalized["decision_sha256"] != _canonical_sha256(payload):
        raise ValueError("action decision digest does not match")
    return normalized


def validate_action_transition(
    value: Any,
    *,
    require_checked: bool = True,
) -> dict[str, Any]:
    fields = {
        "schema",
        "bucket",
        "snapshot_sha256",
        "decision_sha256",
        "step_index",
        "action",
        "mode",
        "outcome",
        "checked",
        "metrics",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action transition fields differ")
    if value.get("schema") != ACTION_TRANSITION_SCHEMA:
        raise ValueError("action transition schema is invalid")
    normalized = dict(value)
    normalized["bucket"] = _bounded_text(value.get("bucket"), name="bucket", limit=160)
    for name in ("snapshot_sha256", "decision_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"action transition {name} is invalid")
    if type(value.get("step_index")) is not int or value["step_index"] < 0:
        raise ValueError("action transition step_index is invalid")
    try:
        action = OperationKind(value.get("action"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action transition action is invalid") from exc
    normalized["action"] = action.value
    normalized["mode"] = _bounded_text(value.get("mode"), name="mode", limit=32)
    normalized["outcome"] = _bounded_text(value.get("outcome"), name="outcome", limit=64)
    if type(value.get("checked")) is not bool:
        raise ValueError("action transition checked flag is invalid")
    if require_checked and value.get("checked") is not True:
        raise ValueError("action transition is not independently checked")
    metrics = value.get("metrics")
    expected_metrics = {
        "verified_delta",
        "information_gain",
        "diversity_gain",
        "unsupported_confidence",
        "cost",
        "gain",
        "reward",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_metrics:
        raise ValueError("action transition metrics fields differ")
    recomputed = transition_reward(
        verified_delta=metrics["verified_delta"],
        information_gain=metrics["information_gain"],
        diversity_gain=metrics["diversity_gain"],
        unsupported_confidence=metrics["unsupported_confidence"],
        cost=metrics["cost"],
    )
    if any(abs(float(metrics[name]) - float(recomputed[name])) > 1e-7 for name in recomputed):
        raise ValueError("action transition reward does not match measured components")
    normalized["metrics"] = recomputed
    return normalized


def validate_action_trace_row(
    value: Any,
    *,
    evidence_snapshot: Mapping[str, Any],
    executors: tuple[OperationKind, ...],
) -> dict[str, Any]:
    """Recompute one worker action from public state and transition signals."""

    base_fields = {
        "decision",
        "transition",
        "state_signal",
        "state_before",
        "state_after",
        "affected_branches",
        "verification",
    }
    constraint_fields = {
        "transient_constraint",
        "transient_constraint_attempt",
    }
    actual_fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if not isinstance(value, Mapping) or actual_fields not in {
        frozenset(base_fields),
        frozenset(base_fields | constraint_fields),
    }:
        raise ValueError("cognitive action trace fields differ")
    transient_constraint = value.get("transient_constraint", {})
    transient_attempt = value.get("transient_constraint_attempt", {})
    if not isinstance(transient_constraint, Mapping) or not isinstance(
        transient_attempt,
        Mapping,
    ):
        raise ValueError("cognitive action transient evidence is invalid")
    for evidence, schema, digest_field in (
        (
            transient_constraint,
            "aura.rlc.transient_constraint_application.v1",
            "application_sha256",
        ),
        (
            transient_attempt,
            "aura.rlc.transient_constraint_attempt.v1",
            "attempt_sha256",
        ),
    ):
        if evidence:
            payload = dict(evidence)
            digest = payload.pop(digest_field, None)
            if (
                payload.get("schema") != schema
                or not isinstance(digest, str)
                or digest != _canonical_sha256(payload)
            ):
                raise ValueError("cognitive action transient evidence digest differs")
    if not isinstance(executors, tuple) or not executors:
        raise ValueError("cognitive action executor inventory is invalid")
    if len(set(executors)) != len(executors) or any(
        not isinstance(action, OperationKind) for action in executors
    ):
        raise ValueError("cognitive action executor inventory is invalid")

    signal = CognitiveStateSignal.from_dict(value.get("state_signal"))
    decision = validate_action_decision(value.get("decision"))
    expected_decision = ValueOfComputationPolicy(evidence_snapshot).choose(
        signal,
        executors=executors,
    )
    if decision != expected_decision:
        raise ValueError("cognitive action decision does not match policy and state")
    transition = validate_action_transition(
        value.get("transition"),
        require_checked=False,
    )
    if (
        transition["decision_sha256"] != decision["decision_sha256"]
        or transition["step_index"] != decision["step_index"]
        or transition["action"] != decision["action"]
        or transition["mode"] != decision["mode"]
    ):
        raise ValueError("cognitive action transition differs from decision")

    def normalized_state(raw: Any, *, after: bool) -> dict[str, float | None]:
        expected = {"residual", "disagreement", "verifier_score"}
        if after:
            expected.add("observed_verifier_score")
        else:
            expected.add("budget_remaining_fraction")
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError("cognitive action public state fields differ")
        normalized = {
            "residual": _finite(
                raw.get("residual"), name="trace residual", minimum=0.0, maximum=1.0
            ),
            "disagreement": _finite(
                raw.get("disagreement"),
                name="trace disagreement",
                minimum=0.0,
                maximum=1.0,
            ),
        }
        verifier_score = raw.get("verifier_score")
        normalized["verifier_score"] = (
            None
            if verifier_score is None
            else _finite(
                verifier_score,
                name="trace verifier score",
                minimum=0.0,
                maximum=1.0,
            )
        )
        if not after:
            normalized["budget_remaining_fraction"] = _finite(
                raw.get("budget_remaining_fraction"),
                name="trace remaining budget",
                minimum=0.0,
                maximum=1.0,
            )
        else:
            observed_score = raw.get("observed_verifier_score")
            normalized["observed_verifier_score"] = (
                None
                if observed_score is None
                else _finite(
                    observed_score,
                    name="trace observed verifier score",
                    minimum=0.0,
                    maximum=1.0,
                )
            )
        return normalized

    before = normalized_state(value.get("state_before"), after=False)
    after = normalized_state(value.get("state_after"), after=True)
    for name, expected in (
        ("residual", signal.residual),
        ("disagreement", signal.disagreement),
        ("budget_remaining_fraction", signal.budget_remaining_fraction),
    ):
        if abs(float(before[name]) - expected) > 1e-7:
            raise ValueError("cognitive action state signal differs from trace")
    if before["verifier_score"] != signal.verifier_score:
        raise ValueError("cognitive action verifier state differs from trace")

    before_score = before["verifier_score"]
    after_score = after["verifier_score"]
    observed_score = after["observed_verifier_score"]
    raw_verification = value.get("verification")
    verification_decision = (
        raw_verification.get("decision") if isinstance(raw_verification, Mapping) else None
    )
    checked = before_score is not None and observed_score is not None
    if transition["checked"] is not checked:
        raise ValueError("cognitive action checked status differs from public state")
    verified_delta = float(observed_score) - float(before_score) if checked else 0.0
    if verification_decision == "preserve_verified":
        if before_score is None or observed_score is None:
            raise ValueError("verified-best preservation lacks comparable scores")
        expected_accepted_score = before_score
    elif verification_decision == "reject_verified_failure":
        expected_accepted_score = before_score
    elif observed_score is None:
        expected_accepted_score = before_score
    elif before_score is not None and observed_score < before_score - 1e-9:
        if "regression_reverted" not in transition["outcome"]:
            raise ValueError("cognitive action verifier regression was not reverted")
        expected_accepted_score = before_score
    else:
        expected_accepted_score = observed_score
    if after_score != expected_accepted_score:
        raise ValueError("cognitive action accepted verifier state is invalid")
    before_uncertainty = max(
        float(before["residual"]),
        float(before["disagreement"]),
        1.0 - float(before_score) if before_score is not None else 1.0,
    )
    after_uncertainty = max(
        float(after["residual"]),
        float(after["disagreement"]),
        1.0 - float(observed_score) if observed_score is not None else before_uncertainty,
    )
    expected_metrics = transition_reward(
        verified_delta=max(-1.0, min(1.0, verified_delta)),
        information_gain=max(
            -1.0,
            min(1.0, before_uncertainty - after_uncertainty),
        ),
        diversity_gain=max(
            -1.0,
            min(
                1.0,
                float(after["disagreement"]) - float(before["disagreement"]),
            ),
        ),
        unsupported_confidence=(max(0.0, min(1.0, -verified_delta)) if checked else 0.0),
        cost=transition["metrics"]["cost"],
    )
    if transition["metrics"] != expected_metrics:
        raise ValueError("cognitive action metrics differ from public transition")
    affected_branches = value.get("affected_branches")
    if type(affected_branches) is not int or not 0 <= affected_branches <= signal.total_branches:
        raise ValueError("cognitive action affected branch count is invalid")
    verification = value.get("verification")
    verification_base_fields = {
        "target_branch",
        "observation",
        "decision",
        "restored",
    }
    verification_lineage_fields = {
        "attempt_parent_state_sha256",
        "constraint_input_state_sha256",
        "candidate_state_sha256",
        "restore_target_state_sha256",
        "kv_boundary_before_sha256",
        "kv_boundary_after_sha256",
        "branch_step_before",
        "branch_step_after",
    }
    if not isinstance(verification, Mapping) or frozenset(verification) not in {
        frozenset(verification_base_fields),
        frozenset(verification_base_fields | verification_lineage_fields),
    }:
        raise ValueError("cognitive action verification fields differ")
    verification = dict(verification)
    has_lineage = set(verification) == (verification_base_fields | verification_lineage_fields)
    if verification["target_branch"] is None:
        if (
            verification["observation"] != {}
            or verification["decision"] != "not_run"
            or verification["restored"] is not False
            or observed_score is not None
            or (
                has_lineage
                and (
                    verification["attempt_parent_state_sha256"]
                    or verification["constraint_input_state_sha256"]
                    or verification["candidate_state_sha256"]
                    or verification["restore_target_state_sha256"]
                    or verification["kv_boundary_before_sha256"]
                    or verification["kv_boundary_after_sha256"]
                    or verification["branch_step_before"] is not None
                    or verification["branch_step_after"] is not None
                )
            )
        ):
            raise ValueError("absent cognitive verification is contradictory")
    else:
        from core.brain.llm.latent_cortex.verified_best import (
            validate_observation,
        )

        if (
            type(verification["target_branch"]) is not int
            or not 0 <= verification["target_branch"] < signal.total_branches
            or verification["decision"]
            not in {
                "ranking_only",
                "promote",
                "preserve_verified",
                "reject_verified_failure",
            }
            or type(verification["restored"]) is not bool
            or (
                has_lineage
                and (
                    not _is_sha256(verification["attempt_parent_state_sha256"])
                    or not _is_sha256(verification["constraint_input_state_sha256"])
                    or not _is_sha256(verification["candidate_state_sha256"])
                    or not _is_sha256(verification["kv_boundary_before_sha256"])
                    or not _is_sha256(verification["kv_boundary_after_sha256"])
                    or type(verification["branch_step_before"]) is not int
                    or verification["branch_step_before"] < 0
                    or verification["branch_step_after"] != verification["branch_step_before"] + 1
                )
            )
        ):
            raise ValueError("cognitive verification decision is invalid")
        verification["observation"] = validate_observation(verification["observation"])
        if (
            observed_score is None
            or abs(float(verification["observation"]["score"]) - float(observed_score)) > 1e-9
            or (
                verification["decision"] == "ranking_only"
                and verification["observation"]["authoritative"]
            )
            or (
                verification["decision"] != "ranking_only"
                and not verification["observation"]["authoritative"]
            )
            or (
                verification["decision"] not in {"preserve_verified", "reject_verified_failure"}
                and verification["restored"]
            )
            or (
                verification["decision"] == "reject_verified_failure"
                and (
                    verification["restored"] is not True
                    or verification["observation"]["basis"]
                    not in {"deterministic_exact", "calibrated_interval"}
                    or float(verification["observation"]["upper_bound"]) > 1e-9
                    or (
                        has_lineage
                        and (
                            not _is_sha256(verification["restore_target_state_sha256"])
                            or verification["restore_target_state_sha256"]
                            != verification["attempt_parent_state_sha256"]
                        )
                    )
                )
            )
            or (
                has_lineage
                and verification["decision"] != "reject_verified_failure"
                and verification["restore_target_state_sha256"]
            )
        ):
            raise ValueError("cognitive verification evidence is contradictory")
    normalized = {
        "decision": decision,
        "transition": transition,
        "state_signal": signal.to_dict(),
        "state_before": before,
        "state_after": after,
        "affected_branches": affected_branches,
        "verification": verification,
    }
    if actual_fields == frozenset(base_fields | constraint_fields):
        normalized["transient_constraint"] = dict(transient_constraint)
        normalized["transient_constraint_attempt"] = dict(transient_attempt)
    return normalized


__all__ = [
    "ACTION_EVIDENCE_SCHEMA",
    "ACTION_TRANSITION_SCHEMA",
    "ACTION_VOCABULARY",
    "MIN_ACTION_TRIALS",
    "VALUE_OF_COMPUTATION_SCHEMA",
    "ActionEvidence",
    "CognitiveStateSignal",
    "ValueOfComputationPolicy",
    "build_evidence_snapshot",
    "feasible_actions",
    "transition_reward",
    "validate_action_decision",
    "validate_action_trace_row",
    "validate_action_transition",
    "validate_evidence_snapshot",
]
