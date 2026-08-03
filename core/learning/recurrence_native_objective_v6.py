"""Generated-prefix recurrent training with exact branch specialization.

Objective v5 fixed the lexical train/live mismatch but could not train the
latent property that virtual width requires: distinct branch trajectories.
This module composes v5's bounded generated-prefix gradient with a second,
bounded exact adjoint over v4's normalized separation hinge. The shared
communication slot is excluded because consensus there is intentional.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v2 import (
    LivePathForward,
    _advance_recurrent_states,
    _prepare_recurrent_prefix,
)
from core.learning.recurrence_native_objective_v4 import (
    DEFAULT_TARGET_SEPARATION,
    pairwise_separations,
)
from core.learning.recurrence_native_objective_v5 import (
    GeneratedRollinLivePathEvaluation,
    GeneratedRollinSelectionConfig,
    generated_rollin_live_path_loss,
    generated_rollin_live_path_value_and_grad,
    validate_generated_rollin_receipt,
)

RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA: Final = (
    "aura.recurrence_native_objective.v6"
)
BRANCH_SPECIALIZATION_CONFIG_SCHEMA: Final = (
    "aura.branch_specialization_config.v1"
)
BRANCH_SPECIALIZATION_RECEIPT_SCHEMA: Final = (
    "aura.branch_specialization_receipt.v1"
)
COMPOSITE_RECEIPT_SCHEMA: Final = (
    "aura.generated_rollin_specialization_receipt.v1"
)
EXACT_ADJOINT_ALGORITHM: Final = (
    "materialized_recurrent_states_single_transition_reverse_v1"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_tokens(tokens: Sequence[int]) -> str:
    normalized = tuple(tokens)
    if not normalized or any(type(token) is not int or token < 0 for token in normalized):
        raise ValueError("prompt tokens must contain non-negative integers")
    return _sha256_json(list(normalized))


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class BranchSpecializationConfig:
    """Weight and target for non-communication branch separation."""

    weight: float = 1.0
    target_separation: float = DEFAULT_TARGET_SEPARATION
    schema: str = BRANCH_SPECIALIZATION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BRANCH_SPECIALIZATION_CONFIG_SCHEMA:
            raise ValueError("branch specialization config schema is invalid")
        for name, value, low, high in (
            ("weight", self.weight, 0.0, 100.0),
            ("target_separation", self.target_separation, 0.0, 2.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not low <= float(value) <= high
            ):
                raise ValueError(f"{name} must be inside [{low}, {high}]")
        if float(self.weight) <= 0.0:
            raise ValueError("weight must be greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "weight": float(self.weight),
            "target_separation": float(self.target_separation),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BranchSpecializationConfig:
        required = {"schema", "weight", "target_separation"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("branch specialization config fields do not match")
        return cls(
            schema=value["schema"],
            weight=value["weight"],
            target_separation=value["target_separation"],
        )


@dataclass(frozen=True, slots=True)
class BranchSpecializationEvaluation:
    """Detached structural value and the atoms needed to replay it."""

    value: float
    raw_penalty: float
    separations: tuple[float, ...]
    branch_indices: tuple[int, ...]
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    prompt_token_count: int
    recurrent_depth: int
    execution_branch_count: int
    comm_slot: int
    config: BranchSpecializationConfig

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": BRANCH_SPECIALIZATION_RECEIPT_SCHEMA,
            "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
            "algorithm": EXACT_ADJOINT_ALGORITHM,
            "value": self.value,
            "raw_penalty": self.raw_penalty,
            "separations": list(self.separations),
            "branch_indices": list(self.branch_indices),
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "prompt_token_count": self.prompt_token_count,
            "recurrent_depth": self.recurrent_depth,
            "execution_branch_count": self.execution_branch_count,
            "comm_slot": self.comm_slot,
            "config": self.config.to_dict(),
            "config_sha256": self.config.sha256,
        }
        return {**body, "receipt_sha256": _sha256_json(body)}


@dataclass(frozen=True, slots=True)
class BranchSpecializationResult:
    evaluation: BranchSpecializationEvaluation
    gradients: Any

    @property
    def value(self) -> float:
        return self.evaluation.value


@dataclass(frozen=True, slots=True)
class GeneratedRollinSpecializationEvaluation:
    generated: GeneratedRollinLivePathEvaluation
    specialization: BranchSpecializationEvaluation

    @property
    def value(self) -> float:
        return self.generated.value + self.specialization.value

    @property
    def branch_values(self) -> tuple[float, ...]:
        return self.generated.branch_values

    @property
    def branch_weights(self) -> tuple[float, ...]:
        return self.generated.branch_weights

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return self.generated.branch_indices

    @property
    def execution_spec_sha256(self) -> str:
        return self.generated.execution_spec_sha256

    @property
    def prompt_tokens_sha256(self) -> str:
        return self.generated.prompt_tokens_sha256

    @property
    def answer_tokens_sha256(self) -> str:
        return self.generated.answer_tokens_sha256

    @property
    def bridge_tokens_sha256(self) -> str:
        return self.generated.bridge_tokens_sha256

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": COMPOSITE_RECEIPT_SCHEMA,
            "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
            "value": self.value,
            "generated_value": self.generated.value,
            "specialization_value": self.specialization.value,
            "generated_receipt": self.generated.receipt(),
            "specialization_receipt": self.specialization.receipt(),
        }
        return {**body, "receipt_sha256": _sha256_json(body)}


@dataclass(frozen=True, slots=True)
class GeneratedRollinSpecializationResult:
    evaluation: GeneratedRollinSpecializationEvaluation
    gradients: Any

    @property
    def value(self) -> float:
        return self.evaluation.value

    @property
    def branch_values(self) -> tuple[float, ...]:
        return self.evaluation.branch_values

    @property
    def branch_weights(self) -> tuple[float, ...]:
        return self.evaluation.branch_weights

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return self.evaluation.branch_indices

    @property
    def execution_spec_sha256(self) -> str:
        return self.evaluation.execution_spec_sha256

    @property
    def prompt_tokens_sha256(self) -> str:
        return self.evaluation.prompt_tokens_sha256

    @property
    def answer_tokens_sha256(self) -> str:
        return self.evaluation.answer_tokens_sha256

    @property
    def bridge_tokens_sha256(self) -> str:
        return self.evaluation.bridge_tokens_sha256


def _branch_indices(
    requested: Sequence[int] | None,
    *,
    branch_count: int,
) -> tuple[int, ...]:
    indices = (
        tuple(range(branch_count))
        if requested is None
        else tuple(requested)
    )
    if (
        len(indices) < 2
        or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < branch_count for index in indices)
    ):
        raise ValueError("branch specialization requires two or more valid branches")
    return indices


def _weighted_penalty(
    states: tuple[Any, ...],
    *,
    indices: tuple[int, ...],
    comm_slot: int,
    config: BranchSpecializationConfig,
) -> Any:
    import mlx.core as mx

    forward = LivePathForward(
        branch_logits=(),
        branch_states=tuple(states[index] for index in indices),
        exchanges=0,
        prompt_tokens=0,
        answer_tokens=0,
        bridge_tokens=0,
    )
    separations = pairwise_separations(forward, comm_slot=comm_slot)
    if not separations:
        raise RuntimeError("branch specialization produced no branch pairs")
    raw = sum(
        mx.maximum(float(config.target_separation) - separation, 0.0)
        for separation in separations
    ) / len(separations)
    return float(config.weight) * raw


def _evaluation(
    states: tuple[Any, ...],
    *,
    prompt_tokens: Sequence[int],
    spec: RLCExecutionSpec,
    indices: tuple[int, ...],
    config: BranchSpecializationConfig,
) -> BranchSpecializationEvaluation:
    import mlx.core as mx

    selected = LivePathForward(
        branch_logits=(),
        branch_states=tuple(states[index] for index in indices),
        exchanges=0,
        prompt_tokens=len(prompt_tokens),
        answer_tokens=0,
        bridge_tokens=0,
    )
    measured = pairwise_separations(selected, comm_slot=spec.comm_slot)
    mx.eval(measured)
    separations = tuple(float(value) for value in measured)
    raw = sum(
        max(float(config.target_separation) - separation, 0.0)
        for separation in separations
    ) / len(separations)
    return BranchSpecializationEvaluation(
        value=float(config.weight) * raw,
        raw_penalty=raw,
        separations=separations,
        branch_indices=indices,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
        prompt_token_count=len(prompt_tokens),
        recurrent_depth=spec.recurrent_steps,
        execution_branch_count=len(states),
        comm_slot=spec.comm_slot,
        config=config,
    )


def _materialized_recurrent_history(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    list[tuple[Any, ...]],
    int,
    int,
]:
    import mlx.core as mx

    (
        _prompt_embeddings,
        _seeds,
        prompts,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)

    def detached(values: Sequence[Any]) -> tuple[Any, ...]:
        result = tuple(mx.stop_gradient(value) for value in values)
        mx.eval(result)
        return result

    frozen_prompts = detached(prompts)
    frozen_anchors = detached(anchors)
    history = [detached(initial_states)]
    for step in range(spec.recurrent_steps):
        outputs = _advance_recurrent_states(
            model,
            frozen_prompts,
            history[-1],
            frozen_anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
        history.append(detached(outputs))
        del outputs
        mx.clear_cache()
    return (
        frozen_prompts,
        frozen_anchors,
        history[0],
        history,
        prelude_end,
        coda_start,
    )


def branch_specialization_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    config: BranchSpecializationConfig | None = None,
    branch_indices: Sequence[int] | None = None,
) -> BranchSpecializationResult:
    """Differentiate final branch separation with one transition graph resident."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    resolved = config or BranchSpecializationConfig()
    indices = _branch_indices(branch_indices, branch_count=len(spec.branch_roles))
    parameters = model.trainable_parameters()
    (
        prompts,
        anchors,
        _initial,
        history,
        prelude_end,
        coda_start,
    ) = _materialized_recurrent_history(
        model,
        prompt_tokens,
        spec=spec,
    )
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    for path, _value in tree_flatten(parameters):
        match = layer_pattern.match(path)
        if match is None or not (
            prelude_end <= int(match.group(1)) < coda_start
        ):
            raise RuntimeError("branch_specialization_requires_window_only_trainables")

    final_states = history[-1]

    def final_objective(states: tuple[Any, ...]) -> Any:
        return _weighted_penalty(
            states,
            indices=indices,
            comm_slot=spec.comm_slot,
            config=resolved,
        )

    value, cotangents = mx.value_and_grad(final_objective)(final_states)
    mx.eval(value, cotangents)
    accumulated: Any | None = None
    for step in range(spec.recurrent_steps - 1, -1, -1):
        prior_states = history[step]

        def transition_pullback(
            parameter_tree: Any,
            states: tuple[Any, ...],
            _step: int = step,
            _cotangents: tuple[Any, ...] = cotangents,
        ) -> Any:
            model.update(parameter_tree)
            outputs = _advance_recurrent_states(
                model,
                prompts,
                states,
                anchors,
                spec,
                _step,
                prelude_end,
                coda_start,
            )
            return sum(
                mx.sum(output * cotangent)
                for output, cotangent in zip(outputs, _cotangents, strict=True)
            )

        _pullback, (parameter_gradient, incoming) = mx.value_and_grad(
            transition_pullback,
            argnums=(0, 1),
        )(parameters, prior_states)
        mx.eval(parameter_gradient, incoming)
        accumulated = (
            parameter_gradient
            if accumulated is None
            else tree_map(
                lambda left, right: left + right,
                accumulated,
                parameter_gradient,
            )
        )
        mx.eval(accumulated)
        cotangents = tuple(mx.stop_gradient(value) for value in incoming)
        mx.eval(cotangents)
        del parameter_gradient, incoming
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("branch specialization parameter gradient is empty")
    finite = [mx.all(mx.isfinite(value)) for _path, value in tree_flatten(accumulated)]
    mx.eval(finite)
    if not finite or not all(bool(flag) for flag in finite):
        raise FloatingPointError("branch specialization gradient is non-finite")
    evaluation = _evaluation(
        final_states,
        prompt_tokens=prompt_tokens,
        spec=spec,
        indices=indices,
        config=resolved,
    )
    if not math.isclose(
        evaluation.value,
        float(value),
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("branch specialization value replay drift")
    return BranchSpecializationResult(
        evaluation=evaluation,
        gradients=accumulated,
    )


def branch_specialization_live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    config: BranchSpecializationConfig | None = None,
    branch_indices: Sequence[int] | None = None,
) -> BranchSpecializationEvaluation:
    """Measure final branch separation without retaining recurrent graphs."""

    resolved = config or BranchSpecializationConfig()
    indices = _branch_indices(branch_indices, branch_count=len(spec.branch_roles))
    *_prefix, history, _prelude_end, _coda_start = _materialized_recurrent_history(
        model,
        prompt_tokens,
        spec=spec,
    )
    return _evaluation(
        history[-1],
        prompt_tokens=prompt_tokens,
        spec=spec,
        indices=indices,
        config=resolved,
    )


def generated_rollin_specialization_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    generated_config: GeneratedRollinSelectionConfig | None = None,
    specialization_config: BranchSpecializationConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinSpecializationResult:
    """Compose lexical and structural gradients without co-resident graphs."""

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_map

    # The structural adjoint is the higher peak at deep recurrence.  Evaluate
    # it first while the MLX allocator is clean, spill only its small adapter
    # gradient to host memory, and destroy the full graph before constructing
    # the generated-rollin objective.  Both gradients are still evaluated at
    # the same unchanged parameters, so their sum is the exact composite
    # gradient rather than a sequential-optimizer approximation.
    specialization = branch_specialization_live_path_value_and_grad(
        model,
        prompt_tokens,
        spec=spec,
        config=specialization_config,
        branch_indices=branch_indices,
    )
    mx.eval(specialization.gradients)
    specialization_evaluation = specialization.evaluation
    structural_host = tree_map(
        lambda value: np.array(value, copy=True), specialization.gradients
    )
    del specialization
    mx.synchronize()
    gc.collect()
    mx.clear_cache()

    generated = generated_rollin_live_path_value_and_grad(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        base_seed=base_seed,
        config=generated_config,
        bridge_tokens=bridge_tokens,
        token_loss_weights=token_loss_weights,
        branch_indices=branch_indices,
    )
    gradients = tree_map(
        lambda lexical, structural: lexical + mx.array(structural),
        generated.gradients,
        structural_host,
    )
    mx.eval(gradients)
    evaluation = GeneratedRollinSpecializationEvaluation(
        generated=generated.evaluation,
        specialization=specialization_evaluation,
    )
    return GeneratedRollinSpecializationResult(
        evaluation=evaluation,
        gradients=gradients,
    )


def generated_rollin_specialization_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    generated_config: GeneratedRollinSelectionConfig | None = None,
    specialization_config: BranchSpecializationConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinSpecializationEvaluation:
    generated = generated_rollin_live_path_loss(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        base_seed=base_seed,
        config=generated_config,
        bridge_tokens=bridge_tokens,
        token_loss_weights=token_loss_weights,
        branch_indices=branch_indices,
    )
    specialization = branch_specialization_live_path_loss(
        model,
        prompt_tokens,
        spec=spec,
        config=specialization_config,
        branch_indices=branch_indices,
    )
    return GeneratedRollinSpecializationEvaluation(
        generated=generated,
        specialization=specialization,
    )


def validate_branch_specialization_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "objective_schema",
        "algorithm",
        "value",
        "raw_penalty",
        "separations",
        "branch_indices",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "prompt_token_count",
        "recurrent_depth",
        "execution_branch_count",
        "comm_slot",
        "config",
        "config_sha256",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("branch specialization receipt fields do not match")
    receipt = dict(value)
    if (
        receipt["schema"] != BRANCH_SPECIALIZATION_RECEIPT_SCHEMA
        or receipt["objective_schema"] != RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA
        or receipt["algorithm"] != EXACT_ADJOINT_ALGORITHM
    ):
        raise ValueError("branch specialization receipt identity is invalid")
    config = BranchSpecializationConfig.from_dict(receipt["config"])
    if receipt["config_sha256"] != config.sha256:
        raise ValueError("branch specialization config commitment mismatch")
    for role in (
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "config_sha256",
        "receipt_sha256",
    ):
        if not _valid_digest(receipt[role]):
            raise ValueError(f"branch specialization {role} is invalid")
    branches = receipt["branch_indices"]
    branch_count = receipt["execution_branch_count"]
    if (
        not isinstance(branches, list)
        or len(branches) < 2
        or len(branches) != len(set(branches))
        or type(branch_count) is not int
        or branch_count < len(branches)
        or any(type(index) is not int or not 0 <= index < branch_count for index in branches)
    ):
        raise ValueError("branch specialization branch identity is invalid")
    separations = receipt["separations"]
    expected_pairs = len(branches) * (len(branches) - 1) // 2
    if (
        not isinstance(separations, list)
        or len(separations) != expected_pairs
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 2.0
            for item in separations
        )
    ):
        raise ValueError("branch specialization separations are invalid")
    for role in ("prompt_token_count", "recurrent_depth"):
        candidate = receipt[role]
        if type(candidate) is not int or candidate < 1:
            raise ValueError(f"branch specialization {role} is invalid")
    if (
        type(receipt["comm_slot"]) is not int
        or receipt["comm_slot"] < 0
        or receipt["comm_slot"] > 1024
    ):
        raise ValueError("branch specialization comm slot is invalid")
    raw = sum(
        max(float(config.target_separation) - float(item), 0.0)
        for item in separations
    ) / len(separations)
    weighted = float(config.weight) * raw
    for role, actual, expected in (
        ("raw penalty", receipt["raw_penalty"], raw),
        ("value", receipt["value"], weighted),
    ):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(f"branch specialization {role} does not replay")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != _sha256_json(body):
        raise ValueError("branch specialization receipt commitment mismatch")
    return receipt


def validate_generated_rollin_specialization_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "objective_schema",
        "value",
        "generated_value",
        "specialization_value",
        "generated_receipt",
        "specialization_receipt",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("generated specialization receipt fields do not match")
    receipt = dict(value)
    if (
        receipt["schema"] != COMPOSITE_RECEIPT_SCHEMA
        or receipt["objective_schema"] != RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA
    ):
        raise ValueError("generated specialization receipt identity is invalid")
    generated = validate_generated_rollin_receipt(receipt["generated_receipt"])
    specialization = validate_branch_specialization_receipt(
        receipt["specialization_receipt"]
    )
    expected = float(generated["value"]) + float(specialization["value"])
    for role, actual, target in (
        ("generated value", receipt["generated_value"], generated["value"]),
        (
            "specialization value",
            receipt["specialization_value"],
            specialization["value"],
        ),
        ("total", receipt["value"], expected),
    ):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), float(target), rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(f"generated specialization {role} does not replay")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if not _valid_digest(receipt["receipt_sha256"]) or receipt[
        "receipt_sha256"
    ] != _sha256_json(body):
        raise ValueError("generated specialization receipt commitment mismatch")
    return receipt


__all__ = [
    "BRANCH_SPECIALIZATION_CONFIG_SCHEMA",
    "BRANCH_SPECIALIZATION_RECEIPT_SCHEMA",
    "COMPOSITE_RECEIPT_SCHEMA",
    "RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA",
    "BranchSpecializationConfig",
    "BranchSpecializationEvaluation",
    "BranchSpecializationResult",
    "GeneratedRollinSpecializationEvaluation",
    "GeneratedRollinSpecializationResult",
    "branch_specialization_live_path_loss",
    "branch_specialization_live_path_value_and_grad",
    "generated_rollin_specialization_loss",
    "generated_rollin_specialization_value_and_grad",
    "validate_branch_specialization_receipt",
    "validate_generated_rollin_specialization_receipt",
]
