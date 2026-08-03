"""Generated-prefix, branch-selective training on the resident RLC backend.

The v1 resident bootstrap optimized an equal mean of teacher-forced terminal
cross-entropies. Live inference does neither: it conditions on its own emitted
prefix and selects one branch. This module closes both mismatches while keeping
the exact KV-cached, recurrent execution path from objective v2.

Generated roll-ins are detached behavior samples. Gold answer tokens remain the
labels, so the objective learns recovery from its own prefixes without treating
mistakes as truth. Branch gradients are materialized one at a time and combined
with detached soft-min weights. The resulting gradient is the derivative of a
soft best-branch objective without retaining every resident branch graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v2 import (
    cached_live_path_token_logprobs,
    generate_cached_live_path_rollin,
)

RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA = "aura.recurrence_native_objective.v5"
GENERATED_ROLLIN_CONFIG_SCHEMA = "aura.generated_rollin_selection_config.v1"
GENERATED_ROLLIN_RECEIPT_SCHEMA = "aura.generated_rollin_selection_receipt.v1"
GENERATED_ROLLIN_TRUST_BOUNDARY = (
    "producer_sealed_tokens_external_policy_replay_required"
)
_ROLLIN_SEED_DOMAIN = b"aura.generated_rollin.branch_seed.v1\0"
_MIX_MASK_DOMAIN = b"aura.generated_rollin.mix_mask.v1\0"


def _sha256_tokens(tokens: Sequence[int], *, allow_empty: bool = False) -> str:
    normalized = list(tokens)
    if (
        (not allow_empty and not normalized)
        or any(type(token) is not int or token < 0 for token in normalized)
    ):
        raise ValueError("tokens must contain non-negative integers")
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_json(value: Any) -> str:
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
class GeneratedRollinSelectionConfig:
    """Bound controls for generated-prefix and branch-selection training."""

    student_forcing_probability: float = 0.5
    sampling_temperature: float = 0.8
    branch_softmin_temperature: float = 0.5
    schema: str = GENERATED_ROLLIN_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GENERATED_ROLLIN_CONFIG_SCHEMA:
            raise ValueError("generated roll-in config schema is unsupported")
        for name, value, lower, upper in (
            (
                "student_forcing_probability",
                self.student_forcing_probability,
                0.0,
                1.0,
            ),
            ("sampling_temperature", self.sampling_temperature, 0.0, 10.0),
            (
                "branch_softmin_temperature",
                self.branch_softmin_temperature,
                1e-3,
                100.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be inside [{lower}, {upper}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "student_forcing_probability": float(
                self.student_forcing_probability
            ),
            "sampling_temperature": float(self.sampling_temperature),
            "branch_softmin_temperature": float(
                self.branch_softmin_temperature
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> GeneratedRollinSelectionConfig:
        required = {
            "schema",
            "student_forcing_probability",
            "sampling_temperature",
            "branch_softmin_temperature",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("generated roll-in config fields do not match")
        return cls(
            schema=value["schema"],
            student_forcing_probability=value["student_forcing_probability"],
            sampling_temperature=value["sampling_temperature"],
            branch_softmin_temperature=value["branch_softmin_temperature"],
        )


@dataclass(frozen=True, slots=True)
class GeneratedRollinBranchEvidence:
    branch_index: int
    branch_seed: int
    loss: float
    selection_weight: float
    generated_tokens_sha256: str
    effective_rollin_sha256: str
    student_forced_positions: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_index": self.branch_index,
            "branch_seed": self.branch_seed,
            "loss": self.loss,
            "selection_weight": self.selection_weight,
            "generated_tokens_sha256": self.generated_tokens_sha256,
            "effective_rollin_sha256": self.effective_rollin_sha256,
            "student_forced_positions": list(self.student_forced_positions),
        }


@dataclass(frozen=True, slots=True)
class GeneratedRollinLivePathEvaluation:
    value: float
    branches: tuple[GeneratedRollinBranchEvidence, ...]
    answer_token_count: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    answer_tokens_sha256: str
    bridge_tokens_sha256: str
    config: GeneratedRollinSelectionConfig
    base_seed: int

    @property
    def config_sha256(self) -> str:
        return self.config.sha256

    @property
    def branch_values(self) -> tuple[float, ...]:
        return tuple(branch.loss for branch in self.branches)

    @property
    def branch_weights(self) -> tuple[float, ...]:
        return tuple(branch.selection_weight for branch in self.branches)

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return tuple(branch.branch_index for branch in self.branches)

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": GENERATED_ROLLIN_RECEIPT_SCHEMA,
            "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA,
            "trust_boundary": GENERATED_ROLLIN_TRUST_BOUNDARY,
            "value": self.value,
            "branches": [branch.to_dict() for branch in self.branches],
            "answer_token_count": self.answer_token_count,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "answer_tokens_sha256": self.answer_tokens_sha256,
            "bridge_tokens_sha256": self.bridge_tokens_sha256,
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "base_seed": self.base_seed,
        }
        return {**body, "receipt_sha256": _sha256_json(body)}


@dataclass(frozen=True, slots=True)
class GeneratedRollinLivePathResult:
    evaluation: GeneratedRollinLivePathEvaluation
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


def detached_softmin_weights(
    losses: Sequence[float],
    *,
    temperature: float,
) -> tuple[float, ...]:
    """Return numerically stable detached best-branch weights."""

    normalized = tuple(float(loss) for loss in losses)
    if (
        not normalized
        or any(not math.isfinite(loss) or loss < 0.0 for loss in normalized)
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 1e-3 <= float(temperature) <= 100.0
    ):
        raise ValueError("softmin losses or temperature are invalid")
    reference = min(normalized)
    exponents = tuple(-(loss - reference) / float(temperature) for loss in normalized)
    if min(exponents) < -700.0:
        raise FloatingPointError("branch loss spread exceeds selection envelope")
    raw = tuple(math.exp(value) for value in exponents)
    total = sum(raw)
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("branch selection weights are non-finite")
    return tuple(value / total for value in raw)


def _softmin_value(losses: Sequence[float], *, temperature: float) -> float:
    reference = min(losses)
    exponents = tuple(-(loss - reference) / temperature for loss in losses)
    if min(exponents) < -700.0:
        raise FloatingPointError("branch loss spread exceeds selection envelope")
    return reference - temperature * math.log(
        sum(math.exp(value) for value in exponents) / len(exponents)
    )


def _branch_seed(
    *,
    base_seed: int,
    branch_index: int,
    execution_spec_sha256: str,
    prompt_tokens_sha256: str,
) -> int:
    payload = {
        "base_seed": base_seed,
        "branch_index": branch_index,
        "execution_spec_sha256": execution_spec_sha256,
        "prompt_tokens_sha256": prompt_tokens_sha256,
    }
    digest = hashlib.sha256()
    digest.update(_ROLLIN_SEED_DOMAIN)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return int.from_bytes(digest.digest()[:4], "big")


def deterministic_mixed_rollin(
    answer_tokens: Sequence[int],
    generated_tokens: Sequence[int],
    *,
    probability: float,
    base_seed: int,
    branch_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mix generated decoder inputs without changing supervised labels."""

    answer = tuple(answer_tokens)
    generated = tuple(generated_tokens)
    if (
        not answer
        or len(answer) != len(generated)
        or any(type(token) is not int or token < 0 for token in (*answer, *generated))
    ):
        raise ValueError("generated roll-in must be answer-aligned")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("student forcing probability must be inside [0, 1]")
    if type(base_seed) is not int or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be inside [0, 2^32-1]")
    if type(branch_index) is not int or branch_index < 0:
        raise ValueError("branch_index must be non-negative")

    mixed: list[int] = []
    selected: list[int] = []
    threshold = int(float(probability) * (1 << 64))
    for position, (target, sampled) in enumerate(zip(answer, generated, strict=True)):
        # The final decoder input has no successor label and is never consumed.
        use_generated = False
        if position + 1 < len(answer):
            digest = hashlib.sha256()
            digest.update(_MIX_MASK_DOMAIN)
            digest.update(base_seed.to_bytes(4, "big"))
            digest.update(branch_index.to_bytes(4, "big"))
            digest.update(position.to_bytes(8, "big"))
            use_generated = int.from_bytes(digest.digest()[:8], "big") < threshold
        mixed.append(sampled if use_generated else target)
        if use_generated:
            selected.append(position)
    return tuple(mixed), tuple(selected)


def _inputs(
    answer_tokens: Sequence[int],
    bridge_tokens: Sequence[int],
    branch_indices: Sequence[int] | None,
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    answer = tuple(answer_tokens)
    bridge = tuple(bridge_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if type(base_seed) is not int or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be inside [0, 2^32-1]")
    indices = (
        tuple(range(len(spec.branch_roles)))
        if branch_indices is None
        else tuple(branch_indices)
    )
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(
            type(index) is not int or not 0 <= index < len(spec.branch_roles)
            for index in indices
        )
    ):
        raise ValueError("branch indices must be unique members of the live branch set")
    return answer, bridge, indices


def _branch_rollin(
    model: Any,
    prompt_tokens: Sequence[int],
    answer: tuple[int, ...],
    bridge: tuple[int, ...],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    base_seed: int,
    config: GeneratedRollinSelectionConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], int, str]:
    prompt_sha256 = _sha256_tokens(prompt_tokens)
    seed = _branch_seed(
        base_seed=base_seed,
        branch_index=branch_index,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=prompt_sha256,
    )
    generated = generate_cached_live_path_rollin(
        model,
        prompt_tokens,
        spec=spec,
        branch_index=branch_index,
        token_count=len(answer),
        seed=seed,
        temperature=config.sampling_temperature,
        bridge_tokens=bridge,
    )
    mixed, positions = deterministic_mixed_rollin(
        answer,
        generated.tokens,
        probability=config.student_forcing_probability,
        base_seed=seed,
        branch_index=branch_index,
    )
    return mixed, positions, seed, generated.tokens_sha256


def generated_rollin_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    config: GeneratedRollinSelectionConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinLivePathResult:
    """Differentiate generated-prefix soft branch selection with bounded memory."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    resolved = config or GeneratedRollinSelectionConfig()
    answer, bridge, indices = _inputs(
        answer_tokens,
        bridge_tokens,
        branch_indices,
        spec=spec,
        base_seed=base_seed,
    )
    weights = (
        tuple(1.0 for _ in answer)
        if token_loss_weights is None
        else tuple(float(value) for value in token_loss_weights)
    )
    weight_total = sum(weights)
    if (
        len(weights) != len(answer)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or weight_total <= 0.0
    ):
        raise ValueError("token loss weights must be finite and answer-aligned")
    weight_tensor = mx.array(weights, dtype=mx.float32)

    gradients_numerator: Any | None = None
    denominator = 0.0
    reference: float | None = None
    records: list[dict[str, Any]] = []
    for branch_index in indices:
        rollin, positions, branch_seed, generated_sha256 = _branch_rollin(
            model,
            prompt_tokens,
            answer,
            bridge,
            spec=spec,
            branch_index=branch_index,
            base_seed=base_seed,
            config=resolved,
        )

        def objective(
            current_model: Any,
            _branch_index: int = branch_index,
            _rollin: tuple[int, ...] = rollin,
        ) -> Any:
            logprobs = cached_live_path_token_logprobs(
                current_model,
                prompt_tokens,
                answer,
                spec=spec,
                branch_index=_branch_index,
                bridge_tokens=bridge,
                adapters_on=True,
                rollin_tokens=_rollin,
            )
            return -mx.sum(logprobs * weight_tensor) / weight_total

        value, gradients = nn.value_and_grad(model, objective)(model)
        finite_flags = [
            mx.all(mx.isfinite(gradient))
            for _path, gradient in tree_flatten(gradients)
        ]
        mx.eval(value, gradients, finite_flags)
        branch_value = float(value)
        if (
            not math.isfinite(branch_value)
            or branch_value < 0.0
            or not finite_flags
            or not all(bool(flag) for flag in finite_flags)
        ):
            raise FloatingPointError("generated roll-in branch gradient is non-finite")

        if reference is None:
            reference = branch_value
            raw_weight = 1.0
        elif branch_value < reference:
            exponent = -(reference - branch_value) / resolved.branch_softmin_temperature
            if exponent < -700.0:
                raise FloatingPointError("branch loss spread exceeds selection envelope")
            rescale = math.exp(exponent)
            if gradients_numerator is not None:
                gradients_numerator = tree_map(
                    lambda total, factor=rescale: total * factor,
                    gradients_numerator,
                )
                mx.eval(gradients_numerator)
            denominator *= rescale
            reference = branch_value
            raw_weight = 1.0
        else:
            exponent = -(branch_value - reference) / resolved.branch_softmin_temperature
            if exponent < -700.0:
                raise FloatingPointError("branch loss spread exceeds selection envelope")
            raw_weight = math.exp(exponent)
        scaled = tree_map(
            lambda gradient, factor=raw_weight: factor * gradient,
            gradients,
        )
        gradients_numerator = (
            scaled
            if gradients_numerator is None
            else tree_map(
                lambda total, gradient: total + gradient,
                gradients_numerator,
                scaled,
            )
        )
        denominator += raw_weight
        mx.eval(gradients_numerator)
        records.append(
            {
                "branch_index": branch_index,
                "branch_seed": branch_seed,
                "loss": branch_value,
                "generated_tokens_sha256": generated_sha256,
                "effective_rollin_sha256": _sha256_tokens(rollin),
                "student_forced_positions": positions,
            }
        )
        del value, gradients, scaled
        mx.clear_cache()

    if (
        gradients_numerator is None
        or reference is None
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise RuntimeError("generated roll-in objective produced no gradient")
    gradients = tree_map(lambda value: value / denominator, gradients_numerator)
    mx.eval(gradients)
    branch_values = tuple(record["loss"] for record in records)
    selection_weights = detached_softmin_weights(
        branch_values,
        temperature=resolved.branch_softmin_temperature,
    )
    branches = tuple(
        GeneratedRollinBranchEvidence(
            branch_index=record["branch_index"],
            branch_seed=record["branch_seed"],
            loss=record["loss"],
            selection_weight=selection_weight,
            generated_tokens_sha256=record["generated_tokens_sha256"],
            effective_rollin_sha256=record["effective_rollin_sha256"],
            student_forced_positions=record["student_forced_positions"],
        )
        for record, selection_weight in zip(records, selection_weights, strict=True)
    )
    evaluation = GeneratedRollinLivePathEvaluation(
        value=_softmin_value(
            branch_values,
            temperature=resolved.branch_softmin_temperature,
        ),
        branches=branches,
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
        answer_tokens_sha256=_sha256_tokens(answer),
        bridge_tokens_sha256=_sha256_tokens(bridge, allow_empty=True),
        config=resolved,
        base_seed=base_seed,
    )
    return GeneratedRollinLivePathResult(
        evaluation=evaluation,
        gradients=gradients,
    )


def generated_rollin_live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    config: GeneratedRollinSelectionConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinLivePathEvaluation:
    """Evaluate the exact generated-prefix branch objective without mutation."""

    import mlx.core as mx

    resolved = config or GeneratedRollinSelectionConfig()
    answer, bridge, indices = _inputs(
        answer_tokens,
        bridge_tokens,
        branch_indices,
        spec=spec,
        base_seed=base_seed,
    )
    weights = (
        tuple(1.0 for _ in answer)
        if token_loss_weights is None
        else tuple(float(value) for value in token_loss_weights)
    )
    weight_total = sum(weights)
    if (
        len(weights) != len(answer)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or weight_total <= 0.0
    ):
        raise ValueError("token loss weights must be finite and answer-aligned")
    weight_tensor = mx.array(weights, dtype=mx.float32)
    records: list[dict[str, Any]] = []
    for branch_index in indices:
        rollin, positions, branch_seed, generated_sha256 = _branch_rollin(
            model,
            prompt_tokens,
            answer,
            bridge,
            spec=spec,
            branch_index=branch_index,
            base_seed=base_seed,
            config=resolved,
        )
        logprobs = cached_live_path_token_logprobs(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge,
            adapters_on=True,
            rollin_tokens=rollin,
        )
        value = -mx.sum(logprobs * weight_tensor) / weight_total
        mx.eval(value)
        branch_value = float(value)
        del value, logprobs
        mx.clear_cache()
        if not math.isfinite(branch_value) or branch_value < 0.0:
            raise FloatingPointError("generated roll-in branch loss is non-finite")
        records.append(
            {
                "branch_index": branch_index,
                "branch_seed": branch_seed,
                "loss": branch_value,
                "generated_tokens_sha256": generated_sha256,
                "effective_rollin_sha256": _sha256_tokens(rollin),
                "student_forced_positions": positions,
            }
        )
    branch_values = tuple(record["loss"] for record in records)
    selection_weights = detached_softmin_weights(
        branch_values,
        temperature=resolved.branch_softmin_temperature,
    )
    return GeneratedRollinLivePathEvaluation(
        value=_softmin_value(
            branch_values,
            temperature=resolved.branch_softmin_temperature,
        ),
        branches=tuple(
            GeneratedRollinBranchEvidence(
                branch_index=record["branch_index"],
                branch_seed=record["branch_seed"],
                loss=record["loss"],
                selection_weight=selection_weight,
                generated_tokens_sha256=record["generated_tokens_sha256"],
                effective_rollin_sha256=record["effective_rollin_sha256"],
                student_forced_positions=record["student_forced_positions"],
            )
            for record, selection_weight in zip(
                records,
                selection_weights,
                strict=True,
            )
        ),
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
        answer_tokens_sha256=_sha256_tokens(answer),
        bridge_tokens_sha256=_sha256_tokens(bridge, allow_empty=True),
        config=resolved,
        base_seed=base_seed,
    )


def validate_generated_rollin_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay all producer arithmetic and reject a merely rehashed receipt."""

    required = {
        "schema",
        "objective_schema",
        "trust_boundary",
        "value",
        "branches",
        "answer_token_count",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "answer_tokens_sha256",
        "bridge_tokens_sha256",
        "config",
        "config_sha256",
        "base_seed",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("generated roll-in receipt fields do not match")
    normalized = dict(value)
    if (
        normalized["schema"] != GENERATED_ROLLIN_RECEIPT_SCHEMA
        or normalized["objective_schema"] != RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA
        or normalized["trust_boundary"] != GENERATED_ROLLIN_TRUST_BOUNDARY
        or type(normalized["answer_token_count"]) is not int
        or normalized["answer_token_count"] < 1
        or type(normalized["base_seed"]) is not int
        or not 0 <= normalized["base_seed"] <= 0xFFFFFFFF
    ):
        raise ValueError("generated roll-in receipt structure is invalid")
    for role in (
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "answer_tokens_sha256",
        "bridge_tokens_sha256",
        "config_sha256",
        "receipt_sha256",
    ):
        candidate = normalized[role]
        if (
            not isinstance(candidate, str)
            or len(candidate) != 64
            or any(character not in "0123456789abcdef" for character in candidate)
        ):
            raise ValueError(f"generated roll-in receipt {role} is invalid")
    branches = normalized["branches"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("generated roll-in receipt branches are invalid")
    branch_fields = {
        "branch_index",
        "branch_seed",
        "loss",
        "selection_weight",
        "generated_tokens_sha256",
        "effective_rollin_sha256",
        "student_forced_positions",
    }
    losses: list[float] = []
    weights: list[float] = []
    indices: list[int] = []
    for branch in branches:
        if not isinstance(branch, Mapping) or set(branch) != branch_fields:
            raise ValueError("generated roll-in branch receipt is invalid")
        index = branch["branch_index"]
        seed = branch["branch_seed"]
        loss = branch["loss"]
        weight = branch["selection_weight"]
        positions = branch["student_forced_positions"]
        if (
            type(index) is not int
            or index < 0
            or type(seed) is not int
            or not 0 <= seed <= 0xFFFFFFFF
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or not 0.0 < float(weight) <= 1.0
            or not isinstance(positions, list)
            or any(
                type(position) is not int
                or not 0 <= position < normalized["answer_token_count"] - 1
                for position in positions
            )
            or positions != sorted(set(positions))
        ):
            raise ValueError("generated roll-in branch values are invalid")
        for role in ("generated_tokens_sha256", "effective_rollin_sha256"):
            digest = branch[role]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("generated roll-in branch digest is invalid")
        indices.append(index)
        losses.append(float(loss))
        weights.append(float(weight))
    if len(indices) != len(set(indices)):
        raise ValueError("generated roll-in branch indices repeat")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("generated roll-in branch weights do not normalize")
    config = GeneratedRollinSelectionConfig.from_dict(normalized["config"])
    if config.sha256 != normalized["config_sha256"]:
        raise ValueError("generated roll-in config commitment mismatch")
    replayed_weights = detached_softmin_weights(
        losses,
        temperature=config.branch_softmin_temperature,
    )
    if any(
        not math.isclose(observed, replayed, rel_tol=0.0, abs_tol=1e-12)
        for observed, replayed in zip(weights, replayed_weights, strict=True)
    ):
        raise ValueError("generated roll-in branch weights do not replay")
    replayed_value = _softmin_value(
        losses,
        temperature=config.branch_softmin_temperature,
    )
    if (
        isinstance(normalized["value"], bool)
        or not isinstance(normalized["value"], (int, float))
        or not math.isfinite(float(normalized["value"]))
        or not math.isclose(
            float(normalized["value"]),
            replayed_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("generated roll-in objective value does not replay")
    body = {key: normalized[key] for key in required - {"receipt_sha256"}}
    if _sha256_json(body) != normalized["receipt_sha256"]:
        raise ValueError("generated roll-in receipt commitment mismatch")
    return normalized


__all__ = [
    "GENERATED_ROLLIN_CONFIG_SCHEMA",
    "GENERATED_ROLLIN_RECEIPT_SCHEMA",
    "GENERATED_ROLLIN_TRUST_BOUNDARY",
    "RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA",
    "GeneratedRollinBranchEvidence",
    "GeneratedRollinLivePathEvaluation",
    "GeneratedRollinLivePathResult",
    "GeneratedRollinSelectionConfig",
    "detached_softmin_weights",
    "deterministic_mixed_rollin",
    "generated_rollin_live_path_loss",
    "generated_rollin_live_path_value_and_grad",
    "validate_generated_rollin_receipt",
]
