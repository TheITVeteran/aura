"""Calibrated contradiction localization over complete recurrent traces."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRADICTION_HEAD_SCHEMA = "aura.rlc.contradiction_tensor_head.v1"
CONTRADICTION_EXAMPLE_SCHEMA = "aura.rlc.contradiction_cell_example.v1"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_FEATURE_WIDTH = 4096
MAX_DATASET_ELEMENTS = 16_777_216
MAX_HEAD_PARAMETERS = 1_048_576
MAX_TRANSITIONS = 256
MAX_POSITIONS = 64
MIN_TRACES_PER_SPLIT = 8
MIN_ERROR_TRACES = 4
MIN_NO_ERROR_TRACES = 2
MIN_TASKS_PER_SPLIT = 4
MIN_MUTATION_FAMILIES = 2
MIN_DOMAINS_PER_SPLIT = 2
LONG_TRACE_MINIMUM = 8


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.strip()) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value.strip()


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    width: int | None = None,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, Sequence | np.ndarray
    ):
        raise ValueError("contradiction feature vector must be a sequence")
    vector = tuple(float(item) for item in value)
    if (
        not vector
        or len(vector) > MAX_FEATURE_WIDTH
        or (width is not None and len(vector) != width)
        or any(
            not math.isfinite(item) or abs(item) > 1_000_000.0
            for item in vector
        )
    ):
        raise ValueError("contradiction feature vector is invalid")
    return vector


def _state_vector(
    value: Sequence[float] | np.ndarray,
    *,
    width: int | None = None,
) -> np.ndarray:
    vector = _finite_vector(value, width=width)
    return np.asarray(vector, dtype=np.float64)


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _cosine_conflict(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = 0.0 if denominator <= 1e-12 else float(left @ right / denominator)
    return max(0.0, min(1.0, (1.0 - max(-1.0, min(1.0, cosine))) / 2.0))


def contradiction_channels(
    prior: Sequence[float] | np.ndarray,
    proposal: Sequence[float] | np.ndarray,
    admitted: Sequence[float] | np.ndarray,
    premise: Sequence[float] | np.ndarray,
    conclusion: Sequence[float] | np.ndarray,
    prefix: Sequence[float] | np.ndarray,
    suffix: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Named, reconstructable evidence channels for one trace-position cell."""

    prior_v = _state_vector(prior)
    width = len(prior_v)
    proposal_v = _state_vector(proposal, width=width)
    admitted_v = _state_vector(admitted, width=width)
    premise_v = _state_vector(premise, width=width)
    conclusion_v = _state_vector(conclusion, width=width)
    prefix_v = _state_vector(prefix, width=width)
    suffix_v = _state_vector(suffix, width=width)
    return {
        "local_discontinuity": round(_rms(prior_v, proposal_v), 10),
        "admission_gap": round(_rms(proposal_v, admitted_v), 10),
        "premise_conflict": round(
            _cosine_conflict(proposal_v, premise_v), 10
        ),
        "conclusion_conflict": round(
            _cosine_conflict(proposal_v, conclusion_v), 10
        ),
        "prefix_conflict": round(
            _cosine_conflict(proposal_v, prefix_v), 10
        ),
        "suffix_conflict": round(
            _cosine_conflict(proposal_v, suffix_v), 10
        ),
        "trajectory_conflict": round(
            _cosine_conflict(prefix_v, suffix_v), 10
        ),
    }


def contradiction_features(
    prior: Sequence[float] | np.ndarray,
    proposal: Sequence[float] | np.ndarray,
    admitted: Sequence[float] | np.ndarray,
    premise: Sequence[float] | np.ndarray,
    conclusion: Sequence[float] | np.ndarray,
    prefix: Sequence[float] | np.ndarray,
    suffix: Sequence[float] | np.ndarray,
    *,
    accepted: bool,
    transition_fraction: float,
    position_fraction: float,
) -> tuple[float, ...]:
    """Exact shared training/runtime feature map for one tensor cell."""

    if (
        type(accepted) is not bool
        or isinstance(transition_fraction, bool)
        or isinstance(position_fraction, bool)
        or not math.isfinite(float(transition_fraction))
        or not math.isfinite(float(position_fraction))
        or not 0.0 <= float(transition_fraction) <= 1.0
        or not 0.0 <= float(position_fraction) <= 1.0
    ):
        raise ValueError("contradiction feature coordinates are invalid")
    prior_v = _state_vector(prior)
    width = len(prior_v)
    vectors = [
        _state_vector(value, width=width)
        for value in (
            proposal,
            admitted,
            premise,
            conclusion,
            prefix,
            suffix,
        )
    ]
    proposal_v, admitted_v, premise_v, conclusion_v, prefix_v, suffix_v = (
        vectors
    )
    pieces = (
        prior_v,
        proposal_v,
        admitted_v,
        proposal_v - prior_v,
        np.abs(proposal_v - prior_v),
        proposal_v - premise_v,
        proposal_v - conclusion_v,
        proposal_v - prefix_v,
        proposal_v - suffix_v,
        prefix_v - suffix_v,
        admitted_v - proposal_v,
    )
    result = tuple(
        float(item)
        for item in np.concatenate(
            (
                *pieces,
                np.asarray(
                    (
                        float(accepted),
                        float(transition_fraction),
                        float(position_fraction),
                    ),
                    dtype=np.float64,
                ),
            )
        )
    )
    return _finite_vector(result)


@dataclass(frozen=True, slots=True)
class ContradictionCellExample:
    """One labelled cell in a complete transition-by-latent-position tensor."""

    example_id: str
    trace_id: str
    task_id: str
    domain_id: str
    relation: str
    mutation_family: str
    transition_index: int
    transition_count: int
    position_index: int
    position_count: int
    contradiction_transition_index: int | None
    contradiction_position_index: int | None
    features: tuple[float, ...]
    trace_receipt_sha256: str
    mutation_receipt_sha256: str
    outcome_verifier_id: str

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "trace_id",
            "task_id",
            "domain_id",
            "mutation_family",
            "outcome_verifier_id",
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name=name),
            )
        if self.relation not in {"train", "in_domain", "out_of_domain"}:
            raise ValueError("relation is invalid")
        if (
            type(self.transition_index) is not int
            or type(self.transition_count) is not int
            or not 1 <= self.transition_count <= MAX_TRANSITIONS
            or not 0 <= self.transition_index < self.transition_count
            or type(self.position_index) is not int
            or type(self.position_count) is not int
            or not 1 <= self.position_count <= MAX_POSITIONS
            or not 0 <= self.position_index < self.position_count
        ):
            raise ValueError("contradiction tensor coordinate is invalid")
        pair = (
            self.contradiction_transition_index,
            self.contradiction_position_index,
        )
        if (pair[0] is None) != (pair[1] is None):
            raise ValueError("contradiction label is incomplete")
        if pair[0] is not None and (
            type(pair[0]) is not int
            or type(pair[1]) is not int
            or not 0 <= pair[0] < self.transition_count
            or not 0 <= pair[1] < self.position_count
        ):
            raise ValueError("contradiction label is outside tensor")
        object.__setattr__(self, "features", _finite_vector(self.features))
        if not _is_sha256(self.trace_receipt_sha256) or not _is_sha256(
            self.mutation_receipt_sha256
        ):
            raise ValueError("contradiction evidence commitment is invalid")

    @property
    def is_contradiction(self) -> bool:
        return (
            self.transition_index == self.contradiction_transition_index
            and self.position_index == self.contradiction_position_index
        )

    @property
    def is_middle_error(self) -> bool:
        index = self.contradiction_transition_index
        if index is None:
            return False
        return self.transition_count // 4 <= index < (
            self.transition_count - self.transition_count // 4
        )

    @property
    def is_long_context(self) -> bool:
        return self.transition_count >= LONG_TRACE_MINIMUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRADICTION_EXAMPLE_SCHEMA,
            "example_id": self.example_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "domain_id": self.domain_id,
            "relation": self.relation,
            "mutation_family": self.mutation_family,
            "transition_index": self.transition_index,
            "transition_count": self.transition_count,
            "position_index": self.position_index,
            "position_count": self.position_count,
            "contradiction_transition_index": self.contradiction_transition_index,
            "contradiction_position_index": self.contradiction_position_index,
            "features_sha256": _sha256(
                [round(value, 10) for value in self.features]
            ),
            "trace_receipt_sha256": self.trace_receipt_sha256,
            "mutation_receipt_sha256": self.mutation_receipt_sha256,
            "outcome_verifier_id": self.outcome_verifier_id,
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positive_count = int((labels == 1.0).sum())
    negative_count = int((labels == 0.0).sum())
    if not positive_count or not negative_count:
        return 0.0
    order = np.argsort(probabilities, kind="mergesort")
    ordered = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1.0].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _group_traces(
    examples: Sequence[ContradictionCellExample],
) -> dict[str, list[ContradictionCellExample]]:
    traces: dict[str, list[ContradictionCellExample]] = {}
    for example in examples:
        traces.setdefault(example.trace_id, []).append(example)
    for rows in traces.values():
        rows.sort(key=lambda row: (row.transition_index, row.position_index))
    return traces


def _validate_split(
    examples: Sequence[ContradictionCellExample],
    *,
    name: str,
    relation: str,
    feature_width: int | None = None,
) -> tuple[int, dict[str, list[ContradictionCellExample]]]:
    if (
        not isinstance(examples, Sequence)
        or isinstance(examples, (str, bytes))
        or not examples
        or len(examples) > 200_000
        or any(not isinstance(row, ContradictionCellExample) for row in examples)
    ):
        raise ValueError(f"{name} contains invalid examples")
    widths = {len(row.features) for row in examples}
    if len(widths) != 1 or (
        feature_width is not None and widths != {feature_width}
    ):
        raise ValueError(f"{name} feature widths differ")
    width = next(iter(widths))
    if len(examples) * width > MAX_DATASET_ELEMENTS:
        raise ValueError(f"{name} exceeds the dataset memory bound")
    if any(row.relation != relation for row in examples):
        raise ValueError(f"{name} relation differs")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError(f"{name} contains duplicate example identity")
    traces = _group_traces(examples)
    if len(traces) < MIN_TRACES_PER_SPLIT:
        raise ValueError(f"{name} lacks trace support")
    if len({rows[0].trace_receipt_sha256 for rows in traces.values()}) != len(
        traces
    ) or len(
        {rows[0].mutation_receipt_sha256 for rows in traces.values()}
    ) != len(traces):
        raise ValueError(f"{name} contains duplicate trace/mutation evidence")
    error_traces = no_error_traces = middle_errors = 0
    long_error = long_no_error = 0
    for trace_id, rows in traces.items():
        first = rows[0]
        invariant = (
            first.task_id,
            first.domain_id,
            first.relation,
            first.mutation_family,
            first.transition_count,
            first.position_count,
            first.contradiction_transition_index,
            first.contradiction_position_index,
            first.trace_receipt_sha256,
            first.mutation_receipt_sha256,
            first.outcome_verifier_id,
        )
        expected_coordinates = [
            (transition, position)
            for transition in range(first.transition_count)
            for position in range(first.position_count)
        ]
        if (
            len(rows) != first.transition_count * first.position_count
            or [
                (row.transition_index, row.position_index) for row in rows
            ]
            != expected_coordinates
            or any(
                (
                    row.task_id,
                    row.domain_id,
                    row.relation,
                    row.mutation_family,
                    row.transition_count,
                    row.position_count,
                    row.contradiction_transition_index,
                    row.contradiction_position_index,
                    row.trace_receipt_sha256,
                    row.mutation_receipt_sha256,
                    row.outcome_verifier_id,
                )
                != invariant
                for row in rows
            )
            or sum(row.is_contradiction for row in rows)
            != int(first.contradiction_transition_index is not None)
        ):
            raise ValueError(f"{name} trace {trace_id!r} is incomplete")
        if first.contradiction_transition_index is None:
            no_error_traces += 1
            long_no_error += int(first.is_long_context)
        else:
            error_traces += 1
            middle_errors += int(first.is_middle_error)
            long_error += int(first.is_long_context)
    if (
        error_traces < MIN_ERROR_TRACES
        or no_error_traces < MIN_NO_ERROR_TRACES
        or middle_errors < 2
        or long_error < 1
        or long_no_error < 1
        or len({row.task_id for row in examples}) < MIN_TASKS_PER_SPLIT
        or len({row.mutation_family for row in examples})
        < MIN_MUTATION_FAMILIES
        or len({row.domain_id for row in examples}) < MIN_DOMAINS_PER_SPLIT
    ):
        raise ValueError(f"{name} lacks class/middle/long-context support")
    for domain in {row.domain_id for row in examples}:
        domain_traces = [
            rows[0] for rows in traces.values() if rows[0].domain_id == domain
        ]
        if (
            not any(row.contradiction_transition_index is None for row in domain_traces)
            or not any(
                row.contradiction_transition_index is not None
                for row in domain_traces
            )
        ):
            raise ValueError(f"{name} domain lacks error/no-error support")
    return width, traces


def _dataset_sha256(examples: Sequence[ContradictionCellExample]) -> str:
    return _sha256(
        [
            row.to_dict()
            for row in sorted(
                examples,
                key=lambda row: (
                    row.trace_id,
                    row.transition_index,
                    row.position_index,
                ),
            )
        ]
    )


def _identity(examples: Sequence[ContradictionCellExample], field: str) -> str:
    return _sha256(sorted({str(getattr(row, field)) for row in examples}))


def _trace_metrics(
    traces: Mapping[str, Sequence[ContradictionCellExample]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    step_probabilities: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, Any]:
    exact = error_exact = step_exact = within_one = no_error_correct = 0
    middle_exact = long_exact = long_no_error_correct = 0
    error_count = no_error_count = middle_count = long_error_count = 0
    long_no_error_count = 0
    scores: list[float] = []
    labels: list[float] = []
    step_scores: list[float] = []
    step_labels: list[float] = []
    for trace_id, rows in traces.items():
        trace_scores = list(probabilities[trace_id])
        trace_step_scores = list(step_probabilities[trace_id])
        if len(trace_scores) != len(rows):
            raise ValueError("contradiction tensor probability coverage differs")
        if len(trace_step_scores) != rows[0].transition_count:
            raise ValueError("contradiction step probability coverage differs")
        best = int(np.argmax(np.asarray(trace_scores, dtype=np.float64)))
        predicted = (
            (
                rows[best].transition_index,
                rows[best].position_index,
            )
            if trace_scores[best] >= threshold
            else None
        )
        expected = (
            None
            if rows[0].contradiction_transition_index is None
            else (
                rows[0].contradiction_transition_index,
                rows[0].contradiction_position_index,
            )
        )
        exact += int(predicted == expected)
        if expected is None:
            no_error_count += 1
            no_error_correct += int(predicted is None)
            if rows[0].is_long_context:
                long_no_error_count += 1
                long_no_error_correct += int(predicted is None)
        else:
            error_count += 1
            error_exact += int(predicted == expected)
            step_exact += int(
                predicted is not None and predicted[0] == expected[0]
            )
            within_one += int(
                predicted is not None and abs(predicted[0] - expected[0]) <= 1
            )
            if rows[0].is_middle_error:
                middle_count += 1
                middle_exact += int(predicted == expected)
            if rows[0].is_long_context:
                long_error_count += 1
                long_exact += int(predicted == expected)
        scores.extend(trace_scores)
        labels.extend(float(row.is_contradiction) for row in rows)
        step_scores.extend(trace_step_scores)
        step_labels.extend(
            float(
                rows[0].contradiction_transition_index == transition_index
            )
            for transition_index in range(rows[0].transition_count)
        )
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    step_score_array = np.asarray(step_scores, dtype=np.float64)
    step_label_array = np.asarray(step_labels, dtype=np.float64)

    def ece(values: np.ndarray, targets: np.ndarray) -> float:
        result = 0.0
        for index in range(10):
            lower = index / 10.0
            upper = (index + 1) / 10.0
            members = (values >= lower) & (
                (values < upper)
                | ((index == 9) & (values == 1.0))
            )
            if members.any():
                result += float(members.mean()) * abs(
                    float(values[members].mean())
                    - float(targets[members].mean())
                )
        return result
    trace_count = len(traces)
    specificity = no_error_correct / max(1, no_error_count)
    return {
        "trace_count": trace_count,
        "error_trace_count": error_count,
        "no_error_trace_count": no_error_count,
        "middle_error_trace_count": middle_count,
        "long_error_trace_count": long_error_count,
        "long_no_error_trace_count": long_no_error_count,
        "exact_cell_accuracy": round(exact / max(1, trace_count), 10),
        "error_exact_cell_accuracy": round(
            error_exact / max(1, error_count), 10
        ),
        "step_exact_accuracy": round(step_exact / max(1, error_count), 10),
        "within_one_step_accuracy": round(within_one / max(1, error_count), 10),
        "no_error_specificity": round(specificity, 10),
        "false_localization_rate": round(1.0 - specificity, 10),
        "middle_exact_cell_accuracy": round(
            middle_exact / max(1, middle_count), 10
        ),
        "long_exact_cell_accuracy": round(
            long_exact / max(1, long_error_count), 10
        ),
        "long_no_error_specificity": round(
            long_no_error_correct / max(1, long_no_error_count), 10
        ),
        "cell_auc": round(_auc(score_array, label_array), 10),
        "cell_brier": round(
            float(np.mean(np.square(score_array - label_array))), 10
        ),
        "cell_ece": round(ece(score_array, label_array), 10),
        "step_auc": round(_auc(step_score_array, step_label_array), 10),
        "step_brier": round(
            float(np.mean(np.square(step_score_array - step_label_array))), 10
        ),
        "step_ece": round(ece(step_score_array, step_label_array), 10),
    }


def _domain_metrics(
    traces: Mapping[str, Sequence[ContradictionCellExample]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    step_probabilities: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    return {
        domain: _trace_metrics(
            {
                trace_id: rows
                for trace_id, rows in traces.items()
                if rows[0].domain_id == domain
            },
            probabilities,
            step_probabilities=step_probabilities,
            threshold=threshold,
        )
        for domain in sorted({rows[0].domain_id for rows in traces.values()})
    }


def _admission_failures(
    in_domain: Mapping[str, Any],
    out_of_domain: Mapping[str, Any],
    in_by_domain: Mapping[str, Mapping[str, Any]],
    out_by_domain: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    checks = (
        ("in_exact_below_limit", in_domain["exact_cell_accuracy"] < 0.70),
        (
            "in_error_exact_below_limit",
            in_domain["error_exact_cell_accuracy"] < 0.70,
        ),
        ("in_step_exact_below_limit", in_domain["step_exact_accuracy"] < 0.80),
        (
            "in_middle_exact_below_limit",
            in_domain["middle_exact_cell_accuracy"] < 0.60,
        ),
        (
            "in_long_exact_below_limit",
            in_domain["long_exact_cell_accuracy"] < 0.60,
        ),
        (
            "in_specificity_below_limit",
            in_domain["no_error_specificity"] < 0.75,
        ),
        ("in_auc_below_limit", in_domain["cell_auc"] < 0.75),
        ("in_brier_above_limit", in_domain["cell_brier"] > 0.20),
        ("in_ece_above_limit", in_domain["cell_ece"] > 0.15),
        ("in_step_auc_below_limit", in_domain["step_auc"] < 0.75),
        ("in_step_brier_above_limit", in_domain["step_brier"] > 0.20),
        ("in_step_ece_above_limit", in_domain["step_ece"] > 0.15),
        ("ood_exact_below_limit", out_of_domain["exact_cell_accuracy"] < 0.60),
        (
            "ood_error_exact_below_limit",
            out_of_domain["error_exact_cell_accuracy"] < 0.60,
        ),
        (
            "ood_step_exact_below_limit",
            out_of_domain["step_exact_accuracy"] < 0.70,
        ),
        (
            "ood_within_one_below_limit",
            out_of_domain["within_one_step_accuracy"] < 0.80,
        ),
        (
            "ood_middle_exact_below_limit",
            out_of_domain["middle_exact_cell_accuracy"] < 0.50,
        ),
        (
            "ood_long_exact_below_limit",
            out_of_domain["long_exact_cell_accuracy"] < 0.50,
        ),
        (
            "ood_long_specificity_below_limit",
            out_of_domain["long_no_error_specificity"] < 0.75,
        ),
        (
            "ood_specificity_below_limit",
            out_of_domain["no_error_specificity"] < 0.75,
        ),
        ("ood_auc_below_limit", out_of_domain["cell_auc"] < 0.75),
        ("ood_brier_above_limit", out_of_domain["cell_brier"] > 0.20),
        ("ood_ece_above_limit", out_of_domain["cell_ece"] > 0.15),
        ("ood_step_auc_below_limit", out_of_domain["step_auc"] < 0.75),
        ("ood_step_brier_above_limit", out_of_domain["step_brier"] > 0.20),
        ("ood_step_ece_above_limit", out_of_domain["step_ece"] > 0.15),
        (
            "in_domain_floor_below_limit",
            any(
                row["exact_cell_accuracy"] < 0.50
                or row["step_exact_accuracy"] < 0.50
                or row["no_error_specificity"] < 0.50
                or row["cell_auc"] < 0.65
                for row in in_by_domain.values()
            ),
        ),
        (
            "ood_domain_floor_below_limit",
            any(
                row["exact_cell_accuracy"] < 0.50
                or row["step_exact_accuracy"] < 0.50
                or row["no_error_specificity"] < 0.50
                or row["cell_auc"] < 0.65
                for row in out_by_domain.values()
            ),
        ),
    )
    return sorted(reason for reason, failed in checks if failed)


@dataclass(slots=True)
class ContradictionTensorHead:
    """Two-layer cell classifier admitted by disjoint ID/OOD trace evidence."""

    means: np.ndarray
    scales: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    temperature: float
    step_temperature: float
    threshold: float
    manifest_data: dict[str, Any]

    @property
    def admitted(self) -> bool:
        return bool(self.manifest_data.get("admitted"))

    @property
    def feature_width(self) -> int:
        return int(self.means.shape[0])

    def logits(self, features: Sequence[float] | np.ndarray) -> float:
        vector = np.asarray(
            _finite_vector(features, width=self.feature_width),
            dtype=np.float64,
        )
        hidden = np.tanh(
            ((vector - self.means) / self.scales) @ self.input_weights
            + self.input_bias
        )
        return float(hidden @ self.output_weights + self.output_bias)

    def probability(self, features: Sequence[float] | np.ndarray) -> float:
        return float(
            _sigmoid(np.asarray([self.logits(features) / self.temperature]))[0]
        )

    def step_probability(
        self,
        feature_rows: Sequence[Sequence[float] | np.ndarray],
    ) -> float:
        if (
            not isinstance(feature_rows, Sequence)
            or isinstance(feature_rows, (str, bytes))
            or not feature_rows
            or len(feature_rows) > MAX_POSITIONS
        ):
            raise ValueError("contradiction step feature rows are invalid")
        maximum = max(self.logits(features) for features in feature_rows)
        return float(
            _sigmoid(np.asarray([maximum / self.step_temperature]))[0]
        )

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest_data))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CONTRADICTION_HEAD_SCHEMA,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "input_weights": self.input_weights.tolist(),
            "input_bias": self.input_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias,
            "temperature": self.temperature,
            "step_temperature": self.step_temperature,
            "threshold": self.threshold,
            "manifest": self.manifest(),
        }
        return {**payload, "content_sha256": _sha256(payload)}

    def save(self, path: str | Path) -> str:
        from core.runtime.atomic_writer import atomic_write_bytes

        self.validate()
        raw = _canonical_bytes(self.to_payload())
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("contradiction tensor artifact exceeds size bound")
        atomic_write_bytes(Path(path), raw, durable=True, mode=0o600)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> ContradictionTensorHead:
        if not _is_sha256(expected_sha256):
            raise ValueError("contradiction tensor artifact pin is invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(Path(path), flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not 1 <= before.st_size <= MAX_ARTIFACT_BYTES
            ):
                raise ValueError("contradiction tensor artifact size/type is invalid")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    raise ValueError("contradiction tensor artifact was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("contradiction tensor artifact grew during read")
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise ValueError("contradiction tensor artifact changed during read")
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("contradiction tensor artifact SHA-256 differs")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("contradiction tensor artifact is invalid JSON") from exc
        fields = {
            "schema",
            "means",
            "scales",
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "temperature",
            "step_temperature",
            "threshold",
            "manifest",
            "content_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("contradiction tensor artifact fields differ")
        content = {key: payload[key] for key in fields - {"content_sha256"}}
        if (
            payload["schema"] != CONTRADICTION_HEAD_SCHEMA
            or payload["content_sha256"] != _sha256(content)
        ):
            raise ValueError("contradiction tensor content commitment is invalid")
        try:
            head = cls(
                means=np.asarray(payload["means"], dtype=np.float64),
                scales=np.asarray(payload["scales"], dtype=np.float64),
                input_weights=np.asarray(payload["input_weights"], dtype=np.float64),
                input_bias=np.asarray(payload["input_bias"], dtype=np.float64),
                output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
                output_bias=float(payload["output_bias"]),
                temperature=float(payload["temperature"]),
                step_temperature=float(payload["step_temperature"]),
                threshold=float(payload["threshold"]),
                manifest_data=dict(payload["manifest"]),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("contradiction tensor arrays are malformed") from exc
        head.validate()
        if not head.admitted:
            raise ValueError("contradiction tensor artifact failed admission")
        return head

    def validate(self) -> None:
        feature_width = (
            int(self.means.shape[0]) if self.means.ndim == 1 else 0
        )
        hidden_width = (
            int(self.input_bias.shape[0]) if self.input_bias.ndim == 1 else 0
        )
        if (
            not 1 <= feature_width <= MAX_FEATURE_WIDTH
            or not 2 <= hidden_width <= 256
            or feature_width * hidden_width > MAX_HEAD_PARAMETERS
            or self.scales.shape != (feature_width,)
            or self.input_weights.shape != (feature_width, hidden_width)
            or self.output_weights.shape != (hidden_width,)
            or any(
                not np.isfinite(array).all()
                for array in (
                    self.means,
                    self.scales,
                    self.input_weights,
                    self.input_bias,
                    self.output_weights,
                )
            )
            or (self.scales <= 0.0).any()
            or not math.isfinite(self.output_bias)
            or not 0.05 <= self.temperature <= 20.0
            or not 0.05 <= self.step_temperature <= 20.0
            or not 0.05 <= self.threshold <= 0.95
        ):
            raise ValueError("contradiction tensor parameter values are invalid")
        fields = {
            "feature_width",
            "hidden_width",
            "train_dataset_sha256",
            "in_domain_dataset_sha256",
            "out_of_domain_dataset_sha256",
            "train_tasks_sha256",
            "in_domain_tasks_sha256",
            "out_of_domain_tasks_sha256",
            "train_domains_sha256",
            "in_domain_domains_sha256",
            "out_of_domain_domains_sha256",
            "train_mutations_sha256",
            "in_domain_mutations_sha256",
            "out_of_domain_mutations_sha256",
            "train_count",
            "in_domain_count",
            "out_of_domain_count",
            "in_domain_metrics",
            "out_of_domain_metrics",
            "in_domain_by_domain",
            "out_of_domain_by_domain",
            "admitted",
            "failure_reasons",
            "attention_perturbation_authorized",
        }
        manifest = self.manifest_data
        if (
            not isinstance(manifest, Mapping)
            or set(manifest) != fields
            or manifest["feature_width"] != feature_width
            or manifest["hidden_width"] != hidden_width
            or any(
                not _is_sha256(manifest[name])
                for name in fields
                if name.endswith("_sha256")
            )
            or any(
                type(manifest[name]) is not int
                or not 1 <= manifest[name] <= 200_000
                for name in ("train_count", "in_domain_count", "out_of_domain_count")
            )
            or len(
                {
                    manifest["train_dataset_sha256"],
                    manifest["in_domain_dataset_sha256"],
                    manifest["out_of_domain_dataset_sha256"],
                }
            )
            != 3
            or len(
                {
                    manifest["train_tasks_sha256"],
                    manifest["in_domain_tasks_sha256"],
                    manifest["out_of_domain_tasks_sha256"],
                }
            )
            != 3
            or manifest["train_domains_sha256"]
            != manifest["in_domain_domains_sha256"]
            or manifest["train_domains_sha256"]
            == manifest["out_of_domain_domains_sha256"]
            or type(manifest["admitted"]) is not bool
            or manifest["attention_perturbation_authorized"] is not False
            or not isinstance(manifest["failure_reasons"], list)
            or any(
                not isinstance(reason, str) or not reason
                for reason in manifest["failure_reasons"]
            )
        ):
            raise ValueError("contradiction tensor manifest is invalid")
        metric_fields = {
            "trace_count",
            "error_trace_count",
            "no_error_trace_count",
            "middle_error_trace_count",
            "long_error_trace_count",
            "long_no_error_trace_count",
            "exact_cell_accuracy",
            "error_exact_cell_accuracy",
            "step_exact_accuracy",
            "within_one_step_accuracy",
            "no_error_specificity",
            "false_localization_rate",
            "middle_exact_cell_accuracy",
            "long_exact_cell_accuracy",
            "long_no_error_specificity",
            "cell_auc",
            "cell_brier",
            "cell_ece",
            "step_auc",
            "step_brier",
            "step_ece",
        }

        def validate_metrics(metrics: Any, *, minimum_traces: int) -> None:
            count_fields = {
                "trace_count",
                "error_trace_count",
                "no_error_trace_count",
                "middle_error_trace_count",
                "long_error_trace_count",
                "long_no_error_trace_count",
            }
            if (
                not isinstance(metrics, Mapping)
                or set(metrics) != metric_fields
                or type(metrics["trace_count"]) is not int
                or not minimum_traces <= metrics["trace_count"] <= 200_000
                or any(
                    type(metrics[name]) is not int or metrics[name] < 0
                    for name in count_fields - {"trace_count"}
                )
                or metrics["error_trace_count"] < 1
                or metrics["no_error_trace_count"] < 1
                or metrics["error_trace_count"] + metrics["no_error_trace_count"]
                != metrics["trace_count"]
                or any(
                    not math.isfinite(float(metrics[name]))
                    or not 0.0 <= float(metrics[name]) <= 1.0
                    for name in metric_fields - count_fields
                )
                or metrics["false_localization_rate"]
                != round(1.0 - metrics["no_error_specificity"], 10)
            ):
                raise ValueError("contradiction tensor metrics are invalid")

        validate_metrics(
            manifest["in_domain_metrics"],
            minimum_traces=MIN_TRACES_PER_SPLIT,
        )
        validate_metrics(
            manifest["out_of_domain_metrics"],
            minimum_traces=MIN_TRACES_PER_SPLIT,
        )
        for name, hash_name in (
            ("in_domain_by_domain", "in_domain_domains_sha256"),
            ("out_of_domain_by_domain", "out_of_domain_domains_sha256"),
        ):
            values = manifest[name]
            if (
                not isinstance(values, Mapping)
                or len(values) < MIN_DOMAINS_PER_SPLIT
                or _sha256(sorted(values)) != manifest[hash_name]
            ):
                raise ValueError("contradiction tensor domain metrics are invalid")
            for domain, metrics in values.items():
                _identifier(domain, name="domain_id")
                validate_metrics(metrics, minimum_traces=2)
        failures = _admission_failures(
            manifest["in_domain_metrics"],
            manifest["out_of_domain_metrics"],
            manifest["in_domain_by_domain"],
            manifest["out_of_domain_by_domain"],
        )
        if (
            manifest["failure_reasons"] != failures
            or manifest["admitted"] is not (not failures)
        ):
            raise ValueError("contradiction tensor admission verdict is invalid")

    @classmethod
    def fit(
        cls,
        train: Sequence[ContradictionCellExample],
        in_domain: Sequence[ContradictionCellExample],
        out_of_domain: Sequence[ContradictionCellExample],
        *,
        hidden_width: int = 16,
        seed: int = 0,
        steps: int = 800,
        learning_rate: float = 0.03,
    ) -> ContradictionTensorHead:
        feature_width, _ = _validate_split(
            train, name="training", relation="train"
        )
        _, in_traces = _validate_split(
            in_domain,
            name="in-domain calibration",
            relation="in_domain",
            feature_width=feature_width,
        )
        _, out_traces = _validate_split(
            out_of_domain,
            name="out-of-domain calibration",
            relation="out_of_domain",
            feature_width=feature_width,
        )
        splits = (train, in_domain, out_of_domain)
        for field in (
            "example_id",
            "trace_id",
            "task_id",
            "trace_receipt_sha256",
            "mutation_receipt_sha256",
        ):
            sets = [{getattr(row, field) for row in split} for split in splits]
            if any(
                left & right
                for index, left in enumerate(sets)
                for right in sets[index + 1 :]
            ):
                raise ValueError("contradiction tensor splits overlap")
        train_domains = {row.domain_id for row in train}
        in_domains = {row.domain_id for row in in_domain}
        out_domains = {row.domain_id for row in out_of_domain}
        if in_domains != train_domains or train_domains & out_domains:
            raise ValueError("contradiction tensor ID/OOD domains are invalid")
        if (
            not 2 <= hidden_width <= min(256, max(2, feature_width * 2))
            or feature_width * hidden_width > MAX_HEAD_PARAMETERS
            or type(seed) is not int
            or type(steps) is not int
            or not 50 <= steps <= 10_000
            or isinstance(learning_rate, bool)
            or not 0.0001 <= float(learning_rate) <= 0.5
        ):
            raise ValueError("contradiction tensor optimizer config is invalid")
        ordered = tuple(
            sorted(
                train,
                key=lambda row: (
                    row.trace_id,
                    row.transition_index,
                    row.position_index,
                ),
            )
        )
        x_train = np.asarray([row.features for row in ordered], dtype=np.float64)
        y_train = np.asarray([float(row.is_contradiction) for row in ordered])
        means = x_train.mean(axis=0)
        scales = np.where(x_train.std(axis=0) < 1e-6, 1.0, x_train.std(axis=0))
        normalized = (x_train - means) / scales
        rng = np.random.default_rng(seed)
        input_weights = rng.normal(
            0.0,
            1.0 / math.sqrt(feature_width),
            size=(feature_width, hidden_width),
        )
        input_bias = np.zeros(hidden_width, dtype=np.float64)
        output_weights = rng.normal(
            0.0, 1.0 / math.sqrt(hidden_width), size=hidden_width
        )
        output_bias = 0.0
        positives = max(1.0, float(y_train.sum()))
        negatives = max(1.0, float(len(y_train) - y_train.sum()))
        sample_weights = np.where(
            y_train == 1.0,
            len(y_train) / (2.0 * positives),
            len(y_train) / (2.0 * negatives),
        )
        for step in range(steps):
            hidden = np.tanh(normalized @ input_weights + input_bias)
            logits = hidden @ output_weights + output_bias
            gradient = (
                (_sigmoid(logits) - y_train)
                * sample_weights
                / len(y_train)
            )
            output_gradient = hidden.T @ gradient + 1e-4 * output_weights
            hidden_gradient = (
                gradient[:, None]
                * output_weights[None, :]
                * (1.0 - hidden * hidden)
            )
            rate = float(learning_rate) / math.sqrt(1.0 + step / 100.0)
            input_weights -= rate * np.clip(
                normalized.T @ hidden_gradient + 1e-4 * input_weights,
                -5.0,
                5.0,
            )
            input_bias -= rate * np.clip(
                hidden_gradient.sum(axis=0), -5.0, 5.0
            )
            output_weights -= rate * np.clip(
                output_gradient, -5.0, 5.0
            )
            output_bias -= rate * max(
                -5.0, min(5.0, float(gradient.sum()))
            )

        def score(
            examples: Sequence[ContradictionCellExample],
        ) -> dict[str, list[float]]:
            result: dict[str, list[float]] = {}
            for row in sorted(
                examples,
                key=lambda item: (
                    item.trace_id,
                    item.transition_index,
                    item.position_index,
                ),
            ):
                vector = np.asarray(row.features, dtype=np.float64)
                hidden = np.tanh(
                    ((vector - means) / scales) @ input_weights + input_bias
                )
                result.setdefault(row.trace_id, []).append(
                    float(hidden @ output_weights + output_bias)
                )
            return result

        in_logits = score(in_domain)
        out_logits = score(out_of_domain)
        in_labels = {
            trace_id: [float(row.is_contradiction) for row in rows]
            for trace_id, rows in in_traces.items()
        }
        temperatures = np.geomspace(0.25, 4.0, num=65)
        temperature = float(
            min(
                temperatures,
                key=lambda candidate: float(
                    np.mean(
                        [
                            (
                                float(
                                    _sigmoid(
                                        np.asarray(
                                            [logit / float(candidate)]
                                        )
                                    )[0]
                                )
                                - label
                            )
                            ** 2
                            for trace_id, logits in in_logits.items()
                            for logit, label in zip(
                                logits, in_labels[trace_id], strict=True
                            )
                        ]
                    )
                ),
            )
        )

        def grouped_step_logits(
            traces: Mapping[str, Sequence[ContradictionCellExample]],
            logits: Mapping[str, Sequence[float]],
        ) -> dict[str, list[float]]:
            result: dict[str, list[float]] = {}
            for trace_id, rows in traces.items():
                values = list(logits[trace_id])
                positions = rows[0].position_count
                result[trace_id] = [
                    max(
                        values[
                            transition * positions :
                            (transition + 1) * positions
                        ]
                    )
                    for transition in range(rows[0].transition_count)
                ]
            return result

        in_step_logits = grouped_step_logits(in_traces, in_logits)
        out_step_logits = grouped_step_logits(out_traces, out_logits)
        in_step_labels = {
            trace_id: [
                float(rows[0].contradiction_transition_index == transition)
                for transition in range(rows[0].transition_count)
            ]
            for trace_id, rows in in_traces.items()
        }
        step_temperature = float(
            min(
                temperatures,
                key=lambda candidate: float(
                    np.mean(
                        [
                            (
                                float(
                                    _sigmoid(
                                        np.asarray(
                                            [logit / float(candidate)]
                                        )
                                    )[0]
                                )
                                - label
                            )
                            ** 2
                            for trace_id, logits in in_step_logits.items()
                            for logit, label in zip(
                                logits,
                                in_step_labels[trace_id],
                                strict=True,
                            )
                        ]
                    )
                ),
            )
        )

        def probabilities(
            logits: Mapping[str, Sequence[float]],
        ) -> dict[str, list[float]]:
            return {
                trace_id: [
                    float(
                        _sigmoid(
                            np.asarray([value / temperature], dtype=np.float64)
                        )[0]
                    )
                    for value in values
                ]
                for trace_id, values in logits.items()
            }

        in_probabilities = probabilities(in_logits)
        out_probabilities = probabilities(out_logits)

        def calibrated_steps(
            logits: Mapping[str, Sequence[float]],
        ) -> dict[str, list[float]]:
            return {
                trace_id: [
                    float(
                        _sigmoid(
                            np.asarray(
                                [value / step_temperature],
                                dtype=np.float64,
                            )
                        )[0]
                    )
                    for value in values
                ]
                for trace_id, values in logits.items()
            }

        in_step_probabilities = calibrated_steps(in_step_logits)
        out_step_probabilities = calibrated_steps(out_step_logits)

        def threshold_score(candidate: float) -> tuple[float, float]:
            metrics = _trace_metrics(
                in_traces,
                in_probabilities,
                step_probabilities=in_step_probabilities,
                threshold=candidate,
            )
            return (
                metrics["exact_cell_accuracy"],
                metrics["no_error_specificity"],
            )

        threshold = float(
            max(
                np.linspace(0.20, 0.80, num=121),
                key=lambda candidate: (
                    *threshold_score(float(candidate)),
                    -abs(float(candidate) - 0.5),
                ),
            )
        )
        in_metrics = _trace_metrics(
            in_traces,
            in_probabilities,
            step_probabilities=in_step_probabilities,
            threshold=threshold,
        )
        out_metrics = _trace_metrics(
            out_traces,
            out_probabilities,
            step_probabilities=out_step_probabilities,
            threshold=threshold,
        )
        in_by_domain = _domain_metrics(
            in_traces,
            in_probabilities,
            step_probabilities=in_step_probabilities,
            threshold=threshold,
        )
        out_by_domain = _domain_metrics(
            out_traces,
            out_probabilities,
            step_probabilities=out_step_probabilities,
            threshold=threshold,
        )
        failures = _admission_failures(
            in_metrics, out_metrics, in_by_domain, out_by_domain
        )
        manifest = {
            "feature_width": feature_width,
            "hidden_width": hidden_width,
            "train_dataset_sha256": _dataset_sha256(train),
            "in_domain_dataset_sha256": _dataset_sha256(in_domain),
            "out_of_domain_dataset_sha256": _dataset_sha256(out_of_domain),
            "train_tasks_sha256": _identity(train, "task_id"),
            "in_domain_tasks_sha256": _identity(in_domain, "task_id"),
            "out_of_domain_tasks_sha256": _identity(out_of_domain, "task_id"),
            "train_domains_sha256": _identity(train, "domain_id"),
            "in_domain_domains_sha256": _identity(in_domain, "domain_id"),
            "out_of_domain_domains_sha256": _identity(
                out_of_domain, "domain_id"
            ),
            "train_mutations_sha256": _identity(train, "mutation_family"),
            "in_domain_mutations_sha256": _identity(
                in_domain, "mutation_family"
            ),
            "out_of_domain_mutations_sha256": _identity(
                out_of_domain, "mutation_family"
            ),
            "train_count": len(train),
            "in_domain_count": len(in_domain),
            "out_of_domain_count": len(out_of_domain),
            "in_domain_metrics": in_metrics,
            "out_of_domain_metrics": out_metrics,
            "in_domain_by_domain": in_by_domain,
            "out_of_domain_by_domain": out_by_domain,
            "admitted": not failures,
            "failure_reasons": failures,
            "attention_perturbation_authorized": False,
        }
        head = cls(
            means=means,
            scales=scales,
            input_weights=input_weights,
            input_bias=input_bias,
            output_weights=output_weights,
            output_bias=output_bias,
            temperature=temperature,
            step_temperature=step_temperature,
            threshold=threshold,
            manifest_data=manifest,
        )
        head.validate()
        return head


__all__ = [
    "CONTRADICTION_EXAMPLE_SCHEMA",
    "CONTRADICTION_HEAD_SCHEMA",
    "LONG_TRACE_MINIMUM",
    "ContradictionCellExample",
    "ContradictionTensorHead",
    "contradiction_channels",
    "contradiction_features",
]
