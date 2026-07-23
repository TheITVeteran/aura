"""Calibrated transition-level mistake localization for recurrent traces."""

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

MISTAKE_LOCATOR_SCHEMA = "aura.rlc.mistake_locator_head.v1"
MISTAKE_EXAMPLE_SCHEMA = "aura.rlc.mistake_transition_example.v1"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_STATE_WIDTH = 16_384
MAX_DATASET_ELEMENTS = 8_388_608
MAX_HEAD_PARAMETERS = 1_048_576
MIN_TRACES_PER_SPLIT = 8
MIN_ERROR_TRACES = 4
MIN_NO_ERROR_TRACES = 2
MIN_TASKS_PER_SPLIT = 4
MIN_MUTATION_FAMILIES = 2
MIN_DOMAINS_PER_SPLIT = 2


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
        raise ValueError("hidden state must be a sequence")
    vector = tuple(float(item) for item in value)
    if (
        not vector
        or len(vector) > MAX_STATE_WIDTH
        or (width is not None and len(vector) != width)
        or any(
            not math.isfinite(item) or abs(item) > 1_000_000.0
            for item in vector
        )
    ):
        raise ValueError("hidden state is empty, non-finite, or wrong-width")
    return vector


def transition_features(
    prior_hidden: Sequence[float] | np.ndarray,
    candidate_hidden: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    """Create the exact training/runtime feature map for one transition."""

    prior = np.asarray(_finite_vector(prior_hidden), dtype=np.float64)
    candidate = np.asarray(
        _finite_vector(candidate_hidden, width=len(prior)),
        dtype=np.float64,
    )
    delta = candidate - prior
    return tuple(
        float(item)
        for item in np.concatenate((prior, candidate, delta, np.abs(delta)))
    )


@dataclass(frozen=True, slots=True)
class MistakeTransitionExample:
    """One transition in a complete, independently labelled reasoning trace."""

    example_id: str
    trace_id: str
    task_id: str
    domain_id: str
    relation: str
    mutation_family: str
    transition_index: int
    transition_count: int
    error_index: int | None
    prior_hidden: tuple[float, ...]
    candidate_hidden: tuple[float, ...]
    trace_receipt_sha256: str
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
            or not 1 <= self.transition_count <= 256
            or not 0 <= self.transition_index < self.transition_count
            or (
                self.error_index is not None
                and (
                    type(self.error_index) is not int
                    or not 0 <= self.error_index < self.transition_count
                )
            )
        ):
            raise ValueError("transition/error index is invalid")
        prior = _finite_vector(self.prior_hidden)
        candidate = _finite_vector(self.candidate_hidden, width=len(prior))
        object.__setattr__(self, "prior_hidden", prior)
        object.__setattr__(self, "candidate_hidden", candidate)
        if not _is_sha256(self.trace_receipt_sha256):
            raise ValueError("trace_receipt_sha256 must be a SHA-256")

    @property
    def is_error(self) -> bool:
        return self.error_index == self.transition_index

    @property
    def features(self) -> tuple[float, ...]:
        return transition_features(self.prior_hidden, self.candidate_hidden)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MISTAKE_EXAMPLE_SCHEMA,
            "example_id": self.example_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "domain_id": self.domain_id,
            "relation": self.relation,
            "mutation_family": self.mutation_family,
            "transition_index": self.transition_index,
            "transition_count": self.transition_count,
            "error_index": self.error_index,
            "feature_sha256": _sha256(
                [round(value, 10) for value in self.features]
            ),
            "trace_receipt_sha256": self.trace_receipt_sha256,
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
    ordered_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and ordered_probabilities[end] == ordered_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1.0].sum())
    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _group_traces(
    examples: Sequence[MistakeTransitionExample],
) -> dict[str, list[MistakeTransitionExample]]:
    traces: dict[str, list[MistakeTransitionExample]] = {}
    for example in examples:
        traces.setdefault(example.trace_id, []).append(example)
    for rows in traces.values():
        rows.sort(key=lambda item: item.transition_index)
    return traces


def _validate_split(
    examples: Sequence[MistakeTransitionExample],
    *,
    name: str,
    relation: str,
    state_width: int | None = None,
) -> tuple[int, dict[str, list[MistakeTransitionExample]]]:
    if (
        not isinstance(examples, Sequence)
        or isinstance(examples, (str, bytes))
        or any(not isinstance(row, MistakeTransitionExample) for row in examples)
    ):
        raise ValueError(f"{name} contains invalid examples")
    if not examples or len(examples) > 100_000:
        raise ValueError(f"{name} example count is outside bounds")
    widths = {len(row.prior_hidden) for row in examples}
    if len(widths) != 1 or (
        state_width is not None and widths != {state_width}
    ):
        raise ValueError(f"{name} hidden-state widths differ")
    width = next(iter(widths))
    if len(examples) * width * 4 > MAX_DATASET_ELEMENTS:
        raise ValueError(f"{name} dataset exceeds memory bound")
    if any(row.relation != relation for row in examples):
        raise ValueError(f"{name} relation differs")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError(f"{name} contains duplicate example identity")
    traces = _group_traces(examples)
    if len(traces) < MIN_TRACES_PER_SPLIT:
        raise ValueError(f"{name} lacks trace support")
    if len({rows[0].trace_receipt_sha256 for rows in traces.values()}) != len(
        traces
    ):
        raise ValueError(f"{name} contains duplicate trace evidence")
    error_traces = no_error_traces = 0
    for trace_id, rows in traces.items():
        first = rows[0]
        invariant = (
            first.task_id,
            first.domain_id,
            first.relation,
            first.mutation_family,
            first.transition_count,
            first.error_index,
            first.trace_receipt_sha256,
            first.outcome_verifier_id,
        )
        if (
            len(rows) != first.transition_count
            or [row.transition_index for row in rows]
            != list(range(first.transition_count))
            or any(
                (
                    row.task_id,
                    row.domain_id,
                    row.relation,
                    row.mutation_family,
                    row.transition_count,
                    row.error_index,
                    row.trace_receipt_sha256,
                    row.outcome_verifier_id,
                )
                != invariant
                for row in rows
            )
        ):
            raise ValueError(f"{name} trace {trace_id!r} is incomplete")
        if first.error_index is None:
            no_error_traces += 1
        else:
            error_traces += 1
    if (
        error_traces < MIN_ERROR_TRACES
        or no_error_traces < MIN_NO_ERROR_TRACES
        or len({row.task_id for row in examples}) < MIN_TASKS_PER_SPLIT
        or len({row.mutation_family for row in examples})
        < MIN_MUTATION_FAMILIES
        or len({row.domain_id for row in examples}) < MIN_DOMAINS_PER_SPLIT
    ):
        raise ValueError(f"{name} lacks task/mutation/class support")
    for domain_id in {row.domain_id for row in examples}:
        domain_rows = [
            rows[0] for rows in traces.values() if rows[0].domain_id == domain_id
        ]
        if (
            not any(row.error_index is None for row in domain_rows)
            or not any(row.error_index is not None for row in domain_rows)
        ):
            raise ValueError(f"{name} domain lacks error/no-error support")
    return width, traces


def _dataset_sha256(examples: Sequence[MistakeTransitionExample]) -> str:
    return _sha256(
        [
            row.to_dict()
            for row in sorted(
                examples,
                key=lambda item: (item.trace_id, item.transition_index),
            )
        ]
    )


def _identity(examples: Sequence[MistakeTransitionExample], field: str) -> str:
    return _sha256(sorted({str(getattr(row, field)) for row in examples}))


def _trace_metrics(
    traces: Mapping[str, Sequence[MistakeTransitionExample]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    threshold: float,
) -> dict[str, Any]:
    exact = error_exact = within_one = no_error_correct = 0
    error_count = no_error_count = 0
    transition_scores: list[float] = []
    transition_labels: list[float] = []
    for trace_id, rows in traces.items():
        scores = list(probabilities[trace_id])
        if len(scores) != len(rows):
            raise ValueError("trace probability coverage differs")
        best_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        predicted = best_index if scores[best_index] >= threshold else None
        expected = rows[0].error_index
        exact += int(predicted == expected)
        if expected is None:
            no_error_count += 1
            no_error_correct += int(predicted is None)
        else:
            error_count += 1
            error_exact += int(predicted == expected)
            within_one += int(
                predicted is not None and abs(predicted - expected) <= 1
            )
        transition_scores.extend(scores)
        transition_labels.extend(float(row.is_error) for row in rows)
    count = len(traces)
    specificity = no_error_correct / max(1, no_error_count)
    scores_array = np.asarray(transition_scores, dtype=np.float64)
    labels_array = np.asarray(transition_labels, dtype=np.float64)
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        members = (scores_array >= lower) & (
            (scores_array < upper)
            | ((index == 9) & (scores_array == 1.0))
        )
        if members.any():
            ece += float(members.mean()) * abs(
                float(scores_array[members].mean())
                - float(labels_array[members].mean())
            )
    return {
        "trace_count": count,
        "error_trace_count": error_count,
        "no_error_trace_count": no_error_count,
        "exact_location_accuracy": round(exact / max(1, count), 10),
        "error_exact_accuracy": round(error_exact / max(1, error_count), 10),
        "within_one_accuracy": round(within_one / max(1, error_count), 10),
        "no_error_specificity": round(specificity, 10),
        "false_localization_rate": round(1.0 - specificity, 10),
        "transition_auc": round(
            _auc(scores_array, labels_array),
            10,
        ),
        "transition_brier": round(
            float(np.mean((scores_array - labels_array) ** 2)),
            10,
        ),
        "transition_ece": round(ece, 10),
    }


def _threshold_score(
    traces: Mapping[str, Sequence[MistakeTransitionExample]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    threshold: float,
) -> tuple[float, float]:
    exact = no_error_correct = no_error_count = 0
    for trace_id, rows in traces.items():
        scores = probabilities[trace_id]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        predicted = best_index if scores[best_index] >= threshold else None
        expected = rows[0].error_index
        exact += int(predicted == expected)
        if expected is None:
            no_error_count += 1
            no_error_correct += int(predicted is None)
    return (
        exact / max(1, len(traces)),
        no_error_correct / max(1, no_error_count),
    )


def _domain_metrics(
    traces: Mapping[str, Sequence[MistakeTransitionExample]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    domains = sorted({rows[0].domain_id for rows in traces.values()})
    return {
        domain_id: _trace_metrics(
            {
                trace_id: rows
                for trace_id, rows in traces.items()
                if rows[0].domain_id == domain_id
            },
            probabilities,
            threshold=threshold,
        )
        for domain_id in domains
    }


def _admission_failures(
    in_domain: Mapping[str, Any],
    out_of_domain: Mapping[str, Any],
    in_domain_by_domain: Mapping[str, Mapping[str, Any]],
    out_of_domain_by_domain: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    checks = (
        ("in_domain_exact_below_limit", in_domain["exact_location_accuracy"] < 0.70),
        ("in_domain_error_exact_below_limit", in_domain["error_exact_accuracy"] < 0.70),
        ("in_domain_specificity_below_limit", in_domain["no_error_specificity"] < 0.75),
        ("in_domain_auc_below_limit", in_domain["transition_auc"] < 0.75),
        ("in_domain_brier_above_limit", in_domain["transition_brier"] > 0.22),
        ("in_domain_ece_above_limit", in_domain["transition_ece"] > 0.15),
        ("out_of_domain_exact_below_limit", out_of_domain["exact_location_accuracy"] < 0.60),
        ("out_of_domain_error_exact_below_limit", out_of_domain["error_exact_accuracy"] < 0.60),
        ("out_of_domain_within_one_below_limit", out_of_domain["within_one_accuracy"] < 0.80),
        ("out_of_domain_specificity_below_limit", out_of_domain["no_error_specificity"] < 0.75),
        ("out_of_domain_auc_below_limit", out_of_domain["transition_auc"] < 0.75),
        ("out_of_domain_brier_above_limit", out_of_domain["transition_brier"] > 0.22),
        ("out_of_domain_ece_above_limit", out_of_domain["transition_ece"] > 0.15),
        (
            "in_domain_domain_floor_below_limit",
            any(
                metrics["exact_location_accuracy"] < 0.50
                or metrics["within_one_accuracy"] < 0.75
                or metrics["no_error_specificity"] < 0.50
                or metrics["transition_auc"] < 0.65
                for metrics in in_domain_by_domain.values()
            ),
        ),
        (
            "out_of_domain_domain_floor_below_limit",
            any(
                metrics["exact_location_accuracy"] < 0.50
                or metrics["within_one_accuracy"] < 0.75
                or metrics["no_error_specificity"] < 0.50
                or metrics["transition_auc"] < 0.65
                for metrics in out_of_domain_by_domain.values()
            ),
        ),
    )
    return sorted(reason for reason, failed in checks if failed)


@dataclass(slots=True)
class MistakeLocatorHead:
    """Two-layer transition classifier admitted by trace-level OOD evidence."""

    means: np.ndarray
    scales: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    temperature: float
    threshold: float
    manifest_data: dict[str, Any]

    @property
    def admitted(self) -> bool:
        return bool(self.manifest_data.get("admitted"))

    @property
    def state_width(self) -> int:
        return int(self.means.shape[0] // 4)

    def logits(
        self,
        prior_hidden: Sequence[float] | np.ndarray,
        candidate_hidden: Sequence[float] | np.ndarray,
    ) -> float:
        vector = np.asarray(
            transition_features(prior_hidden, candidate_hidden),
            dtype=np.float64,
        )
        if vector.shape != self.means.shape:
            raise ValueError("mistake-locator state width differs")
        normalized = (vector - self.means) / self.scales
        hidden = np.tanh(normalized @ self.input_weights + self.input_bias)
        return float(hidden @ self.output_weights + self.output_bias)

    def probability(
        self,
        prior_hidden: Sequence[float] | np.ndarray,
        candidate_hidden: Sequence[float] | np.ndarray,
    ) -> float:
        return float(
            _sigmoid(
                np.asarray(
                    [self.logits(prior_hidden, candidate_hidden) / self.temperature]
                )
            )[0]
        )

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest_data))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MISTAKE_LOCATOR_SCHEMA,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "input_weights": self.input_weights.tolist(),
            "input_bias": self.input_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "manifest": self.manifest(),
        }
        return {**payload, "content_sha256": _sha256(payload)}

    def save(self, path: str | Path) -> str:
        from core.runtime.atomic_writer import atomic_write_bytes

        self.validate()
        raw = _canonical_bytes(self.to_payload())
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("mistake-locator artifact exceeds size bound")
        atomic_write_bytes(Path(path), raw, durable=True, mode=0o600)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> MistakeLocatorHead:
        if not _is_sha256(expected_sha256):
            raise ValueError("mistake-locator artifact pin is invalid")
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
                raise ValueError("mistake-locator artifact size/type is invalid")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    raise ValueError("mistake-locator artifact was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("mistake-locator artifact grew during read")
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise ValueError("mistake-locator artifact changed during read")
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("mistake-locator artifact SHA-256 differs")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("mistake-locator artifact is not valid JSON") from exc
        fields = {
            "schema",
            "means",
            "scales",
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "temperature",
            "threshold",
            "manifest",
            "content_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("mistake-locator artifact fields differ")
        content = {key: payload[key] for key in fields - {"content_sha256"}}
        if (
            payload["schema"] != MISTAKE_LOCATOR_SCHEMA
            or payload["content_sha256"] != _sha256(content)
        ):
            raise ValueError("mistake-locator content commitment is invalid")
        try:
            head = cls(
                means=np.asarray(payload["means"], dtype=np.float64),
                scales=np.asarray(payload["scales"], dtype=np.float64),
                input_weights=np.asarray(payload["input_weights"], dtype=np.float64),
                input_bias=np.asarray(payload["input_bias"], dtype=np.float64),
                output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
                output_bias=float(payload["output_bias"]),
                temperature=float(payload["temperature"]),
                threshold=float(payload["threshold"]),
                manifest_data=dict(payload["manifest"]),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mistake-locator arrays are malformed") from exc
        head.validate()
        if not head.admitted:
            raise ValueError("mistake-locator artifact failed admission")
        return head

    def validate(self) -> None:
        feature_width = int(self.means.shape[0]) if self.means.ndim == 1 else 0
        state_width = feature_width // 4
        hidden_width = (
            int(self.input_bias.shape[0]) if self.input_bias.ndim == 1 else 0
        )
        if (
            feature_width != state_width * 4
            or not 1 <= state_width <= MAX_STATE_WIDTH
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
            or not 0.05 <= self.threshold <= 0.95
        ):
            raise ValueError("mistake-locator parameter shapes/values are invalid")
        fields = {
            "state_width",
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
            "train_count",
            "in_domain_count",
            "out_of_domain_count",
            "in_domain_metrics",
            "out_of_domain_metrics",
            "in_domain_by_domain",
            "out_of_domain_by_domain",
            "admitted",
            "failure_reasons",
            "repair_steering_authorized",
        }
        manifest = self.manifest_data
        if (
            not isinstance(manifest, Mapping)
            or set(manifest) != fields
            or manifest["state_width"] != state_width
            or manifest["hidden_width"] != hidden_width
            or any(
                not _is_sha256(manifest[name])
                for name in fields
                if name.endswith("_sha256")
            )
            or any(
                type(manifest[name]) is not int
                or not 1 <= manifest[name] <= 100_000
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
            or manifest["repair_steering_authorized"] is not False
            or not isinstance(manifest["failure_reasons"], list)
            or any(
                not isinstance(reason, str) or not reason
                for reason in manifest["failure_reasons"]
            )
        ):
            raise ValueError("mistake-locator manifest is invalid")
        metric_fields = {
            "trace_count",
            "error_trace_count",
            "no_error_trace_count",
            "exact_location_accuracy",
            "error_exact_accuracy",
            "within_one_accuracy",
            "no_error_specificity",
            "false_localization_rate",
            "transition_auc",
            "transition_brier",
            "transition_ece",
        }

        def validate_metrics(metrics: Any, *, minimum_traces: int) -> None:
            if (
                not isinstance(metrics, Mapping)
                or set(metrics) != metric_fields
                or any(
                    type(metrics[key]) is not int or metrics[key] < 1
                    for key in (
                        "trace_count",
                        "error_trace_count",
                        "no_error_trace_count",
                    )
                )
                or metrics["error_trace_count"] + metrics["no_error_trace_count"]
                != metrics["trace_count"]
                or not minimum_traces <= metrics["trace_count"] <= 100_000
                or any(
                    not math.isfinite(float(metrics[key]))
                    or not 0.0 <= float(metrics[key]) <= 1.0
                    for key in metric_fields
                    - {
                        "trace_count",
                        "error_trace_count",
                        "no_error_trace_count",
                    }
                )
                or metrics["false_localization_rate"]
                != round(1.0 - metrics["no_error_specificity"], 10)
            ):
                raise ValueError("mistake-locator metrics are invalid")

        for name in ("in_domain_metrics", "out_of_domain_metrics"):
            validate_metrics(
                manifest[name],
                minimum_traces=MIN_TRACES_PER_SPLIT,
            )
        for name, domain_hash_name in (
            ("in_domain_by_domain", "in_domain_domains_sha256"),
            ("out_of_domain_by_domain", "out_of_domain_domains_sha256"),
        ):
            domain_rows = manifest[name]
            if (
                not isinstance(domain_rows, Mapping)
                or len(domain_rows) < MIN_DOMAINS_PER_SPLIT
                or _sha256(sorted(domain_rows)) != manifest[domain_hash_name]
            ):
                raise ValueError("mistake-locator domain metrics are invalid")
            for domain_id, metrics in domain_rows.items():
                _identifier(domain_id, name="domain_id")
                validate_metrics(metrics, minimum_traces=2)
        if (
            not manifest["in_domain_metrics"]["trace_count"]
            <= manifest["in_domain_count"]
            <= 256 * manifest["in_domain_metrics"]["trace_count"]
            or not manifest["out_of_domain_metrics"]["trace_count"]
            <= manifest["out_of_domain_count"]
            <= 256 * manifest["out_of_domain_metrics"]["trace_count"]
        ):
            raise ValueError("mistake-locator metric coverage is invalid")
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
            raise ValueError("mistake-locator admission verdict is invalid")

    @classmethod
    def fit(
        cls,
        train: Sequence[MistakeTransitionExample],
        in_domain: Sequence[MistakeTransitionExample],
        out_of_domain: Sequence[MistakeTransitionExample],
        *,
        hidden_width: int = 16,
        seed: int = 0,
        steps: int = 600,
        learning_rate: float = 0.03,
    ) -> MistakeLocatorHead:
        state_width, _train_traces = _validate_split(
            train, name="training", relation="train"
        )
        _, in_domain_traces = _validate_split(
            in_domain,
            name="in-domain calibration",
            relation="in_domain",
            state_width=state_width,
        )
        _, out_domain_traces = _validate_split(
            out_of_domain,
            name="out-of-domain calibration",
            relation="out_of_domain",
            state_width=state_width,
        )
        splits = (train, in_domain, out_of_domain)
        example_sets = [{row.example_id for row in split} for split in splits]
        trace_sets = [{row.trace_id for row in split} for split in splits]
        task_sets = [{row.task_id for row in split} for split in splits]
        receipt_sets = [
            {row.trace_receipt_sha256 for row in split} for split in splits
        ]
        if any(
            left & right
            for collection in (
                example_sets,
                trace_sets,
                task_sets,
                receipt_sets,
            )
            for index, left in enumerate(collection)
            for right in collection[index + 1 :]
        ):
            raise ValueError("mistake-locator splits overlap")
        train_domains = {row.domain_id for row in train}
        in_domains = {row.domain_id for row in in_domain}
        out_domains = {row.domain_id for row in out_of_domain}
        if in_domains != train_domains or train_domains & out_domains:
            raise ValueError("in-domain/OOD domain identities are invalid")
        feature_width = state_width * 4
        if (
            not 2 <= hidden_width <= min(256, max(2, feature_width * 2))
            or feature_width * hidden_width > MAX_HEAD_PARAMETERS
            or type(seed) is not int
            or type(steps) is not int
            or not 50 <= steps <= 10_000
            or isinstance(learning_rate, bool)
            or not 0.0001 <= float(learning_rate) <= 0.5
        ):
            raise ValueError("mistake-locator optimizer configuration is invalid")
        ordered_train = tuple(
            sorted(train, key=lambda row: (row.trace_id, row.transition_index))
        )
        x_train = np.asarray([row.features for row in ordered_train], dtype=np.float64)
        y_train = np.asarray([float(row.is_error) for row in ordered_train])
        means = x_train.mean(axis=0)
        scales = x_train.std(axis=0)
        scales = np.where(scales < 1e-6, 1.0, scales)
        normalized = (x_train - means) / scales
        rng = np.random.default_rng(seed)
        input_weights = rng.normal(
            0.0, 1.0 / math.sqrt(feature_width), size=(feature_width, hidden_width)
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
            input_bias -= rate * np.clip(hidden_gradient.sum(axis=0), -5.0, 5.0)
            output_weights -= rate * np.clip(output_gradient, -5.0, 5.0)
            output_bias -= rate * max(-5.0, min(5.0, float(gradient.sum())))

        def score(
            examples: Sequence[MistakeTransitionExample],
        ) -> dict[str, list[float]]:
            values: dict[str, list[float]] = {}
            for row in sorted(
                examples, key=lambda item: (item.trace_id, item.transition_index)
            ):
                vector = np.asarray(row.features, dtype=np.float64)
                hidden = np.tanh(
                    ((vector - means) / scales) @ input_weights + input_bias
                )
                logit = float(hidden @ output_weights + output_bias)
                values.setdefault(row.trace_id, []).append(logit)
            return values

        in_logits = score(in_domain)
        out_logits = score(out_of_domain)
        in_labels = {
            trace_id: [float(row.is_error) for row in rows]
            for trace_id, rows in in_domain_traces.items()
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
                                        np.asarray([logit / float(candidate)])
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

        def probabilities(
            logits: Mapping[str, Sequence[float]],
        ) -> dict[str, list[float]]:
            return {
                trace_id: [
                    float(_sigmoid(np.asarray([value / temperature]))[0])
                    for value in values
                ]
                for trace_id, values in logits.items()
            }

        in_probabilities = probabilities(in_logits)
        out_probabilities = probabilities(out_logits)
        thresholds = np.linspace(0.20, 0.80, num=121)
        threshold = float(
            max(
                thresholds,
                key=lambda candidate: (
                    _threshold_score(
                        in_domain_traces,
                        in_probabilities,
                        threshold=float(candidate),
                    )[0],
                    _threshold_score(
                        in_domain_traces,
                        in_probabilities,
                        threshold=float(candidate),
                    )[1],
                    -abs(float(candidate) - 0.5),
                ),
            )
        )
        in_metrics = _trace_metrics(
            in_domain_traces, in_probabilities, threshold=threshold
        )
        out_metrics = _trace_metrics(
            out_domain_traces, out_probabilities, threshold=threshold
        )
        in_domain_by_domain = _domain_metrics(
            in_domain_traces,
            in_probabilities,
            threshold=threshold,
        )
        out_of_domain_by_domain = _domain_metrics(
            out_domain_traces,
            out_probabilities,
            threshold=threshold,
        )
        failures = _admission_failures(
            in_metrics,
            out_metrics,
            in_domain_by_domain,
            out_of_domain_by_domain,
        )
        manifest = {
            "state_width": state_width,
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
            "train_count": len(train),
            "in_domain_count": len(in_domain),
            "out_of_domain_count": len(out_of_domain),
            "in_domain_metrics": in_metrics,
            "out_of_domain_metrics": out_metrics,
            "in_domain_by_domain": in_domain_by_domain,
            "out_of_domain_by_domain": out_of_domain_by_domain,
            "admitted": not failures,
            "failure_reasons": failures,
            "repair_steering_authorized": False,
        }
        head = cls(
            means=means,
            scales=scales,
            input_weights=input_weights,
            input_bias=input_bias,
            output_weights=output_weights,
            output_bias=output_bias,
            temperature=temperature,
            threshold=threshold,
            manifest_data=manifest,
        )
        head.validate()
        return head


__all__ = [
    "MIN_ERROR_TRACES",
    "MIN_NO_ERROR_TRACES",
    "MIN_TRACES_PER_SPLIT",
    "MISTAKE_EXAMPLE_SCHEMA",
    "MISTAKE_LOCATOR_SCHEMA",
    "MistakeLocatorHead",
    "MistakeTransitionExample",
    "transition_features",
]
