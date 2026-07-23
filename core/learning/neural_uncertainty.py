"""Calibrated correctness uncertainty learned directly from hidden states."""

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

NEURAL_UNCERTAINTY_SCHEMA = "aura.rlc.neural_uncertainty_head.v1"
HIDDEN_STATE_EXAMPLE_SCHEMA = "aura.rlc.hidden_state_correctness_example.v1"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_INPUT_WIDTH = 16_384
MAX_DATASET_ELEMENTS = 8_388_608
MAX_HEAD_PARAMETERS = 1_048_576
MIN_TRAIN_EXAMPLES = 32
MIN_CALIBRATION_EXAMPLES = 32
MIN_CLASS_EXAMPLES = 8
MIN_TASKS_PER_SPLIT = 4
RELIABILITY_BINS = 5


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


def _probability(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    width: int | None = None,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence | np.ndarray):
        raise ValueError("hidden-state vector must be a sequence")
    vector = tuple(float(item) for item in value)
    if (
        not vector
        or len(vector) > MAX_INPUT_WIDTH
        or (width is not None and len(vector) != width)
        or any(not math.isfinite(item) or abs(item) > 1_000_000.0 for item in vector)
    ):
        raise ValueError("hidden-state vector is empty, non-finite, or wrong-width")
    return vector


@dataclass(frozen=True, slots=True)
class HiddenStateCorrectnessExample:
    """One hidden state labelled by an independent task outcome."""

    example_id: str
    task_id: str
    hidden_state: tuple[float, ...]
    correct: bool
    state_sha256: str
    outcome_receipt_sha256: str
    outcome_verifier_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "example_id",
            _identifier(self.example_id, name="example_id"),
        )
        object.__setattr__(
            self,
            "task_id",
            _identifier(self.task_id, name="task_id"),
        )
        object.__setattr__(
            self,
            "hidden_state",
            _finite_vector(self.hidden_state),
        )
        if type(self.correct) is not bool:
            raise ValueError("correctness label must be boolean")
        if not _is_sha256(self.state_sha256):
            raise ValueError("state_sha256 must be a SHA-256")
        if not _is_sha256(self.outcome_receipt_sha256):
            raise ValueError("outcome_receipt_sha256 must be a SHA-256")
        object.__setattr__(
            self,
            "outcome_verifier_id",
            _identifier(self.outcome_verifier_id, name="outcome_verifier_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HIDDEN_STATE_EXAMPLE_SCHEMA,
            "example_id": self.example_id,
            "task_id": self.task_id,
            "hidden_state_sha256": _sha256(
                [round(value, 10) for value in self.hidden_state]
            ),
            "correct": self.correct,
            "state_sha256": self.state_sha256,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
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


def _classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = labels == 1.0
    positive = max(1, int(truth.sum()))
    negative = max(1, int((~truth).sum()))
    true_positive = int((predictions & truth).sum())
    true_negative = int((~predictions & ~truth).sum())
    false_positive = int((predictions & ~truth).sum())
    false_negative = int((~predictions & truth).sum())
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        members = (probabilities >= lower) & (
            (probabilities < upper) | ((index == 9) & (probabilities == 1.0))
        )
        if members.any():
            ece += float(members.mean()) * abs(
                float(probabilities[members].mean())
                - float(labels[members].mean())
            )
    return {
        "auc": round(_auc(probabilities, labels), 10),
        "balanced_accuracy": round(
            0.5 * (true_positive / positive + true_negative / negative),
            10,
        ),
        "brier": round(float(np.mean((probabilities - labels) ** 2)), 10),
        "ece": round(ece, 10),
        "false_positive_rate": round(false_positive / negative, 10),
        "false_negative_rate": round(false_negative / positive, 10),
    }


def _wilson(successes: int, total: int, *, upper: bool) -> float:
    if total <= 0:
        return 1.0 if upper else 0.0
    z = 1.6448536269514722
    point = successes / total
    denominator = 1.0 + z * z / total
    centre = point + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (point * (1.0 - point) + z * z / (4.0 * total)) / total
    )
    return max(
        0.0,
        min(1.0, (centre + margin if upper else centre - margin) / denominator),
    )


def _reliability(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(RELIABILITY_BINS):
        lower = index / RELIABILITY_BINS
        upper = (index + 1) / RELIABILITY_BINS
        members = (probabilities >= lower) & (
            (probabilities < upper)
            | ((index == RELIABILITY_BINS - 1) & (probabilities == 1.0))
        )
        count = int(members.sum())
        successes = int(labels[members].sum())
        rows.append(
            {
                "index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "successes": successes,
                "observed_rate": (
                    None if count == 0 else round(successes / count, 10)
                ),
                "confidence_lower": round(_wilson(successes, count, upper=False), 10),
                "confidence_upper": round(_wilson(successes, count, upper=True), 10),
            }
        )
    return rows


def _dataset_identity(
    examples: Sequence[HiddenStateCorrectnessExample],
) -> str:
    return _sha256([example.to_dict() for example in examples])


def _task_identity(examples: Sequence[HiddenStateCorrectnessExample]) -> str:
    return _sha256(sorted({example.task_id for example in examples}))


def _validate_split(
    examples: Sequence[HiddenStateCorrectnessExample],
    *,
    name: str,
    minimum: int,
    width: int | None = None,
) -> int:
    if len(examples) < minimum or len(examples) > 100_000:
        raise ValueError(f"{name} example count is outside bounds")
    if any(not isinstance(example, HiddenStateCorrectnessExample) for example in examples):
        raise ValueError(f"{name} contains an invalid example")
    widths = {len(example.hidden_state) for example in examples}
    if len(widths) != 1 or (width is not None and widths != {width}):
        raise ValueError(f"{name} hidden-state widths differ")
    observed_width = next(iter(widths))
    if len(examples) * observed_width > MAX_DATASET_ELEMENTS:
        raise ValueError(f"{name} hidden-state dataset exceeds memory bound")
    identifiers = [example.example_id for example in examples]
    state_hashes = [example.state_sha256 for example in examples]
    outcome_hashes = [example.outcome_receipt_sha256 for example in examples]
    if any(
        len(values) != len(set(values))
        for values in (identifiers, state_hashes, outcome_hashes)
    ):
        raise ValueError(f"{name} contains duplicate evidence identity")
    positives = sum(example.correct for example in examples)
    negatives = len(examples) - positives
    if min(positives, negatives) < MIN_CLASS_EXAMPLES:
        raise ValueError(f"{name} lacks class support")
    if len({example.task_id for example in examples}) < MIN_TASKS_PER_SPLIT:
        raise ValueError(f"{name} lacks task support")
    return observed_width


@dataclass(slots=True)
class NeuralUncertaintyHead:
    """Two-layer correctness head with held-out calibration evidence."""

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
    def calibrated(self) -> bool:
        return bool(self.manifest_data.get("calibrated"))

    @property
    def input_width(self) -> int:
        return int(self.means.shape[0])

    def logits(self, hidden_state: Sequence[float] | np.ndarray) -> float:
        vector = np.asarray(
            _finite_vector(hidden_state, width=self.input_width),
            dtype=np.float64,
        )
        normalized = (vector - self.means) / self.scales
        hidden = np.tanh(normalized @ self.input_weights + self.input_bias)
        return float(hidden @ self.output_weights + self.output_bias)

    def probability(self, hidden_state: Sequence[float] | np.ndarray) -> float:
        return float(_sigmoid(np.asarray([self.logits(hidden_state) / self.temperature]))[0])

    def estimate(self, hidden_state: Sequence[float] | np.ndarray) -> dict[str, Any]:
        probability = self.probability(hidden_state)
        entropy = 0.0
        if 0.0 < probability < 1.0:
            entropy = -(
                probability * math.log2(probability)
                + (1.0 - probability) * math.log2(1.0 - probability)
            )
        index = min(RELIABILITY_BINS - 1, int(probability * RELIABILITY_BINS))
        reliability = self.manifest_data["reliability_bins"][index]
        supported = (
            self.calibrated
            and int(reliability["count"]) >= MIN_CLASS_EXAMPLES
        )
        return {
            "correctness_probability": round(probability, 10),
            "predictive_entropy": round(entropy, 10),
            "confidence_lower": float(reliability["confidence_lower"]),
            "confidence_upper": float(reliability["confidence_upper"]),
            "calibration_bin": index,
            "calibration_samples": int(reliability["count"]),
            "supported": supported,
            "abstention_reason": "" if supported else "sparse_calibration_bin",
        }

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest_data))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": NEURAL_UNCERTAINTY_SCHEMA,
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
        target = Path(path)
        raw = _canonical_bytes(self.to_payload())
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("neural-uncertainty artifact exceeds size bound")
        atomic_write_bytes(target, raw, durable=True, mode=0o600)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> NeuralUncertaintyHead:
        if not _is_sha256(expected_sha256):
            raise ValueError("neural-uncertainty artifact pin is invalid")
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
                raise ValueError("neural-uncertainty artifact size/type is invalid")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    raise ValueError(
                        "neural-uncertainty artifact was truncated"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("neural-uncertainty artifact grew during read")
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise ValueError("neural-uncertainty artifact changed during read")
        finally:
            os.close(descriptor)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("neural-uncertainty artifact SHA-256 differs")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("neural-uncertainty artifact is not valid JSON") from exc
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
            raise ValueError("neural-uncertainty artifact fields differ")
        content = {key: payload[key] for key in fields - {"content_sha256"}}
        if (
            payload["schema"] != NEURAL_UNCERTAINTY_SCHEMA
            or payload["content_sha256"] != _sha256(content)
        ):
            raise ValueError("neural-uncertainty content commitment is invalid")
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
            raise ValueError("neural-uncertainty arrays are malformed") from exc
        head.validate()
        if not head.calibrated:
            raise ValueError("neural-uncertainty artifact failed calibration")
        return head

    def validate(self) -> None:
        width = int(self.means.shape[0]) if self.means.ndim == 1 else 0
        hidden_width = int(self.input_bias.shape[0]) if self.input_bias.ndim == 1 else 0
        if (
            not 1 <= width <= MAX_INPUT_WIDTH
            or not 2 <= hidden_width <= 256
            or width * hidden_width > MAX_HEAD_PARAMETERS
            or self.scales.shape != (width,)
            or self.input_weights.shape != (width, hidden_width)
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
            raise ValueError("neural-uncertainty parameter shapes/values are invalid")
        manifest_fields = {
            "input_width",
            "hidden_width",
            "train_dataset_sha256",
            "calibration_dataset_sha256",
            "train_tasks_sha256",
            "calibration_tasks_sha256",
            "train_count",
            "calibration_count",
            "metrics",
            "reliability_bins",
            "calibrated",
            "failure_reasons",
        }
        manifest = self.manifest_data
        if (
            not isinstance(manifest, Mapping)
            or set(manifest) != manifest_fields
            or manifest["input_width"] != width
            or manifest["hidden_width"] != hidden_width
            or not _is_sha256(manifest["train_dataset_sha256"])
            or not _is_sha256(manifest["calibration_dataset_sha256"])
            or not _is_sha256(manifest["train_tasks_sha256"])
            or not _is_sha256(manifest["calibration_tasks_sha256"])
            or type(manifest["train_count"]) is not int
            or type(manifest["calibration_count"]) is not int
            or not MIN_TRAIN_EXAMPLES <= manifest["train_count"] <= 100_000
            or not MIN_CALIBRATION_EXAMPLES
            <= manifest["calibration_count"]
            <= 100_000
            or manifest["train_dataset_sha256"]
            == manifest["calibration_dataset_sha256"]
            or manifest["train_tasks_sha256"]
            == manifest["calibration_tasks_sha256"]
            or type(manifest["calibrated"]) is not bool
            or not isinstance(manifest["failure_reasons"], list)
            or any(
                not isinstance(reason, str) or not reason
                for reason in manifest["failure_reasons"]
            )
            or not isinstance(manifest["reliability_bins"], list)
            or len(manifest["reliability_bins"]) != RELIABILITY_BINS
        ):
            raise ValueError("neural-uncertainty manifest is invalid")
        metrics = manifest["metrics"]
        required_metrics = {
            "auc",
            "balanced_accuracy",
            "brier",
            "ece",
            "false_positive_rate",
            "false_negative_rate",
        }
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != required_metrics
            or any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in metrics.values()
            )
        ):
            raise ValueError("neural-uncertainty metrics are invalid")
        expected_failures = sorted(
            reason
            for reason, failed in (
                ("auc_below_limit", metrics["auc"] < 0.75),
                (
                    "balanced_accuracy_below_limit",
                    metrics["balanced_accuracy"] < 0.70,
                ),
                ("brier_above_limit", metrics["brier"] > 0.22),
                ("ece_above_limit", metrics["ece"] > 0.15),
                (
                    "false_positive_rate_above_limit",
                    metrics["false_positive_rate"] > 0.25,
                ),
            )
            if failed
        )
        if (
            manifest["failure_reasons"] != expected_failures
            or manifest["calibrated"] is not (not expected_failures)
        ):
            raise ValueError("neural-uncertainty admission verdict is invalid")
        reliability_count = reliability_successes = 0
        for index, row in enumerate(manifest["reliability_bins"]):
            fields = {
                "index",
                "lower",
                "upper",
                "count",
                "successes",
                "observed_rate",
                "confidence_lower",
                "confidence_upper",
            }
            if (
                not isinstance(row, Mapping)
                or set(row) != fields
                or row["index"] != index
                or type(row["count"]) is not int
                or type(row["successes"]) is not int
                or not 0 <= row["successes"] <= row["count"]
            ):
                raise ValueError("neural-uncertainty reliability evidence is invalid")
            for name in ("lower", "upper", "confidence_lower", "confidence_upper"):
                _probability(row[name], name=f"reliability.{name}")
            count = row["count"]
            successes = row["successes"]
            expected_rate = None if count == 0 else round(successes / count, 10)
            if (
                row["lower"] != index / RELIABILITY_BINS
                or row["upper"] != (index + 1) / RELIABILITY_BINS
                or row["observed_rate"] != expected_rate
                or row["confidence_lower"]
                != round(_wilson(successes, count, upper=False), 10)
                or row["confidence_upper"]
                != round(_wilson(successes, count, upper=True), 10)
            ):
                raise ValueError(
                    "neural-uncertainty reliability derivation is invalid"
                )
            reliability_count += count
            reliability_successes += successes
        if (
            reliability_count != manifest["calibration_count"]
            or not MIN_CLASS_EXAMPLES
            <= reliability_successes
            <= reliability_count - MIN_CLASS_EXAMPLES
        ):
            raise ValueError("neural-uncertainty reliability totals are invalid")

    @classmethod
    def fit(
        cls,
        train: Sequence[HiddenStateCorrectnessExample],
        calibration: Sequence[HiddenStateCorrectnessExample],
        *,
        hidden_width: int = 16,
        seed: int = 0,
        steps: int = 500,
        learning_rate: float = 0.03,
    ) -> NeuralUncertaintyHead:
        width = _validate_split(
            train,
            name="training",
            minimum=MIN_TRAIN_EXAMPLES,
        )
        _validate_split(
            calibration,
            name="calibration",
            minimum=MIN_CALIBRATION_EXAMPLES,
            width=width,
        )
        train_ids = {example.example_id for example in train}
        calibration_ids = {example.example_id for example in calibration}
        train_tasks = {example.task_id for example in train}
        calibration_tasks = {example.task_id for example in calibration}
        if train_ids & calibration_ids or train_tasks & calibration_tasks:
            raise ValueError("training and calibration splits overlap")
        if (
            not 2 <= hidden_width <= min(256, max(2, width * 2))
            or width * hidden_width > MAX_HEAD_PARAMETERS
        ):
            raise ValueError("neural-uncertainty hidden width is invalid")
        if (
            type(seed) is not int
            or type(steps) is not int
            or not 50 <= steps <= 10_000
            or isinstance(learning_rate, bool)
            or not 0.0001 <= float(learning_rate) <= 0.5
        ):
            raise ValueError("neural-uncertainty optimizer configuration is invalid")
        ordered_train = tuple(sorted(train, key=lambda example: example.example_id))
        ordered_calibration = tuple(
            sorted(calibration, key=lambda example: example.example_id)
        )
        x_train = np.asarray(
            [example.hidden_state for example in ordered_train],
            dtype=np.float64,
        )
        y_train = np.asarray(
            [float(example.correct) for example in ordered_train],
            dtype=np.float64,
        )
        x_calibration = np.asarray(
            [example.hidden_state for example in ordered_calibration],
            dtype=np.float64,
        )
        y_calibration = np.asarray(
            [float(example.correct) for example in ordered_calibration],
            dtype=np.float64,
        )
        means = x_train.mean(axis=0)
        scales = x_train.std(axis=0)
        scales = np.where(scales < 1e-6, 1.0, scales)
        normalized = (x_train - means) / scales
        rng = np.random.default_rng(seed)
        input_weights = rng.normal(
            0.0,
            1.0 / math.sqrt(width),
            size=(width, hidden_width),
        )
        input_bias = np.zeros(hidden_width, dtype=np.float64)
        output_weights = rng.normal(
            0.0,
            1.0 / math.sqrt(hidden_width),
            size=hidden_width,
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
            probabilities = _sigmoid(logits)
            gradient = (probabilities - y_train) * sample_weights / len(y_train)
            output_gradient = hidden.T @ gradient + 1e-4 * output_weights
            output_bias_gradient = float(gradient.sum())
            hidden_gradient = (
                gradient[:, None]
                * output_weights[None, :]
                * (1.0 - hidden * hidden)
            )
            input_gradient = normalized.T @ hidden_gradient + 1e-4 * input_weights
            input_bias_gradient = hidden_gradient.sum(axis=0)
            rate = float(learning_rate) / math.sqrt(1.0 + step / 100.0)
            input_weights -= rate * np.clip(input_gradient, -5.0, 5.0)
            input_bias -= rate * np.clip(input_bias_gradient, -5.0, 5.0)
            output_weights -= rate * np.clip(output_gradient, -5.0, 5.0)
            output_bias -= rate * max(-5.0, min(5.0, output_bias_gradient))
        normalized_calibration = (x_calibration - means) / scales
        calibration_hidden = np.tanh(
            normalized_calibration @ input_weights + input_bias
        )
        raw_logits = calibration_hidden @ output_weights + output_bias
        temperatures = np.geomspace(0.25, 4.0, num=65)
        temperature = float(
            min(
                temperatures,
                key=lambda candidate: float(
                    np.mean(
                        (
                            _sigmoid(raw_logits / float(candidate))
                            - y_calibration
                        )
                        ** 2
                    )
                ),
            )
        )
        probabilities = _sigmoid(raw_logits / temperature)
        thresholds = np.linspace(0.25, 0.75, num=101)
        threshold_metrics = {
            float(candidate): _classification_metrics(
                probabilities,
                y_calibration,
                threshold=float(candidate),
            )
            for candidate in thresholds
        }
        admissible_thresholds = [
            candidate
            for candidate in thresholds
            if threshold_metrics[float(candidate)]["false_positive_rate"] <= 0.25
        ]
        threshold_candidates = admissible_thresholds or list(thresholds)
        threshold = float(
            max(
                threshold_candidates,
                key=lambda candidate: (
                    threshold_metrics[float(candidate)]["balanced_accuracy"],
                    -abs(float(candidate) - 0.5),
                ),
            )
        )
        metrics = _classification_metrics(
            probabilities,
            y_calibration,
            threshold=threshold,
        )
        failures = sorted(
            reason
            for reason, failed in (
                ("auc_below_limit", metrics["auc"] < 0.75),
                (
                    "balanced_accuracy_below_limit",
                    metrics["balanced_accuracy"] < 0.70,
                ),
                ("brier_above_limit", metrics["brier"] > 0.22),
                ("ece_above_limit", metrics["ece"] > 0.15),
                (
                    "false_positive_rate_above_limit",
                    metrics["false_positive_rate"] > 0.25,
                ),
            )
            if failed
        )
        manifest = {
            "input_width": width,
            "hidden_width": hidden_width,
            "train_dataset_sha256": _dataset_identity(ordered_train),
            "calibration_dataset_sha256": _dataset_identity(ordered_calibration),
            "train_tasks_sha256": _task_identity(ordered_train),
            "calibration_tasks_sha256": _task_identity(ordered_calibration),
            "train_count": len(ordered_train),
            "calibration_count": len(ordered_calibration),
            "metrics": metrics,
            "reliability_bins": _reliability(probabilities, y_calibration),
            "calibrated": not failures,
            "failure_reasons": failures,
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
    "HIDDEN_STATE_EXAMPLE_SCHEMA",
    "MIN_CALIBRATION_EXAMPLES",
    "MIN_CLASS_EXAMPLES",
    "MIN_TASKS_PER_SPLIT",
    "MIN_TRAIN_EXAMPLES",
    "NEURAL_UNCERTAINTY_SCHEMA",
    "HiddenStateCorrectnessExample",
    "NeuralUncertaintyHead",
]
