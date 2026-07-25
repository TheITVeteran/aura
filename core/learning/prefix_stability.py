"""Task-disjoint calibration for regenerated-conclusion recurrence.

This is deliberately not a correctness model.  It maps a bounded runtime
stability statistic to the probability that a later, independently generated
continuation lands on the same conclusion signature.  Fit and calibration
tasks must be disjoint, artifacts are content-addressed, and runtime consumers
may only use an admitted held-out calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREFIX_STABILITY_CALIBRATOR_SCHEMA = "aura.rlc.prefix_stability_calibrator.v1"
PREFIX_STABILITY_EXAMPLE_SCHEMA = "aura.rlc.prefix_stability_example.v1"
CALIBRATION_TARGET = "future_conclusion_recurrence_not_correctness"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_EXAMPLES = 100_000
MIN_FIT_EXAMPLES = 32
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


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
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


@dataclass(frozen=True, slots=True)
class PrefixStabilityExample:
    """One probe paired with a later, independent recurrence observation."""

    example_id: str
    task_id: str
    domain: str
    raw_stability: float
    future_conclusion_match: bool
    probe_receipt_sha256: str
    future_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _identifier(self.example_id, name="example_id"))
        object.__setattr__(self, "task_id", _identifier(self.task_id, name="task_id"))
        object.__setattr__(self, "domain", _identifier(self.domain, name="domain"))
        object.__setattr__(
            self,
            "raw_stability",
            _probability(self.raw_stability, name="raw_stability"),
        )
        if type(self.future_conclusion_match) is not bool:
            raise ValueError("future_conclusion_match must be boolean")
        if not _is_sha(self.probe_receipt_sha256):
            raise ValueError("probe_receipt_sha256 must be a SHA-256")
        if not _is_sha(self.future_receipt_sha256):
            raise ValueError("future_receipt_sha256 must be a SHA-256")
        if self.probe_receipt_sha256 == self.future_receipt_sha256:
            raise ValueError("probe and future evidence must be distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PREFIX_STABILITY_EXAMPLE_SCHEMA,
            "target": CALIBRATION_TARGET,
            "example_id": self.example_id,
            "task_id": self.task_id,
            "domain": self.domain,
            "raw_stability": round(self.raw_stability, 10),
            "future_conclusion_match": self.future_conclusion_match,
            "probe_receipt_sha256": self.probe_receipt_sha256,
            "future_receipt_sha256": self.future_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PrefixStabilityExample:
        fields = {
            "schema",
            "target",
            "example_id",
            "task_id",
            "domain",
            "raw_stability",
            "future_conclusion_match",
            "probe_receipt_sha256",
            "future_receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("prefix-stability example fields differ")
        if (
            value["schema"] != PREFIX_STABILITY_EXAMPLE_SCHEMA
            or value["target"] != CALIBRATION_TARGET
        ):
            raise ValueError("prefix-stability example schema or target differs")
        return cls(
            example_id=value["example_id"],
            task_id=value["task_id"],
            domain=value["domain"],
            raw_stability=value["raw_stability"],
            future_conclusion_match=value["future_conclusion_match"],
            probe_receipt_sha256=value["probe_receipt_sha256"],
            future_receipt_sha256=value["future_receipt_sha256"],
        )


def _dataset_sha(examples: Sequence[PrefixStabilityExample]) -> str:
    return _sha([example.to_dict() for example in examples])


def _task_sha(examples: Sequence[PrefixStabilityExample]) -> str:
    return _sha(sorted({example.task_id for example in examples}))


def _validate_split(
    examples: Sequence[PrefixStabilityExample],
    *,
    name: str,
    minimum: int,
) -> None:
    if not minimum <= len(examples) <= MAX_EXAMPLES:
        raise ValueError(f"{name} example count is outside bounds")
    if any(not isinstance(example, PrefixStabilityExample) for example in examples):
        raise ValueError(f"{name} contains an invalid example")
    if len({example.task_id for example in examples}) < MIN_TASKS_PER_SPLIT:
        raise ValueError(f"{name} lacks task support")
    positives = sum(example.future_conclusion_match for example in examples)
    if min(positives, len(examples) - positives) < MIN_CLASS_EXAMPLES:
        raise ValueError(f"{name} lacks recurrence class support")
    identities = (
        [example.example_id for example in examples],
        [example.probe_receipt_sha256 for example in examples],
        [example.future_receipt_sha256 for example in examples],
    )
    if any(len(values) != len(set(values)) for values in identities):
        raise ValueError(f"{name} contains duplicate evidence identity")


def _fit_isotonic(examples: Sequence[PrefixStabilityExample]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda row: (row.raw_stability, row.example_id)):
        value = round(example.raw_stability, 10)
        if grouped and grouped[-1]["upper"] == value:
            grouped[-1]["count"] += 1
            grouped[-1]["successes"] += int(example.future_conclusion_match)
        else:
            grouped.append(
                {
                    "lower": value,
                    "upper": value,
                    "count": 1,
                    "successes": int(example.future_conclusion_match),
                }
            )
    blocks: list[dict[str, Any]] = []
    for group in grouped:
        blocks.append(group)
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_rate = left["successes"] / left["count"]
            right_rate = right["successes"] / right["count"]
            if left_rate <= right_rate:
                break
            blocks[-2:] = [
                {
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "count": left["count"] + right["count"],
                    "successes": left["successes"] + right["successes"],
                }
            ]
    return [
        {
            **block,
            "probability": round(block["successes"] / block["count"], 10),
        }
        for block in blocks
    ]


def _predict(blocks: Sequence[Mapping[str, Any]], value: float) -> float:
    for block in blocks:
        if value <= float(block["upper"]):
            return float(block["probability"])
    return float(blocks[-1]["probability"])


def _auc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return 0.0
    ordered = sorted(zip(probabilities, labels, strict=True))
    wins = 0.0
    negatives_before = 0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        group = ordered[cursor:end]
        group_positive = sum(label for _probability, label in group)
        group_negative = len(group) - group_positive
        wins += group_positive * negatives_before
        wins += 0.5 * group_positive * group_negative
        negatives_before += group_negative
        cursor = end
    return wins / (positive * negative)


def _metrics(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    brier = sum(
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        members = [
            position
            for position, probability in enumerate(probabilities)
            if probability >= lower
            and (probability < upper or (index == 9 and probability == 1.0))
        ]
        if members:
            predicted = sum(probabilities[position] for position in members) / len(members)
            observed = sum(labels[position] for position in members) / len(members)
            ece += len(members) / len(labels) * abs(predicted - observed)
    base_rate = sum(labels) / len(labels)
    baseline_brier = sum((base_rate - label) ** 2 for label in labels) / len(labels)
    return {
        "auc": round(_auc(probabilities, labels), 10),
        "brier": round(brier, 10),
        "ece": round(ece, 10),
        "heldout_base_rate": round(base_rate, 10),
        "constant_baseline_brier": round(baseline_brier, 10),
    }


def _reliability(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(RELIABILITY_BINS):
        lower = index / RELIABILITY_BINS
        upper = (index + 1) / RELIABILITY_BINS
        members = [
            position
            for position, probability in enumerate(probabilities)
            if probability >= lower
            and (
                probability < upper
                or (index == RELIABILITY_BINS - 1 and probability == 1.0)
            )
        ]
        successes = sum(labels[position] for position in members)
        rows.append(
            {
                "index": index,
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "successes": successes,
                "observed_rate": (
                    None if not members else round(successes / len(members), 10)
                ),
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class PrefixStabilityCalibrator:
    """Monotonic recurrence calibrator with held-out admission evidence."""

    blocks: tuple[dict[str, Any], ...]
    manifest_data: dict[str, Any]

    @property
    def admitted(self) -> bool:
        return bool(self.manifest_data.get("admitted"))

    def probability(self, raw_stability: float) -> float:
        value = _probability(raw_stability, name="raw_stability")
        return _predict(self.blocks, value)

    def estimate(self, raw_stability: float) -> dict[str, Any]:
        value = _probability(raw_stability, name="raw_stability")
        return {
            "target": CALIBRATION_TARGET,
            "raw_stability": round(value, 10),
            "future_recurrence_probability": round(self.probability(value), 10),
            "calibrated": self.admitted,
            "selection_authority_admitted": False,
            "correctness_authority_admitted": False,
        }

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest_data))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PREFIX_STABILITY_CALIBRATOR_SCHEMA,
            "target": CALIBRATION_TARGET,
            "blocks": [dict(block) for block in self.blocks],
            "manifest": self.manifest(),
        }
        return {**payload, "content_sha256": _sha(payload)}

    def validate(self) -> None:
        if not self.blocks or len(self.blocks) > MAX_FIT_BLOCKS:
            raise ValueError("prefix-stability calibrator block inventory is invalid")
        previous_upper = -1.0
        previous_probability = -1.0
        fit_total = fit_successes = 0
        for block in self.blocks:
            fields = {"lower", "upper", "count", "successes", "probability"}
            if not isinstance(block, Mapping) or set(block) != fields:
                raise ValueError("prefix-stability calibrator block fields differ")
            lower = _probability(block["lower"], name="block.lower")
            upper = _probability(block["upper"], name="block.upper")
            probability = _probability(block["probability"], name="block.probability")
            count = block["count"]
            successes = block["successes"]
            if (
                type(count) is not int
                or type(successes) is not int
                or not 1 <= count <= MAX_EXAMPLES
                or not 0 <= successes <= count
                or lower > upper
                or lower <= previous_upper
                or probability < previous_probability
                or probability != round(successes / count, 10)
            ):
                raise ValueError("prefix-stability calibrator block is invalid")
            previous_upper = upper
            previous_probability = probability
            fit_total += count
            fit_successes += successes
        manifest = self.manifest_data
        fields = {
            "target",
            "fit_dataset_sha256",
            "calibration_dataset_sha256",
            "fit_tasks_sha256",
            "calibration_tasks_sha256",
            "fit_count",
            "fit_successes",
            "calibration_count",
            "calibration_successes",
            "metrics",
            "reliability_bins",
            "failure_reasons",
            "admitted",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != fields:
            raise ValueError("prefix-stability calibrator manifest fields differ")
        if (
            manifest["target"] != CALIBRATION_TARGET
            or any(
                not _is_sha(manifest[name])
                for name in (
                    "fit_dataset_sha256",
                    "calibration_dataset_sha256",
                    "fit_tasks_sha256",
                    "calibration_tasks_sha256",
                )
            )
            or manifest["fit_dataset_sha256"] == manifest["calibration_dataset_sha256"]
            or manifest["fit_tasks_sha256"] == manifest["calibration_tasks_sha256"]
            or manifest["fit_count"] != fit_total
            or manifest["fit_successes"] != fit_successes
            or not MIN_FIT_EXAMPLES <= manifest["fit_count"] <= MAX_EXAMPLES
            or not MIN_CLASS_EXAMPLES
            <= manifest["fit_successes"]
            <= manifest["fit_count"] - MIN_CLASS_EXAMPLES
            or not MIN_CALIBRATION_EXAMPLES
            <= manifest["calibration_count"]
            <= MAX_EXAMPLES
            or type(manifest["calibration_successes"]) is not int
            or not MIN_CLASS_EXAMPLES
            <= manifest["calibration_successes"]
            <= manifest["calibration_count"] - MIN_CLASS_EXAMPLES
            or type(manifest["admitted"]) is not bool
            or not isinstance(manifest["failure_reasons"], list)
            or any(not isinstance(reason, str) or not reason for reason in manifest["failure_reasons"])
        ):
            raise ValueError("prefix-stability calibrator manifest is invalid")
        metrics = manifest["metrics"]
        metric_fields = {
            "auc",
            "brier",
            "ece",
            "heldout_base_rate",
            "constant_baseline_brier",
        }
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != metric_fields
            or any(
                _probability(value, name=f"metrics.{name}") != float(value)
                for name, value in metrics.items()
            )
        ):
            raise ValueError("prefix-stability calibration metrics are invalid")
        expected_failures = sorted(
            reason
            for reason, failed in (
                ("auc_below_limit", metrics["auc"] < 0.55),
                ("brier_above_limit", metrics["brier"] > 0.25),
                ("ece_above_limit", metrics["ece"] > 0.20),
                (
                    "constant_baseline_not_matched",
                    metrics["brier"] > metrics["constant_baseline_brier"] + 0.02,
                ),
            )
            if failed
        )
        if (
            manifest["failure_reasons"] != expected_failures
            or manifest["admitted"] is not (not expected_failures)
        ):
            raise ValueError("prefix-stability calibration verdict is invalid")
        rows = manifest["reliability_bins"]
        if not isinstance(rows, list) or len(rows) != RELIABILITY_BINS:
            raise ValueError("prefix-stability reliability bins are invalid")
        count = successes = 0
        for index, row in enumerate(rows):
            fields = {"index", "lower", "upper", "count", "successes", "observed_rate"}
            if not isinstance(row, Mapping) or set(row) != fields:
                raise ValueError("prefix-stability reliability row fields differ")
            if (
                row["index"] != index
                or row["lower"] != index / RELIABILITY_BINS
                or row["upper"] != (index + 1) / RELIABILITY_BINS
                or type(row["count"]) is not int
                or type(row["successes"]) is not int
                or not 0 <= row["successes"] <= row["count"]
                or row["observed_rate"]
                != (
                    None
                    if row["count"] == 0
                    else round(row["successes"] / row["count"], 10)
                )
            ):
                raise ValueError("prefix-stability reliability row is invalid")
            count += row["count"]
            successes += row["successes"]
        if (
            count != manifest["calibration_count"]
            or successes != manifest["calibration_successes"]
        ):
            raise ValueError("prefix-stability reliability totals differ")

    @classmethod
    def fit(
        cls,
        fit_examples: Sequence[PrefixStabilityExample],
        calibration_examples: Sequence[PrefixStabilityExample],
    ) -> PrefixStabilityCalibrator:
        _validate_split(fit_examples, name="fit", minimum=MIN_FIT_EXAMPLES)
        _validate_split(
            calibration_examples,
            name="calibration",
            minimum=MIN_CALIBRATION_EXAMPLES,
        )
        fit_tasks = {example.task_id for example in fit_examples}
        calibration_tasks = {example.task_id for example in calibration_examples}
        fit_evidence = {
            value
            for example in fit_examples
            for value in (
                example.probe_receipt_sha256,
                example.future_receipt_sha256,
            )
        }
        calibration_evidence = {
            value
            for example in calibration_examples
            for value in (
                example.probe_receipt_sha256,
                example.future_receipt_sha256,
            )
        }
        if fit_tasks & calibration_tasks or fit_evidence & calibration_evidence:
            raise ValueError("fit and calibration evidence overlap")
        ordered_fit = tuple(sorted(fit_examples, key=lambda row: row.example_id))
        ordered_calibration = tuple(
            sorted(calibration_examples, key=lambda row: row.example_id)
        )
        blocks = _fit_isotonic(ordered_fit)
        probabilities = [
            _predict(blocks, example.raw_stability)
            for example in ordered_calibration
        ]
        labels = [
            int(example.future_conclusion_match)
            for example in ordered_calibration
        ]
        metrics = _metrics(probabilities, labels)
        failures = sorted(
            reason
            for reason, failed in (
                ("auc_below_limit", metrics["auc"] < 0.55),
                ("brier_above_limit", metrics["brier"] > 0.25),
                ("ece_above_limit", metrics["ece"] > 0.20),
                (
                    "constant_baseline_not_matched",
                    metrics["brier"] > metrics["constant_baseline_brier"] + 0.02,
                ),
            )
            if failed
        )
        manifest = {
            "target": CALIBRATION_TARGET,
            "fit_dataset_sha256": _dataset_sha(ordered_fit),
            "calibration_dataset_sha256": _dataset_sha(ordered_calibration),
            "fit_tasks_sha256": _task_sha(ordered_fit),
            "calibration_tasks_sha256": _task_sha(ordered_calibration),
            "fit_count": len(ordered_fit),
            "fit_successes": sum(example.future_conclusion_match for example in ordered_fit),
            "calibration_count": len(ordered_calibration),
            "calibration_successes": sum(labels),
            "metrics": metrics,
            "reliability_bins": _reliability(probabilities, labels),
            "failure_reasons": failures,
            "admitted": not failures,
        }
        calibrator = cls(
            blocks=tuple(dict(block) for block in blocks),
            manifest_data=manifest,
        )
        calibrator.validate()
        return calibrator

    def save(self, path: str | Path) -> str:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        self.validate()
        raw = _canonical_bytes(self.to_payload())
        if not 1 <= len(raw) <= MAX_ARTIFACT_BYTES:
            raise ValueError("prefix-stability artifact exceeds size bound")
        with local_internal_governed_scope(
            "prefix_stability.calibrator",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                Path(path),
                raw,
                source="prefix_stability.calibrator",
            )
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> PrefixStabilityCalibrator:
        from core.runtime.file_read_gateway import read_stable_bytes

        if not _is_sha(expected_sha256):
            raise ValueError("prefix-stability artifact pin is invalid")
        raw = read_stable_bytes(Path(path), max_bytes=MAX_ARTIFACT_BYTES)
        if not raw:
            raise ValueError("prefix-stability artifact is empty")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("prefix-stability artifact SHA-256 differs")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("prefix-stability artifact is not valid JSON") from exc
        fields = {"schema", "target", "blocks", "manifest", "content_sha256"}
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("prefix-stability artifact fields differ")
        content = {key: payload[key] for key in fields - {"content_sha256"}}
        if (
            payload["schema"] != PREFIX_STABILITY_CALIBRATOR_SCHEMA
            or payload["target"] != CALIBRATION_TARGET
            or payload["content_sha256"] != _sha(content)
            or not isinstance(payload["blocks"], list)
            or not isinstance(payload["manifest"], Mapping)
        ):
            raise ValueError("prefix-stability artifact commitment is invalid")
        calibrator = cls(
            blocks=tuple(dict(block) for block in payload["blocks"]),
            manifest_data=dict(payload["manifest"]),
        )
        calibrator.validate()
        if not calibrator.admitted:
            raise ValueError("prefix-stability artifact failed held-out calibration")
        return calibrator


MAX_FIT_BLOCKS = MAX_EXAMPLES


__all__ = [
    "CALIBRATION_TARGET",
    "PREFIX_STABILITY_CALIBRATOR_SCHEMA",
    "PREFIX_STABILITY_EXAMPLE_SCHEMA",
    "PrefixStabilityCalibrator",
    "PrefixStabilityExample",
]
