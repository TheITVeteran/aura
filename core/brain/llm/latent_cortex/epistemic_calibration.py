"""Measured calibration profiles for RLC claim uncertainty.

Profiles are immutable certificates over independently graded held-out
predictions. They report proper scoring and reliability metrics, expire, and
map a raw claim signal to a Wilson-bounded empirical interval. Sparse, stale,
non-discriminative, or poorly calibrated profiles return abstention rather than
manufacturing confidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

CALIBRATION_PROFILE_SCHEMA = "aura.rlc.epistemic_calibration.v1"
MAX_CALIBRATION_OBSERVATIONS = 4_096
MAX_CALIBRATION_BINS = 20

_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,95}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CalibrationError(ValueError):
    """A calibration profile or estimate violated its trust contract."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"calibration is not serializable: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise CalibrationError(f"{name} is not a valid bounded identifier")
    return value


def _text(value: Any, *, name: str, limit: int = 256) -> str:
    if not isinstance(value, str):
        raise CalibrationError(f"{name} must be a string")
    rendered = value.strip()
    if not rendered or len(rendered) > limit or _CONTROL_RE.search(rendered):
        raise CalibrationError(f"{name} is empty, oversized, or contains controls")
    return rendered


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise CalibrationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise CalibrationError(f"{name} must be finite and in [0, 1]")
    return parsed


def _time(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise CalibrationError(f"{name} must be finite and nonnegative")
    return parsed


def _exact_fields(data: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    if not isinstance(data, Mapping):
        raise CalibrationError(f"{name} must be an object")
    actual = set(data)
    if actual != fields:
        raise CalibrationError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )


def _wire_list(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise CalibrationError(f"{name} must be an array")
    return tuple(value)


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _wilson(successes: int, total: int, *, z: float, upper: bool) -> float:
    if total <= 0:
        return 1.0 if upper else 0.0
    point = successes / total
    denominator = 1.0 + z * z / total
    centre = point + z * z / (2.0 * total)
    margin = z * math.sqrt((point * (1.0 - point) + z * z / (4.0 * total)) / total)
    bound = (centre + margin if upper else centre - margin) / denominator
    return _rounded(min(1.0, max(0.0, bound)))


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    bins: int = 5
    min_samples: int = 40
    min_bin_samples: int = 12
    max_brier: float = 0.20
    max_ece: float = 0.10
    support_lower_bound: float = 0.70
    wilson_z: float = 1.6449

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("bins", self.bins, 2, MAX_CALIBRATION_BINS),
            ("min_samples", self.min_samples, 2, MAX_CALIBRATION_OBSERVATIONS),
            ("min_bin_samples", self.min_bin_samples, 2, MAX_CALIBRATION_OBSERVATIONS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise CalibrationError(f"policy.{name} must be an integer in [{low}, {high}]")
        if self.min_bin_samples > self.min_samples:
            raise CalibrationError("policy min_bin_samples exceeds min_samples")
        object.__setattr__(self, "max_brier", _probability(self.max_brier, name="policy.max_brier"))
        object.__setattr__(self, "max_ece", _probability(self.max_ece, name="policy.max_ece"))
        object.__setattr__(
            self,
            "support_lower_bound",
            _probability(
                self.support_lower_bound,
                name="policy.support_lower_bound",
            ),
        )
        if isinstance(self.wilson_z, bool) or not isinstance(self.wilson_z, (int, float)):
            raise CalibrationError("policy.wilson_z must be numeric")
        z = float(self.wilson_z)
        if not math.isfinite(z) or not 0.0 < z <= 5.0:
            raise CalibrationError("policy.wilson_z must be finite and in (0, 5]")
        object.__setattr__(self, "wilson_z", z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "min_samples": self.min_samples,
            "min_bin_samples": self.min_bin_samples,
            "max_brier": self.max_brier,
            "max_ece": self.max_ece,
            "support_lower_bound": self.support_lower_bound,
            "wilson_z": self.wilson_z,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationPolicy:
        fields = {
            "bins",
            "min_samples",
            "min_bin_samples",
            "max_brier",
            "max_ece",
            "support_lower_bound",
            "wilson_z",
        }
        _exact_fields(data, fields, name="calibration.policy")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    observation_id: str
    domain: str
    predicted_probability: float
    outcome: bool
    prediction_receipt_sha256: str
    outcome_receipt_sha256: str
    outcome_verifier_id: str
    outcome_verifier_version: str
    observed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _identifier(self.observation_id, name="observation_id"),
        )
        object.__setattr__(self, "domain", _identifier(self.domain, name="observation.domain"))
        object.__setattr__(
            self,
            "predicted_probability",
            _probability(
                self.predicted_probability,
                name="observation.predicted_probability",
            ),
        )
        if type(self.outcome) is not bool:
            raise CalibrationError("observation.outcome must be boolean ground truth")
        _digest(
            self.prediction_receipt_sha256,
            name="observation.prediction_receipt_sha256",
        )
        _digest(
            self.outcome_receipt_sha256,
            name="observation.outcome_receipt_sha256",
        )
        object.__setattr__(
            self,
            "outcome_verifier_id",
            _text(self.outcome_verifier_id, name="observation.outcome_verifier_id"),
        )
        object.__setattr__(
            self,
            "outcome_verifier_version",
            _text(
                self.outcome_verifier_version,
                name="observation.outcome_verifier_version",
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            _time(self.observed_at, name="observation.observed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "domain": self.domain,
            "predicted_probability": self.predicted_probability,
            "outcome": self.outcome,
            "prediction_receipt_sha256": self.prediction_receipt_sha256,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
            "outcome_verifier_id": self.outcome_verifier_id,
            "outcome_verifier_version": self.outcome_verifier_version,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationObservation:
        fields = {
            "observation_id",
            "domain",
            "predicted_probability",
            "outcome",
            "prediction_receipt_sha256",
            "outcome_receipt_sha256",
            "outcome_verifier_id",
            "outcome_verifier_version",
            "observed_at",
        }
        _exact_fields(data, fields, name="calibration.observation")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    index: int
    lower: float
    upper: float
    count: int
    successes: int
    mean_prediction: float | None
    observed_rate: float | None
    lower_bound: float
    upper_bound: float
    calibration_gap: float | None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise CalibrationError("calibration bin index must be nonnegative")
        lower = _probability(self.lower, name="calibration.bin.lower")
        upper = _probability(self.upper, name="calibration.bin.upper")
        if upper <= lower:
            raise CalibrationError("calibration bin bounds are invalid")
        for name, value in (("count", self.count), ("successes", self.successes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CalibrationError(f"calibration bin {name} must be nonnegative")
        if self.successes > self.count:
            raise CalibrationError("calibration bin successes exceed count")
        optional = (
            ("mean_prediction", self.mean_prediction),
            ("observed_rate", self.observed_rate),
            ("calibration_gap", self.calibration_gap),
        )
        if self.count == 0 and any(value is not None for _, value in optional):
            raise CalibrationError("empty calibration bin contains derived values")
        if self.count > 0 and any(value is None for _, value in optional):
            raise CalibrationError("populated calibration bin omits derived values")
        for name, value in optional:
            if value is not None:
                _probability(value, name=f"calibration.bin.{name}")
        lower_bound = _probability(
            self.lower_bound,
            name="calibration.bin.lower_bound",
        )
        upper_bound = _probability(
            self.upper_bound,
            name="calibration.bin.upper_bound",
        )
        if lower_bound > upper_bound:
            raise CalibrationError("calibration confidence bounds are reversed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "successes": self.successes,
            "mean_prediction": self.mean_prediction,
            "observed_rate": self.observed_rate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "calibration_gap": self.calibration_gap,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReliabilityBin:
        fields = {
            "index",
            "lower",
            "upper",
            "count",
            "successes",
            "mean_prediction",
            "observed_rate",
            "lower_bound",
            "upper_bound",
            "calibration_gap",
        }
        _exact_fields(data, fields, name="calibration.bin")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True, slots=True)
class CalibrationEstimate:
    raw_probability: float
    lower: float
    point: float
    upper: float
    sample_count: int
    profile_id: str
    profile_sha256: str
    evaluated_at: float
    bin_index: int
    supported: bool
    abstention_reason: str

    def __post_init__(self) -> None:
        raw = _probability(self.raw_probability, name="estimate.raw_probability")
        lower = _probability(self.lower, name="estimate.lower")
        point = _probability(self.point, name="estimate.point")
        upper = _probability(self.upper, name="estimate.upper")
        if not lower <= point <= upper:
            raise CalibrationError("estimate interval is reversed")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise CalibrationError("estimate sample_count must be nonnegative")
        _identifier(self.profile_id, name="estimate.profile_id")
        _digest(self.profile_sha256, name="estimate.profile_sha256")
        _time(self.evaluated_at, name="estimate.evaluated_at")
        if (
            isinstance(self.bin_index, bool)
            or not isinstance(self.bin_index, int)
            or self.bin_index < 0
        ):
            raise CalibrationError("estimate bin_index must be nonnegative")
        if not isinstance(self.supported, bool):
            raise CalibrationError("estimate supported must be boolean")
        if self.supported and self.abstention_reason:
            raise CalibrationError("supported estimate cannot include abstention reason")
        if not self.supported and not self.abstention_reason:
            raise CalibrationError("unsupported estimate requires abstention reason")
        object.__setattr__(self, "raw_probability", raw)


def _derive(
    observations: tuple[CalibrationObservation, ...],
    policy: CalibrationPolicy,
) -> tuple[tuple[ReliabilityBin, ...], float, float, float, float, bool, tuple[str, ...]]:
    total = len(observations)
    if total == 0:
        raise CalibrationError("cannot derive calibration from no observations")
    truths = [1.0 if item.outcome else 0.0 for item in observations]
    predictions = [item.predicted_probability for item in observations]
    base_rate = sum(truths) / total
    brier = _rounded(
        sum(
            (prediction - truth) ** 2 for prediction, truth in zip(predictions, truths, strict=True)
        )
        / total
    )
    baseline_brier = _rounded(sum((base_rate - truth) ** 2 for truth in truths) / total)
    reliability: list[ReliabilityBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(policy.bins):
        lower = index / policy.bins
        upper = (index + 1) / policy.bins
        members = [
            item
            for item in observations
            if lower <= item.predicted_probability < upper
            or (index == policy.bins - 1 and item.predicted_probability == 1.0)
        ]
        count = len(members)
        successes = sum(item.outcome for item in members)
        if members:
            mean_prediction = _rounded(sum(item.predicted_probability for item in members) / count)
            observed_rate = _rounded(successes / count)
            gap = _rounded(abs(mean_prediction - observed_rate))
            weighted_gap += count / total * gap
            maximum_gap = max(maximum_gap, gap)
        else:
            mean_prediction = None
            observed_rate = None
            gap = None
        reliability.append(
            ReliabilityBin(
                index=index,
                lower=_rounded(lower),
                upper=_rounded(upper),
                count=count,
                successes=successes,
                mean_prediction=mean_prediction,
                observed_rate=observed_rate,
                lower_bound=_wilson(
                    successes,
                    count,
                    z=policy.wilson_z,
                    upper=False,
                ),
                upper_bound=_wilson(
                    successes,
                    count,
                    z=policy.wilson_z,
                    upper=True,
                ),
                calibration_gap=gap,
            )
        )
    ece = _rounded(weighted_gap)
    mce = _rounded(maximum_gap)
    failures: list[str] = []
    if total < policy.min_samples:
        failures.append("insufficient_samples")
    if len(set(truths)) < 2:
        failures.append("single_outcome_class")
    if brier > policy.max_brier:
        failures.append("brier_above_limit")
    if ece > policy.max_ece:
        failures.append("ece_above_limit")
    if brier >= baseline_brier:
        failures.append("does_not_beat_constant_predictor")
    reasons = tuple(sorted(failures))
    return (
        tuple(reliability),
        brier,
        baseline_brier,
        ece,
        mce,
        not reasons,
        reasons,
    )


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    schema: str
    profile_id: str
    estimator_id: str
    estimator_version: str
    domain: str
    dataset_sha256: str
    split_manifest_sha256: str
    trained_at: float
    expires_at: float
    policy: CalibrationPolicy
    observations: tuple[CalibrationObservation, ...]
    reliability_bins: tuple[ReliabilityBin, ...]
    brier_score: float
    baseline_brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    passed: bool
    failure_reasons: tuple[str, ...]
    profile_sha256: str

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_PROFILE_SCHEMA:
            raise CalibrationError("unsupported calibration profile schema")
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, name="profile_id"))
        object.__setattr__(self, "estimator_id", _text(self.estimator_id, name="estimator_id"))
        object.__setattr__(
            self,
            "estimator_version",
            _text(self.estimator_version, name="estimator_version"),
        )
        object.__setattr__(self, "domain", _identifier(self.domain, name="profile.domain"))
        _digest(self.dataset_sha256, name="profile.dataset_sha256")
        _digest(self.split_manifest_sha256, name="profile.split_manifest_sha256")
        trained_at = _time(self.trained_at, name="profile.trained_at")
        expires_at = _time(self.expires_at, name="profile.expires_at")
        if expires_at <= trained_at:
            raise CalibrationError("calibration profile must expire after training")
        object.__setattr__(self, "trained_at", trained_at)
        object.__setattr__(self, "expires_at", expires_at)
        if not isinstance(self.policy, CalibrationPolicy):
            raise CalibrationError("profile.policy must be a CalibrationPolicy")
        supplied_observations = tuple(self.observations)
        if not supplied_observations or (len(supplied_observations) > MAX_CALIBRATION_OBSERVATIONS):
            raise CalibrationError("calibration observations are empty or out of bounds")
        if any(not isinstance(item, CalibrationObservation) for item in supplied_observations):
            raise CalibrationError("profile contains an invalid observation")
        observations = tuple(sorted(supplied_observations, key=lambda item: item.observation_id))
        identifiers = [item.observation_id for item in observations]
        prediction_receipts = [item.prediction_receipt_sha256 for item in observations]
        outcome_receipts = [item.outcome_receipt_sha256 for item in observations]
        if any(
            len(set(values)) != len(values)
            for values in (identifiers, prediction_receipts, outcome_receipts)
        ):
            raise CalibrationError(
                "calibration observations contain duplicate identity or receipts"
            )
        for item in observations:
            if item.domain != self.domain:
                raise CalibrationError("calibration observation domain drift")
            if item.outcome_verifier_id == self.estimator_id:
                raise CalibrationError("calibration outcome verifier must differ from estimator")
            if item.observed_at > trained_at:
                raise CalibrationError("calibration observation occurs after profile training")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(
            self,
            "brier_score",
            _probability(self.brier_score, name="profile.brier_score"),
        )
        object.__setattr__(
            self,
            "baseline_brier_score",
            _probability(
                self.baseline_brier_score,
                name="profile.baseline_brier_score",
            ),
        )
        object.__setattr__(
            self,
            "expected_calibration_error",
            _probability(
                self.expected_calibration_error,
                name="profile.expected_calibration_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_calibration_error",
            _probability(
                self.maximum_calibration_error,
                name="profile.maximum_calibration_error",
            ),
        )
        if not isinstance(self.passed, bool):
            raise CalibrationError("profile.passed must be boolean")
        reasons = tuple(
            sorted(
                _identifier(reason, name="profile failure reason")
                for reason in self.failure_reasons
            )
        )
        if len(reasons) != len(set(reasons)):
            raise CalibrationError("profile failure reasons contain duplicates")
        object.__setattr__(self, "failure_reasons", reasons)
        reliability_bins = tuple(self.reliability_bins)
        if len(reliability_bins) != self.policy.bins or any(
            not isinstance(item, ReliabilityBin) for item in reliability_bins
        ):
            raise CalibrationError("profile reliability bins are invalid")
        object.__setattr__(self, "reliability_bins", reliability_bins)
        expected = _derive(observations, self.policy)
        actual = (
            self.reliability_bins,
            self.brier_score,
            self.baseline_brier_score,
            self.expected_calibration_error,
            self.maximum_calibration_error,
            self.passed,
            self.failure_reasons,
        )
        if actual != expected:
            raise CalibrationError("calibration metrics do not match graded observations")
        expected_hash = _sha256(self.to_dict(include_hash=False))
        if _digest(self.profile_sha256, name="profile.profile_sha256") != expected_hash:
            raise CalibrationError("calibration profile hash does not match content")

    @classmethod
    def fit(
        cls,
        *,
        profile_id: str,
        estimator_id: str,
        estimator_version: str,
        domain: str,
        dataset_sha256: str,
        split_manifest_sha256: str,
        trained_at: float,
        expires_at: float,
        observations: Iterable[CalibrationObservation],
        policy: CalibrationPolicy | None = None,
    ) -> CalibrationProfile:
        selected_policy = policy or CalibrationPolicy()
        supplied = tuple(observations)
        if not supplied or len(supplied) > MAX_CALIBRATION_OBSERVATIONS:
            raise CalibrationError("calibration observations are empty or out of bounds")
        if any(not isinstance(item, CalibrationObservation) for item in supplied):
            raise CalibrationError("profile contains an invalid observation")
        items = tuple(sorted(supplied, key=lambda item: item.observation_id))
        derived = _derive(items, selected_policy)
        values = {
            "schema": CALIBRATION_PROFILE_SCHEMA,
            "profile_id": profile_id,
            "estimator_id": estimator_id,
            "estimator_version": estimator_version,
            "domain": domain,
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "trained_at": trained_at,
            "expires_at": expires_at,
            "policy": selected_policy,
            "observations": items,
            "reliability_bins": derived[0],
            "brier_score": derived[1],
            "baseline_brier_score": derived[2],
            "expected_calibration_error": derived[3],
            "maximum_calibration_error": derived[4],
            "passed": derived[5],
            "failure_reasons": derived[6],
        }
        provisional = cls.__new__(cls)
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        digest = _sha256(provisional.to_dict(include_hash=False))
        return cls(profile_sha256=digest, **values)

    def estimate(self, raw_probability: float, *, evaluated_at: float) -> CalibrationEstimate:
        raw = _probability(raw_probability, name="estimate.raw_probability")
        checked_at = _time(evaluated_at, name="estimate.evaluated_at")
        index = min(int(raw * self.policy.bins), self.policy.bins - 1)
        cell = self.reliability_bins[index]
        reasons: list[str] = []
        if checked_at < self.trained_at:
            reasons.append("profile_not_yet_valid")
        if checked_at > self.expires_at:
            reasons.append("profile_expired")
        if not self.passed:
            reasons.append("profile_not_admitted")
        if cell.count < self.policy.min_bin_samples:
            reasons.append("sparse_calibration_bin")
        if cell.lower_bound < self.policy.support_lower_bound:
            reasons.append("support_lower_bound_not_met")
        abstention_reason = "+".join(sorted(set(reasons)))
        return CalibrationEstimate(
            raw_probability=raw,
            lower=cell.lower_bound,
            point=cell.observed_rate if cell.observed_rate is not None else raw,
            upper=cell.upper_bound,
            sample_count=cell.count,
            profile_id=self.profile_id,
            profile_sha256=self.profile_sha256,
            evaluated_at=checked_at,
            bin_index=index,
            supported=not abstention_reason,
            abstention_reason=abstention_reason,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "domain": self.domain,
            "dataset_sha256": self.dataset_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "trained_at": self.trained_at,
            "expires_at": self.expires_at,
            "policy": self.policy.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "reliability_bins": [item.to_dict() for item in self.reliability_bins],
            "brier_score": self.brier_score,
            "baseline_brier_score": self.baseline_brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }
        if include_hash:
            result["profile_sha256"] = self.profile_sha256
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationProfile:
        fields = {
            "schema",
            "profile_id",
            "estimator_id",
            "estimator_version",
            "domain",
            "dataset_sha256",
            "split_manifest_sha256",
            "trained_at",
            "expires_at",
            "policy",
            "observations",
            "reliability_bins",
            "brier_score",
            "baseline_brier_score",
            "expected_calibration_error",
            "maximum_calibration_error",
            "passed",
            "failure_reasons",
            "profile_sha256",
        }
        _exact_fields(data, fields, name="calibration.profile")
        if not isinstance(data["passed"], bool):
            raise CalibrationError("calibration profile passed must be boolean")
        return cls(
            schema=data["schema"],
            profile_id=data["profile_id"],
            estimator_id=data["estimator_id"],
            estimator_version=data["estimator_version"],
            domain=data["domain"],
            dataset_sha256=data["dataset_sha256"],
            split_manifest_sha256=data["split_manifest_sha256"],
            trained_at=data["trained_at"],
            expires_at=data["expires_at"],
            policy=CalibrationPolicy.from_dict(data["policy"]),
            observations=tuple(
                CalibrationObservation.from_dict(item)
                for item in _wire_list(
                    data["observations"],
                    name="calibration.observations",
                )
            ),
            reliability_bins=tuple(
                ReliabilityBin.from_dict(item)
                for item in _wire_list(
                    data["reliability_bins"],
                    name="calibration.reliability_bins",
                )
            ),
            brier_score=data["brier_score"],
            baseline_brier_score=data["baseline_brier_score"],
            expected_calibration_error=data["expected_calibration_error"],
            maximum_calibration_error=data["maximum_calibration_error"],
            passed=data["passed"],
            failure_reasons=tuple(
                _wire_list(
                    data["failure_reasons"],
                    name="calibration.failure_reasons",
                )
            ),
            profile_sha256=data["profile_sha256"],
        )


__all__ = [
    "CALIBRATION_PROFILE_SCHEMA",
    "CalibrationError",
    "CalibrationEstimate",
    "CalibrationObservation",
    "CalibrationPolicy",
    "CalibrationProfile",
    "ReliabilityBin",
]
