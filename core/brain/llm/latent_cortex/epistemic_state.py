"""Strict transactional epistemic state for one latent-cortex episode.

This is the shared data substrate for Spark reasoning. It deliberately contains
no model calls and no prose parser: producers must submit typed claims, evidence,
hypotheses, operations, budgets, and answer dependencies. Every state is deeply
immutable, bounded, canonically serialized, content-addressed, and validated as
a dependency graph before it can become current.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from core.brain.llm.latent_cortex.epistemic_calibration import (
    MAX_CALIBRATION_OBSERVATIONS,
    CalibrationEstimate,
    CalibrationProfile,
)

EPISTEMIC_STATE_SCHEMA = "aura.rlc.epistemic_state.v5"

MAX_OBJECTIVE_CHARS = 16_384
MAX_TEXT_CHARS = 8_192
MAX_SUMMARY_CHARS = 2_048
MAX_CONSTRAINTS = 128
MAX_CALIBRATIONS = 32
MAX_EVIDENCE = 256
MAX_HYPOTHESES = 64
MAX_CLAIMS = 512
MAX_OPERATIONS = 512
MAX_REFS = 128
MAX_OPERATION_ATTEMPTS = 3
MIN_HYPOTHESIS_MASS = 0.02
MAX_MINORITY_MASS = 0.25
FAVORED_HYPOTHESIS_MASS = 0.50
PORTFOLIO_SUM_TOLERANCE = 1e-9

_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,95}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class EpistemicStateError(ValueError):
    """Base error for invalid or stale state transitions."""


class StaleEpistemicTransactionError(EpistemicStateError):
    """A transaction tried to replace a state other than its base."""


class EpistemicStatePersistence(Protocol):
    """Durable write-ahead authority used by the state machine."""

    def bootstrap(self, genesis: EpistemicState) -> EpistemicState: ...

    def append(
        self,
        *,
        expected_base: EpistemicState,
        candidate: EpistemicState,
    ) -> None: ...


class EvidenceKind(StrEnum):
    IMMUTABLE_PROBLEM = "immutable_problem"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    CALCULATION = "calculation"
    PROOF = "proof"
    SIMULATION = "simulation"
    OBSERVATION = "observation"
    MEMORY = "memory"


class EvidenceVerification(StrEnum):
    UNVERIFIED = "unverified"
    SOURCE_BOUND = "source_bound"
    INDEPENDENT = "independent"


class EvidencePurpose(StrEnum):
    IMMUTABLE_PROBLEM = "immutable_problem"
    CLAIM_TEST = "claim_test"
    CONTEXT_ONLY = "context_only"


class UncertaintyBasis(StrEnum):
    UNCALIBRATED = "uncalibrated"
    EMPIRICAL = "empirical"
    EXACT = "exact"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    MINORITY = "minority"
    FAVORED = "favored"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class OperationKind(StrEnum):
    DECOMPOSE = "decompose"
    BLIND_RESOLVE = "blind_resolve"
    BRANCH = "branch"
    SEARCH_MEMORY = "search_memory"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    EXECUTE = "execute"
    SIMULATE = "simulate"
    FALSIFY = "falsify"
    CHECK_ASSUMPTION = "check_assumption"
    REGENERATE_FROM_PREFIX = "regenerate_from_prefix"
    FORMALIZE = "formalize"
    COMPARE = "compare"
    BACKTRACK = "backtrack"
    COMPRESS_STATE = "compress_state"
    ANSWER = "answer"
    ABSTAIN = "abstain"


class OperationOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


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
        raise EpistemicStateError(f"state is not canonically serializable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _strict_text(value: Any, *, name: str, limit: int, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise EpistemicStateError(f"{name} must be a string")
    rendered = value.strip()
    if not rendered and not empty:
        raise EpistemicStateError(f"{name} must not be empty")
    if len(rendered) > limit:
        raise EpistemicStateError(f"{name} exceeds {limit} characters")
    if _CONTROL_RE.search(rendered):
        raise EpistemicStateError(f"{name} contains control characters")
    return rendered


def _strict_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EpistemicStateError(f"{name} is not a valid bounded identifier")
    return value


def _strict_digest(value: Any, *, name: str, empty: bool = False) -> str:
    if empty and value == "":
        return ""
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise EpistemicStateError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unit(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EpistemicStateError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise EpistemicStateError(f"{name} must be finite and in [0, 1]")
    return parsed


def _nonnegative(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EpistemicStateError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise EpistemicStateError(f"{name} must be finite and nonnegative")
    return parsed


def _bounded_ids(values: Iterable[str], *, name: str, limit: int = MAX_REFS) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise EpistemicStateError(f"{name} must be a sequence of identifiers")
    result = tuple(_strict_id(value, name=f"{name} item") for value in values)
    if len(result) > limit:
        raise EpistemicStateError(f"{name} exceeds {limit} references")
    if len(set(result)) != len(result):
        raise EpistemicStateError(f"{name} contains duplicate references")
    return tuple(sorted(result))


def _exact_fields(data: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    if not isinstance(data, Mapping):
        raise EpistemicStateError(f"{name} must be an object")
    actual = set(data)
    if actual != fields:
        raise EpistemicStateError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )


def _wire_list(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise EpistemicStateError(f"{name} must be an array")
    return tuple(value)


def _wire_enum[EnumT: StrEnum](enum_type: type[EnumT], value: Any, *, name: str) -> EnumT:
    if not isinstance(value, str):
        raise EpistemicStateError(f"{name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise EpistemicStateError(f"{name} is not a supported value: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ProbabilityInterval:
    lower: float
    point: float
    upper: float
    method: str
    evidence_count: int
    basis: UncertaintyBasis = UncertaintyBasis.UNCALIBRATED
    raw_probability: float | None = None
    calibration_id: str = ""
    calibration_sha256: str = ""
    signal_evidence_ids: tuple[str, ...] = ()
    evaluated_at: float | None = None
    abstain: bool = True
    abstention_reason: str = "uncalibrated"

    def __post_init__(self) -> None:
        lower = _unit(self.lower, name="uncertainty.lower")
        point = _unit(self.point, name="uncertainty.point")
        upper = _unit(self.upper, name="uncertainty.upper")
        if not lower <= point <= upper:
            raise EpistemicStateError("uncertainty interval must satisfy lower <= point <= upper")
        if not isinstance(self.evidence_count, int) or isinstance(self.evidence_count, bool):
            raise EpistemicStateError("uncertainty.evidence_count must be an integer")
        if not 0 <= self.evidence_count <= MAX_CALIBRATION_OBSERVATIONS:
            raise EpistemicStateError("uncertainty.evidence_count is out of bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self, "method", _strict_text(self.method, name="uncertainty.method", limit=96)
        )
        if not isinstance(self.basis, UncertaintyBasis):
            raise EpistemicStateError("uncertainty.basis must be an UncertaintyBasis")
        object.__setattr__(
            self,
            "signal_evidence_ids",
            _bounded_ids(
                self.signal_evidence_ids,
                name="uncertainty signal evidence",
            ),
        )
        if not isinstance(self.abstain, bool):
            raise EpistemicStateError("uncertainty.abstain must be boolean")
        reason = _strict_text(
            self.abstention_reason,
            name="uncertainty.abstention_reason",
            limit=512,
            empty=True,
        )
        object.__setattr__(self, "abstention_reason", reason)
        if self.basis is UncertaintyBasis.UNCALIBRATED:
            if any(
                (
                    self.evidence_count != 0,
                    self.raw_probability is not None,
                    bool(self.calibration_id),
                    bool(self.calibration_sha256),
                    bool(self.signal_evidence_ids),
                    self.evaluated_at is not None,
                    not self.abstain,
                    not reason,
                )
            ):
                raise EpistemicStateError(
                    "uncalibrated uncertainty must remain an explicit abstention"
                )
        elif self.basis is UncertaintyBasis.EMPIRICAL:
            if self.raw_probability is None or self.evaluated_at is None:
                raise EpistemicStateError(
                    "empirical uncertainty requires raw probability and evaluation time"
                )
            object.__setattr__(
                self,
                "raw_probability",
                _unit(self.raw_probability, name="uncertainty.raw_probability"),
            )
            object.__setattr__(
                self,
                "calibration_id",
                _strict_id(self.calibration_id, name="uncertainty.calibration_id"),
            )
            _strict_digest(
                self.calibration_sha256,
                name="uncertainty.calibration_sha256",
            )
            if not self.signal_evidence_ids:
                raise EpistemicStateError("empirical uncertainty requires measured signal evidence")
            object.__setattr__(
                self,
                "evaluated_at",
                _nonnegative(self.evaluated_at, name="uncertainty.evaluated_at"),
            )
            if self.abstain != bool(reason):
                raise EpistemicStateError("empirical abstention and reason must agree")
        else:
            if (lower, point, upper) != (1.0, 1.0, 1.0):
                raise EpistemicStateError("exact uncertainty must equal one")
            if self.raw_probability != 1.0 or self.evaluated_at is None:
                raise EpistemicStateError("exact uncertainty requires a measured exact evaluation")
            if self.calibration_id or self.calibration_sha256:
                raise EpistemicStateError("exact uncertainty cannot reference an empirical profile")
            if not self.signal_evidence_ids or self.evidence_count != len(self.signal_evidence_ids):
                raise EpistemicStateError("exact uncertainty requires counted signal evidence")
            object.__setattr__(
                self,
                "evaluated_at",
                _nonnegative(self.evaluated_at, name="uncertainty.evaluated_at"),
            )
            if self.abstain or reason:
                raise EpistemicStateError("exact uncertainty cannot abstain")

    @classmethod
    def from_calibration_estimate(
        cls,
        estimate: CalibrationEstimate,
        *,
        signal_evidence_ids: Iterable[str],
    ) -> ProbabilityInterval:
        if not isinstance(estimate, CalibrationEstimate):
            raise TypeError("estimate must be a CalibrationEstimate")
        return cls(
            lower=estimate.lower,
            point=estimate.point,
            upper=estimate.upper,
            method=f"empirical_reliability_bin_{estimate.bin_index}",
            evidence_count=estimate.sample_count,
            basis=UncertaintyBasis.EMPIRICAL,
            raw_probability=estimate.raw_probability,
            calibration_id=estimate.profile_id,
            calibration_sha256=estimate.profile_sha256,
            signal_evidence_ids=tuple(signal_evidence_ids),
            evaluated_at=estimate.evaluated_at,
            abstain=not estimate.supported,
            abstention_reason=estimate.abstention_reason,
        )

    @classmethod
    def exact(
        cls,
        *,
        signal_evidence_ids: Iterable[str],
        evaluated_at: float,
    ) -> ProbabilityInterval:
        evidence_ids = tuple(signal_evidence_ids)
        return cls(
            lower=1.0,
            point=1.0,
            upper=1.0,
            method="independently_verified_exact_evidence",
            evidence_count=len(evidence_ids),
            basis=UncertaintyBasis.EXACT,
            raw_probability=1.0,
            signal_evidence_ids=evidence_ids,
            evaluated_at=evaluated_at,
            abstain=False,
            abstention_reason="",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "point": self.point,
            "upper": self.upper,
            "method": self.method,
            "evidence_count": self.evidence_count,
            "basis": self.basis.value,
            "raw_probability": self.raw_probability,
            "calibration_id": self.calibration_id,
            "calibration_sha256": self.calibration_sha256,
            "signal_evidence_ids": list(self.signal_evidence_ids),
            "evaluated_at": self.evaluated_at,
            "abstain": self.abstain,
            "abstention_reason": self.abstention_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbabilityInterval:
        fields = {
            "lower",
            "point",
            "upper",
            "method",
            "evidence_count",
            "basis",
            "raw_probability",
            "calibration_id",
            "calibration_sha256",
            "signal_evidence_ids",
            "evaluated_at",
            "abstain",
            "abstention_reason",
        }
        _exact_fields(data, fields, name="uncertainty")
        return cls(
            lower=data["lower"],
            point=data["point"],
            upper=data["upper"],
            method=data["method"],
            evidence_count=data["evidence_count"],
            basis=_wire_enum(
                UncertaintyBasis,
                data["basis"],
                name="uncertainty.basis",
            ),
            raw_probability=data["raw_probability"],
            calibration_id=data["calibration_id"],
            calibration_sha256=data["calibration_sha256"],
            signal_evidence_ids=_wire_list(
                data["signal_evidence_ids"],
                name="uncertainty.signal_evidence_ids",
            ),
            evaluated_at=data["evaluated_at"],
            abstain=data["abstain"],
            abstention_reason=data["abstention_reason"],
        )


@dataclass(frozen=True, slots=True)
class ProblemFrame:
    objective: str
    objective_sha256: str
    constraints: tuple[str, ...] = ()
    immutable_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        objective = _strict_text(
            self.objective, name="problem.objective", limit=MAX_OBJECTIVE_CHARS
        )
        if _strict_digest(self.objective_sha256, name="problem.objective_sha256") != text_sha256(
            objective
        ):
            raise EpistemicStateError("problem objective digest does not match objective")
        if isinstance(self.constraints, (str, bytes)):
            raise EpistemicStateError("problem constraints must be a sequence")
        constraints = tuple(
            _strict_text(item, name="problem constraint", limit=MAX_SUMMARY_CHARS)
            for item in self.constraints
        )
        if len(constraints) > MAX_CONSTRAINTS or len(set(constraints)) != len(constraints):
            raise EpistemicStateError("problem constraints are duplicate or out of bounds")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(
            self,
            "immutable_evidence_ids",
            _bounded_ids(self.immutable_evidence_ids, name="problem immutable evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "objective_sha256": self.objective_sha256,
            "constraints": list(self.constraints),
            "immutable_evidence_ids": list(self.immutable_evidence_ids),
        }

    @classmethod
    def create(cls, objective: str, *, constraints: Iterable[str] = ()) -> ProblemFrame:
        rendered = _strict_text(objective, name="problem.objective", limit=MAX_OBJECTIVE_CHARS)
        return cls(rendered, text_sha256(rendered), tuple(constraints), ())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProblemFrame:
        fields = {
            "objective",
            "objective_sha256",
            "constraints",
            "immutable_evidence_ids",
        }
        _exact_fields(data, fields, name="problem")
        return cls(
            objective=data["objective"],
            objective_sha256=data["objective_sha256"],
            constraints=_wire_list(data["constraints"], name="problem.constraints"),
            immutable_evidence_ids=_wire_list(
                data["immutable_evidence_ids"],
                name="problem.immutable_evidence_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    source_id: str
    source_version: str
    invocation_sha256: str
    receipt_sha256: str
    verification: EvidenceVerification
    verifier_id: str = ""
    verifier_version: str = ""
    verification_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _strict_text(self.source_id, name="provenance.source_id", limit=512),
        )
        object.__setattr__(
            self,
            "source_version",
            _strict_text(
                self.source_version,
                name="provenance.source_version",
                limit=256,
            ),
        )
        _strict_digest(
            self.invocation_sha256,
            name="provenance.invocation_sha256",
        )
        _strict_digest(self.receipt_sha256, name="provenance.receipt_sha256")
        if not isinstance(self.verification, EvidenceVerification):
            raise EpistemicStateError("provenance.verification must be an EvidenceVerification")
        verifier_id = _strict_text(
            self.verifier_id,
            name="provenance.verifier_id",
            limit=512,
            empty=True,
        )
        verifier_version = _strict_text(
            self.verifier_version,
            name="provenance.verifier_version",
            limit=256,
            empty=True,
        )
        verification_receipt = _strict_digest(
            self.verification_receipt_sha256,
            name="provenance.verification_receipt_sha256",
            empty=True,
        )
        verifier_fields = (verifier_id, verifier_version, verification_receipt)
        if self.verification is EvidenceVerification.INDEPENDENT:
            if not all(verifier_fields):
                raise EpistemicStateError(
                    "independent verification requires verifier identity and receipt"
                )
            if verifier_id == self.source_id:
                raise EpistemicStateError(
                    "independent verifier must differ from the evidence producer"
                )
        elif any(verifier_fields):
            raise EpistemicStateError(
                "non-independent evidence cannot claim an independent verifier"
            )
        object.__setattr__(self, "verifier_id", verifier_id)
        object.__setattr__(self, "verifier_version", verifier_version)
        object.__setattr__(
            self,
            "verification_receipt_sha256",
            verification_receipt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "invocation_sha256": self.invocation_sha256,
            "receipt_sha256": self.receipt_sha256,
            "verification": self.verification.value,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceProvenance:
        fields = {
            "source_id",
            "source_version",
            "invocation_sha256",
            "receipt_sha256",
            "verification",
            "verifier_id",
            "verifier_version",
            "verification_receipt_sha256",
        }
        _exact_fields(data, fields, name="evidence.provenance")
        return cls(
            source_id=data["source_id"],
            source_version=data["source_version"],
            invocation_sha256=data["invocation_sha256"],
            receipt_sha256=data["receipt_sha256"],
            verification=_wire_enum(
                EvidenceVerification,
                data["verification"],
                name="provenance.verification",
            ),
            verifier_id=data["verifier_id"],
            verifier_version=data["verifier_version"],
            verification_receipt_sha256=data["verification_receipt_sha256"],
        )


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    episode_id: str
    objective_sha256: str
    claim_ids: tuple[str, ...]
    purpose: EvidencePurpose

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_id",
            _strict_id(self.episode_id, name="evidence.scope.episode_id"),
        )
        _strict_digest(
            self.objective_sha256,
            name="evidence.scope.objective_sha256",
        )
        object.__setattr__(
            self,
            "claim_ids",
            _bounded_ids(self.claim_ids, name="evidence scope claims"),
        )
        if not isinstance(self.purpose, EvidencePurpose):
            raise EpistemicStateError("evidence.scope.purpose must be an EvidencePurpose")
        if self.purpose is EvidencePurpose.CLAIM_TEST and not self.claim_ids:
            raise EpistemicStateError("claim-test evidence scope requires a claim")
        if self.purpose is not EvidencePurpose.CLAIM_TEST and self.claim_ids:
            raise EpistemicStateError(
                "only claim-test evidence scope may contain claim identifiers"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "objective_sha256": self.objective_sha256,
            "claim_ids": list(self.claim_ids),
            "purpose": self.purpose.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceScope:
        fields = {"episode_id", "objective_sha256", "claim_ids", "purpose"}
        _exact_fields(data, fields, name="evidence.scope")
        return cls(
            episode_id=data["episode_id"],
            objective_sha256=data["objective_sha256"],
            claim_ids=_wire_list(data["claim_ids"], name="evidence.scope.claim_ids"),
            purpose=_wire_enum(
                EvidencePurpose,
                data["purpose"],
                name="evidence.scope.purpose",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
    summary: str
    content_sha256: str
    provenance: EvidenceProvenance
    scope: EvidenceScope
    observed_at: float
    expires_at: float | None = None
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _strict_id(self.evidence_id, name="evidence_id"))
        if not isinstance(self.kind, EvidenceKind):
            raise EpistemicStateError("evidence.kind must be an EvidenceKind")
        object.__setattr__(
            self,
            "summary",
            _strict_text(self.summary, name="evidence.summary", limit=MAX_SUMMARY_CHARS),
        )
        _strict_digest(self.content_sha256, name="evidence.content_sha256")
        if not isinstance(self.provenance, EvidenceProvenance):
            raise EpistemicStateError("evidence.provenance must be an EvidenceProvenance")
        if not isinstance(self.scope, EvidenceScope):
            raise EpistemicStateError("evidence.scope must be an EvidenceScope")
        observed = _nonnegative(self.observed_at, name="evidence.observed_at")
        object.__setattr__(self, "observed_at", observed)
        if self.expires_at is not None:
            expires = _nonnegative(self.expires_at, name="evidence.expires_at")
            if expires < observed:
                raise EpistemicStateError("evidence expires before it was observed")
            object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "supports", _bounded_ids(self.supports, name="evidence supports"))
        object.__setattr__(
            self, "contradicts", _bounded_ids(self.contradicts, name="evidence contradicts")
        )
        if set(self.supports) & set(self.contradicts):
            raise EpistemicStateError("evidence cannot both support and contradict one claim")
        linked_claims = set(self.supports) | set(self.contradicts)
        if linked_claims != set(self.scope.claim_ids):
            raise EpistemicStateError("evidence claim links must exactly match its declared scope")
        if self.kind is EvidenceKind.IMMUTABLE_PROBLEM:
            if self.scope.purpose is not EvidencePurpose.IMMUTABLE_PROBLEM:
                raise EpistemicStateError(
                    "immutable problem evidence requires immutable-problem scope"
                )
        elif self.scope.purpose is EvidencePurpose.IMMUTABLE_PROBLEM:
            raise EpistemicStateError("immutable-problem scope requires immutable problem evidence")
        if self.kind is EvidenceKind.MEMORY:
            if self.scope.purpose is not EvidencePurpose.CONTEXT_ONLY:
                raise EpistemicStateError(
                    "memory evidence is context-only until independently reverified"
                )
        if linked_claims and self.provenance.verification is EvidenceVerification.UNVERIFIED:
            raise EpistemicStateError("unverified evidence cannot support or contradict a claim")

    def is_fresh(self, at_time: float) -> bool:
        """Return whether this evidence existed and remained valid at a time."""

        checked = _nonnegative(at_time, name="evidence freshness time")
        return checked >= self.observed_at and (
            self.expires_at is None or checked <= self.expires_at
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "summary": self.summary,
            "content_sha256": self.content_sha256,
            "provenance": self.provenance.to_dict(),
            "scope": self.scope.to_dict(),
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        fields = {
            "evidence_id",
            "kind",
            "summary",
            "content_sha256",
            "provenance",
            "scope",
            "observed_at",
            "expires_at",
            "supports",
            "contradicts",
        }
        _exact_fields(data, fields, name="evidence")
        return cls(
            evidence_id=data["evidence_id"],
            kind=_wire_enum(EvidenceKind, data["kind"], name="evidence.kind"),
            summary=data["summary"],
            content_sha256=data["content_sha256"],
            provenance=EvidenceProvenance.from_dict(data["provenance"]),
            scope=EvidenceScope.from_dict(data["scope"]),
            observed_at=data["observed_at"],
            expires_at=data["expires_at"],
            supports=_wire_list(data["supports"], name="evidence.supports"),
            contradicts=_wire_list(data["contradicts"], name="evidence.contradicts"),
        )


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    claim_id: str
    text: str
    status: ClaimStatus
    uncertainty: ProbabilityInterval
    premises: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    failure_condition: str = ""
    answer_relevant: bool = False
    domain: str = "general"

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _strict_id(self.claim_id, name="claim_id"))
        object.__setattr__(
            self, "text", _strict_text(self.text, name="claim.text", limit=MAX_TEXT_CHARS)
        )
        if not isinstance(self.status, ClaimStatus):
            raise EpistemicStateError("claim.status must be a ClaimStatus")
        if not isinstance(self.uncertainty, ProbabilityInterval):
            raise EpistemicStateError("claim.uncertainty must be a ProbabilityInterval")
        object.__setattr__(self, "premises", _bounded_ids(self.premises, name="claim premises"))
        object.__setattr__(
            self, "evidence_ids", _bounded_ids(self.evidence_ids, name="claim evidence")
        )
        object.__setattr__(
            self, "contradictions", _bounded_ids(self.contradictions, name="claim contradictions")
        )
        object.__setattr__(
            self,
            "failure_condition",
            _strict_text(
                self.failure_condition,
                name="claim.failure_condition",
                limit=MAX_SUMMARY_CHARS,
                empty=True,
            ),
        )
        if not isinstance(self.answer_relevant, bool):
            raise EpistemicStateError("claim.answer_relevant must be boolean")
        object.__setattr__(
            self,
            "domain",
            _strict_id(self.domain, name="claim.domain"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "status": self.status.value,
            "uncertainty": self.uncertainty.to_dict(),
            "premises": list(self.premises),
            "evidence_ids": list(self.evidence_ids),
            "contradictions": list(self.contradictions),
            "failure_condition": self.failure_condition,
            "answer_relevant": self.answer_relevant,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClaimRecord:
        fields = {
            "claim_id",
            "text",
            "status",
            "uncertainty",
            "premises",
            "evidence_ids",
            "contradictions",
            "failure_condition",
            "answer_relevant",
            "domain",
        }
        _exact_fields(data, fields, name="claim")
        return cls(
            claim_id=data["claim_id"],
            text=data["text"],
            status=_wire_enum(ClaimStatus, data["status"], name="claim.status"),
            uncertainty=ProbabilityInterval.from_dict(data["uncertainty"]),
            premises=_wire_list(data["premises"], name="claim.premises"),
            evidence_ids=_wire_list(data["evidence_ids"], name="claim.evidence_ids"),
            contradictions=_wire_list(
                data["contradictions"],
                name="claim.contradictions",
            ),
            failure_condition=data["failure_condition"],
            answer_relevant=data["answer_relevant"],
            domain=data["domain"],
        )


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    hypothesis_id: str
    statement: str
    posterior: ProbabilityInterval
    status: HypothesisStatus
    claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hypothesis_id", _strict_id(self.hypothesis_id, name="hypothesis_id")
        )
        object.__setattr__(
            self,
            "statement",
            _strict_text(self.statement, name="hypothesis.statement", limit=MAX_TEXT_CHARS),
        )
        if not isinstance(self.posterior, ProbabilityInterval):
            raise EpistemicStateError("hypothesis.posterior must be a ProbabilityInterval")
        if not isinstance(self.status, HypothesisStatus):
            raise EpistemicStateError("hypothesis.status must be a HypothesisStatus")
        object.__setattr__(
            self, "claim_ids", _bounded_ids(self.claim_ids, name="hypothesis claims")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "posterior": self.posterior.to_dict(),
            "status": self.status.value,
            "claim_ids": list(self.claim_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HypothesisRecord:
        fields = {"hypothesis_id", "statement", "posterior", "status", "claim_ids"}
        _exact_fields(data, fields, name="hypothesis")
        return cls(
            hypothesis_id=data["hypothesis_id"],
            statement=data["statement"],
            posterior=ProbabilityInterval.from_dict(data["posterior"]),
            status=_wire_enum(
                HypothesisStatus,
                data["status"],
                name="hypothesis.status",
            ),
            claim_ids=_wire_list(data["claim_ids"], name="hypothesis.claim_ids"),
        )


@dataclass(frozen=True, slots=True)
class OperationAdmission:
    allowed: bool
    reason: str
    attempt_sha256: str
    prior_attempt_count: int
    retry_of_operation_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise EpistemicStateError("operation admission allowed must be boolean")
        object.__setattr__(self, "reason", _strict_id(self.reason, name="admission.reason"))
        _strict_digest(self.attempt_sha256, name="admission.attempt_sha256")
        if (
            not isinstance(self.prior_attempt_count, int)
            or isinstance(self.prior_attempt_count, bool)
            or not 0 <= self.prior_attempt_count <= MAX_OPERATION_ATTEMPTS
        ):
            raise EpistemicStateError("operation admission attempt count is out of bounds")
        if self.retry_of_operation_id:
            object.__setattr__(
                self,
                "retry_of_operation_id",
                _strict_id(
                    self.retry_of_operation_id,
                    name="admission.retry_of_operation_id",
                ),
            )


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    kind: OperationKind
    outcome: OperationOutcome
    input_state_sha256: str
    cost: float
    operator_id: str
    operator_version: str
    input_payload_sha256: str
    attempt_sha256: str
    started_at: float
    completed_at: float
    input_claim_ids: tuple[str, ...] = ()
    input_hypothesis_ids: tuple[str, ...] = ()
    input_evidence_ids: tuple[str, ...] = ()
    affected_claim_ids: tuple[str, ...] = ()
    affected_hypothesis_ids: tuple[str, ...] = ()
    evidence_gained: tuple[str, ...] = ()
    retry_of_operation_id: str = ""
    failure_code: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _strict_id(self.operation_id, name="operation_id"))
        if not isinstance(self.kind, OperationKind) or not isinstance(
            self.outcome, OperationOutcome
        ):
            raise EpistemicStateError("operation kind/outcome use invalid enums")
        _strict_digest(self.input_state_sha256, name="operation.input_state_sha256")
        object.__setattr__(self, "cost", _nonnegative(self.cost, name="operation.cost"))
        object.__setattr__(
            self,
            "operator_id",
            _strict_id(self.operator_id, name="operation.operator_id"),
        )
        object.__setattr__(
            self,
            "operator_version",
            _strict_id(self.operator_version, name="operation.operator_version"),
        )
        _strict_digest(self.input_payload_sha256, name="operation.input_payload_sha256")
        object.__setattr__(
            self,
            "input_claim_ids",
            _bounded_ids(
                self.input_claim_ids,
                name="operation input claims",
                limit=MAX_CLAIMS,
            ),
        )
        object.__setattr__(
            self,
            "input_hypothesis_ids",
            _bounded_ids(
                self.input_hypothesis_ids,
                name="operation input hypotheses",
                limit=MAX_HYPOTHESES,
            ),
        )
        object.__setattr__(
            self,
            "input_evidence_ids",
            _bounded_ids(self.input_evidence_ids, name="operation input evidence"),
        )
        object.__setattr__(
            self,
            "affected_claim_ids",
            _bounded_ids(
                self.affected_claim_ids,
                name="operation affected claims",
                limit=MAX_CLAIMS,
            ),
        )
        object.__setattr__(
            self,
            "affected_hypothesis_ids",
            _bounded_ids(
                self.affected_hypothesis_ids,
                name="operation affected hypotheses",
                limit=MAX_HYPOTHESES,
            ),
        )
        object.__setattr__(
            self, "evidence_gained", _bounded_ids(self.evidence_gained, name="operation evidence")
        )
        started = _nonnegative(self.started_at, name="operation.started_at")
        completed = _nonnegative(self.completed_at, name="operation.completed_at")
        if completed < started:
            raise EpistemicStateError("operation completed before it started")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "completed_at", completed)
        expected_attempt = self.compute_attempt_sha256(
            kind=self.kind,
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            input_payload_sha256=self.input_payload_sha256,
            input_claim_ids=self.input_claim_ids,
            input_hypothesis_ids=self.input_hypothesis_ids,
            input_evidence_ids=self.input_evidence_ids,
        )
        if _strict_digest(self.attempt_sha256, name="operation.attempt_sha256") != expected_attempt:
            raise EpistemicStateError("operation attempt digest does not match canonical inputs")
        if self.retry_of_operation_id:
            object.__setattr__(
                self,
                "retry_of_operation_id",
                _strict_id(
                    self.retry_of_operation_id,
                    name="operation.retry_of_operation_id",
                ),
            )
            if self.retry_of_operation_id == self.operation_id:
                raise EpistemicStateError("operation cannot retry itself")
        failure_code = _strict_text(
            self.failure_code,
            name="operation.failure_code",
            limit=96,
            empty=True,
        )
        if failure_code and not _ID_RE.fullmatch(failure_code):
            raise EpistemicStateError("operation.failure_code is not a bounded identifier")
        object.__setattr__(self, "failure_code", failure_code)
        if self.outcome is OperationOutcome.SUCCEEDED and failure_code:
            raise EpistemicStateError("successful operation cannot carry a failure code")
        if self.outcome is not OperationOutcome.SUCCEEDED and not failure_code:
            raise EpistemicStateError("unsuccessful operation requires a failure code")
        object.__setattr__(
            self,
            "detail",
            _strict_text(self.detail, name="operation.detail", limit=MAX_SUMMARY_CHARS, empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "outcome": self.outcome.value,
            "input_state_sha256": self.input_state_sha256,
            "cost": self.cost,
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "input_payload_sha256": self.input_payload_sha256,
            "attempt_sha256": self.attempt_sha256,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_claim_ids": list(self.input_claim_ids),
            "input_hypothesis_ids": list(self.input_hypothesis_ids),
            "input_evidence_ids": list(self.input_evidence_ids),
            "affected_claim_ids": list(self.affected_claim_ids),
            "affected_hypothesis_ids": list(self.affected_hypothesis_ids),
            "evidence_gained": list(self.evidence_gained),
            "retry_of_operation_id": self.retry_of_operation_id,
            "failure_code": self.failure_code,
            "detail": self.detail,
        }

    @staticmethod
    def compute_attempt_sha256(
        *,
        kind: OperationKind,
        operator_id: str,
        operator_version: str,
        input_payload_sha256: str,
        input_claim_ids: Iterable[str] = (),
        input_hypothesis_ids: Iterable[str] = (),
        input_evidence_ids: Iterable[str] = (),
    ) -> str:
        if not isinstance(kind, OperationKind):
            raise EpistemicStateError("operation attempt kind must be an OperationKind")
        normalized_operator_id = _strict_id(operator_id, name="operation.operator_id")
        normalized_operator_version = _strict_id(
            operator_version,
            name="operation.operator_version",
        )
        normalized_payload = _strict_digest(
            input_payload_sha256,
            name="operation.input_payload_sha256",
        )
        return canonical_sha256(
            {
                "kind": kind.value,
                "operator_id": normalized_operator_id,
                "operator_version": normalized_operator_version,
                "input_payload_sha256": normalized_payload,
                "input_claim_ids": list(
                    _bounded_ids(
                        input_claim_ids,
                        name="operation input claims",
                        limit=MAX_CLAIMS,
                    )
                ),
                "input_hypothesis_ids": list(
                    _bounded_ids(
                        input_hypothesis_ids,
                        name="operation input hypotheses",
                        limit=MAX_HYPOTHESES,
                    )
                ),
                "input_evidence_ids": list(
                    _bounded_ids(input_evidence_ids, name="operation input evidence")
                ),
            }
        )

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        kind: OperationKind,
        outcome: OperationOutcome,
        input_state_sha256: str,
        cost: float,
        operator_id: str,
        operator_version: str,
        input_payload_sha256: str,
        started_at: float,
        completed_at: float,
        input_claim_ids: Iterable[str] = (),
        input_hypothesis_ids: Iterable[str] = (),
        input_evidence_ids: Iterable[str] = (),
        affected_claim_ids: Iterable[str] = (),
        affected_hypothesis_ids: Iterable[str] = (),
        evidence_gained: Iterable[str] = (),
        retry_of_operation_id: str = "",
        failure_code: str = "",
        detail: str = "",
    ) -> OperationRecord:
        input_claim_ids_tuple = tuple(input_claim_ids)
        input_hypothesis_ids_tuple = tuple(input_hypothesis_ids)
        input_evidence_ids_tuple = tuple(input_evidence_ids)
        return cls(
            operation_id=operation_id,
            kind=kind,
            outcome=outcome,
            input_state_sha256=input_state_sha256,
            cost=cost,
            operator_id=operator_id,
            operator_version=operator_version,
            input_payload_sha256=input_payload_sha256,
            attempt_sha256=cls.compute_attempt_sha256(
                kind=kind,
                operator_id=operator_id,
                operator_version=operator_version,
                input_payload_sha256=input_payload_sha256,
                input_claim_ids=input_claim_ids_tuple,
                input_hypothesis_ids=input_hypothesis_ids_tuple,
                input_evidence_ids=input_evidence_ids_tuple,
            ),
            started_at=started_at,
            completed_at=completed_at,
            input_claim_ids=input_claim_ids_tuple,
            input_hypothesis_ids=input_hypothesis_ids_tuple,
            input_evidence_ids=input_evidence_ids_tuple,
            affected_claim_ids=tuple(affected_claim_ids),
            affected_hypothesis_ids=tuple(affected_hypothesis_ids),
            evidence_gained=tuple(evidence_gained),
            retry_of_operation_id=retry_of_operation_id,
            failure_code=failure_code,
            detail=detail,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperationRecord:
        fields = {
            "operation_id",
            "kind",
            "outcome",
            "input_state_sha256",
            "cost",
            "operator_id",
            "operator_version",
            "input_payload_sha256",
            "attempt_sha256",
            "started_at",
            "completed_at",
            "input_claim_ids",
            "input_hypothesis_ids",
            "input_evidence_ids",
            "affected_claim_ids",
            "affected_hypothesis_ids",
            "evidence_gained",
            "retry_of_operation_id",
            "failure_code",
            "detail",
        }
        _exact_fields(data, fields, name="operation")
        return cls(
            operation_id=data["operation_id"],
            kind=_wire_enum(OperationKind, data["kind"], name="operation.kind"),
            outcome=_wire_enum(
                OperationOutcome,
                data["outcome"],
                name="operation.outcome",
            ),
            input_state_sha256=data["input_state_sha256"],
            cost=data["cost"],
            operator_id=data["operator_id"],
            operator_version=data["operator_version"],
            input_payload_sha256=data["input_payload_sha256"],
            attempt_sha256=data["attempt_sha256"],
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            input_claim_ids=_wire_list(
                data["input_claim_ids"],
                name="operation.input_claim_ids",
            ),
            input_hypothesis_ids=_wire_list(
                data["input_hypothesis_ids"],
                name="operation.input_hypothesis_ids",
            ),
            input_evidence_ids=_wire_list(
                data["input_evidence_ids"],
                name="operation.input_evidence_ids",
            ),
            affected_claim_ids=_wire_list(
                data["affected_claim_ids"],
                name="operation.affected_claim_ids",
            ),
            affected_hypothesis_ids=_wire_list(
                data["affected_hypothesis_ids"],
                name="operation.affected_hypothesis_ids",
            ),
            evidence_gained=_wire_list(
                data["evidence_gained"],
                name="operation.evidence_gained",
            ),
            retry_of_operation_id=data["retry_of_operation_id"],
            failure_code=data["failure_code"],
            detail=data["detail"],
        )


@dataclass(frozen=True, slots=True)
class ComputeBudgetState:
    total: float
    used: float = 0.0
    tool_calls_total: int = 0
    tool_calls_used: int = 0

    def __post_init__(self) -> None:
        total = _nonnegative(self.total, name="budget.total")
        used = _nonnegative(self.used, name="budget.used")
        if used > total:
            raise EpistemicStateError("budget.used exceeds budget.total")
        for name, value in (
            ("tool_calls_total", self.tool_calls_total),
            ("tool_calls_used", self.tool_calls_used),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EpistemicStateError(f"budget.{name} must be a nonnegative integer")
        if self.tool_calls_used > self.tool_calls_total:
            raise EpistemicStateError("used tool calls exceed the tool-call budget")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "used", used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "used": self.used,
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_used": self.tool_calls_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComputeBudgetState:
        fields = {"total", "used", "tool_calls_total", "tool_calls_used"}
        _exact_fields(data, fields, name="budget")
        return cls(**{key: data[key] for key in fields})


@dataclass(frozen=True, slots=True)
class AcceptedAnswer:
    text: str
    text_sha256: str
    claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: ProbabilityInterval
    accepted_at: float

    def __post_init__(self) -> None:
        text = _strict_text(self.text, name="answer.text", limit=MAX_TEXT_CHARS)
        if _strict_digest(self.text_sha256, name="answer.text_sha256") != text_sha256(text):
            raise EpistemicStateError("answer text digest does not match")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "claim_ids", _bounded_ids(self.claim_ids, name="answer claims"))
        object.__setattr__(
            self, "evidence_ids", _bounded_ids(self.evidence_ids, name="answer evidence")
        )
        if not isinstance(self.confidence, ProbabilityInterval):
            raise EpistemicStateError("answer.confidence must be a ProbabilityInterval")
        object.__setattr__(
            self,
            "accepted_at",
            _nonnegative(self.accepted_at, name="answer.accepted_at"),
        )
        if not self.claim_ids:
            raise EpistemicStateError("answer must cite at least one claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.text_sha256,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence.to_dict(),
            "accepted_at": self.accepted_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AcceptedAnswer:
        fields = {
            "text",
            "text_sha256",
            "claim_ids",
            "evidence_ids",
            "confidence",
            "accepted_at",
        }
        _exact_fields(data, fields, name="accepted_answer")
        return cls(
            text=data["text"],
            text_sha256=data["text_sha256"],
            claim_ids=_wire_list(data["claim_ids"], name="answer.claim_ids"),
            evidence_ids=_wire_list(data["evidence_ids"], name="answer.evidence_ids"),
            confidence=ProbabilityInterval.from_dict(data["confidence"]),
            accepted_at=data["accepted_at"],
        )


def _unique_by_id(
    values: Iterable[Any],
    *,
    expected_type: type[Any],
    attr: str,
    limit: int,
    name: str,
) -> tuple[Any, ...]:
    items = tuple(values)
    if len(items) > limit:
        raise EpistemicStateError(f"{name} exceeds {limit} records")
    if any(not isinstance(item, expected_type) for item in items):
        raise EpistemicStateError(f"{name} contains an invalid record type")
    identifiers = [getattr(item, attr) for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise EpistemicStateError(f"{name} contains duplicate identifiers")
    return tuple(sorted(items, key=lambda item: getattr(item, attr)))


@dataclass(frozen=True, slots=True)
class EpistemicState:
    schema: str
    episode_id: str
    version: int
    parent_sha256: str
    problem: ProblemFrame
    calibrations: tuple[CalibrationProfile, ...]
    evidence: tuple[EvidenceRecord, ...]
    hypotheses: tuple[HypothesisRecord, ...]
    claims: tuple[ClaimRecord, ...]
    operations: tuple[OperationRecord, ...]
    budget: ComputeBudgetState
    accepted_answer: AcceptedAnswer | None
    state_sha256: str

    def __post_init__(self) -> None:
        if self.schema != EPISTEMIC_STATE_SCHEMA:
            raise EpistemicStateError("unsupported epistemic-state schema")
        object.__setattr__(self, "episode_id", _strict_id(self.episode_id, name="episode_id"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise EpistemicStateError("state.version must be a nonnegative integer")
        _strict_digest(self.parent_sha256, name="state.parent_sha256", empty=self.version == 0)
        if self.version == 0 and self.parent_sha256:
            raise EpistemicStateError("genesis state cannot have a parent")
        if self.version > 0 and not self.parent_sha256:
            raise EpistemicStateError("non-genesis state requires a parent")
        if not isinstance(self.problem, ProblemFrame) or not isinstance(
            self.budget, ComputeBudgetState
        ):
            raise EpistemicStateError("state problem/budget types are invalid")
        object.__setattr__(
            self,
            "calibrations",
            _unique_by_id(
                self.calibrations,
                expected_type=CalibrationProfile,
                attr="profile_id",
                limit=MAX_CALIBRATIONS,
                name="calibrations",
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            _unique_by_id(
                self.evidence,
                expected_type=EvidenceRecord,
                attr="evidence_id",
                limit=MAX_EVIDENCE,
                name="evidence",
            ),
        )
        object.__setattr__(
            self,
            "hypotheses",
            _unique_by_id(
                self.hypotheses,
                expected_type=HypothesisRecord,
                attr="hypothesis_id",
                limit=MAX_HYPOTHESES,
                name="hypotheses",
            ),
        )
        object.__setattr__(
            self,
            "claims",
            _unique_by_id(
                self.claims,
                expected_type=ClaimRecord,
                attr="claim_id",
                limit=MAX_CLAIMS,
                name="claims",
            ),
        )
        object.__setattr__(
            self,
            "operations",
            _unique_by_id(
                self.operations,
                expected_type=OperationRecord,
                attr="operation_id",
                limit=MAX_OPERATIONS,
                name="operations",
            ),
        )
        if self.accepted_answer is not None and not isinstance(
            self.accepted_answer, AcceptedAnswer
        ):
            raise EpistemicStateError("state accepted_answer type is invalid")
        self._validate_references()
        expected = canonical_sha256(self.to_dict(include_hash=False))
        if _strict_digest(self.state_sha256, name="state.state_sha256") != expected:
            raise EpistemicStateError("state hash does not match canonical content")

    def _validate_references(self) -> None:
        claim_map = {item.claim_id: item for item in self.claims}
        evidence_map = {item.evidence_id: item for item in self.evidence}
        calibration_map = {item.profile_id: item for item in self.calibrations}
        evidence_ids = set(evidence_map)
        claim_ids = set(claim_map)
        immutable_ids = set(self.problem.immutable_evidence_ids)
        if not immutable_ids <= evidence_ids:
            raise EpistemicStateError("problem references missing immutable evidence")
        for evidence in self.evidence:
            refs = set(evidence.supports) | set(evidence.contradicts)
            if not refs <= claim_ids:
                raise EpistemicStateError("evidence references an unknown claim")
            if evidence.scope.episode_id != self.episode_id:
                raise EpistemicStateError("evidence scope belongs to another episode")
            if evidence.scope.objective_sha256 != self.problem.objective_sha256:
                raise EpistemicStateError("evidence scope belongs to another objective")
            if (evidence.evidence_id in immutable_ids) != (
                evidence.kind is EvidenceKind.IMMUTABLE_PROBLEM
            ):
                raise EpistemicStateError(
                    "problem evidence membership and immutable kind must match"
                )
        blocked = {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}
        established = {ClaimStatus.SUPPORTED, ClaimStatus.VERIFIED}
        for claim in self.claims:
            if claim.claim_id in claim.premises or claim.claim_id in claim.contradictions:
                raise EpistemicStateError("claim cannot depend on or contradict itself")
            if not set(claim.premises) <= claim_ids:
                raise EpistemicStateError("claim references an unknown premise")
            if not set(claim.evidence_ids) <= evidence_ids:
                raise EpistemicStateError("claim references unknown evidence")
            for evidence_id in claim.evidence_ids:
                linked = evidence_map[evidence_id]
                if claim.claim_id not in set(linked.supports) | set(linked.contradicts):
                    raise EpistemicStateError("claim and evidence links must be bidirectional")
            if not set(claim.contradictions) <= claim_ids:
                raise EpistemicStateError("claim references an unknown contradiction")
            for other in claim.contradictions:
                if claim.claim_id not in claim_map[other].contradictions:
                    raise EpistemicStateError("claim contradictions must be symmetric")
            if claim.status in established and any(
                claim_map[premise].status not in established for premise in claim.premises
            ):
                raise EpistemicStateError(
                    "supported or verified claim depends on an unestablished premise"
                )
            if claim.status in established and any(
                claim_map[other].status in established for other in claim.contradictions
            ):
                raise EpistemicStateError(
                    "mutually contradictory claims cannot both be established"
                )
            self._validate_claim_uncertainty(
                claim,
                evidence_map=evidence_map,
                calibration_map=calibration_map,
            )
            if claim.status in established and (
                claim.uncertainty.basis is UncertaintyBasis.UNCALIBRATED
                or claim.uncertainty.abstain
            ):
                raise EpistemicStateError(
                    "supported or verified claim lacks validated uncertainty support"
                )
        for evidence in self.evidence:
            for claim_id in (*evidence.supports, *evidence.contradicts):
                if evidence.evidence_id not in claim_map[claim_id].evidence_ids:
                    raise EpistemicStateError("evidence and claim links must be bidirectional")
        self._validate_claim_dag(claim_map)
        for hypothesis in self.hypotheses:
            if not set(hypothesis.claim_ids) <= claim_ids:
                raise EpistemicStateError("hypothesis references an unknown claim")
            if hypothesis.status is HypothesisStatus.FAVORED and any(
                claim_map[claim_id].status not in established for claim_id in hypothesis.claim_ids
            ):
                raise EpistemicStateError("favored hypothesis depends on an unestablished claim")
        self._validate_hypothesis_portfolio(self.hypotheses, claim_map=claim_map)
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        for operation in self.operations:
            if not set(operation.input_claim_ids) <= claim_ids:
                raise EpistemicStateError("operation input references an unknown claim")
            if not set(operation.input_hypothesis_ids) <= hypothesis_ids:
                raise EpistemicStateError("operation input references an unknown hypothesis")
            if not set(operation.input_evidence_ids) <= evidence_ids:
                raise EpistemicStateError("operation input references unknown evidence")
            if not set(operation.affected_claim_ids) <= claim_ids:
                raise EpistemicStateError("operation references an unknown claim")
            if not set(operation.affected_hypothesis_ids) <= hypothesis_ids:
                raise EpistemicStateError("operation references an unknown hypothesis")
            if not set(operation.evidence_gained) <= evidence_ids:
                raise EpistemicStateError("operation references unknown evidence")
        self._validate_operation_history(self.operations)
        operation_cost = math.fsum(operation.cost for operation in self.operations)
        if not math.isclose(
            operation_cost,
            self.budget.used,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EpistemicStateError("compute budget usage does not equal recorded operation cost")
        if self.accepted_answer is not None:
            if not set(self.accepted_answer.claim_ids) <= claim_ids:
                raise EpistemicStateError("answer references an unknown claim")
            if not set(self.accepted_answer.evidence_ids) <= evidence_ids:
                raise EpistemicStateError("answer references unknown evidence")
            if any(claim_map[cid].status in blocked for cid in self.accepted_answer.claim_ids):
                raise EpistemicStateError("answer depends on rejected or contradicted claims")
            if any(
                claim_map[claim_id].status not in established
                for claim_id in self.accepted_answer.claim_ids
            ):
                raise EpistemicStateError("answer depends on unresolved claims")
            if any(
                not claim_map[claim_id].answer_relevant
                for claim_id in self.accepted_answer.claim_ids
            ):
                raise EpistemicStateError("answer depends on a claim not marked answer-relevant")
            dependency_claims = self._claim_ancestors(
                claim_map,
                self.accepted_answer.claim_ids,
            )
            required_evidence = {
                evidence_id
                for claim_id in dependency_claims
                for evidence_id in claim_map[claim_id].evidence_ids
            }
            if set(self.accepted_answer.evidence_ids) != required_evidence:
                raise EpistemicStateError(
                    "answer evidence must exactly cover transitive claim dependencies"
                )
            for evidence_id in required_evidence:
                evidence = evidence_map[evidence_id]
                if not evidence.is_fresh(self.accepted_answer.accepted_at):
                    raise EpistemicStateError(
                        "answer depends on stale or not-yet-observed evidence"
                    )
                if set(evidence.contradicts) & dependency_claims:
                    raise EpistemicStateError(
                        "answer depends on a claim with unresolved contradictory evidence"
                    )
            uncertainty_claims = [claim_map[claim_id] for claim_id in dependency_claims]
            weakest = min(
                uncertainty_claims,
                key=lambda claim: (
                    claim.uncertainty.lower,
                    claim.uncertainty.point,
                    claim.uncertainty.upper,
                    claim.claim_id,
                ),
            ).uncertainty
            if self.accepted_answer.confidence != weakest:
                raise EpistemicStateError(
                    "answer confidence must equal its weakest calibrated dependency"
                )
            for claim in uncertainty_claims:
                uncertainty = claim.uncertainty
                if uncertainty.evaluated_at is None or (
                    uncertainty.evaluated_at > self.accepted_answer.accepted_at
                ):
                    raise EpistemicStateError("answer predates its claim uncertainty measurement")
                if uncertainty.basis is UncertaintyBasis.EMPIRICAL:
                    profile = calibration_map[uncertainty.calibration_id]
                    if self.accepted_answer.accepted_at > profile.expires_at:
                        raise EpistemicStateError(
                            "answer depends on an expired calibration profile"
                        )

    @staticmethod
    def _validate_hypothesis_portfolio(
        hypotheses: Iterable[HypothesisRecord],
        *,
        claim_map: Mapping[str, ClaimRecord],
    ) -> None:
        portfolio = tuple(hypotheses)
        if not portfolio:
            return

        blocked = {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}
        known_claim_ids = set(claim_map)
        live = []
        favored = []
        for hypothesis in portfolio:
            if not set(hypothesis.claim_ids) <= known_claim_ids:
                raise EpistemicStateError("hypothesis references an unknown claim")
            if hypothesis.status is HypothesisStatus.REFUTED:
                if (
                    hypothesis.posterior.lower != 0.0
                    or hypothesis.posterior.point != 0.0
                    or hypothesis.posterior.upper != 0.0
                ):
                    raise EpistemicStateError("refuted hypothesis must have zero posterior mass")
                if not any(
                    claim_map[claim_id].status in blocked for claim_id in hypothesis.claim_ids
                ):
                    raise EpistemicStateError(
                        "refuted hypothesis requires a rejected or contradicted claim"
                    )
                continue
            live.append(hypothesis)
            if hypothesis.status is HypothesisStatus.FAVORED:
                favored.append(hypothesis)
            if hypothesis.status is HypothesisStatus.MINORITY and (
                hypothesis.posterior.point <= 0.0 or hypothesis.posterior.point > MAX_MINORITY_MASS
            ):
                raise EpistemicStateError(
                    "minority hypothesis posterior must be positive and at most "
                    f"{MAX_MINORITY_MASS}"
                )

        if not live:
            return
        total = math.fsum(hypothesis.posterior.point for hypothesis in live)
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=PORTFOLIO_SUM_TOLERANCE,
        ):
            raise EpistemicStateError(
                f"non-refuted hypothesis posterior mass must sum to one, got {total:.12g}"
            )
        if len(live) > 1 and any(
            hypothesis.posterior.point < MIN_HYPOTHESIS_MASS for hypothesis in live
        ):
            raise EpistemicStateError(
                "live hypothesis posterior fell below the protected minority floor"
            )
        if len(favored) > 1:
            raise EpistemicStateError("hypothesis portfolio cannot contain multiple favorites")
        if favored:
            winner = favored[0]
            if winner.posterior.point < FAVORED_HYPOTHESIS_MASS:
                raise EpistemicStateError(
                    "favored hypothesis posterior is below the favored threshold"
                )
            if any(
                other.hypothesis_id != winner.hypothesis_id
                and other.posterior.point >= winner.posterior.point
                for other in live
            ):
                raise EpistemicStateError(
                    "favored hypothesis must be the unique highest-mass hypothesis"
                )

    @staticmethod
    def _validate_operation_history(operations: Iterable[OperationRecord]) -> None:
        history = tuple(operations)
        if not history:
            return
        operation_map = {operation.operation_id: operation for operation in history}
        children: dict[str, str] = {}
        groups: dict[str, list[OperationRecord]] = {}
        for operation in history:
            groups.setdefault(operation.attempt_sha256, []).append(operation)
            if not operation.retry_of_operation_id:
                continue
            parent = operation_map.get(operation.retry_of_operation_id)
            if parent is None:
                raise EpistemicStateError("operation retry references an unknown attempt")
            if parent.attempt_sha256 != operation.attempt_sha256:
                raise EpistemicStateError("operation retry changes canonical inputs")
            if parent.outcome is OperationOutcome.SUCCEEDED:
                raise EpistemicStateError("successful operation cannot be retried")
            if operation.started_at < parent.completed_at:
                raise EpistemicStateError("operation retry started before its parent completed")
            if parent.operation_id in children:
                raise EpistemicStateError("operation retry history forks from one attempt")
            children[parent.operation_id] = operation.operation_id

        for attempts in groups.values():
            roots = [attempt for attempt in attempts if not attempt.retry_of_operation_id]
            if len(roots) != 1:
                raise EpistemicStateError("repeated operation requires one explicit retry lineage")
            visited: set[str] = set()
            current = roots[0]
            while current is not None:
                if current.operation_id in visited:
                    raise EpistemicStateError("operation retry history contains a cycle")
                visited.add(current.operation_id)
                child_id = children.get(current.operation_id)
                current = operation_map[child_id] if child_id is not None else None
            if len(visited) != len(attempts):
                raise EpistemicStateError("repeated operation requires one explicit retry lineage")
            if len(visited) > MAX_OPERATION_ATTEMPTS:
                raise EpistemicStateError("operation retry budget is exhausted")

    def operation_attempts(self, attempt_sha256: str) -> tuple[OperationRecord, ...]:
        """Return one canonical retry lineage in execution order."""

        attempt_sha256 = _strict_digest(
            attempt_sha256,
            name="operation attempt query",
        )
        matching = {
            operation.operation_id: operation
            for operation in self.operations
            if operation.attempt_sha256 == attempt_sha256
        }
        if not matching:
            return ()
        roots = [
            operation for operation in matching.values() if not operation.retry_of_operation_id
        ]
        if len(roots) != 1:
            raise EpistemicStateError("operation history has no unique retry root")
        child_by_parent = {
            operation.retry_of_operation_id: operation
            for operation in matching.values()
            if operation.retry_of_operation_id
        }
        ordered = []
        current = roots[0]
        while current is not None:
            ordered.append(current)
            current = child_by_parent.get(current.operation_id)
        if len(ordered) != len(matching):
            raise EpistemicStateError("operation history is not one retry lineage")
        return tuple(ordered)

    def operation_admission(
        self,
        attempt_sha256: str,
        *,
        retry_of_operation_id: str = "",
    ) -> OperationAdmission:
        """Decide whether a new or explicitly linked retry may be recorded."""

        attempt_sha256 = _strict_digest(
            attempt_sha256,
            name="operation admission attempt",
        )
        if retry_of_operation_id:
            retry_of_operation_id = _strict_id(
                retry_of_operation_id,
                name="operation admission retry parent",
            )
        attempts = self.operation_attempts(attempt_sha256)
        if not attempts:
            return OperationAdmission(
                allowed=not retry_of_operation_id,
                reason=("new_operation" if not retry_of_operation_id else "retry_parent_unknown"),
                attempt_sha256=attempt_sha256,
                prior_attempt_count=0,
            )
        latest = attempts[-1]
        if latest.outcome is OperationOutcome.SUCCEEDED:
            return OperationAdmission(
                allowed=False,
                reason="operation_already_succeeded",
                attempt_sha256=attempt_sha256,
                prior_attempt_count=len(attempts),
                retry_of_operation_id=latest.operation_id,
            )
        if len(attempts) >= MAX_OPERATION_ATTEMPTS:
            return OperationAdmission(
                allowed=False,
                reason="operation_retry_budget_exhausted",
                attempt_sha256=attempt_sha256,
                prior_attempt_count=len(attempts),
                retry_of_operation_id=latest.operation_id,
            )
        if not retry_of_operation_id:
            return OperationAdmission(
                allowed=False,
                reason="explicit_retry_link_required",
                attempt_sha256=attempt_sha256,
                prior_attempt_count=len(attempts),
                retry_of_operation_id=latest.operation_id,
            )
        if retry_of_operation_id != latest.operation_id:
            return OperationAdmission(
                allowed=False,
                reason="stale_retry_parent",
                attempt_sha256=attempt_sha256,
                prior_attempt_count=len(attempts),
                retry_of_operation_id=latest.operation_id,
            )
        return OperationAdmission(
            allowed=True,
            reason="explicit_retry_admitted",
            attempt_sha256=attempt_sha256,
            prior_attempt_count=len(attempts),
            retry_of_operation_id=latest.operation_id,
        )

    @staticmethod
    def _validate_claim_uncertainty(
        claim: ClaimRecord,
        *,
        evidence_map: Mapping[str, EvidenceRecord],
        calibration_map: Mapping[str, CalibrationProfile],
    ) -> None:
        uncertainty = claim.uncertainty
        if uncertainty.basis is UncertaintyBasis.UNCALIBRATED:
            return
        signal_ids = set(uncertainty.signal_evidence_ids)
        if not signal_ids <= set(claim.evidence_ids):
            raise EpistemicStateError("claim uncertainty references evidence outside the claim")
        for evidence_id in signal_ids:
            evidence = evidence_map[evidence_id]
            if claim.claim_id not in evidence.supports:
                raise EpistemicStateError("claim uncertainty signal does not support the claim")
            if uncertainty.evaluated_at is None or not evidence.is_fresh(uncertainty.evaluated_at):
                raise EpistemicStateError("claim uncertainty uses stale or future signal evidence")
        if uncertainty.basis is UncertaintyBasis.EXACT:
            for evidence_id in signal_ids:
                evidence = evidence_map[evidence_id]
                if evidence.kind not in {
                    EvidenceKind.CALCULATION,
                    EvidenceKind.PROOF,
                } or (evidence.provenance.verification is not EvidenceVerification.INDEPENDENT):
                    raise EpistemicStateError(
                        "exact uncertainty requires independently verified proof or calculation evidence"
                    )
            return
        profile = calibration_map.get(uncertainty.calibration_id)
        if profile is None:
            raise EpistemicStateError("claim uncertainty references an unknown calibration profile")
        if profile.profile_sha256 != uncertainty.calibration_sha256:
            raise EpistemicStateError("claim calibration profile digest mismatch")
        if profile.domain != claim.domain:
            raise EpistemicStateError("claim calibration profile domain mismatch")
        expected = ProbabilityInterval.from_calibration_estimate(
            profile.estimate(
                uncertainty.raw_probability,
                evaluated_at=uncertainty.evaluated_at,
            ),
            signal_evidence_ids=uncertainty.signal_evidence_ids,
        )
        if uncertainty != expected:
            raise EpistemicStateError(
                "claim uncertainty does not match its measured calibration profile"
            )

    @staticmethod
    def _validate_claim_dag(claim_map: Mapping[str, ClaimRecord]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(claim_id: str) -> None:
            if claim_id in visiting:
                raise EpistemicStateError("claim dependency graph contains a cycle")
            if claim_id in visited:
                return
            visiting.add(claim_id)
            for premise in claim_map[claim_id].premises:
                visit(premise)
            visiting.remove(claim_id)
            visited.add(claim_id)

        for claim_id in sorted(claim_map):
            visit(claim_id)

    @staticmethod
    def _claim_descendants(claim_map: Mapping[str, ClaimRecord], claim_id: str) -> tuple[str, ...]:
        if claim_id not in claim_map:
            raise EpistemicStateError(f"claim does not exist: {claim_id}")
        reverse: dict[str, list[str]] = {key: [] for key in claim_map}
        for candidate in claim_map.values():
            for premise in candidate.premises:
                if premise not in reverse:
                    raise EpistemicStateError("claim references an unknown premise")
                reverse[premise].append(candidate.claim_id)
        EpistemicState._validate_claim_dag(claim_map)
        descendants: set[str] = set()
        pending = list(sorted(reverse[claim_id], reverse=True))
        while pending:
            current = pending.pop()
            if current in descendants:
                continue
            descendants.add(current)
            pending.extend(sorted(reverse[current], reverse=True))
        return tuple(sorted(descendants))

    @staticmethod
    def _claim_ancestors(
        claim_map: Mapping[str, ClaimRecord],
        claim_ids: Iterable[str],
    ) -> set[str]:
        ancestors: set[str] = set()
        pending = list(claim_ids)
        while pending:
            current = pending.pop()
            if current in ancestors:
                continue
            if current not in claim_map:
                raise EpistemicStateError("answer references an unknown claim")
            ancestors.add(current)
            pending.extend(claim_map[current].premises)
        return ancestors

    def claim_descendants(self, claim_id: str) -> tuple[str, ...]:
        """Return every transitive dependent of a claim in stable order."""

        return self._claim_descendants(
            {item.claim_id: item for item in self.claims},
            _strict_id(claim_id, name="claim_id"),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "version": self.version,
            "parent_sha256": self.parent_sha256,
            "problem": self.problem.to_dict(),
            "calibrations": [item.to_dict() for item in self.calibrations],
            "evidence": [item.to_dict() for item in self.evidence],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "claims": [item.to_dict() for item in self.claims],
            "operations": [item.to_dict() for item in self.operations],
            "budget": self.budget.to_dict(),
            "accepted_answer": self.accepted_answer.to_dict() if self.accepted_answer else None,
        }
        if include_hash:
            result["state_sha256"] = self.state_sha256
        return result

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EpistemicState:
        fields = {
            "schema",
            "episode_id",
            "version",
            "parent_sha256",
            "problem",
            "calibrations",
            "evidence",
            "hypotheses",
            "claims",
            "operations",
            "budget",
            "accepted_answer",
            "state_sha256",
        }
        _exact_fields(data, fields, name="epistemic_state")
        answer = data["accepted_answer"]
        if answer is not None and not isinstance(answer, Mapping):
            raise EpistemicStateError("accepted_answer must be an object or null")
        return cls(
            schema=data["schema"],
            episode_id=data["episode_id"],
            version=data["version"],
            parent_sha256=data["parent_sha256"],
            problem=ProblemFrame.from_dict(data["problem"]),
            calibrations=tuple(
                CalibrationProfile.from_dict(item)
                for item in _wire_list(
                    data["calibrations"],
                    name="state.calibrations",
                )
            ),
            evidence=tuple(
                EvidenceRecord.from_dict(item)
                for item in _wire_list(data["evidence"], name="state.evidence")
            ),
            hypotheses=tuple(
                HypothesisRecord.from_dict(item)
                for item in _wire_list(data["hypotheses"], name="state.hypotheses")
            ),
            claims=tuple(
                ClaimRecord.from_dict(item)
                for item in _wire_list(data["claims"], name="state.claims")
            ),
            operations=tuple(
                OperationRecord.from_dict(item)
                for item in _wire_list(data["operations"], name="state.operations")
            ),
            budget=ComputeBudgetState.from_dict(data["budget"]),
            accepted_answer=AcceptedAnswer.from_dict(answer) if answer is not None else None,
            state_sha256=data["state_sha256"],
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> EpistemicState:
        if not isinstance(payload, str):
            raise EpistemicStateError("canonical state payload must be a string")
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise EpistemicStateError(f"invalid canonical state JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise EpistemicStateError("canonical state payload must contain an object")
        state = cls.from_dict(decoded)
        if state.to_canonical_json() != payload:
            raise EpistemicStateError("state JSON is valid but not canonical")
        return state

    @classmethod
    def genesis(
        cls,
        *,
        episode_id: str,
        problem: ProblemFrame,
        budget: ComputeBudgetState,
        calibrations: Iterable[CalibrationProfile] = (),
        evidence: Iterable[EvidenceRecord] = (),
    ) -> EpistemicState:
        items = tuple(evidence)
        problem_with_evidence = replace(
            problem,
            immutable_evidence_ids=tuple(
                item.evidence_id for item in items if item.kind is EvidenceKind.IMMUTABLE_PROBLEM
            ),
        )
        return cls._build(
            episode_id=episode_id,
            version=0,
            parent_sha256="",
            problem=problem_with_evidence,
            calibrations=tuple(calibrations),
            evidence=items,
            hypotheses=(),
            claims=(),
            operations=(),
            budget=budget,
            accepted_answer=None,
        )

    @classmethod
    def _build(cls, **values: Any) -> EpistemicState:
        normalized = {
            **values,
            "evidence": _unique_by_id(
                values["evidence"],
                expected_type=EvidenceRecord,
                attr="evidence_id",
                limit=MAX_EVIDENCE,
                name="evidence",
            ),
            "calibrations": _unique_by_id(
                values["calibrations"],
                expected_type=CalibrationProfile,
                attr="profile_id",
                limit=MAX_CALIBRATIONS,
                name="calibrations",
            ),
            "hypotheses": _unique_by_id(
                values["hypotheses"],
                expected_type=HypothesisRecord,
                attr="hypothesis_id",
                limit=MAX_HYPOTHESES,
                name="hypotheses",
            ),
            "claims": _unique_by_id(
                values["claims"],
                expected_type=ClaimRecord,
                attr="claim_id",
                limit=MAX_CLAIMS,
                name="claims",
            ),
            "operations": _unique_by_id(
                values["operations"],
                expected_type=OperationRecord,
                attr="operation_id",
                limit=MAX_OPERATIONS,
                name="operations",
            ),
        }
        payload = {
            "schema": EPISTEMIC_STATE_SCHEMA,
            **normalized,
        }
        provisional = cls.__new__(cls)
        for key, value in payload.items():
            object.__setattr__(provisional, key, value)
        object.__setattr__(provisional, "state_sha256", "0" * 64)
        digest = canonical_sha256(provisional.to_dict(include_hash=False))
        return cls(state_sha256=digest, **payload)


class EpistemicTransaction:
    """Copy-on-write transaction over one immutable epistemic state."""

    def __init__(self, base: EpistemicState) -> None:
        if not isinstance(base, EpistemicState):
            raise TypeError("base must be an EpistemicState")
        self.base = base
        self._calibrations = {item.profile_id: item for item in base.calibrations}
        self._evidence = {item.evidence_id: item for item in base.evidence}
        self._hypotheses = {item.hypothesis_id: item for item in base.hypotheses}
        self._claims = {item.claim_id: item for item in base.claims}
        self._operations = {item.operation_id: item for item in base.operations}
        self._budget = base.budget
        self._accepted_answer = base.accepted_answer
        self._replaced_claim_ids: set[str] = set()
        self._replaced_hypothesis_ids: set[str] = set()
        self._closed = False

    def _open(self) -> None:
        if self._closed:
            raise EpistemicStateError("epistemic transaction is already closed")

    @staticmethod
    def _insert(target: dict[str, Any], key: str, value: Any, *, name: str) -> None:
        if key in target:
            raise EpistemicStateError(f"{name} identifier already exists: {key}")
        target[key] = value

    def _validated_operation_budget(
        self,
        operation: OperationRecord,
        *,
        claim_ids: Iterable[str] | None = None,
        hypothesis_ids: Iterable[str] | None = None,
        evidence_ids: Iterable[str] | None = None,
        budget_error: str = "operation exceeds compute budget",
    ) -> float:
        if operation.input_state_sha256 != self.base.state_sha256:
            raise EpistemicStateError("operation input hash does not match transaction base")
        if operation.operation_id in self._operations:
            raise EpistemicStateError(
                f"operation identifier already exists: {operation.operation_id}"
            )
        known_claims = set(self._claims if claim_ids is None else claim_ids)
        known_hypotheses = set(self._hypotheses if hypothesis_ids is None else hypothesis_ids)
        known_evidence = set(self._evidence if evidence_ids is None else evidence_ids)
        if not set((*operation.input_claim_ids, *operation.affected_claim_ids)) <= known_claims:
            raise EpistemicStateError("operation references an unknown claim")
        if (
            not set((*operation.input_hypothesis_ids, *operation.affected_hypothesis_ids))
            <= known_hypotheses
        ):
            raise EpistemicStateError("operation references an unknown hypothesis")
        if not set((*operation.input_evidence_ids, *operation.evidence_gained)) <= known_evidence:
            raise EpistemicStateError("operation references unknown evidence")
        EpistemicState._validate_operation_history((*self._operations.values(), operation))
        next_used = self._budget.used + operation.cost
        if next_used > self._budget.total:
            raise EpistemicStateError(budget_error)
        return next_used

    def add_evidence(self, evidence: EvidenceRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("evidence must be an EvidenceRecord")
        self._insert(self._evidence, evidence.evidence_id, evidence, name="evidence")
        return self

    def add_calibration(
        self,
        calibration: CalibrationProfile,
    ) -> EpistemicTransaction:
        self._open()
        if not isinstance(calibration, CalibrationProfile):
            raise TypeError("calibration must be a CalibrationProfile")
        self._insert(
            self._calibrations,
            calibration.profile_id,
            calibration,
            name="calibration",
        )
        return self

    def add_claim(self, claim: ClaimRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(claim, ClaimRecord):
            raise TypeError("claim must be a ClaimRecord")
        self._insert(self._claims, claim.claim_id, claim, name="claim")
        return self

    def replace_claim(self, claim: ClaimRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(claim, ClaimRecord):
            raise TypeError("claim must be a ClaimRecord")
        if claim.claim_id not in self._claims:
            raise EpistemicStateError(f"claim does not exist: {claim.claim_id}")
        self._claims[claim.claim_id] = claim
        self._replaced_claim_ids.add(claim.claim_id)
        return self

    def invalidate_claim(
        self,
        claim_id: str,
        *,
        operation_id: str,
        started_at: float,
        completed_at: float,
        status: ClaimStatus = ClaimStatus.REJECTED,
        cost: float = 0.0,
        operator_id: str = "epistemic_transaction",
        operator_version: str = "v1",
        detail: str = "",
    ) -> tuple[str, ...]:
        """Invalidate a claim, its descendants, hypotheses, and accepted answer."""

        self._open()
        claim_id = _strict_id(claim_id, name="claim_id")
        if not isinstance(status, ClaimStatus):
            raise EpistemicStateError("claim invalidation status must be a ClaimStatus")
        if status not in {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}:
            raise EpistemicStateError("claim invalidation status must be rejected or contradicted")
        target = self._claims.get(claim_id)
        if target is None:
            raise EpistemicStateError(f"claim does not exist: {claim_id}")
        if target.status in {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}:
            raise EpistemicStateError("claim is already invalidated")

        descendants = EpistemicState._claim_descendants(self._claims, claim_id)
        affected = tuple(sorted((claim_id, *descendants)))
        revised_claims = dict(self._claims)
        revised_claims[claim_id] = replace(target, status=status)
        blocked = {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}
        for descendant_id in descendants:
            descendant = revised_claims[descendant_id]
            if descendant.status not in blocked:
                revised_claims[descendant_id] = replace(
                    descendant,
                    status=ClaimStatus.UNRESOLVED,
                )
        revised_hypotheses = dict(self._hypotheses)
        affected_set = set(affected)
        affected_hypothesis_ids = []
        for hypothesis_id, hypothesis in revised_hypotheses.items():
            if affected_set.intersection(hypothesis.claim_ids) and hypothesis.status not in {
                HypothesisStatus.REFUTED,
                HypothesisStatus.UNRESOLVED,
            }:
                revised_hypotheses[hypothesis_id] = replace(
                    hypothesis,
                    status=HypothesisStatus.UNRESOLVED,
                )
                affected_hypothesis_ids.append(hypothesis_id)
        affected_hypothesis_ids_tuple = tuple(sorted(affected_hypothesis_ids))
        operation = OperationRecord.create(
            operation_id=operation_id,
            kind=(
                OperationKind.FALSIFY
                if status is ClaimStatus.CONTRADICTED
                else OperationKind.BACKTRACK
            ),
            outcome=OperationOutcome.SUCCEEDED,
            input_state_sha256=self.base.state_sha256,
            cost=cost,
            operator_id=operator_id,
            operator_version=operator_version,
            input_payload_sha256=canonical_sha256(
                {
                    "claim_id": claim_id,
                    "status": status.value,
                }
            ),
            started_at=started_at,
            completed_at=completed_at,
            input_claim_ids=(claim_id,),
            affected_claim_ids=affected,
            affected_hypothesis_ids=affected_hypothesis_ids_tuple,
            detail=detail or f"invalidated {claim_id} and its dependent claims",
        )
        next_used = self._validated_operation_budget(
            operation,
            hypothesis_ids=revised_hypotheses,
            budget_error="claim invalidation exceeds compute budget",
        )
        answer = self._accepted_answer
        if answer is not None and affected_set.intersection(answer.claim_ids):
            answer = None

        self._claims = revised_claims
        self._hypotheses = revised_hypotheses
        self._operations[operation.operation_id] = operation
        self._budget = replace(self._budget, used=next_used)
        self._accepted_answer = answer
        self._replaced_claim_ids.update(affected)
        self._replaced_hypothesis_ids.update(affected_hypothesis_ids_tuple)
        return affected

    def add_hypothesis(self, hypothesis: HypothesisRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(hypothesis, HypothesisRecord):
            raise TypeError("hypothesis must be a HypothesisRecord")
        if self.base.hypotheses:
            raise EpistemicStateError(
                "existing hypothesis portfolio must be changed through complete revision"
            )
        self._insert(self._hypotheses, hypothesis.hypothesis_id, hypothesis, name="hypothesis")
        return self

    def revise_hypothesis_portfolio(
        self,
        replacements: Iterable[HypothesisRecord],
        *,
        operation_id: str,
        started_at: float,
        completed_at: float,
        input_evidence_ids: Iterable[str] = (),
        evidence_gained: Iterable[str] = (),
        cost: float = 0.0,
        operator_id: str = "epistemic_transaction",
        operator_version: str = "v1",
        detail: str = "",
    ) -> tuple[str, ...]:
        """Atomically revise the complete weighted hypothesis portfolio."""

        self._open()
        proposed = _unique_by_id(
            replacements,
            expected_type=HypothesisRecord,
            attr="hypothesis_id",
            limit=MAX_HYPOTHESES,
            name="hypotheses",
        )
        proposed_map = {item.hypothesis_id: item for item in proposed}
        missing = sorted(set(self._hypotheses) - set(proposed_map))
        if missing:
            raise EpistemicStateError(f"portfolio revision cannot delete hypotheses: {missing}")
        for hypothesis_id, previous in self._hypotheses.items():
            replacement = proposed_map[hypothesis_id]
            if (
                replacement.statement != previous.statement
                or replacement.claim_ids != previous.claim_ids
            ):
                raise EpistemicStateError(
                    "portfolio revision cannot rewrite hypothesis identity or claim scope"
                )
            if (
                previous.status is HypothesisStatus.REFUTED
                and replacement.status is not HypothesisStatus.REFUTED
                and any(
                    self._claims[claim_id].status
                    in {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}
                    for claim_id in replacement.claim_ids
                )
            ):
                raise EpistemicStateError(
                    "portfolio revision cannot revive a hypothesis while its refuting claim is blocked"
                )

        changed = tuple(
            sorted(
                hypothesis_id
                for hypothesis_id, hypothesis in proposed_map.items()
                if self._hypotheses.get(hypothesis_id) != hypothesis
            )
        )
        if not changed:
            raise EpistemicStateError("portfolio revision did not change any hypothesis")
        EpistemicState._validate_hypothesis_portfolio(
            proposed,
            claim_map=self._claims,
        )
        input_evidence_ids_tuple = tuple(input_evidence_ids)
        operation = OperationRecord.create(
            operation_id=operation_id,
            kind=OperationKind.COMPARE,
            outcome=OperationOutcome.SUCCEEDED,
            input_state_sha256=self.base.state_sha256,
            cost=cost,
            operator_id=operator_id,
            operator_version=operator_version,
            input_payload_sha256=canonical_sha256(
                {
                    "portfolio": [item.to_dict() for item in proposed],
                }
            ),
            started_at=started_at,
            completed_at=completed_at,
            input_hypothesis_ids=tuple(self._hypotheses),
            input_evidence_ids=input_evidence_ids_tuple,
            affected_hypothesis_ids=changed,
            evidence_gained=tuple(evidence_gained),
            detail=detail or "revised weighted hypothesis portfolio",
        )
        next_used = self._validated_operation_budget(
            operation,
            hypothesis_ids=proposed_map,
            budget_error="portfolio revision exceeds compute budget",
        )

        self._hypotheses = proposed_map
        self._operations[operation.operation_id] = operation
        self._budget = replace(self._budget, used=next_used)
        self._replaced_hypothesis_ids.update(changed)
        return changed

    def add_operation(self, operation: OperationRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(operation, OperationRecord):
            raise TypeError("operation must be an OperationRecord")
        next_used = self._validated_operation_budget(operation)
        self._operations[operation.operation_id] = operation
        self._budget = replace(self._budget, used=next_used)
        return self

    def set_budget(self, budget: ComputeBudgetState) -> EpistemicTransaction:
        self._open()
        if not isinstance(budget, ComputeBudgetState):
            raise TypeError("budget must be a ComputeBudgetState")
        if (
            budget.total != self.base.budget.total
            or budget.tool_calls_total != self.base.budget.tool_calls_total
        ):
            raise EpistemicStateError("transaction cannot change preregistered budget caps")
        if budget.used != self._budget.used:
            raise EpistemicStateError(
                "transaction cannot change compute usage outside operation history"
            )
        if budget.tool_calls_used < self.base.budget.tool_calls_used:
            raise EpistemicStateError("transaction cannot refund consumed budget")
        self._budget = budget
        return self

    def accept_answer(self, answer: AcceptedAnswer | None) -> EpistemicTransaction:
        self._open()
        if answer is not None and not isinstance(answer, AcceptedAnswer):
            raise TypeError("answer must be an AcceptedAnswer or None")
        self._accepted_answer = answer
        return self

    def _prepare(self) -> EpistemicState:
        self._open()
        candidate = EpistemicState._build(
            episode_id=self.base.episode_id,
            version=self.base.version + 1,
            parent_sha256=self.base.state_sha256,
            problem=self.base.problem,
            calibrations=tuple(self._calibrations.values()),
            evidence=tuple(self._evidence.values()),
            hypotheses=tuple(self._hypotheses.values()),
            claims=tuple(self._claims.values()),
            operations=tuple(self._operations.values()),
            budget=self._budget,
            accepted_answer=self._accepted_answer,
        )
        base_operation_ids = {item.operation_id for item in self.base.operations}
        covered_claim_ids = {
            claim_id
            for operation in candidate.operations
            if operation.operation_id not in base_operation_ids
            and operation.outcome is OperationOutcome.SUCCEEDED
            for claim_id in operation.affected_claim_ids
        }
        if not self._replaced_claim_ids <= covered_claim_ids:
            missing = sorted(self._replaced_claim_ids - covered_claim_ids)
            raise EpistemicStateError(
                f"claim revisions lack a successful operation receipt: {missing}"
            )
        covered_hypothesis_ids = {
            hypothesis_id
            for operation in candidate.operations
            if operation.operation_id not in base_operation_ids
            and operation.outcome is OperationOutcome.SUCCEEDED
            for hypothesis_id in operation.affected_hypothesis_ids
        }
        if not self._replaced_hypothesis_ids <= covered_hypothesis_ids:
            missing = sorted(self._replaced_hypothesis_ids - covered_hypothesis_ids)
            raise EpistemicStateError(
                f"hypothesis revisions lack a successful operation receipt: {missing}"
            )
        return candidate

    def commit(self) -> EpistemicState:
        candidate = self._prepare()
        self._closed = True
        return candidate


class EpistemicStateMachine:
    """Atomic in-memory authority for one episode's current state."""

    def __init__(
        self,
        genesis: EpistemicState,
        *,
        persistence: EpistemicStatePersistence | None = None,
    ) -> None:
        if not isinstance(genesis, EpistemicState):
            raise TypeError("genesis must be an EpistemicState")
        if genesis.version != 0:
            raise EpistemicStateError("state machine requires a genesis state")
        self._lock = threading.RLock()
        self._persistence = persistence
        current = persistence.bootstrap(genesis) if persistence is not None else genesis
        if not isinstance(current, EpistemicState):
            raise EpistemicStateError("persistence returned an invalid state")
        if current.episode_id != genesis.episode_id or current.problem != genesis.problem:
            raise EpistemicStateError("recovered state does not match genesis identity")
        if (
            current.budget.total != genesis.budget.total
            or current.budget.tool_calls_total != genesis.budget.tool_calls_total
        ):
            raise EpistemicStateError("recovered state changed preregistered budget caps")
        self._current = current

    def snapshot(self) -> EpistemicState:
        with self._lock:
            return self._current

    def begin(self) -> EpistemicTransaction:
        return EpistemicTransaction(self.snapshot())

    def commit(self, transaction: EpistemicTransaction) -> EpistemicState:
        if not isinstance(transaction, EpistemicTransaction):
            raise TypeError("transaction must be an EpistemicTransaction")
        with self._lock:
            if transaction.base.state_sha256 != self._current.state_sha256:
                raise StaleEpistemicTransactionError("transaction base is not current")
            candidate = transaction._prepare()
            if self._persistence is not None:
                self._persistence.append(
                    expected_base=self._current,
                    candidate=candidate,
                )
            transaction._closed = True
            self._current = candidate
            return candidate

    async def commit_async(self, transaction: EpistemicTransaction) -> EpistemicState:
        """Commit on a worker thread so durable fsync never blocks the event loop."""

        import asyncio

        return await asyncio.to_thread(self.commit, transaction)


__all__ = [
    "AcceptedAnswer",
    "CalibrationProfile",
    "ClaimRecord",
    "ClaimStatus",
    "ComputeBudgetState",
    "EPISTEMIC_STATE_SCHEMA",
    "EpistemicState",
    "EpistemicStateError",
    "EpistemicStateMachine",
    "EpistemicStatePersistence",
    "EpistemicTransaction",
    "EvidenceKind",
    "EvidenceProvenance",
    "EvidencePurpose",
    "EvidenceRecord",
    "EvidenceScope",
    "EvidenceVerification",
    "HypothesisRecord",
    "HypothesisStatus",
    "OperationKind",
    "OperationAdmission",
    "OperationOutcome",
    "OperationRecord",
    "ProbabilityInterval",
    "ProblemFrame",
    "StaleEpistemicTransactionError",
    "UncertaintyBasis",
    "canonical_sha256",
    "text_sha256",
]
