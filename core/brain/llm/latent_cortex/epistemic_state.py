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
from typing import Any

EPISTEMIC_STATE_SCHEMA = "aura.rlc.epistemic_state.v1"

MAX_OBJECTIVE_CHARS = 16_384
MAX_TEXT_CHARS = 8_192
MAX_SUMMARY_CHARS = 2_048
MAX_CONSTRAINTS = 128
MAX_EVIDENCE = 256
MAX_HYPOTHESES = 64
MAX_CLAIMS = 512
MAX_OPERATIONS = 512
MAX_REFS = 128

_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,95}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class EpistemicStateError(ValueError):
    """Base error for invalid or stale state transitions."""


class StaleEpistemicTransactionError(EpistemicStateError):
    """A transaction tried to replace a state other than its base."""


class EvidenceKind(StrEnum):
    IMMUTABLE_PROBLEM = "immutable_problem"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    CALCULATION = "calculation"
    PROOF = "proof"
    SIMULATION = "simulation"
    OBSERVATION = "observation"
    MEMORY = "memory"


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


def _bounded_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise EpistemicStateError(f"{name} must be a sequence of identifiers")
    result = tuple(_strict_id(value, name=f"{name} item") for value in values)
    if len(result) > MAX_REFS:
        raise EpistemicStateError(f"{name} exceeds {MAX_REFS} references")
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


def _wire_enum[EnumT: StrEnum](
    enum_type: type[EnumT], value: Any, *, name: str
) -> EnumT:
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

    def __post_init__(self) -> None:
        lower = _unit(self.lower, name="uncertainty.lower")
        point = _unit(self.point, name="uncertainty.point")
        upper = _unit(self.upper, name="uncertainty.upper")
        if not lower <= point <= upper:
            raise EpistemicStateError("uncertainty interval must satisfy lower <= point <= upper")
        if not isinstance(self.evidence_count, int) or isinstance(self.evidence_count, bool):
            raise EpistemicStateError("uncertainty.evidence_count must be an integer")
        if not 0 <= self.evidence_count <= MAX_EVIDENCE:
            raise EpistemicStateError("uncertainty.evidence_count is out of bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "method", _strict_text(self.method, name="uncertainty.method", limit=96))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "point": self.point,
            "upper": self.upper,
            "method": self.method,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbabilityInterval:
        fields = {"lower", "point", "upper", "method", "evidence_count"}
        _exact_fields(data, fields, name="uncertainty")
        return cls(**{key: data[key] for key in fields})


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
        if _strict_digest(self.objective_sha256, name="problem.objective_sha256") != text_sha256(objective):
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
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
    summary: str
    content_sha256: str
    source: str
    observed_at: float
    expires_at: float | None = None
    receipt_sha256: str = ""
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _strict_id(self.evidence_id, name="evidence_id"))
        if not isinstance(self.kind, EvidenceKind):
            raise EpistemicStateError("evidence.kind must be an EvidenceKind")
        object.__setattr__(self, "summary", _strict_text(self.summary, name="evidence.summary", limit=MAX_SUMMARY_CHARS))
        _strict_digest(self.content_sha256, name="evidence.content_sha256")
        object.__setattr__(self, "source", _strict_text(self.source, name="evidence.source", limit=512))
        observed = _nonnegative(self.observed_at, name="evidence.observed_at")
        object.__setattr__(self, "observed_at", observed)
        if self.expires_at is not None:
            expires = _nonnegative(self.expires_at, name="evidence.expires_at")
            if expires < observed:
                raise EpistemicStateError("evidence expires before it was observed")
            object.__setattr__(self, "expires_at", expires)
        _strict_digest(self.receipt_sha256, name="evidence.receipt_sha256", empty=True)
        object.__setattr__(self, "supports", _bounded_ids(self.supports, name="evidence supports"))
        object.__setattr__(self, "contradicts", _bounded_ids(self.contradicts, name="evidence contradicts"))
        if set(self.supports) & set(self.contradicts):
            raise EpistemicStateError("evidence cannot both support and contradict one claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "summary": self.summary,
            "content_sha256": self.content_sha256,
            "source": self.source,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "receipt_sha256": self.receipt_sha256,
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
            "source",
            "observed_at",
            "expires_at",
            "receipt_sha256",
            "supports",
            "contradicts",
        }
        _exact_fields(data, fields, name="evidence")
        return cls(
            evidence_id=data["evidence_id"],
            kind=_wire_enum(EvidenceKind, data["kind"], name="evidence.kind"),
            summary=data["summary"],
            content_sha256=data["content_sha256"],
            source=data["source"],
            observed_at=data["observed_at"],
            expires_at=data["expires_at"],
            receipt_sha256=data["receipt_sha256"],
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _strict_id(self.claim_id, name="claim_id"))
        object.__setattr__(self, "text", _strict_text(self.text, name="claim.text", limit=MAX_TEXT_CHARS))
        if not isinstance(self.status, ClaimStatus):
            raise EpistemicStateError("claim.status must be a ClaimStatus")
        if not isinstance(self.uncertainty, ProbabilityInterval):
            raise EpistemicStateError("claim.uncertainty must be a ProbabilityInterval")
        object.__setattr__(self, "premises", _bounded_ids(self.premises, name="claim premises"))
        object.__setattr__(self, "evidence_ids", _bounded_ids(self.evidence_ids, name="claim evidence"))
        object.__setattr__(self, "contradictions", _bounded_ids(self.contradictions, name="claim contradictions"))
        object.__setattr__(self, "failure_condition", _strict_text(self.failure_condition, name="claim.failure_condition", limit=MAX_SUMMARY_CHARS, empty=True))
        if not isinstance(self.answer_relevant, bool):
            raise EpistemicStateError("claim.answer_relevant must be boolean")

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
        )


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    hypothesis_id: str
    statement: str
    posterior: ProbabilityInterval
    status: HypothesisStatus
    claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _strict_id(self.hypothesis_id, name="hypothesis_id"))
        object.__setattr__(self, "statement", _strict_text(self.statement, name="hypothesis.statement", limit=MAX_TEXT_CHARS))
        if not isinstance(self.posterior, ProbabilityInterval):
            raise EpistemicStateError("hypothesis.posterior must be a ProbabilityInterval")
        if not isinstance(self.status, HypothesisStatus):
            raise EpistemicStateError("hypothesis.status must be a HypothesisStatus")
        object.__setattr__(self, "claim_ids", _bounded_ids(self.claim_ids, name="hypothesis claims"))

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
class OperationRecord:
    operation_id: str
    kind: OperationKind
    outcome: OperationOutcome
    input_state_sha256: str
    cost: float
    affected_claim_ids: tuple[str, ...] = ()
    evidence_gained: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _strict_id(self.operation_id, name="operation_id"))
        if not isinstance(self.kind, OperationKind) or not isinstance(self.outcome, OperationOutcome):
            raise EpistemicStateError("operation kind/outcome use invalid enums")
        _strict_digest(self.input_state_sha256, name="operation.input_state_sha256")
        object.__setattr__(self, "cost", _nonnegative(self.cost, name="operation.cost"))
        object.__setattr__(self, "affected_claim_ids", _bounded_ids(self.affected_claim_ids, name="operation affected claims"))
        object.__setattr__(self, "evidence_gained", _bounded_ids(self.evidence_gained, name="operation evidence"))
        object.__setattr__(self, "detail", _strict_text(self.detail, name="operation.detail", limit=MAX_SUMMARY_CHARS, empty=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "outcome": self.outcome.value,
            "input_state_sha256": self.input_state_sha256,
            "cost": self.cost,
            "affected_claim_ids": list(self.affected_claim_ids),
            "evidence_gained": list(self.evidence_gained),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperationRecord:
        fields = {
            "operation_id",
            "kind",
            "outcome",
            "input_state_sha256",
            "cost",
            "affected_claim_ids",
            "evidence_gained",
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
            affected_claim_ids=_wire_list(
                data["affected_claim_ids"],
                name="operation.affected_claim_ids",
            ),
            evidence_gained=_wire_list(
                data["evidence_gained"],
                name="operation.evidence_gained",
            ),
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

    def __post_init__(self) -> None:
        text = _strict_text(self.text, name="answer.text", limit=MAX_TEXT_CHARS)
        if _strict_digest(self.text_sha256, name="answer.text_sha256") != text_sha256(text):
            raise EpistemicStateError("answer text digest does not match")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "claim_ids", _bounded_ids(self.claim_ids, name="answer claims"))
        object.__setattr__(self, "evidence_ids", _bounded_ids(self.evidence_ids, name="answer evidence"))
        if not isinstance(self.confidence, ProbabilityInterval):
            raise EpistemicStateError("answer.confidence must be a ProbabilityInterval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.text_sha256,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AcceptedAnswer:
        fields = {"text", "text_sha256", "claim_ids", "evidence_ids", "confidence"}
        _exact_fields(data, fields, name="accepted_answer")
        return cls(
            text=data["text"],
            text_sha256=data["text_sha256"],
            claim_ids=_wire_list(data["claim_ids"], name="answer.claim_ids"),
            evidence_ids=_wire_list(data["evidence_ids"], name="answer.evidence_ids"),
            confidence=ProbabilityInterval.from_dict(data["confidence"]),
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
        if not isinstance(self.problem, ProblemFrame) or not isinstance(self.budget, ComputeBudgetState):
            raise EpistemicStateError("state problem/budget types are invalid")
        object.__setattr__(self, "evidence", _unique_by_id(self.evidence, expected_type=EvidenceRecord, attr="evidence_id", limit=MAX_EVIDENCE, name="evidence"))
        object.__setattr__(self, "hypotheses", _unique_by_id(self.hypotheses, expected_type=HypothesisRecord, attr="hypothesis_id", limit=MAX_HYPOTHESES, name="hypotheses"))
        object.__setattr__(self, "claims", _unique_by_id(self.claims, expected_type=ClaimRecord, attr="claim_id", limit=MAX_CLAIMS, name="claims"))
        object.__setattr__(self, "operations", _unique_by_id(self.operations, expected_type=OperationRecord, attr="operation_id", limit=MAX_OPERATIONS, name="operations"))
        if self.accepted_answer is not None and not isinstance(self.accepted_answer, AcceptedAnswer):
            raise EpistemicStateError("state accepted_answer type is invalid")
        self._validate_references()
        expected = canonical_sha256(self.to_dict(include_hash=False))
        if _strict_digest(self.state_sha256, name="state.state_sha256") != expected:
            raise EpistemicStateError("state hash does not match canonical content")

    def _validate_references(self) -> None:
        claim_map = {item.claim_id: item for item in self.claims}
        evidence_ids = {item.evidence_id for item in self.evidence}
        claim_ids = set(claim_map)
        immutable_ids = set(self.problem.immutable_evidence_ids)
        if not immutable_ids <= evidence_ids:
            raise EpistemicStateError("problem references missing immutable evidence")
        for evidence in self.evidence:
            refs = set(evidence.supports) | set(evidence.contradicts)
            if not refs <= claim_ids:
                raise EpistemicStateError("evidence references an unknown claim")
            if evidence.evidence_id in immutable_ids and evidence.kind is not EvidenceKind.IMMUTABLE_PROBLEM:
                raise EpistemicStateError("problem evidence must use immutable_problem kind")
        for claim in self.claims:
            if claim.claim_id in claim.premises or claim.claim_id in claim.contradictions:
                raise EpistemicStateError("claim cannot depend on or contradict itself")
            if not set(claim.premises) <= claim_ids:
                raise EpistemicStateError("claim references an unknown premise")
            if not set(claim.evidence_ids) <= evidence_ids:
                raise EpistemicStateError("claim references unknown evidence")
            if not set(claim.contradictions) <= claim_ids:
                raise EpistemicStateError("claim references an unknown contradiction")
            for other in claim.contradictions:
                if claim.claim_id not in claim_map[other].contradictions:
                    raise EpistemicStateError("claim contradictions must be symmetric")
        self._validate_claim_dag(claim_map)
        for hypothesis in self.hypotheses:
            if not set(hypothesis.claim_ids) <= claim_ids:
                raise EpistemicStateError("hypothesis references an unknown claim")
        for operation in self.operations:
            if not set(operation.affected_claim_ids) <= claim_ids:
                raise EpistemicStateError("operation references an unknown claim")
            if not set(operation.evidence_gained) <= evidence_ids:
                raise EpistemicStateError("operation references unknown evidence")
        if self.accepted_answer is not None:
            if not set(self.accepted_answer.claim_ids) <= claim_ids:
                raise EpistemicStateError("answer references an unknown claim")
            if not set(self.accepted_answer.evidence_ids) <= evidence_ids:
                raise EpistemicStateError("answer references unknown evidence")
            blocked = {ClaimStatus.REJECTED, ClaimStatus.CONTRADICTED}
            if any(claim_map[cid].status in blocked for cid in self.accepted_answer.claim_ids):
                raise EpistemicStateError("answer depends on rejected or contradicted claims")

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

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "version": self.version,
            "parent_sha256": self.parent_sha256,
            "problem": self.problem.to_dict(),
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
        evidence: Iterable[EvidenceRecord] = (),
    ) -> EpistemicState:
        items = tuple(evidence)
        problem_with_evidence = replace(
            problem,
            immutable_evidence_ids=tuple(
                item.evidence_id
                for item in items
                if item.kind is EvidenceKind.IMMUTABLE_PROBLEM
            ),
        )
        return cls._build(
            episode_id=episode_id,
            version=0,
            parent_sha256="",
            problem=problem_with_evidence,
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
        self._evidence = {item.evidence_id: item for item in base.evidence}
        self._hypotheses = {item.hypothesis_id: item for item in base.hypotheses}
        self._claims = {item.claim_id: item for item in base.claims}
        self._operations = {item.operation_id: item for item in base.operations}
        self._budget = base.budget
        self._accepted_answer = base.accepted_answer
        self._closed = False

    def _open(self) -> None:
        if self._closed:
            raise EpistemicStateError("epistemic transaction is already closed")

    @staticmethod
    def _insert(target: dict[str, Any], key: str, value: Any, *, name: str) -> None:
        if key in target:
            raise EpistemicStateError(f"{name} identifier already exists: {key}")
        target[key] = value

    def add_evidence(self, evidence: EvidenceRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("evidence must be an EvidenceRecord")
        self._insert(self._evidence, evidence.evidence_id, evidence, name="evidence")
        return self

    def add_claim(self, claim: ClaimRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(claim, ClaimRecord):
            raise TypeError("claim must be a ClaimRecord")
        self._insert(self._claims, claim.claim_id, claim, name="claim")
        return self

    def replace_claim(self, claim: ClaimRecord) -> EpistemicTransaction:
        self._open()
        if claim.claim_id not in self._claims:
            raise EpistemicStateError(f"claim does not exist: {claim.claim_id}")
        self._claims[claim.claim_id] = claim
        return self

    def add_hypothesis(self, hypothesis: HypothesisRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(hypothesis, HypothesisRecord):
            raise TypeError("hypothesis must be a HypothesisRecord")
        self._insert(self._hypotheses, hypothesis.hypothesis_id, hypothesis, name="hypothesis")
        return self

    def add_operation(self, operation: OperationRecord) -> EpistemicTransaction:
        self._open()
        if not isinstance(operation, OperationRecord):
            raise TypeError("operation must be an OperationRecord")
        if operation.input_state_sha256 != self.base.state_sha256:
            raise EpistemicStateError("operation input hash does not match transaction base")
        self._insert(self._operations, operation.operation_id, operation, name="operation")
        return self

    def set_budget(self, budget: ComputeBudgetState) -> EpistemicTransaction:
        self._open()
        if not isinstance(budget, ComputeBudgetState):
            raise TypeError("budget must be a ComputeBudgetState")
        if budget.total != self.base.budget.total or budget.tool_calls_total != self.base.budget.tool_calls_total:
            raise EpistemicStateError("transaction cannot change preregistered budget caps")
        if budget.used < self.base.budget.used or budget.tool_calls_used < self.base.budget.tool_calls_used:
            raise EpistemicStateError("transaction cannot refund consumed budget")
        self._budget = budget
        return self

    def accept_answer(self, answer: AcceptedAnswer | None) -> EpistemicTransaction:
        self._open()
        if answer is not None and not isinstance(answer, AcceptedAnswer):
            raise TypeError("answer must be an AcceptedAnswer or None")
        self._accepted_answer = answer
        return self

    def commit(self) -> EpistemicState:
        self._open()
        candidate = EpistemicState._build(
            episode_id=self.base.episode_id,
            version=self.base.version + 1,
            parent_sha256=self.base.state_sha256,
            problem=self.base.problem,
            evidence=tuple(self._evidence.values()),
            hypotheses=tuple(self._hypotheses.values()),
            claims=tuple(self._claims.values()),
            operations=tuple(self._operations.values()),
            budget=self._budget,
            accepted_answer=self._accepted_answer,
        )
        self._closed = True
        return candidate


class EpistemicStateMachine:
    """Atomic in-memory authority for one episode's current state."""

    def __init__(self, genesis: EpistemicState) -> None:
        if genesis.version != 0:
            raise EpistemicStateError("state machine requires a genesis state")
        self._current = genesis
        self._lock = threading.RLock()

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
            candidate = transaction.commit()
            self._current = candidate
            return candidate


__all__ = [
    "AcceptedAnswer",
    "ClaimRecord",
    "ClaimStatus",
    "ComputeBudgetState",
    "EPISTEMIC_STATE_SCHEMA",
    "EpistemicState",
    "EpistemicStateError",
    "EpistemicStateMachine",
    "EpistemicTransaction",
    "EvidenceKind",
    "EvidenceRecord",
    "HypothesisRecord",
    "HypothesisStatus",
    "OperationKind",
    "OperationOutcome",
    "OperationRecord",
    "ProbabilityInterval",
    "ProblemFrame",
    "StaleEpistemicTransactionError",
    "canonical_sha256",
    "text_sha256",
]
