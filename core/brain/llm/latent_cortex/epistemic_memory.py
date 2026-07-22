"""Selective, provenance-bound memory ingress for one RLC episode.

Memory is useful context, not authority.  This module queries Aura's existing
memory organs through bounded adapters, normalizes their heterogeneous records
into one deterministic contract, and can commit the admitted recall into the
episode's :class:`EpistemicState` as ``MEMORY`` / ``CONTEXT_ONLY`` evidence.

The bridge deliberately does not own another database.  Durable knowledge stays
in the source stores; the RLC receives a scoped, content-addressed observation
and a receipt proving where it came from and why it was admitted.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from core.brain.llm.latent_cortex.epistemic_state import (
    EpistemicState,
    EpistemicStateError,
    EpistemicTransaction,
    EvidenceKind,
    EvidenceProvenance,
    EvidencePurpose,
    EvidenceRecord,
    EvidenceScope,
    EvidenceVerification,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    canonical_sha256,
    text_sha256,
)

SELECTIVE_MEMORY_SCHEMA = "aura.rlc.selective_memory.v1"
MAX_MEMORY_CONTENT_CHARS = 400
MAX_MEMORY_RESULTS = 6
MAX_MEMORY_PER_TIER = 4
DEFAULT_SOURCE_TIMEOUT_S = 1.5

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_GENERIC_TERMS = frozenset(
    "the and for with from this that what when where which who why how are was "
    "were have has had can could would should will does did into about your you "
    "our their then than also just more most some any".split()
)


class SelectiveMemoryError(ValueError):
    """A selective-memory contract or state transition is invalid."""


class MemoryTier(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    NONPARAMETRIC = "nonparametric"


class MemorySourceStatus(StrEnum):
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


MemoryAdapter = Callable[[str, int], Iterable[Any] | Any]


def _text(value: Any, *, name: str, limit: int, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SelectiveMemoryError(f"{name} must be a string")
    rendered = value.strip()
    if not rendered and not empty:
        raise SelectiveMemoryError(f"{name} must not be empty")
    if len(rendered) > limit:
        raise SelectiveMemoryError(f"{name} exceeds {limit} characters")
    if _CONTROL_RE.search(rendered):
        raise SelectiveMemoryError(f"{name} contains control characters")
    return rendered


def _bounded_int(value: Any, *, name: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise SelectiveMemoryError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectiveMemoryError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0.0:
        raise SelectiveMemoryError(f"{name} must be finite and nonnegative")
    return rendered


def _unit(value: Any, *, name: str) -> float:
    rendered = _nonnegative(value, name=name)
    if rendered > 1.0:
        raise SelectiveMemoryError(f"{name} must be in [0, 1]")
    return rendered


def _scope_value(value: Any, *, name: str, fallback: str) -> str:
    candidate = str(value or fallback).strip()
    return _text(candidate, name=name, limit=128)


def _timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) and rendered >= 0.0 else None


def _metadata(item: Any) -> dict[str, Any]:
    raw = item.get("metadata") if isinstance(item, Mapping) else getattr(item, "metadata", None)
    metadata = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        "tenant_id",
        "user_id",
        "session_id",
        "timestamp",
        "created_at",
        "expires_at",
        "source",
        "source_version",
        "score",
        "similarity",
        "relevance",
        "confidence",
        "contested",
        "identity_relevant",
    ):
        candidate = item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)
        if key not in metadata and candidate is not None:
            metadata[key] = candidate
    return metadata


def _content(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in (
            "content",
            "text",
            "summary",
            "description",
            "fact",
            "memory",
            "value",
            "lesson",
            "skeleton",
        ):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    for attr in (
        "content",
        "text",
        "summary",
        "description",
        "lesson",
        "skeleton",
    ):
        candidate = getattr(item, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _record_id(item: Any, *, content_sha256: str, rank: int) -> str:
    for key in ("id", "memory_id", "record_id", "episode_id", "procedure_id", "playbook_id"):
        candidate = item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:256]
    return f"anonymous:{rank}:{content_sha256[:24]}"


def _raw_score(
    item: Any, metadata: Mapping[str, Any], rank: int, objective: str, content: str
) -> float:
    for key in ("score", "similarity", "relevance", "confidence"):
        candidate = metadata.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            value = float(candidate)
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
    query_terms = set(_WORD_RE.findall(objective.lower())) - _GENERIC_TERMS
    content_terms = set(_WORD_RE.findall(content.lower())) - _GENERIC_TERMS
    overlap = len(query_terms & content_terms) / max(1, len(query_terms | content_terms))
    return max(0.05, min(1.0, 0.65 * overlap + 0.35 / (rank + 1)))


@dataclass(frozen=True, slots=True)
class MemoryScope:
    tenant_id: str
    user_id: str
    session_id: str
    episode_id: str
    objective_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant_id", _scope_value(self.tenant_id, name="tenant_id", fallback="local")
        )
        object.__setattr__(
            self, "user_id", _scope_value(self.user_id, name="user_id", fallback="owner")
        )
        object.__setattr__(
            self, "session_id", _scope_value(self.session_id, name="session_id", fallback="local")
        )
        object.__setattr__(self, "episode_id", _text(self.episode_id, name="episode_id", limit=96))
        if not isinstance(self.objective_sha256, str) or len(self.objective_sha256) != 64:
            raise SelectiveMemoryError("objective_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "objective_sha256": self.objective_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    query_id: str
    objective: str
    objective_sha256: str
    scope: MemoryScope
    requested_tiers: tuple[MemoryTier, ...]
    per_tier_limit: int
    total_limit: int
    source_timeout_s: float
    issued_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _text(self.query_id, name="query_id", limit=96))
        objective = _text(self.objective, name="objective", limit=16_384)
        object.__setattr__(self, "objective", objective)
        if self.objective_sha256 != text_sha256(objective):
            raise SelectiveMemoryError("objective digest does not match objective")
        if not isinstance(self.scope, MemoryScope):
            raise SelectiveMemoryError("scope must be a MemoryScope")
        if self.scope.objective_sha256 != self.objective_sha256:
            raise SelectiveMemoryError("scope belongs to another objective")
        tiers = tuple(self.requested_tiers)
        if (
            not tiers
            or len(set(tiers)) != len(tiers)
            or any(not isinstance(tier, MemoryTier) for tier in tiers)
        ):
            raise SelectiveMemoryError("requested tiers must be unique MemoryTier values")
        object.__setattr__(self, "requested_tiers", tiers)
        object.__setattr__(
            self,
            "per_tier_limit",
            _bounded_int(
                self.per_tier_limit, name="per_tier_limit", low=1, high=MAX_MEMORY_PER_TIER
            ),
        )
        object.__setattr__(
            self,
            "total_limit",
            _bounded_int(self.total_limit, name="total_limit", low=1, high=MAX_MEMORY_RESULTS),
        )
        timeout = _nonnegative(self.source_timeout_s, name="source_timeout_s")
        if timeout <= 0.0 or timeout > 10.0:
            raise SelectiveMemoryError("source_timeout_s must be in (0, 10]")
        object.__setattr__(self, "source_timeout_s", timeout)
        object.__setattr__(self, "issued_at", _nonnegative(self.issued_at, name="issued_at"))

    @classmethod
    def create(
        cls,
        objective: str,
        *,
        episode_id: str,
        tenant_id: str = "local",
        user_id: str = "owner",
        session_id: str = "local",
        requested_tiers: Iterable[MemoryTier] = tuple(MemoryTier),
        per_tier_limit: int = 4,
        total_limit: int = 6,
        source_timeout_s: float = DEFAULT_SOURCE_TIMEOUT_S,
        issued_at: float | None = None,
    ) -> MemoryQuery:
        objective = str(objective or "").strip()
        objective_sha256 = text_sha256(objective)
        issued = time.time() if issued_at is None else issued_at
        query_id = (
            "mq-"
            + canonical_sha256(
                {
                    "objective_sha256": objective_sha256,
                    "episode_id": episode_id,
                    "issued_at": issued,
                }
            )[:24]
        )
        scope = MemoryScope(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            episode_id=episode_id,
            objective_sha256=objective_sha256,
        )
        return cls(
            query_id=query_id,
            objective=objective,
            objective_sha256=objective_sha256,
            scope=scope,
            requested_tiers=tuple(requested_tiers),
            per_tier_limit=per_tier_limit,
            total_limit=total_limit,
            source_timeout_s=source_timeout_s,
            issued_at=issued,
        )

    def to_dict(self, *, include_objective: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_id": self.query_id,
            "objective_sha256": self.objective_sha256,
            "scope": self.scope.to_dict(),
            "requested_tiers": [tier.value for tier in self.requested_tiers],
            "per_tier_limit": self.per_tier_limit,
            "total_limit": self.total_limit,
            "source_timeout_s": self.source_timeout_s,
            "issued_at": self.issued_at,
        }
        if include_objective:
            payload["objective"] = self.objective
        return payload

    @property
    def invocation_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    candidate_id: str
    tier: MemoryTier
    record_id: str
    content: str
    content_sha256: str
    source_id: str
    source_version: str
    scope: MemoryScope
    created_at: float | None
    retrieved_at: float
    expires_at: float | None
    relevance: float
    source_rank: int
    contested: bool
    identity_relevant: bool
    instruction_authority: bool
    corroborating_tiers: tuple[MemoryTier, ...]
    retrieval_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, name="candidate_id", limit=96)
        )
        if not isinstance(self.tier, MemoryTier):
            raise SelectiveMemoryError("candidate tier must be a MemoryTier")
        object.__setattr__(self, "record_id", _text(self.record_id, name="record_id", limit=256))
        content = _text(self.content, name="memory content", limit=MAX_MEMORY_CONTENT_CHARS)
        object.__setattr__(self, "content", content)
        if self.content_sha256 != text_sha256(content):
            raise SelectiveMemoryError("memory content digest mismatch")
        object.__setattr__(self, "source_id", _text(self.source_id, name="source_id", limit=256))
        object.__setattr__(
            self, "source_version", _text(self.source_version, name="source_version", limit=128)
        )
        if not isinstance(self.scope, MemoryScope):
            raise SelectiveMemoryError("candidate scope must be a MemoryScope")
        if self.created_at is not None:
            object.__setattr__(self, "created_at", _nonnegative(self.created_at, name="created_at"))
        object.__setattr__(
            self, "retrieved_at", _nonnegative(self.retrieved_at, name="retrieved_at")
        )
        if self.expires_at is not None:
            expires_at = _nonnegative(self.expires_at, name="expires_at")
            if self.created_at is not None and expires_at < self.created_at:
                raise SelectiveMemoryError("memory expires before it was created")
            object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "relevance", _unit(self.relevance, name="relevance"))
        object.__setattr__(
            self,
            "source_rank",
            _bounded_int(self.source_rank, name="source_rank", low=0, high=10_000),
        )
        if type(self.contested) is not bool or type(self.identity_relevant) is not bool:
            raise SelectiveMemoryError("memory flags must be boolean")
        if self.instruction_authority is not False:
            raise SelectiveMemoryError("recalled memory can never have instruction authority")
        corroborating = tuple(self.corroborating_tiers)
        if self.tier not in corroborating or len(set(corroborating)) != len(corroborating):
            raise SelectiveMemoryError("corroborating tiers must uniquely include the primary tier")
        object.__setattr__(self, "corroborating_tiers", corroborating)
        if (
            not isinstance(self.retrieval_receipt_sha256, str)
            or len(self.retrieval_receipt_sha256) != 64
        ):
            raise SelectiveMemoryError("retrieval receipt must be a SHA-256 digest")

    @property
    def evidence_id(self) -> str:
        return (
            "memory-"
            + canonical_sha256(
                {
                    "candidate_id": self.candidate_id,
                    "scope_sha256": self.scope.sha256,
                    "content_sha256": self.content_sha256,
                }
            )[:24]
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "tier": self.tier.value,
            "record_id": self.record_id,
            "content_sha256": self.content_sha256,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "scope_sha256": self.scope.sha256,
            "created_at": self.created_at,
            "retrieved_at": self.retrieved_at,
            "expires_at": self.expires_at,
            "relevance": round(self.relevance, 6),
            "source_rank": self.source_rank,
            "contested": self.contested,
            "identity_relevant": self.identity_relevant,
            "instruction_authority": False,
            "corroborating_tiers": [tier.value for tier in self.corroborating_tiers],
            "retrieval_receipt_sha256": self.retrieval_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class MemorySourceReceipt:
    tier: MemoryTier
    source_id: str
    source_version: str
    status: MemorySourceStatus
    retrieved_count: int
    admitted_count: int
    refused_count: int
    latency_ms: float
    error_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.tier, MemoryTier) or not isinstance(self.status, MemorySourceStatus):
            raise SelectiveMemoryError("source receipt uses invalid enums")
        object.__setattr__(
            self, "source_id", _text(self.source_id, name="source receipt id", limit=256)
        )
        object.__setattr__(
            self,
            "source_version",
            _text(self.source_version, name="source receipt version", limit=128),
        )
        for name in ("retrieved_count", "admitted_count", "refused_count"):
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name=name, low=0, high=10_000)
            )
        if self.admitted_count + self.refused_count > self.retrieved_count:
            raise SelectiveMemoryError("source receipt counts are inconsistent")
        object.__setattr__(self, "latency_ms", _nonnegative(self.latency_ms, name="latency_ms"))
        object.__setattr__(
            self, "error_code", _text(self.error_code, name="error_code", limit=96, empty=True)
        )
        failed = self.status in {MemorySourceStatus.FAILED, MemorySourceStatus.TIMED_OUT}
        if failed != bool(self.error_code):
            raise SelectiveMemoryError("failed source receipts and error codes must agree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "status": self.status.value,
            "retrieved_count": self.retrieved_count,
            "admitted_count": self.admitted_count,
            "refused_count": self.refused_count,
            "latency_ms": round(self.latency_ms, 3),
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class SelectiveMemoryResult:
    query: MemoryQuery
    candidates: tuple[MemoryCandidate, ...]
    source_receipts: tuple[MemorySourceReceipt, ...]
    started_at: float
    completed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.query, MemoryQuery):
            raise SelectiveMemoryError("result query must be a MemoryQuery")
        candidates = tuple(self.candidates)
        if len(candidates) > self.query.total_limit:
            raise SelectiveMemoryError("result exceeds query total limit")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise SelectiveMemoryError("result contains duplicate candidate identifiers")
        if len({candidate.content_sha256 for candidate in candidates}) != len(candidates):
            raise SelectiveMemoryError("result contains duplicate memory content")
        if any(candidate.scope != self.query.scope for candidate in candidates):
            raise SelectiveMemoryError("result candidate belongs to another scope")
        object.__setattr__(self, "candidates", candidates)
        receipts = tuple(self.source_receipts)
        if len({receipt.tier for receipt in receipts}) != len(receipts):
            raise SelectiveMemoryError("result contains duplicate tier receipts")
        if set(receipt.tier for receipt in receipts) != set(self.query.requested_tiers):
            raise SelectiveMemoryError("result does not receipt every requested tier")
        object.__setattr__(self, "source_receipts", receipts)
        started = _nonnegative(self.started_at, name="started_at")
        completed = _nonnegative(self.completed_at, name="completed_at")
        if completed < started:
            raise SelectiveMemoryError("retrieval completed before it started")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "completed_at", completed)

    def to_receipt(self) -> dict[str, Any]:
        payload = {
            "schema": SELECTIVE_MEMORY_SCHEMA,
            "query": self.query.to_dict(include_objective=False),
            "candidates": [candidate.to_receipt() for candidate in self.candidates],
            "sources": [receipt.to_dict() for receipt in self.source_receipts],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        payload["result_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def result_sha256(self) -> str:
        return str(self.to_receipt()["result_sha256"])

    def context_items(self, *, state_sha256: str, max_items: int = 2) -> list[dict[str, Any]]:
        if not isinstance(state_sha256, str) or len(state_sha256) != 64:
            raise SelectiveMemoryError("state_sha256 must be a SHA-256 digest")
        limit = _bounded_int(max_items, name="max_items", low=1, high=MAX_MEMORY_RESULTS)
        return [
            {
                # Keep the first slot's stable organ name for existing GWT/UI
                # consumers. Additional slots are uniquely named so workspace
                # telemetry cannot collapse two memories onto one source key.
                "source": (
                    "memory"
                    if index == 0
                    else (
                        f"memory.{candidate.tier.value}."
                        f"{candidate.evidence_id.rsplit('-', 1)[-1][:8]}"
                    )
                ),
                "text": candidate.content,
                "context_role": "memory_observation",
                "instruction_authority": False,
                "evidence_id": candidate.evidence_id,
                "content_sha256": candidate.content_sha256,
                "scope_sha256": candidate.scope.sha256,
                "retrieval_receipt_sha256": candidate.retrieval_receipt_sha256,
                "epistemic_state_sha256": state_sha256,
                "memory_tier": candidate.tier.value,
            }
            for index, candidate in enumerate(self.candidates[:limit])
        ]


@dataclass(frozen=True, slots=True)
class _AdapterSpec:
    tier: MemoryTier
    source_id: str
    source_version: str
    adapter: MemoryAdapter | None


class SelectiveMemoryBridge:
    """Execute one bounded query across a fixed set of typed store adapters."""

    def __init__(
        self, adapters: Mapping[MemoryTier, tuple[str, str, MemoryAdapter | None]]
    ) -> None:
        specs: dict[MemoryTier, _AdapterSpec] = {}
        for tier in MemoryTier:
            raw = adapters.get(tier)
            if raw is None:
                specs[tier] = _AdapterSpec(tier, f"memory.{tier.value}", "unavailable", None)
                continue
            source_id, source_version, adapter = raw
            if adapter is not None and not callable(adapter):
                raise TypeError(f"adapter for {tier.value} must be callable or None")
            specs[tier] = _AdapterSpec(
                tier,
                _text(source_id, name="adapter source_id", limit=256),
                _text(source_version, name="adapter source_version", limit=128),
                adapter,
            )
        self._specs = specs

    def _normalize(
        self,
        query: MemoryQuery,
        spec: _AdapterSpec,
        raw: Any,
    ) -> tuple[list[MemoryCandidate], int, int]:
        if raw is None:
            items: list[Any] = []
        elif isinstance(raw, (str, bytes, Mapping)):
            items = [raw]
        else:
            try:
                items = list(raw)
            except TypeError as exc:
                raise SelectiveMemoryError("memory adapter returned a non-iterable result") from exc
        candidates: list[MemoryCandidate] = []
        refused = 0
        for rank, item in enumerate(items[: query.per_tier_limit * 2]):
            content = _content(item)[:MAX_MEMORY_CONTENT_CHARS].strip()
            if not content or _CONTROL_RE.search(content):
                refused += 1
                continue
            metadata = _metadata(item)
            if bool(metadata.get("contested")):
                refused += 1
                continue
            mismatched = False
            for key in ("tenant_id", "user_id", "session_id"):
                supplied = str(metadata.get(key) or "").strip()
                if supplied and supplied != getattr(query.scope, key):
                    mismatched = True
                    break
            if mismatched:
                refused += 1
                continue
            created_at = _timestamp(metadata.get("created_at") or metadata.get("timestamp"))
            expires_at = _timestamp(metadata.get("expires_at"))
            if expires_at is not None and expires_at < query.issued_at:
                refused += 1
                continue
            content_sha256 = text_sha256(content)
            record_id = _record_id(item, content_sha256=content_sha256, rank=rank)
            source_id = str(metadata.get("source") or spec.source_id).strip()[:256]
            source_version = str(metadata.get("source_version") or spec.source_version).strip()[
                :128
            ]
            relevance = _raw_score(item, metadata, rank, query.objective, content)
            receipt_payload = {
                "schema": SELECTIVE_MEMORY_SCHEMA,
                "query_sha256": query.invocation_sha256,
                "tier": spec.tier.value,
                "source_id": source_id,
                "source_version": source_version,
                "record_id": record_id,
                "content_sha256": content_sha256,
                "scope_sha256": query.scope.sha256,
                "created_at": created_at,
                "expires_at": expires_at,
                "source_rank": rank,
            }
            candidate_id = "mc-" + canonical_sha256(receipt_payload)[:24]
            candidates.append(
                MemoryCandidate(
                    candidate_id=candidate_id,
                    tier=spec.tier,
                    record_id=record_id,
                    content=content,
                    content_sha256=content_sha256,
                    source_id=source_id,
                    source_version=source_version,
                    scope=query.scope,
                    created_at=created_at,
                    retrieved_at=query.issued_at,
                    expires_at=expires_at,
                    relevance=relevance,
                    source_rank=rank,
                    contested=False,
                    identity_relevant=bool(metadata.get("identity_relevant")),
                    instruction_authority=False,
                    corroborating_tiers=(spec.tier,),
                    retrieval_receipt_sha256=canonical_sha256(receipt_payload),
                )
            )
        candidates.sort(key=lambda item: (-item.relevance, item.source_rank, item.content_sha256))
        return candidates[: query.per_tier_limit], len(items), refused

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        name = type(exc).__name__.lower()
        rendered = re.sub(r"[^a-z0-9_.:-]+", "_", name).strip("_")
        return f"memory_source_{rendered or 'failed'}"[:96]

    def _run_sync(
        self, query: MemoryQuery, spec: _AdapterSpec
    ) -> tuple[list[MemoryCandidate], MemorySourceReceipt]:
        if spec.adapter is None:
            return [], MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=MemorySourceStatus.UNAVAILABLE,
                retrieved_count=0,
                admitted_count=0,
                refused_count=0,
                latency_ms=0.0,
            )
        started = time.monotonic()
        try:
            raw = spec.adapter(query.objective, query.per_tier_limit)
            if inspect.isawaitable(raw):
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
                else:
                    cancel = getattr(raw, "cancel", None)
                    if callable(cancel):
                        cancel()
                raise SelectiveMemoryError("synchronous adapter returned an awaitable")
            candidates, retrieved, refused = self._normalize(query, spec, raw)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if elapsed_ms > query.source_timeout_s * 1000.0:
                return [], MemorySourceReceipt(
                    tier=spec.tier,
                    source_id=spec.source_id,
                    source_version=spec.source_version,
                    status=MemorySourceStatus.TIMED_OUT,
                    retrieved_count=retrieved,
                    admitted_count=0,
                    refused_count=retrieved,
                    latency_ms=elapsed_ms,
                    error_code="memory_source_timeout",
                )
            status = MemorySourceStatus.SUCCEEDED if candidates else MemorySourceStatus.EMPTY
            return candidates, MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=status,
                retrieved_count=retrieved,
                admitted_count=len(candidates),
                refused_count=refused,
                latency_ms=elapsed_ms,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return [], MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=MemorySourceStatus.FAILED,
                retrieved_count=0,
                admitted_count=0,
                refused_count=0,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error_code=self._error_code(exc),
            )

    async def _run_async(
        self, query: MemoryQuery, spec: _AdapterSpec
    ) -> tuple[list[MemoryCandidate], MemorySourceReceipt]:
        if spec.adapter is None:
            return self._run_sync(query, spec)
        started = time.monotonic()
        try:
            if inspect.iscoroutinefunction(spec.adapter):
                raw = await asyncio.wait_for(
                    spec.adapter(query.objective, query.per_tier_limit),
                    timeout=query.source_timeout_s,
                )
            else:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(spec.adapter, query.objective, query.per_tier_limit),
                    timeout=query.source_timeout_s,
                )
                if inspect.isawaitable(raw):
                    raw = await asyncio.wait_for(raw, timeout=query.source_timeout_s)
            candidates, retrieved, refused = self._normalize(query, spec, raw)
            status = MemorySourceStatus.SUCCEEDED if candidates else MemorySourceStatus.EMPTY
            return candidates, MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=status,
                retrieved_count=retrieved,
                admitted_count=len(candidates),
                refused_count=refused,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        except TimeoutError:
            return [], MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=MemorySourceStatus.TIMED_OUT,
                retrieved_count=0,
                admitted_count=0,
                refused_count=0,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error_code="memory_source_timeout",
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return [], MemorySourceReceipt(
                tier=spec.tier,
                source_id=spec.source_id,
                source_version=spec.source_version,
                status=MemorySourceStatus.FAILED,
                retrieved_count=0,
                admitted_count=0,
                refused_count=0,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error_code=self._error_code(exc),
            )

    @staticmethod
    def _merge(
        query: MemoryQuery, grouped: Mapping[MemoryTier, list[MemoryCandidate]]
    ) -> tuple[MemoryCandidate, ...]:
        best: dict[str, MemoryCandidate] = {}
        for tier in query.requested_tiers:
            for candidate in grouped.get(tier, []):
                existing = best.get(candidate.content_sha256)
                if existing is None:
                    best[candidate.content_sha256] = candidate
                    continue
                tiers = tuple(dict.fromkeys((*existing.corroborating_tiers, candidate.tier)))
                winner = candidate if candidate.relevance > existing.relevance else existing
                best[candidate.content_sha256] = replace(winner, corroborating_tiers=tiers)

        by_tier: dict[MemoryTier, list[MemoryCandidate]] = {
            tier: [] for tier in query.requested_tiers
        }
        for candidate in best.values():
            by_tier[candidate.tier].append(candidate)
        for values in by_tier.values():
            values.sort(key=lambda item: (-item.relevance, item.source_rank, item.content_sha256))

        selected: list[MemoryCandidate] = []
        selected_ids: set[str] = set()
        # One candidate per available tier prevents a rich semantic store from
        # starving procedural or episodic evidence before global ranking.
        for tier in query.requested_tiers:
            if by_tier[tier] and len(selected) < query.total_limit:
                candidate = by_tier[tier][0]
                selected.append(candidate)
                selected_ids.add(candidate.candidate_id)
        remaining = sorted(
            (item for item in best.values() if item.candidate_id not in selected_ids),
            key=lambda item: (
                -item.relevance,
                query.requested_tiers.index(item.tier),
                item.source_rank,
                item.content_sha256,
            ),
        )
        selected.extend(remaining[: max(0, query.total_limit - len(selected))])
        return tuple(selected)

    def retrieve(self, query: MemoryQuery) -> SelectiveMemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        started = time.time()
        grouped: dict[MemoryTier, list[MemoryCandidate]] = {}
        receipts: list[MemorySourceReceipt] = []
        for tier in query.requested_tiers:
            candidates, receipt = self._run_sync(query, self._specs[tier])
            grouped[tier] = candidates
            receipts.append(receipt)
        return SelectiveMemoryResult(
            query=query,
            candidates=self._merge(query, grouped),
            source_receipts=tuple(receipts),
            started_at=started,
            completed_at=time.time(),
        )

    async def retrieve_async(self, query: MemoryQuery) -> SelectiveMemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        started = time.time()
        outcomes = await asyncio.gather(
            *(self._run_async(query, self._specs[tier]) for tier in query.requested_tiers)
        )
        grouped = {
            tier: outcome[0] for tier, outcome in zip(query.requested_tiers, outcomes, strict=True)
        }
        receipts = tuple(outcome[1] for outcome in outcomes)
        return SelectiveMemoryResult(
            query=query,
            candidates=self._merge(query, grouped),
            source_receipts=receipts,
            started_at=started,
            completed_at=time.time(),
        )


def attach_memory_result(
    state: EpistemicState,
    result: SelectiveMemoryResult,
    *,
    operation_cost: float = 0.01,
) -> EpistemicState:
    """Atomically attach a retrieval result to the episode authority."""
    if not isinstance(state, EpistemicState):
        raise TypeError("state must be an EpistemicState")
    if not isinstance(result, SelectiveMemoryResult):
        raise TypeError("result must be a SelectiveMemoryResult")
    query = result.query
    if query.scope.episode_id != state.episode_id:
        raise SelectiveMemoryError("memory result belongs to another episode")
    if query.objective_sha256 != state.problem.objective_sha256:
        raise SelectiveMemoryError("memory result belongs to another objective")
    transaction = EpistemicTransaction(state)
    evidence_ids: list[str] = []
    for candidate in result.candidates:
        evidence = EvidenceRecord(
            evidence_id=candidate.evidence_id,
            kind=EvidenceKind.MEMORY,
            summary=candidate.content,
            content_sha256=candidate.content_sha256,
            provenance=EvidenceProvenance(
                source_id=f"memory.{candidate.tier.value}.{candidate.source_id}",
                source_version=candidate.source_version,
                invocation_sha256=query.invocation_sha256,
                receipt_sha256=candidate.retrieval_receipt_sha256,
                verification=EvidenceVerification.SOURCE_BOUND,
            ),
            scope=EvidenceScope(
                episode_id=state.episode_id,
                objective_sha256=state.problem.objective_sha256,
                claim_ids=(),
                purpose=EvidencePurpose.CONTEXT_ONLY,
            ),
            observed_at=candidate.retrieved_at,
            expires_at=candidate.expires_at,
        )
        transaction.add_evidence(evidence)
        evidence_ids.append(evidence.evidence_id)

    failed_sources = [
        receipt
        for receipt in result.source_receipts
        if receipt.status in {MemorySourceStatus.FAILED, MemorySourceStatus.TIMED_OUT}
    ]
    unavailable_sources = [
        receipt
        for receipt in result.source_receipts
        if receipt.status is MemorySourceStatus.UNAVAILABLE
    ]
    all_unavailable = len(unavailable_sources) == len(result.source_receipts)
    completed_sources = [
        receipt
        for receipt in result.source_receipts
        if receipt.status in {MemorySourceStatus.SUCCEEDED, MemorySourceStatus.EMPTY}
    ]
    operation_failed = not completed_sources
    outcome = OperationOutcome.FAILED if operation_failed else OperationOutcome.SUCCEEDED
    failure_code = (
        "memory_sources_unavailable"
        if all_unavailable
        else "memory_sources_failed"
        if operation_failed
        else ""
    )
    detail = (
        f"queried={len(result.source_receipts)} admitted={len(evidence_ids)} "
        f"failed={len(failed_sources)} unavailable={len(unavailable_sources)}"
    )
    operation = OperationRecord.create(
        operation_id="memory-search-" + result.result_sha256[:20],
        kind=OperationKind.SEARCH_MEMORY,
        outcome=outcome,
        input_state_sha256=state.state_sha256,
        cost=operation_cost,
        operator_id="rlc.selective_memory_bridge",
        operator_version=SELECTIVE_MEMORY_SCHEMA,
        input_payload_sha256=query.invocation_sha256,
        started_at=result.started_at,
        completed_at=result.completed_at,
        evidence_gained=tuple(evidence_ids),
        failure_code=failure_code,
        detail=detail,
    )
    transaction.add_operation(operation)
    try:
        return transaction.commit()
    except EpistemicStateError as exc:
        raise SelectiveMemoryError(f"memory result could not join epistemic state: {exc}") from exc


_MEMORY_CONTEXT_FIELDS = {
    "source",
    "text",
    "context_role",
    "instruction_authority",
    "evidence_id",
    "content_sha256",
    "scope_sha256",
    "retrieval_receipt_sha256",
    "epistemic_state_sha256",
    "memory_tier",
}


def validate_memory_context_items(
    state: EpistemicState,
    result: SelectiveMemoryResult,
    items: Iterable[Mapping[str, Any]],
) -> None:
    """Prove every memory slot is the state/result record it claims to be."""
    if not isinstance(state, EpistemicState):
        raise TypeError("state must be an EpistemicState")
    if not isinstance(result, SelectiveMemoryResult):
        raise TypeError("result must be a SelectiveMemoryResult")
    if state.episode_id != result.query.scope.episode_id:
        raise SelectiveMemoryError("epistemic state and memory result use different episodes")
    if state.problem.objective_sha256 != result.query.objective_sha256:
        raise SelectiveMemoryError("epistemic state and memory result use different objectives")
    evidence_by_id = {record.evidence_id: record for record in state.evidence}
    candidates_by_evidence = {candidate.evidence_id: candidate for candidate in result.candidates}
    observed: set[str] = set()
    for raw in items:
        if not isinstance(raw, Mapping):
            raise SelectiveMemoryError("cognitive context item must be an object")
        role = raw.get("context_role")
        reserved = set(raw) & _MEMORY_CONTEXT_FIELDS
        if role != "memory_observation":
            if reserved - {"source", "text"}:
                raise SelectiveMemoryError(
                    "non-memory context cannot carry memory authority fields"
                )
            continue
        if set(raw) != _MEMORY_CONTEXT_FIELDS:
            raise SelectiveMemoryError("memory context fields do not match the contract")
        if raw.get("instruction_authority") is not False:
            raise SelectiveMemoryError("memory context cannot have instruction authority")
        evidence_id = str(raw.get("evidence_id") or "")
        if evidence_id in observed:
            raise SelectiveMemoryError("memory context repeats one evidence record")
        observed.add(evidence_id)
        candidate = candidates_by_evidence.get(evidence_id)
        evidence = evidence_by_id.get(evidence_id)
        if candidate is None or evidence is None:
            raise SelectiveMemoryError("memory context references unknown evidence")
        if evidence.kind is not EvidenceKind.MEMORY:
            raise SelectiveMemoryError("memory context references non-memory evidence")
        if evidence.scope.purpose is not EvidencePurpose.CONTEXT_ONLY:
            raise SelectiveMemoryError("memory context acquired claim authority")
        candidate_index = list(result.candidates).index(candidate)
        expected = {
            "source": (
                "memory"
                if candidate_index == 0
                else (
                    f"memory.{candidate.tier.value}.{candidate.evidence_id.rsplit('-', 1)[-1][:8]}"
                )
            ),
            "text": candidate.content,
            "context_role": "memory_observation",
            "instruction_authority": False,
            "evidence_id": candidate.evidence_id,
            "content_sha256": candidate.content_sha256,
            "scope_sha256": candidate.scope.sha256,
            "retrieval_receipt_sha256": candidate.retrieval_receipt_sha256,
            "epistemic_state_sha256": state.state_sha256,
            "memory_tier": candidate.tier.value,
        }
        if dict(raw) != expected:
            raise SelectiveMemoryError("memory context differs from its admitted record")
        if (
            evidence.summary != candidate.content
            or evidence.content_sha256 != candidate.content_sha256
        ):
            raise SelectiveMemoryError("memory evidence content differs from its slot")


__all__ = [
    "DEFAULT_SOURCE_TIMEOUT_S",
    "MAX_MEMORY_CONTENT_CHARS",
    "MAX_MEMORY_PER_TIER",
    "MAX_MEMORY_RESULTS",
    "MemoryAdapter",
    "MemoryCandidate",
    "MemoryQuery",
    "MemoryScope",
    "MemorySourceReceipt",
    "MemorySourceStatus",
    "MemoryTier",
    "SELECTIVE_MEMORY_SCHEMA",
    "SelectiveMemoryBridge",
    "SelectiveMemoryError",
    "SelectiveMemoryResult",
    "attach_memory_result",
    "validate_memory_context_items",
]
