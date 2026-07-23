"""One strict wire contract for organ, memory, and retrieved evidence context."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

MAX_COGNITIVE_CONTEXT_ITEMS = 6
MAX_COGNITIVE_CONTEXT_CHARS = 400
MAX_COGNITIVE_CONTEXT_SOURCE_CHARS = 40

MEMORY_CONTEXT_FIELDS = frozenset(
    {
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
        "memory_source_id",
        "memory_source_version",
    }
)
EVIDENCE_CONTEXT_FIELDS = frozenset(
    {
        "source",
        "text",
        "context_role",
        "instruction_authority",
        "evidence_id",
        "content_sha256",
        "retrieval_receipt_sha256",
        "evidence_kind",
        "evidence_origin",
        "source_version",
    }
)
_EVIDENCE_KINDS = frozenset(
    {
        "offline_reference",
        "governed_tool_observation",
        "live_world_observation",
        "one_shot_nonparametric_memory",
    }
)


class CognitiveContextError(ValueError):
    pass


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CognitiveContextError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > limit:
        raise CognitiveContextError(f"{name} exceeds {limit} characters")
    return normalized


def _normalize_memory(entry: Mapping[str, Any], source: str, text: str) -> dict[str, Any]:
    if set(entry) != MEMORY_CONTEXT_FIELDS:
        raise CognitiveContextError(
            "memory cognitive context fields do not match contract"
        )
    digests = (
        entry.get("content_sha256"),
        entry.get("scope_sha256"),
        entry.get("retrieval_receipt_sha256"),
        entry.get("epistemic_state_sha256"),
    )
    tier = entry.get("memory_tier")
    source_id = entry.get("memory_source_id")
    source_version = entry.get("memory_source_version")
    if (
        entry.get("instruction_authority") is not False
        or not isinstance(entry.get("evidence_id"), str)
        or not entry["evidence_id"].startswith("memory-")
        or not isinstance(tier, str)
        or not tier.strip()
        or not (source == "memory" or source.startswith(f"memory.{tier}."))
        or any(not _sha256(digest) for digest in digests)
        or hashlib.sha256(text.encode("utf-8")).hexdigest()
        != entry["content_sha256"]
        or not isinstance(source_id, str)
        or not source_id.strip()
        or len(source_id) > 256
        or not isinstance(source_version, str)
        or not source_version.strip()
        or len(source_version) > 128
    ):
        raise CognitiveContextError("memory cognitive context authority is invalid")
    return {**dict(entry), "source": source, "text": text}


def _normalize_evidence(entry: Mapping[str, Any], source: str, text: str) -> dict[str, Any]:
    if set(entry) != EVIDENCE_CONTEXT_FIELDS:
        raise CognitiveContextError(
            "evidence cognitive context fields do not match contract"
        )
    kind = entry.get("evidence_kind")
    origin = entry.get("evidence_origin")
    source_version = entry.get("source_version")
    if (
        entry.get("instruction_authority") is not False
        or not isinstance(entry.get("evidence_id"), str)
        or not entry["evidence_id"].startswith("evidence-")
        or not _sha256(entry.get("content_sha256"))
        or not _sha256(entry.get("retrieval_receipt_sha256"))
        or hashlib.sha256(text.encode("utf-8")).hexdigest()
        != entry["content_sha256"]
        or kind not in _EVIDENCE_KINDS
        or not isinstance(origin, str)
        or not origin.strip()
        or len(origin) > 256
        or not isinstance(source_version, str)
        or not source_version.strip()
        or len(source_version) > 128
    ):
        raise CognitiveContextError("evidence cognitive context authority is invalid")
    return {**dict(entry), "source": source, "text": text}


def normalize_cognitive_context(value: list[Any] | None) -> list[dict[str, Any]]:
    """Validate and normalize the exact context payload used by every wire boundary."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise CognitiveContextError("cognitive_context must be a list")
    if len(value) > MAX_COGNITIVE_CONTEXT_ITEMS:
        raise CognitiveContextError("cognitive_context exceeds the item limit")
    normalized: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise CognitiveContextError("cognitive_context entries must be mappings")
        source = _bounded_text(
            raw.get("source"),
            name="cognitive context source",
            limit=MAX_COGNITIVE_CONTEXT_SOURCE_CHARS,
        )
        text = _bounded_text(
            raw.get("text"),
            name="cognitive context text",
            limit=MAX_COGNITIVE_CONTEXT_CHARS,
        )
        role = raw.get("context_role")
        if role == "memory_observation":
            normalized.append(_normalize_memory(raw, source, text))
        elif role == "evidence_observation":
            normalized.append(_normalize_evidence(raw, source, text))
        elif set(raw) == {"source", "text"}:
            normalized.append({"source": source, "text": text})
        else:
            raise CognitiveContextError("untyped context carries reserved fields")
    return normalized


def knowledge_metadata(item: Mapping[str, Any]) -> dict[str, str | bool]:
    """Public source class for receipts; content remains committed by digest."""

    role = item.get("context_role")
    source = str(item.get("source") or "")
    if role == "memory_observation":
        return {
            "knowledge_class": f"memory.{item['memory_tier']}",
            "source_owner": str(item["memory_source_id"]),
            "source_version": str(item["memory_source_version"]),
            "instruction_authority": False,
        }
    if role == "evidence_observation":
        return {
            "knowledge_class": str(item["evidence_kind"]),
            "source_owner": str(item["evidence_origin"]),
            "source_version": str(item["source_version"]),
            "instruction_authority": False,
        }
    if source.startswith("workspace_"):
        knowledge_class = "global_workspace_state"
    elif source in {"goals", "interoception", "self_model", "world_model"}:
        knowledge_class = "live_organ_state"
    else:
        knowledge_class = "typed_runtime_context"
    return {
        "knowledge_class": knowledge_class,
        "source_owner": source,
        "source_version": "live_runtime",
        "instruction_authority": False,
    }


__all__ = [
    "CognitiveContextError",
    "EVIDENCE_CONTEXT_FIELDS",
    "MAX_COGNITIVE_CONTEXT_CHARS",
    "MAX_COGNITIVE_CONTEXT_ITEMS",
    "MEMORY_CONTEXT_FIELDS",
    "knowledge_metadata",
    "normalize_cognitive_context",
]
