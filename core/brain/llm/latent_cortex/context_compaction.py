"""Deterministic context compaction for resident-scale latent episodes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

POLICY_VERSION = "resident_latent_salience_v1"
_OMISSION_MARKER = "\n\n[... context omitted by resident latent salience budget ...]\n\n"
_TERMS = re.compile(r"[a-z0-9_]{3,}")
_STOP_WORDS = {
    "and",
    "are",
    "but",
    "can",
    "for",
    "from",
    "how",
    "into",
    "not",
    "that",
    "the",
    "then",
    "this",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}
_HIGH_VALUE_MARKERS = (
    "LIVE MIND",
    "CURRENT FUNCTIONAL STATE",
    "CANONICAL",
    "RESPONSE CONTRACT",
    "RUNTIME",
    "SKILL RESULT",
    "INTERNAL MEMORY",
    "CAUSAL",
)


def _canonical_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_ends(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_OMISSION_MARKER) + 2:
        return text[: max(0, limit)]
    remaining = limit - len(_OMISSION_MARKER)
    head = max(1, remaining * 2 // 3)
    tail = max(1, remaining - head)
    return f"{text[:head]}{_OMISSION_MARKER}{text[-tail:]}"


def _normalize_messages(messages: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "aura":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("latent context has no valid messages")
    return normalized


def compact_latent_messages(
    messages: list[Any],
    *,
    max_chars: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select a bounded structured context and return a hash-bound receipt."""

    if type(max_chars) is not int or not 2048 <= max_chars <= 65536:
        raise ValueError("max_chars must be an integer inside [2048, 65536]")
    original = _normalize_messages(messages)
    original_chars = sum(len(item["content"]) for item in original)
    original_sha256 = _canonical_sha256(original)
    if original_chars <= max_chars:
        compacted = [dict(item) for item in original]
    else:
        system_index = next(
            (
                index
                for index, item in enumerate(original)
                if item["role"] == "system"
            ),
            None,
        )
        user_index = next(
            (
                index
                for index in range(len(original) - 1, -1, -1)
                if original[index]["role"] == "user"
            ),
            None,
        )
        mandatory_indices = {
            index for index in (system_index, user_index) if index is not None
        }
        if not mandatory_indices:
            mandatory_indices.add(len(original) - 1)
        selected: dict[int, dict[str, str]] = {}
        if system_index is not None and user_index is not None:
            mandatory_limits = {
                system_index: max(1024, int(max_chars * 0.45)),
                user_index: max(512, int(max_chars * 0.30)),
            }
        else:
            mandatory_limits = {next(iter(mandatory_indices)): max_chars}
        for index in sorted(mandatory_indices):
            content = original[index]["content"]
            fitted = _fit_ends(content, min(len(content), mandatory_limits[index]))
            selected[index] = {"role": original[index]["role"], "content": fitted}
        remaining = max_chars - sum(
            len(item["content"]) for item in selected.values()
        )
        objective_index = user_index if user_index is not None else max(mandatory_indices)
        user_text = original[objective_index]["content"]
        objective_terms = {
            term
            for term in _TERMS.findall(user_text.lower())
            if term not in _STOP_WORDS
        }

        ranked: list[tuple[int, int]] = []
        for index, item in enumerate(original):
            if index in mandatory_indices:
                continue
            content = item["content"]
            lowered = content.lower()
            overlap = sum(1 for term in objective_terms if term in lowered)
            marker_hits = sum(1 for marker in _HIGH_VALUE_MARKERS if marker in content.upper())
            role_weight = (
                40
                if item["role"] == "system"
                else 20
                if item["role"] == "assistant"
                else 10
            )
            recency = index
            ranked.append((marker_hits * 1000 + overlap * 100 + role_weight + recency, index))

        for _score, index in sorted(ranked, reverse=True):
            if remaining < 256:
                break
            content_limit = min(len(original[index]["content"]), remaining, 1800)
            content = _fit_ends(original[index]["content"], content_limit)
            selected[index] = {
                "role": original[index]["role"],
                "content": content,
            }
            remaining -= len(content)

        compacted = [selected[index] for index in sorted(selected)]

    compacted_chars = sum(len(item["content"]) for item in compacted)
    receipt = {
        "schema": "aura.latent_context_compaction.v1",
        "policy": POLICY_VERSION,
        "applied": compacted != original,
        "max_chars": max_chars,
        "original_message_count": len(original),
        "original_char_count": original_chars,
        "original_sha256": original_sha256,
        "compacted_message_count": len(compacted),
        "compacted_char_count": compacted_chars,
        "compacted_sha256": _canonical_sha256(compacted),
        "omitted_char_count": max(0, original_chars - compacted_chars),
    }
    return compacted, receipt


__all__ = ["POLICY_VERSION", "compact_latent_messages"]
