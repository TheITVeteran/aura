"""Bind governed capability results into recurrent cognition as observations.

Capability execution remains owned by CapabilityEngine and the Will.  This
module does not execute tools.  It admits a successful, current-turn result as
non-authoritative evidence so the Recursive Latent Cortex can reason over the
same web, local-compute, and sensory facts already available to ordinary chat.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from core.brain.epistemic_firewall import EpistemicFirewall, EvidenceItem
from core.brain.llm.latent_cortex.cognitive_context import (
    MAX_COGNITIVE_CONTEXT_CHARS,
    MAX_COGNITIVE_CONTEXT_ITEMS,
    normalize_cognitive_context,
)
from core.brain.prompts.sanitizer import ContextGuard
from core.skills.catalog_policy import resolve_skill_policy

CAPABILITY_EVIDENCE_SCHEMA = "aura.rlc.capability_evidence.v1"
CAPABILITY_CONTEXT_MERGE_SCHEMA = "aura.rlc.capability_context_merge.v1"

_ADMISSIBLE_EFFECT_SCOPES = frozenset(
    {"status", "read_only", "pure_compute", "sandboxed_compute"}
)
_BROWSER_OBSERVATION_SKILLS = frozenset({"sovereign_browser"})
_WEB_SKILLS = frozenset(
    {
        "free_search",
        "grounded_search",
        "search_web",
        "sovereign_browser",
        "web_search",
    }
)
_COMPUTE_SCOPES = frozenset({"pure_compute", "sandboxed_compute"})


@dataclass(frozen=True, slots=True)
class CapabilityEvidenceBundle:
    items: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _objective_sha256(value: Any) -> str:
    return _sha256_text(" ".join(str(value or "").split()).strip())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=lambda item: f"<{type(item).__module__}.{type(item).__qualname__}>",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit].rstrip()


def _freshness_basis(modifiers: dict[str, Any], objective: str) -> str:
    expected = _objective_sha256(objective)
    observed = str(modifiers.get("last_skill_objective_hash") or "").strip()
    if expected and observed == expected:
        return "objective_sha256"
    turn_marker = str(modifiers.get("evidence_turn_marker") or "").strip()
    observed_marker = str(modifiers.get("last_skill_turn_marker") or "").strip()
    if turn_marker and observed_marker == turn_marker:
        return "turn_marker"
    return ""


def _candidate_rows(skill: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract bounded semantic observations without serializing raw payloads."""

    rows: list[tuple[str, str]] = []
    if skill in _WEB_SKILLS:
        for item in list(payload.get("results") or [])[:3]:
            if not isinstance(item, dict):
                continue
            title = _bounded(item.get("title"), 100)
            snippet = _bounded(item.get("snippet"), 240)
            url = _bounded(item.get("url"), 180)
            text = ". ".join(part for part in (title, snippet) if part)
            if url:
                text = f"{text} Source: {url}" if text else f"Source: {url}"
            if text:
                rows.append((text, url or f"capability:{skill}"))
        summary = _bounded(
            payload.get("answer")
            or payload.get("summary")
            or payload.get("content")
            or payload.get("result"),
            500,
        )
        source = _bounded(payload.get("source") or payload.get("url"), 180)
        if summary:
            rows.insert(0, (summary, source or f"capability:{skill}"))
        return rows[:4]

    if skill in {"run_code", "code_repl", "internal_sandbox"}:
        try:
            exit_code = int(payload.get("exit_code", payload.get("return_code", 0)) or 0)
        except (TypeError, ValueError, OverflowError):
            return []
        if exit_code != 0 or payload.get("ok") is False:
            return []
        stdout = _bounded(payload.get("stdout"), 600)
        summary = _bounded(payload.get("summary") or payload.get("result"), 300)
        text = stdout or summary
        return [(text, f"capability:{skill}")] if text else []

    summary = _bounded(
        payload.get("summary")
        or payload.get("content")
        or payload.get("result")
        or payload.get("message")
        or payload.get("readable"),
        600,
    )
    source = _bounded(payload.get("source") or payload.get("url"), 180)
    return [(summary, source or f"capability:{skill}")] if summary else []


def _public_guard_receipt(receipt: Any) -> dict[str, Any]:
    return {
        "policy_version": str(getattr(receipt, "policy_version", "")),
        "quarantined": bool(getattr(receipt, "quarantined", False)),
        "fail_closed": bool(getattr(receipt, "fail_closed", False)),
        "categories": sorted(
            {
                str(getattr(item, "category", ""))
                for item in list(getattr(receipt, "detections", []) or [])
                if str(getattr(item, "category", ""))
            }
        ),
    }


def _public_firewall_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": str(receipt.get("schema") or ""),
        "admitted_count": len(list(receipt.get("admitted") or [])),
        "refused_count": len(list(receipt.get("refused") or [])),
        "conflict_count": len(list(receipt.get("conflicts") or [])),
        "coverage": receipt.get("coverage"),
        "abstain": bool(receipt.get("abstain", False)),
        "needs_more_retrieval": bool(receipt.get("needs_more_retrieval", False)),
        "reasons": list(receipt.get("reasons") or [])[:8],
    }


def build_current_turn_capability_evidence(
    modifiers: dict[str, Any] | None,
    objective: str,
) -> CapabilityEvidenceBundle:
    """Create at most one content-addressed current-turn evidence slot."""

    values = dict(modifiers or {})
    skill = str(values.get("last_skill_run") or "").strip()
    base = {
        "schema": CAPABILITY_EVIDENCE_SCHEMA,
        "skill": skill,
        "objective_sha256": _objective_sha256(objective),
        "admitted": False,
        "reason": "",
        "freshness_basis": "",
        "effect_scope": "",
        "candidate_count": 0,
        "guard": [],
        "firewall": {},
        "item_sha256s": [],
    }
    if not skill:
        return CapabilityEvidenceBundle((), {**base, "reason": "no_skill_result"})
    if values.get("last_skill_ok") is not True:
        return CapabilityEvidenceBundle((), {**base, "reason": "skill_not_successful"})
    payload = values.get("last_skill_result_payload")
    if not isinstance(payload, dict) or not payload:
        return CapabilityEvidenceBundle((), {**base, "reason": "skill_payload_missing"})

    freshness = _freshness_basis(values, objective)
    if not freshness:
        return CapabilityEvidenceBundle((), {**base, "reason": "stale_skill_result"})
    base["freshness_basis"] = freshness

    policy = resolve_skill_policy(skill)
    scope = policy.effect_scope if policy is not None else ""
    base["effect_scope"] = scope
    if scope not in _ADMISSIBLE_EFFECT_SCOPES and skill not in _BROWSER_OBSERVATION_SKILLS:
        return CapabilityEvidenceBundle((), {**base, "reason": "effect_scope_not_observational"})

    raw_rows = _candidate_rows(skill, payload)
    base["candidate_count"] = len(raw_rows)
    if not raw_rows:
        return CapabilityEvidenceBundle((), {**base, "reason": "no_observation_content"})

    evidence: list[EvidenceItem] = []
    guard_receipts: list[dict[str, Any]] = []
    for index, (text, origin) in enumerate(raw_rows):
        guarded = ContextGuard.guard(
            text,
            role="retrieved",
            request_id=f"rlc-capability:{skill}:{index}",
        )
        public_guard = _public_guard_receipt(guarded.receipt)
        guard_receipts.append(public_guard)
        if public_guard["quarantined"] or public_guard["fail_closed"]:
            continue
        clean = _bounded(guarded.text, 800)
        if not clean:
            continue
        evidence.append(
            EvidenceItem(
                text=clean,
                origin=origin,
                channel=f"capability.{skill}",
                kind="observed_fact" if scope in _COMPUTE_SCOPES or scope == "status" else "claim",
                trust=0.9 if scope in _COMPUTE_SCOPES or scope == "status" else 0.6,
            )
        )
    base["guard"] = guard_receipts
    if not evidence:
        return CapabilityEvidenceBundle((), {**base, "reason": "all_content_quarantined"})

    firewall = EpistemicFirewall(
        max_admitted=3,
        min_coverage=0.0 if scope in _COMPUTE_SCOPES or scope == "status" else 0.2,
        min_item_relevance=0.0 if scope in _COMPUTE_SCOPES or scope == "status" else 0.12,
    ).review(objective, evidence)
    firewall_receipt = firewall.to_receipt()
    base["firewall"] = _public_firewall_receipt(firewall_receipt)
    admitted = firewall.admitted_texts()
    if not admitted:
        reason = "epistemic_conflict" if firewall.abstain else "epistemic_admission_empty"
        return CapabilityEvidenceBundle((), {**base, "reason": reason})

    text = " | ".join(admitted)
    text = _bounded(text, MAX_COGNITIVE_CONTEXT_CHARS)
    content_sha256 = _sha256_text(text)
    private_receipt = {
        "schema": CAPABILITY_EVIDENCE_SCHEMA,
        "skill": skill,
        "objective_sha256": base["objective_sha256"],
        "freshness_basis": freshness,
        "effect_scope": scope,
        "payload_sha256": _canonical_sha256(payload),
        "guard_receipts": [guarded for guarded in guard_receipts],
        "firewall_receipt": firewall_receipt,
        "content_sha256": content_sha256,
    }
    retrieval_receipt_sha256 = _canonical_sha256(private_receipt)
    source = f"capability.{skill}"[:40]
    source_version = f"aura.capability-result.v1:{scope}"[:128]
    identity = _sha256_text(
        f"{skill}:{content_sha256}:{retrieval_receipt_sha256}:{source_version}"
    )
    item = {
        "source": source,
        "text": text,
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": f"evidence-{identity[:24]}",
        "content_sha256": content_sha256,
        "retrieval_receipt_sha256": retrieval_receipt_sha256,
        "evidence_kind": "governed_tool_observation",
        "evidence_origin": f"core.capability_engine:{skill}",
        "source_version": source_version,
    }
    normalized = normalize_cognitive_context([item])
    receipt = {
        **base,
        "admitted": True,
        "reason": "current_turn_observation_admitted",
        "item_sha256s": [content_sha256],
        "receipt_sha256": retrieval_receipt_sha256,
    }
    return CapabilityEvidenceBundle(tuple(normalized), receipt)


def merge_capability_evidence(
    cognitive_context: list[dict[str, Any]] | None,
    bundle: CapabilityEvidenceBundle,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Reserve capacity for current evidence and receipt any displaced organ slot."""

    existing = normalize_cognitive_context(list(cognitive_context or []))
    capability_items = normalize_cognitive_context(list(bundle.items))
    requested = [*existing, *capability_items]
    displaced: list[dict[str, str]] = []
    while len(requested) > MAX_COGNITIVE_CONTEXT_ITEMS:
        candidates = [
            (index, _context_priority(item))
            for index, item in enumerate(requested)
            if item.get("evidence_kind") != "governed_tool_observation"
        ]
        if not candidates:
            raise ValueError("capability evidence exceeds the cognitive context limit")
        drop_index, _priority = min(candidates, key=lambda row: (row[1], -row[0]))
        dropped = requested.pop(drop_index)
        displaced.append(
            {
                "source": str(dropped.get("source") or "")[:40],
                "content_sha256": _sha256_text(str(dropped.get("text") or "")),
            }
        )
    admitted = normalize_cognitive_context(requested) or None
    body = {
        "schema": CAPABILITY_CONTEXT_MERGE_SCHEMA,
        "requested_items": len(existing) + len(capability_items),
        "admitted_items": len(admitted or []),
        "capability_items": len(capability_items),
        "displaced": displaced,
        "complete": not displaced,
    }
    return admitted, {**body, "receipt_sha256": _canonical_sha256(body)}


def _context_priority(item: dict[str, Any]) -> int:
    source = str(item.get("source") or "")
    if item.get("context_role") == "memory_observation":
        return 100
    if source == "epistemic_caution":
        return 95
    if source == "interoception":
        return 90
    if item.get("context_role") == "evidence_observation":
        return 85
    if source == "goals":
        return 75
    if source == "self_model":
        return 70
    if source == "world_model":
        return 65
    return 60


__all__ = [
    "CAPABILITY_CONTEXT_MERGE_SCHEMA",
    "CAPABILITY_EVIDENCE_SCHEMA",
    "CapabilityEvidenceBundle",
    "build_current_turn_capability_evidence",
    "merge_capability_evidence",
]
