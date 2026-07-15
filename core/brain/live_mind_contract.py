"""Shared live-mind generation contract helpers.

The desktop lane treats live-mind metadata as proof material. Parent context
may reconcile a stale structural-bound bit, but it must never manufacture a
worker application or quality success that the worker did not report.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from typing import Any

REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "clean_user_surface_recurrent_loops",
        "clean_user_surface_steering_alpha",
    }
)

_MAX_TEXT_MUTATIONS = 64
_CONTENT_FREE_REASON_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,96}$")


def _content_free_reason(value: Any) -> str:
    """Keep categorical reasons while replacing arbitrary detail with a digest."""

    if isinstance(value, Mapping):
        for key in ("code", "category", "rule", "type", "name"):
            candidate = str(value.get(key) or "").strip()
            if _CONTENT_FREE_REASON_RE.fullmatch(candidate):
                return candidate[:96]
        value = repr(sorted((str(key), repr(item)) for key, item in value.items()))
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if _CONTENT_FREE_REASON_RE.fullmatch(candidate):
        return candidate[:96]
    digest = hashlib.sha256(candidate.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"detail_sha256:{digest}"


def normalize_text_mutations(value: Any) -> list[dict[str, Any]]:
    """Return bounded, content-free provenance for visible-text mutations."""

    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for fallback_sequence, item in enumerate(
        value[-_MAX_TEXT_MUTATIONS:],
        start=1,
    ):
        if not isinstance(item, Mapping):
            continue
        stage = str(item.get("stage") or "").strip()[:96]
        method = str(item.get("method") or "").strip()[:96]
        reasons = [
            normalized_reason
            for reason in (item.get("reasons") or [])
            if (normalized_reason := _content_free_reason(reason))
        ][:16]
        if not stage or not method:
            continue
        entry: dict[str, Any] = {
            "sequence": fallback_sequence,
            "stage": stage,
            "method": method,
            "reasons": reasons,
            "deterministic": bool(item.get("deterministic", False)),
        }
        event_id = str(item.get("event_id") or "").strip()[:64]
        if event_id:
            entry["event_id"] = event_id
        try:
            entry["sequence"] = max(1, int(item.get("sequence") or fallback_sequence))
        except (TypeError, ValueError, OverflowError):
            entry["sequence"] = fallback_sequence
        for key in ("before_chars", "after_chars"):
            try:
                entry[key] = max(0, int(item.get(key) or 0))
            except (TypeError, ValueError, OverflowError):
                entry[key] = 0
        normalized.append(entry)
    return normalized


def merge_text_mutations(*values: Any) -> list[dict[str, Any]]:
    """Merge ordered mutation ledgers while removing copied receipt duplicates."""

    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        for entry in normalize_text_mutations(value):
            event_id = str(entry.get("event_id") or "")
            identity = (
                ("event", event_id)
                if event_id
                else (
                    "legacy",
                    entry["sequence"],
                    entry["stage"],
                    entry["method"],
                    tuple(entry["reasons"]),
                    entry["deterministic"],
                    entry["before_chars"],
                    entry["after_chars"],
                )
            )
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(entry)
    bounded = merged[-_MAX_TEXT_MUTATIONS:]
    return [dict(entry, sequence=index) for index, entry in enumerate(bounded, start=1)]


def append_text_mutation(
    container: dict[str, Any],
    *,
    stage: str,
    method: str,
    reasons: Any,
    before: Any,
    after: Any,
    deterministic: bool,
) -> dict[str, Any]:
    """Append one mutation only when the visible bytes actually changed."""

    before_text = str(before or "")
    after_text = str(after or "")
    if before_text == after_text:
        return container
    if isinstance(reasons, str):
        reason_items = [reasons]
    else:
        reason_items = list(reasons or [])
    entry = {
        "event_id": uuid.uuid4().hex,
        "sequence": 1
        + max(
            (
                int(item.get("sequence") or 0)
                for item in normalize_text_mutations(container.get("text_mutations"))
            ),
            default=0,
        ),
        "stage": stage,
        "method": method,
        "reasons": reason_items,
        "deterministic": bool(deterministic),
        "before_chars": len(before_text),
        "after_chars": len(after_text),
    }
    mutations = merge_text_mutations(container.get("text_mutations"), [entry])
    container["text_mutations"] = mutations
    container["text_mutation_count"] = len(mutations)
    container["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in mutations
    )
    return container


def live_mind_generation_controls_present(generation_controls: Any) -> bool:
    return bool(
        isinstance(generation_controls, Mapping)
        and REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
            generation_controls.keys()
        )
    )


def normalize_live_mind_surface_control_receipt(
    receipt: Any,
    *,
    controls_bound: bool,
    generation_controls: Any,
    surface_quality_gate_passed: bool | None = None,
    source: str,
) -> dict[str, Any]:
    """Return a coherent receipt for an already-bound live-mind turn.

    The worker reports whether it applied CAA/recurrent surface controls, while
    the CognitiveEngine owns whether the live mind controls were structurally
    bound for the turn. If the worker receipt is otherwise successful but omits
    that structural bit, normalize it before the chat contract evaluates the
    full-mind path.
    """

    normalized = dict(receipt) if isinstance(receipt, Mapping) else {}
    controls_present = live_mind_generation_controls_present(generation_controls)
    quality_passed = bool(
        normalized.get("surface_quality_gate_passed") is True
        if surface_quality_gate_passed is None
        else (
            surface_quality_gate_passed
            and normalized.get("surface_quality_gate_passed") is True
        )
    )

    generation_required = normalized.get("generation_required") is not False
    if not generation_required:
        return {
            **normalized,
            "enabled": False,
            "applied": False,
            "generation_required": False,
            "application_status": "not_applicable_structured_floor",
            "live_mind_controls_bound": bool(controls_bound and controls_present),
            "clean_user_surface_contract": True,
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": bool(quality_passed),
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "source": normalized.get("source") or source,
        }

    if not normalized:
        return {
            "enabled": False,
            "applied": False,
            "generation_required": True,
            "application_status": "worker_receipt_missing",
            "live_mind_controls_bound": bool(controls_bound and controls_present),
            "clean_user_surface_contract": False,
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": False,
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": ["worker_receipt_missing"],
            "source": source,
        }

    worker_applied = normalized.get("applied") is True
    clean_surface_proven = normalized.get("clean_user_surface_contract") is True
    worker_success = bool(worker_applied and clean_surface_proven and quality_passed)
    structurally_bound = bool(controls_bound and controls_present and worker_success)

    if structurally_bound and normalized.get("live_mind_controls_bound") is True:
        return normalized

    if not worker_success:
        if not worker_applied:
            failure_status = "worker_controls_not_applied"
        elif not clean_surface_proven:
            failure_status = "worker_clean_surface_unproven"
        else:
            failure_status = "worker_surface_quality_unproven"
    elif not structurally_bound:
        failure_status = "live_mind_controls_unbound"
    else:
        failure_status = "applied"

    return {
        **normalized,
        "enabled": bool(normalized.get("enabled", False)),
        "applied": worker_applied,
        "live_mind_controls_bound": structurally_bound,
        "clean_user_surface_contract": clean_surface_proven,
        "surface_quality_gate_enabled": bool(
            normalized.get("surface_quality_gate_enabled", False)
        ),
        "surface_quality_gate_passed": quality_passed,
        "surface_quality_gate_attempts": int(
            normalized.get("surface_quality_gate_attempts", 0) or 0
        ),
        "surface_quality_gate_reasons": list(
            normalized.get("surface_quality_gate_reasons", []) or []
        ),
        "application_status": normalized.get("application_status") or failure_status,
        "source": normalized.get("source") or source,
    }
