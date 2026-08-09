"""Bounded governed web evidence acquisition for foreground recurrent thought.

The resident model selects a retrieval-class cognitive action.  This service-
side broker, never the MLX worker, may satisfy that request through the
canonical CapabilityEngine path when local reference evidence is stale or
absent.  Results return as non-authoritative typed evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from core.brain.capability_evidence_context import (
    CapabilityEvidenceBundle,
    build_current_turn_capability_evidence,
    merge_capability_evidence,
)
from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

CORTEX_WEB_ACQUISITION_SCHEMA = "aura.rlc.web_acquisition.v1"
WEB_ACQUISITION_TIMEOUT_S = 20.0

_LIVE_EVIDENCE_RE = re.compile(
    r"\b(?:today|tonight|yesterday|latest|current(?:ly)?|recent(?:ly)?|"
    r"live|breaking|news|online|internet|web|website|url|source|citation|"
    r"price|weather|schedule|score|version|release|updated?)\b",
    re.IGNORECASE,
)
_LOCAL_ONLY_RE = re.compile(
    r"\b(?:my|our|private|confidential|local file|on (?:my|this) (?:computer|mac)|"
    r"in (?:my|our) (?:notes|email|messages|memory)|you remember)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CortexWebAcquisition:
    context: list[dict[str, Any]] | None
    receipt: dict[str, Any]


def should_acquire_live_web(
    objective: str,
    retrieval_query: str,
    *,
    local_context_is_new: bool,
) -> tuple[bool, str]:
    """Select live evidence for temporal/source requests or local misses."""

    text = f"{objective}\n{retrieval_query}"[:2_400]
    from core.security.structural_redaction import redact_text

    _redacted, sensitive = redact_text(text)
    if sensitive or _LOCAL_ONLY_RE.search(text):
        return False, "private_or_local_objective"
    if _LIVE_EVIDENCE_RE.search(text):
        return True, "live_or_source_sensitive_objective"
    if not local_context_is_new:
        return True, "local_reference_uncovered"
    return False, "local_reference_sufficient"


def _base_receipt(objective: str, query: str, reason: str) -> dict[str, Any]:
    return {
        "schema": CORTEX_WEB_ACQUISITION_SCHEMA,
        "objective_sha256": hashlib.sha256(objective.encode("utf-8")).hexdigest(),
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "selection_reason": reason,
        "attempted": False,
        "completed": False,
        "status": "not_attempted",
        "tool": "web_search",
        "worker_performed_io": False,
        "service_performed_io": False,
        "capability_receipt": {},
        "merge_receipt": {},
        "result_sha256": None,
    }


def _finish_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


async def acquire_live_web_evidence(
    orchestrator: Any,
    *,
    objective: str,
    retrieval_query: str,
    cognitive_context: list[dict[str, Any]] | None,
    selection_reason: str,
    timeout_s: float = WEB_ACQUISITION_TIMEOUT_S,
) -> CortexWebAcquisition:
    """Execute one read-only web search and merge its admitted observation."""

    objective = str(objective or "").strip()
    # The first episode's tentative answer belongs in Aura's private
    # recurrence receipt, not in a third-party search query. The public
    # objective is sufficient to retrieve corroborating evidence.
    query = objective[:600]
    receipt = _base_receipt(objective, query, selection_reason)
    executor = getattr(orchestrator, "execute_tool", None)
    if not objective or not query:
        receipt["status"] = "invalid_query"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))
    if not callable(executor):
        receipt["status"] = "executor_unavailable"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))

    receipt["attempted"] = True
    receipt["service_performed_io"] = True
    context = {
        "origin": "latent_cortex",
        "source": "latent_cortex",
        "objective": objective[:500],
        "message": objective[:500],
        "reason": "foreground_recurrent_evidence_acquisition",
        "effect_scope": "read_only",
        "risk_level": "low",
        "foreground_request": True,
        "foreground_cognitive_acquisition": True,
        "user_explicitly_authorized": False,
        "user_requested_action": False,
    }
    try:
        result = await asyncio.wait_for(
            executor(
                "web_search",
                {
                    "query": query,
                    "deep": False,
                    "num_results": 4,
                    "retain": False,
                },
                origin="latent_cortex",
                payload_context=context,
            ),
            timeout=max(1.0, min(float(timeout_s), WEB_ACQUISITION_TIMEOUT_S)),
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        receipt["status"] = "timeout"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))
    except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        receipt["status"] = f"execution_error:{type(exc).__name__.lower()}"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))

    if not isinstance(result, dict):
        receipt["status"] = "invalid_result"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))
    try:
        receipt["result_sha256"] = canonical_sha256(result)
    except ValueError:
        receipt["status"] = "noncanonical_result"
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))
    if result.get("ok") is not True:
        receipt["status"] = str(result.get("status") or "tool_failed")[:96]
        return CortexWebAcquisition(cognitive_context, _finish_receipt(receipt))

    bundle: CapabilityEvidenceBundle = build_current_turn_capability_evidence(
        {
            "last_skill_run": "web_search",
            "last_skill_ok": True,
            "last_skill_objective_hash": hashlib.sha256(
                " ".join(objective.split()).encode("utf-8")
            ).hexdigest(),
            "last_skill_result_payload": result,
        },
        objective,
    )
    merged, merge_receipt = merge_capability_evidence(cognitive_context, bundle)
    receipt["capability_receipt"] = bundle.receipt
    receipt["merge_receipt"] = merge_receipt
    receipt["completed"] = bool(bundle.items)
    receipt["status"] = (
        "completed_new_context"
        if bundle.items
        else f"evidence_not_admitted:{bundle.receipt.get('reason') or 'unknown'}"
    )
    return CortexWebAcquisition(merged, _finish_receipt(receipt))


__all__ = [
    "CORTEX_WEB_ACQUISITION_SCHEMA",
    "CortexWebAcquisition",
    "WEB_ACQUISITION_TIMEOUT_S",
    "acquire_live_web_evidence",
    "should_acquire_live_web",
]
