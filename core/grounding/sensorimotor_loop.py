"""core/grounding/sensorimotor_loop.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action → prediction → verification → surprise: real grounding over the
digital world she actually inhabits.

Symbolic reflexes execute an action and trust the return value. Grounded
action commits to an EXPECTED world-state first (an outcome-ledger
receipt), executes, then checks REALITY with its own senses (filesystem
stats, not the tool's claim) and resolves the receipt with what was
actually observed. The ledger computes prediction error and calibrates
expectations over time — surprise becomes a learning signal instead of
a silent discrepancy.

The sharpest edge is the claim/reality diff: a tool that REPORTS success
while the predicted world-state is absent is a confabulated action —
recorded as fault ACTION-CLAIM-MISMATCH, the exact failure class where
an agent believes its own unexecuted intentions.

Scope (honest): predicates cover filesystem effects first — the
archetypal verifiable action lane. The predicate registry is the
extension seam for scripting/GUI observation lanes.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Aura.Grounding.Sensorimotor")

# Predicate: () -> (verified: bool, detail: str). Built BEFORE execution,
# closed over expected world-state, checked against reality AFTER.
Predicate = Callable[[], tuple[bool, str]]


def _file_predicate(params: dict[str, Any]) -> Predicate | None:
    action = str(params.get("action") or params.get("operation") or "").lower()
    path = str(params.get("path") or params.get("full_path") or "").strip()
    if not path:
        return None
    expanded = os.path.expanduser(path)

    if action in {"write", "create", "append", "save", "export"}:
        content = params.get("content") or params.get("text") or ""
        min_size = max(0, len(str(content).encode("utf-8", errors="replace")) // 2)

        def _written() -> tuple[bool, str]:
            if not os.path.exists(expanded):
                return False, f"predicted file absent: {expanded}"
            size = os.path.getsize(expanded)
            if size < min_size:
                return False, f"file exists but only {size}B (predicted ≥{min_size}B)"
            return True, f"verified: {expanded} exists with {size}B"

        return _written

    if action in {"mkdir", "create_folder", "make_directory"}:
        def _made() -> tuple[bool, str]:
            if os.path.isdir(expanded):
                return True, f"verified: directory {expanded} exists"
            return False, f"predicted directory absent: {expanded}"

        return _made

    if action in {"delete", "remove"}:
        def _gone() -> tuple[bool, str]:
            if os.path.exists(expanded):
                return False, f"predicted deletion, but {expanded} still exists"
            return True, f"verified: {expanded} removed"

        return _gone

    return None


_PREDICATE_BUILDERS: dict[str, Callable[[dict[str, Any]], Predicate | None]] = {
    "file_operation": _file_predicate,
}


def register_predicate_builder(
    tool_name: str, builder: Callable[[dict[str, Any]], Predicate | None],
) -> None:
    """Extension seam: new observation lanes plug in per tool."""
    _PREDICATE_BUILDERS[tool_name] = builder


def build_predicate(tool_name: str, params: Any) -> Predicate | None:
    builder = _PREDICATE_BUILDERS.get(str(tool_name or ""))
    if builder is None or not isinstance(params, dict):
        return None
    try:
        return builder(params)
    except (TypeError, ValueError, AttributeError) as exc:
        logger.debug("Predicate build failed for %s: %s", tool_name, exc)
        return None


def ground_result(
    tool_name: str,
    params: Any,
    result: Any,
    predicate: Predicate,
    receipt_id: str | None,
) -> None:
    """Verify reality against the pre-execution prediction; learn from it."""
    claimed_ok = bool(isinstance(result, dict) and result.get("ok", result.get("success")))
    try:
        verified, detail = predicate()
    except (OSError, TypeError, ValueError) as exc:
        verified, detail = False, f"observation failed: {exc}"

    observed = 1.0 if verified else 0.0
    if receipt_id:
        try:
            from core.cognition.outcome_ledger import get_outcome_ledger

            get_outcome_ledger().resolve(receipt_id, observed, note=detail[:180])
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Outcome ledger resolve skipped: %s", exc)

    if claimed_ok and not verified:
        # The confabulated-action class: the tool said yes, the world says no.
        logger.warning(
            "ACTION-CLAIM-MISMATCH: %s claimed success but %s", tool_name, detail,
        )
        try:
            from core.resilience.fault_taxonomy import get_fault_registry

            get_fault_registry().record_fault(
                "ACTION-CLAIM-MISMATCH",
                subsystem=f"grounding.{tool_name}",
                details=detail[:200],
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Fault registry unavailable for claim mismatch: %s", exc)
    elif verified:
        logger.debug("Grounded %s: %s", tool_name, detail)


def open_grounding(tool_name: str, params: Any) -> tuple[Predicate | None, str | None]:
    """Pre-execution half: build the predicate and commit the expectation."""
    predicate = build_predicate(tool_name, params)
    if predicate is None:
        return None, None
    receipt_id: str | None = None
    try:
        from core.cognition.outcome_ledger import get_outcome_ledger

        receipt_id = get_outcome_ledger().open(
            action=f"sensorimotor:{tool_name}:{str(params.get('action', ''))[:24]}",
            expected=0.9,  # calibrated over time by the ledger itself
            category="sensorimotor",
            horizon_s=300.0,
            context={"tool": tool_name},
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Outcome ledger open skipped: %s", exc)
    return predicate, receipt_id
