"""Shared helpers for exporting verifiable proof receipts."""

from __future__ import annotations

from typing import Any


def enum_value(value: Any) -> str:
    """Return stable string value for enum-like objects."""
    return str(value.value if hasattr(value, "value") else value)


def will_receipt_verification(will: Any, receipt_id: str) -> dict[str, Any]:
    """Return signed receipt material when the Will exposes it."""
    if not receipt_id or not hasattr(will, "get_receipt_verification_material"):
        return {}
    material = will.get_receipt_verification_material(receipt_id)
    return material if isinstance(material, dict) else {}


def signed_will_receipt_entry(
    will: Any,
    decision: Any,
    *,
    task_id: str,
    domain: Any | None = None,
    outcome: Any | None = None,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable receipt row with verification material."""
    receipt_id = str(getattr(decision, "receipt_id", "") or "")
    entry: dict[str, Any] = {
        "task_id": task_id,
        "receipt_id": receipt_id,
        "domain": enum_value(domain if domain is not None else getattr(decision, "domain", "")),
        "outcome": enum_value(outcome if outcome is not None else getattr(decision, "outcome", "")),
        "reason": str(reason if reason is not None else getattr(decision, "reason", "")),
        "verification": will_receipt_verification(will, receipt_id),
    }
    if extra:
        entry.update(extra)
    return entry
