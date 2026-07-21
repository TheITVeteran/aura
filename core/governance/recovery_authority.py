"""Narrow authority contract for internal homeostatic recovery actions.

Recovery work must remain possible when welfare pressure is high, but a generic
``state_mutation`` must not be able to label itself a repair and bypass the
normal action policy. This module keeps the trusted sources, operations, and
negative constraints in one place for Will, BeingRuntime, and repair callers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ALLOWED_RECOVERY_OPERATIONS: dict[str, frozenset[str]] = {
    "adaptive_immune_system": frozenset(
        {
            "adaptive_immune_behavioral_rule",
            "clear_cache",
            "halt_runaway",
            "patch_proposal",
            "quarantine",
            "reduce_load",
            "restart_component",
            "restore_checkpoint",
            "revoke_tool",
            "schema_migration",
        }
    ),
    "autopoiesis_engine": frozenset(
        {
            "checkpoint",
            "clear_cache",
            "heal",
            "isolate",
            "reduce_load",
            "restart",
        }
    ),
}

_PROHIBITED_RECOVERY_MARKERS = frozenset(
    {
        "belief_update",
        "constitutional_change",
        "desktop_control",
        "external_action",
        "file_write",
        "identity_rewrite",
        "memory_write",
        "network_call",
        "policy_change",
        "public_action",
        "self_modification",
        "social_action",
        "world_affecting",
    }
)


def normalize_recovery_source(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def normalize_recovery_operation(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def build_internal_recovery_context(
    source: str,
    operation: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed context for one allow-listed recovery operation."""

    source_name = normalize_recovery_source(source)
    operation_name = normalize_recovery_operation(operation)
    if operation_name not in _ALLOWED_RECOVERY_OPERATIONS.get(source_name, frozenset()):
        raise ValueError(
            "unrecognized internal recovery operation: "
            f"{source_name or 'unknown'}:{operation_name or 'unknown'}"
        )
    payload = dict(evidence or {})
    if any(bool(payload.get(marker)) for marker in _PROHIBITED_RECOVERY_MARKERS):
        raise ValueError("internal recovery evidence requests a prohibited effect")
    payload.update(
        {
            "source": source_name,
            "internal_recovery_action": True,
            "recovery_operation": operation_name,
            "effect_scope": "internal_runtime_recovery",
            "no_external_effects": True,
        }
    )
    return payload


def is_internal_recovery_context(
    domain: Any,
    context: Mapping[str, Any] | None,
) -> bool:
    """Return whether a context is an allow-listed, internal-only repair."""

    domain_name = str(getattr(domain, "value", domain) or "").strip().lower()
    if domain_name != "state_mutation":
        return False
    payload = dict(context or {})
    if not bool(payload.get("internal_recovery_action")):
        return False
    if payload.get("effect_scope") != "internal_runtime_recovery":
        return False
    if not bool(payload.get("no_external_effects")):
        return False
    if any(bool(payload.get(marker)) for marker in _PROHIBITED_RECOVERY_MARKERS):
        return False
    source_name = normalize_recovery_source(payload.get("source"))
    operation_name = normalize_recovery_operation(payload.get("recovery_operation"))
    return operation_name in _ALLOWED_RECOVERY_OPERATIONS.get(source_name, frozenset())
