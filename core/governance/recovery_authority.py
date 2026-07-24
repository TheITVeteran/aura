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

# Sleep-class restoration: internal work that welfare recovery DEPENDS on.
# The live 7/17-7/21 sessions measured the deadlock this exists to break:
# recovery_drive > 0.6 deferred every consequential action (11,738 defers)
# INCLUDING dream consolidation (2,428 blocks) — but consolidation is the
# mechanism that lowers recovery drive, so the interior life stayed locked
# for entire sessions: no memory writes, no belief updates, no initiative.
# This lane is deliberately narrow: allowlisted sources, internal-only,
# and it exempts ONLY the recovery-drive defer — integrity guards, action
# inhibition, and every external-effect rule still apply in full.
RESTORATIVE_CONSOLIDATION_SOURCES = frozenset(
    {
        "mind_tick.dream_consolidation",
        "memory.consolidate_working_memory",
    }
)

# Consolidation reorganizes internal memory, so the memory_write marker is
# permitted here; everything identity- or world-touching stays prohibited.
_PROHIBITED_RESTORATION_MARKERS = _PROHIBITED_RECOVERY_MARKERS - {"memory_write"}


def is_restorative_consolidation(
    source: Any,
    context: Mapping[str, Any] | None,
) -> bool:
    """Return whether a context is allow-listed sleep-class restoration.

    A generic mutation must not be able to self-label as restoration: the
    source must be on the allowlist, the context must declare itself
    internal-only, and no external-effect marker may be present.
    """

    payload = dict(context or {})
    if str(payload.get("effect_scope")) != "internal_restoration":
        return False
    if not bool(payload.get("no_external_effects")):
        return False
    if any(bool(payload.get(marker)) for marker in _PROHIBITED_RESTORATION_MARKERS):
        return False
    source_name = str(payload.get("source") or source or "").strip().lower()
    return source_name in RESTORATIVE_CONSOLIDATION_SOURCES


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
