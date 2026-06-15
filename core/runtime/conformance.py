"""Aura Conformance v1 — runtime invariant proofs.

The audit lists ten invariants that must hold:

  1. runtime singularity (one runtime owner)
  2. service graph (each service registered once with known aliases)
  3. governance (no consequential action without receipt)
  4. boot readiness (READY impossible until critical probes pass)
  5. persistence (every durable write atomic + schema-versioned + one gateway)
  6. event delivery (delivered, dropped-with-audit, or rejected — never silent)
  7. shutdown ordering (output -> memory -> state -> actors -> model -> bus)
  8. self-repair (patches climb every rung)
  9. launch authority (every mode uses the same boot helper)
 10. strict mode (degraded/fail-open behavior is impossible)

This module exposes runnable check functions for each invariant. Each
function returns ``ConformanceResult`` so the conformance test suite and
the abuse gauntlet runner can both consume the same evidence.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConformanceResult:
    name: str
    ok: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Invariant proofs
# ---------------------------------------------------------------------------


def proof_runtime_singularity(registered: dict[str, Any]) -> ConformanceResult:
    from core.runtime.service_manifest import (
        SERVICE_MANIFEST,
        critical_violations,
        verify_manifest,
    )

    runtime_role = SERVICE_MANIFEST["runtime"]
    crit = critical_violations(
        verify_manifest(
            registered,
            strict=True,
            manifest={"runtime": runtime_role},
        )
    )
    if crit:
        return ConformanceResult(
            "runtime_singularity",
            ok=False,
            detail="; ".join(f"{v.role}: {v.reason}" for v in crit),
        )
    owner = registered.get(runtime_role.canonical_owner)
    if owner is None:
        return ConformanceResult(
            "runtime_singularity",
            ok=False,
            detail=f"runtime owner '{runtime_role.canonical_owner}' not registered",
        )
    owner_type = type(owner)
    resolved_aliases = sorted(
        name
        for name in runtime_role.aliases
        if registered.get(name) is owner
    )
    return ConformanceResult(
        "runtime_singularity",
        ok=True,
        evidence={
            "owner": runtime_role.canonical_owner,
            "owner_type": f"{owner_type.__module__}.{owner_type.__qualname__}",
            "resolved_aliases": resolved_aliases,
        },
    )


def proof_service_graph(registered: dict[str, Any]) -> ConformanceResult:
    from core.runtime.service_manifest import (
        SERVICE_MANIFEST,
        critical_violations,
        verify_manifest,
    )

    violations = verify_manifest(registered, strict=True)
    critical = critical_violations(violations)
    if critical:
        return ConformanceResult(
            "service_graph",
            ok=False,
            detail="; ".join(f"{v.role}: {v.reason}" for v in critical),
        )
    aliases_seen: dict[str, str] = {}
    for role in SERVICE_MANIFEST.values():
        if role.canonical_owner in registered:
            aliases_seen[role.canonical_owner] = role.name
        for alias in role.aliases:
            if alias in registered and alias != role.canonical_owner:
                aliases_seen[alias] = role.name
    return ConformanceResult(
        "service_graph",
        ok=True,
        evidence={"aliases": aliases_seen},
    )


async def proof_governance_receipt(action_runner: Callable[[], Awaitable[Any]]) -> ConformanceResult:
    """``action_runner`` must return a WillTransaction-shaped object after
    performing the consequential action. We assert the transaction has a
    receipt and a recorded result."""
    try:
        txn = await action_runner()
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        return ConformanceResult(
            "governance",
            ok=False,
            detail=f"governed action failed: {exc}",
        )
    if txn is None:
        return ConformanceResult("governance", ok=False, detail="no transaction returned")
    if not bool(getattr(txn, "approved", False)):
        record = getattr(txn, "record", None)
        failure = getattr(record, "failure", None)
        detail = f"transaction was not approved: {failure}" if failure else "transaction was not approved"
        return ConformanceResult("governance", ok=False, detail=detail)
    receipt_id = getattr(txn, "receipt_id", None)
    record = getattr(txn, "record", None)
    has_result = (
        record is not None
        and isinstance(getattr(record, "result", None), dict)
        and bool(record.result)
    )
    if not receipt_id:
        return ConformanceResult("governance", ok=False, detail="missing receipt_id")
    if not has_result:
        return ConformanceResult(
            "governance",
            ok=False,
            detail="action ran without non-empty post-action effect evidence",
        )
    if getattr(record, "finished_at", None) is None:
        return ConformanceResult(
            "governance",
            ok=False,
            detail="transaction returned before its governed context closed",
        )
    if getattr(record, "failure", None):
        return ConformanceResult(
            "governance",
            ok=False,
            detail=f"transaction recorded failure: {record.failure}",
        )
    will = getattr(txn, "_will", None)
    verify_receipt = getattr(will, "verify_receipt", None)
    if not callable(verify_receipt):
        return ConformanceResult(
            "governance",
            ok=False,
            detail="receipt authority verifier unavailable",
        )
    try:
        receipt_verified = bool(verify_receipt(str(receipt_id)))
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        return ConformanceResult(
            "governance",
            ok=False,
            detail=f"receipt authority verification failed: {exc}",
        )
    if not receipt_verified:
        return ConformanceResult(
            "governance",
            ok=False,
            detail="receipt was not recognized by the issuing authority",
        )
    verify_signature = getattr(will, "verify_receipt_signature", None)
    signature_verified: bool | None = None
    if callable(verify_signature):
        try:
            signature_verified = bool(verify_signature(str(receipt_id)))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return ConformanceResult(
                "governance",
                ok=False,
                detail=f"receipt signature verification failed: {exc}",
            )
        if not signature_verified:
            return ConformanceResult(
                "governance",
                ok=False,
                detail="receipt signature was invalid",
            )
    return ConformanceResult(
        "governance",
        ok=True,
        evidence={
            "receipt_id": receipt_id,
            "result": record.result,
            "authority_verified": True,
            "signature_verified": signature_verified,
        },
    )


def proof_boot_readiness(boot_phase: str, critical_probes: dict[str, bool]) -> ConformanceResult:
    """READY must be impossible while any critical probe is failing."""
    normalized_phase = str(boot_phase or "").strip().lower()
    if normalized_phase == "ready" and not critical_probes:
        return ConformanceResult(
            "boot_readiness",
            ok=False,
            detail="READY reached without any critical probe evidence",
        )
    if normalized_phase == "ready" and not all(critical_probes.values()):
        failed = [name for name, ok in critical_probes.items() if not ok]
        return ConformanceResult(
            "boot_readiness",
            ok=False,
            detail=f"READY reached with failing critical probes: {failed}",
        )
    return ConformanceResult(
        "boot_readiness",
        ok=True,
        evidence={"phase": normalized_phase, "probes": critical_probes},
    )


def proof_persistence_atomic(target_dir: Path) -> ConformanceResult:
    """Every persistent file in ``target_dir`` must be either committed
    or absent — no temp leftovers."""
    from core.runtime.atomic_writer import DEFAULT_TEMP_PREFIX

    if not target_dir.exists():
        return ConformanceResult(
            "persistence",
            ok=True,
            evidence={"target_exists": False, "files": 0},
        )
    leftovers = [p.name for p in target_dir.iterdir() if p.name.startswith(DEFAULT_TEMP_PREFIX)]
    if leftovers:
        return ConformanceResult(
            "persistence",
            ok=False,
            detail=f"unfinished atomic temp files present: {leftovers}",
        )
    return ConformanceResult(
        "persistence",
        ok=True,
        evidence={
            "target_exists": True,
            "files": len(list(target_dir.iterdir())),
        },
    )


def proof_event_delivery(
    audit_log: Iterable[dict[str, Any]],
    dispatched: int,
) -> ConformanceResult:
    """Every dispatched event must appear in the audit log as either
    delivered, dropped-with-reason, or explicitly rejected. Silent loss
    is forbidden."""
    accounted_entries: list[dict[str, Any]] = []
    for entry in audit_log:
        status = entry.get("status")
        if status in {"delivered", "dropped", "rejected"} and entry.get("reason") is not None:
            accounted_entries.append(entry)
        elif status == "delivered":
            accounted_entries.append(entry)
    event_ids = [
        str(entry["event_id"])
        for entry in accounted_entries
        if entry.get("event_id") not in (None, "")
    ]
    if event_ids and len(event_ids) != len(accounted_entries):
        return ConformanceResult(
            "event_delivery",
            ok=False,
            detail="mixed identified and unidentified audit entries cannot prove delivery",
        )
    if len(event_ids) != len(set(event_ids)):
        return ConformanceResult(
            "event_delivery",
            ok=False,
            detail="duplicate event_id entries cannot count as distinct delivery evidence",
        )
    accounted = len(event_ids) if event_ids else len(accounted_entries)
    if accounted < dispatched:
        return ConformanceResult(
            "event_delivery",
            ok=False,
            detail=f"only {accounted}/{dispatched} events accounted for in audit log",
        )
    return ConformanceResult(
        "event_delivery",
        ok=True,
        evidence={
            "dispatched": dispatched,
            "accounted": accounted,
            "identity_verified": bool(event_ids),
        },
    )


def proof_shutdown_ordering(observed_phases: list[str]) -> ConformanceResult:
    from core.runtime.shutdown_coordinator import SHUTDOWN_PHASES

    expected = list(SHUTDOWN_PHASES)
    if observed_phases != expected:
        return ConformanceResult(
            "shutdown_ordering",
            ok=False,
            detail=f"expected complete canonical sequence {expected}; observed {observed_phases}",
        )
    return ConformanceResult(
        "shutdown_ordering",
        ok=True,
        evidence={"phases": observed_phases, "complete": True},
    )


async def proof_self_repair(report: Any) -> ConformanceResult:
    from core.runtime.self_repair_ladder import patch_is_acceptable

    if not patch_is_acceptable(report):
        failed = [r.rung for r in getattr(report, "rungs", []) if not r.ok]
        return ConformanceResult(
            "self_repair",
            ok=False,
            detail=f"patch did not pass all rungs (missing/failed: {failed})",
        )
    return ConformanceResult("self_repair", ok=True)


def proof_launch_authority(main_source: str) -> ConformanceResult:
    """Every launch surface must use ``boot_aura_runtime``."""
    if "boot_aura_runtime" not in main_source:
        return ConformanceResult(
            "launch_authority", ok=False, detail="canonical boot helper missing"
        )
    if "create_orchestrator()" in main_source and main_source.count("create_orchestrator()") > 1:
        return ConformanceResult(
            "launch_authority",
            ok=False,
            detail="multiple create_orchestrator() call sites suggest split runtime ownership",
        )
    return ConformanceResult("launch_authority", ok=True)


def proof_strict_mode(strict_violations: list[str]) -> ConformanceResult:
    """Strict mode must never silently degrade. ``strict_violations`` is
    a list of degraded-event reasons that fired during a strict-mode boot.
    Any non-empty list is a failure."""
    if strict_violations:
        return ConformanceResult(
            "strict_mode",
            ok=False,
            detail=f"strict-mode degradations observed: {strict_violations}",
        )
    return ConformanceResult("strict_mode", ok=True)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass
class ConformanceReport:
    results: list[ConformanceResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results) and bool(self.results)

    def failures(self) -> list[ConformanceResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [
                {"name": r.name, "ok": r.ok, "detail": r.detail, "evidence": r.evidence}
                for r in self.results
            ],
        }
