"""Concrete MemoryWriteGateway adapter.

Implements the abstract `core.runtime.gateways.MemoryWriteGateway` and
routes every memory write through:

  1. governance check (fail-closed if no authority is wired)
  2. atomic_writer durability (temp + fsync + rename, schema-versioned)
  3. universal MemoryWriteReceipt emission
  4. optional registration into the existing memory_facade for retrieval

Concrete and load-bearing — flagship modules (BryanModelEngine,
AbstractionEngine, EnhancedMemorySystem) should call this gateway
instead of writing JSON to disk directly.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.runtime.atomic_writer import atomic_write_json
from core.runtime.gateways import (
    MemoryWriteGateway as MemoryWriteGatewayBase,
    MemoryWriteReceipt as MemoryWriteReceiptDC,
    MemoryWriteRequest,
)
from core.runtime.receipts import (
    MemoryWriteReceipt,
    get_receipt_store,
)

logger = logging.getLogger("Aura.MemoryWriteGateway")
_GOVERNANCE_DECISION_ERRORS = (
    AttributeError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


SCHEMA_VERSIONS = {
    "user_model": 1,
    "principle": 1,
    "episodic": 1,
    "skill_memory": 1,
    "movie_session": 1,
    "default": 1,
}


class ConcreteMemoryWriteGateway(MemoryWriteGatewayBase):
    """Single canonical memory write authority.

    Each write is staged through atomic_writer and recorded as a
    MemoryWriteReceipt in the central receipt store. A governance
    authority must approve before persistence.
    """

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        governance_decide: Optional[Callable[..., Any]] = None,
    ):
        self.root = Path(root) if root else (Path.home() / ".aura" / "memory")
        self.root.mkdir(parents=True, exist_ok=True)
        self._governance = governance_decide
        self._quarantine_dir = self.root / "_quarantine"
        self._quarantine_dir.mkdir(exist_ok=True)

    def is_ready(self) -> bool:
        """Health-contract liveness probe for the canonical memory write path."""
        return bool(
            self.root.exists()
            and self.root.is_dir()
            and self._quarantine_dir.exists()
            and self._quarantine_dir.is_dir()
            and callable(self._governance)
        )

    async def write(self, request: MemoryWriteRequest) -> MemoryWriteReceiptDC:
        family = (request.metadata or {}).get("family", "episodic")
        record_id = (request.metadata or {}).get("record_id") or f"mem-{uuid.uuid4()}"
        approved, gov_receipt_id = await self._authorize(family, request)
        if not approved:
            raise PermissionError(
                f"MemoryWriteGateway: governance denied write to family '{family}'"
            )
        target = self.root / family / f"{record_id}.json"
        payload = {
            "content": request.content,
            "metadata": request.metadata or {},
            "cause": request.cause,
            "governance_receipt_id": gov_receipt_id or request.receipt_id,
            "written_at": time.time(),
        }
        schema_version = SCHEMA_VERSIONS.get(family, SCHEMA_VERSIONS["default"])
        # Off-loop write: this is the per-turn durable memory lane; an
        # on-loop fsync here stalls the whole runtime under disk pressure.
        from core.runtime.atomic_writer import async_atomic_write_json

        await async_atomic_write_json(
            target, payload, schema_version=schema_version, schema_name=f"memory.{family}"
        )
        bytes_written = target.stat().st_size

        receipt_store = get_receipt_store()
        receipt_store.emit(
            MemoryWriteReceipt(
                receipt_id=f"memwr-{uuid.uuid4()}",
                cause=request.cause,
                family=family,
                record_id=record_id,
                bytes_written=bytes_written,
                schema_version=schema_version,
                governance_receipt_id=gov_receipt_id or request.receipt_id,
                metadata={"path": str(target)},
            )
        )
        return MemoryWriteReceiptDC(
            record_id=record_id,
            receipt_id=gov_receipt_id or request.receipt_id or "rcpt-pending",
            bytes_written=bytes_written,
            schema_version=schema_version,
        )

    async def quarantine(self, record_id: str, reason: str) -> None:
        # Find the record across families.
        for family_dir in self.root.iterdir():
            if not family_dir.is_dir() or family_dir.name.startswith("_"):
                continue
            candidate = family_dir / f"{record_id}.json"
            if candidate.exists():
                target = self._quarantine_dir / f"{family_dir.name}_{record_id}.json"
                candidate.replace(target)
                logger.warning(
                    "MemoryWriteGateway quarantined record %s/%s: %s",
                    family_dir.name, record_id, reason,
                )
                return

    async def _authorize(self, family: str, request: MemoryWriteRequest):
        if self._governance is None:
            logger.warning(
                "MemoryWriteGateway has no governance authority; denying write to family '%s' (fail-closed).",
                family,
            )
            return False, None
        try:
            decision = self._governance(
                domain="memory_write",
                action=family,
                cause=request.cause,
                context={
                    "family": family,
                    "record_id": (request.metadata or {}).get("record_id"),
                    "memory_type": str(family or "").strip().lower(),
                    "memory_source": str((request.metadata or {}).get("source") or "").strip().lower().replace("-", "_"),
                    "memory_metadata": dict(request.metadata or {}),
                    "explicit_observational_memory_write": bool(
                        (request.metadata or {}).get("explicit_memory_request")
                        or (request.metadata or {}).get("session_memory_pin")
                    ),
                    "user_facing_memory_write": bool(
                        (request.metadata or {}).get("explicit_memory_request")
                        or (request.metadata or {}).get("session_memory_pin")
                        or (request.metadata or {}).get("source") in {"user", "chat_api", "desktop_ui", "session_memory_pin"}
                    ),
                    "high_risk_memory_write": bool(
                        (request.metadata or {}).get("belief_update")
                        or (request.metadata or {}).get("identity_rewrite")
                        or (request.metadata or {}).get("self_model_write")
                    ),
                    "objective": str(request.content or "")[:400],
                },
            )
            if asyncio.iscoroutine(decision):
                decision = await decision
        except _GOVERNANCE_DECISION_ERRORS as exc:
            record_degradation('memory_write_gateway', exc)
            logger.warning(
                "MemoryWriteGateway governance call failed; denying write (fail-closed): %s",
                exc,
            )
            return False, None
        if isinstance(decision, dict):
            return bool(decision.get("approved")), decision.get("receipt_id")
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            return bool(approved()), getattr(decision, "receipt_id", None)
        return bool(decision), None


_global: Optional[ConcreteMemoryWriteGateway] = None


async def _default_memory_governance_decide(**kwargs: Any) -> Dict[str, Any]:
    from core.governance.will_client import WillClient, WillRequest

    decision = await WillClient().decide_async(
        WillRequest(
            content=f"memory write:{kwargs.get('action', 'unknown')}",
            source="memory_write_gateway",
            domain="memory_write",
            priority=0.7,
            context=dict(kwargs),
        )
    )
    return {
        "approved": WillClient.is_approved(decision),
        "receipt_id": getattr(decision, "receipt_id", None),
    }


def get_memory_write_gateway(*, root: Optional[Path] = None) -> ConcreteMemoryWriteGateway:
    """Return the process gateway, honoring explicit roots as contracts.

    A caller passing an explicit root (proof scenarios, sandboxes) must
    get a gateway bound to THAT root — the old accessor silently
    returned the pre-existing singleton, so a unified-scenario write
    'succeeded' into ~/.aura/memory while the scenario's continuity
    check correctly found its own root empty (fake-pass shape). Explicit
    -root callers now get a dedicated instance; the global singleton
    stays reserved for default-root callers and is never hijacked.
    """
    global _global
    if root is not None:
        resolved = Path(root)
        if _global is not None and Path(_global.root) == resolved:
            return _global
        return ConcreteMemoryWriteGateway(
            root=resolved, governance_decide=_default_memory_governance_decide
        )
    if _global is None:
        _global = ConcreteMemoryWriteGateway(root=None, governance_decide=_default_memory_governance_decide)
    return _global


def reset_memory_write_gateway() -> None:
    global _global
    _global = None
