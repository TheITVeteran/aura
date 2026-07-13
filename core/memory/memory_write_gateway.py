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

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import (
    async_durable_replace,
    async_durable_unlink,
    read_json_envelope,
)
from core.runtime.errors import record_degradation
from core.runtime.gateways import (
    MemoryWriteGateway as MemoryWriteGatewayBase,
)
from core.runtime.gateways import (
    MemoryWriteReceipt as MemoryWriteReceiptDC,
)
from core.runtime.gateways import (
    MemoryWriteRequest,
)
from core.runtime.receipts import (
    MemoryWriteReceipt,
    get_receipt_store,
)

logger = logging.getLogger("Aura.MemoryWriteGateway")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
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


def _default_memory_root() -> Path:
    test_runtime_root = str(os.environ.get("AURA_TEST_RUNTIME_ROOT") or "").strip()
    if test_runtime_root:
        return Path(test_runtime_root) / "memory"
    return Path.home() / ".aura" / "memory"


class ConcreteMemoryWriteGateway(MemoryWriteGatewayBase):
    """Single canonical memory write authority.

    Each write is staged through atomic_writer and recorded as a
    MemoryWriteReceipt in the central receipt store. A governance
    authority must approve before persistence.
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        governance_decide: Callable[..., Any] | None = None,
    ):
        self.root = Path(root) if root else _default_memory_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._governance = governance_decide
        self._mutation_lock = asyncio.Lock()
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
        family = _safe_component((request.metadata or {}).get("family", "episodic"), "family")
        record_id = _safe_component(
            (request.metadata or {}).get("record_id") or f"mem-{uuid.uuid4()}",
            "record_id",
        )
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
        from core.runtime.atomic_writer import async_atomic_write_json

        async with self._mutation_lock:
            existed, old_payload = await asyncio.to_thread(_read_memory_payload, target)
            await async_atomic_write_json(
                target,
                payload,
                schema_version=schema_version,
                schema_name=f"memory.{family}",
            )
            bytes_written = await asyncio.to_thread(lambda: target.stat().st_size)

            receipt_store = get_receipt_store()
            try:
                emitted_receipt = await asyncio.to_thread(
                    receipt_store.emit,
                    MemoryWriteReceipt(
                        receipt_id=f"memwr-{uuid.uuid4()}",
                        cause=request.cause,
                        family=family,
                        record_id=record_id,
                        bytes_written=bytes_written,
                        schema_version=schema_version,
                        governance_receipt_id=gov_receipt_id or request.receipt_id,
                        metadata={"path": str(target)},
                    ),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                try:
                    await _restore_memory_payload(
                        target,
                        existed=existed,
                        payload=old_payload,
                        schema_version=schema_version,
                        schema_name=f"memory.{family}",
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                    record_degradation(
                        "memory_write_gateway",
                        rollback_exc,
                        severity="critical",
                        action="memory receipt failed and compensating rollback also failed",
                        enforce_failure_policy=False,
                        extra={"target": str(target), "family": family, "record_id": record_id},
                    )
                    raise RuntimeError(
                        "memory_write_receipt_failed_and_rollback_failed:"
                        f"{exc}; rollback={rollback_exc}"
                    ) from rollback_exc
                raise RuntimeError(f"memory_write_receipt_failed_rolled_back:{exc}") from exc
        return MemoryWriteReceiptDC(
            record_id=record_id,
            receipt_id=emitted_receipt.receipt_id,
            bytes_written=bytes_written,
            schema_version=schema_version,
        )

    async def quarantine(self, record_id: str, reason: str) -> None:
        record_id = _safe_component(record_id, "record_id")
        located = await asyncio.to_thread(_find_memory_record, self.root, record_id)
        if located is None:
            return
        family, candidate = located
        request = MemoryWriteRequest(
            content="",
            metadata={
                "family": family,
                "record_id": record_id,
                "source": "memory_quarantine",
                "quarantine": True,
            },
            cause="memory_write_gateway.quarantine",
        )
        approved, gov_receipt_id = await self._authorize(family, request)
        if not approved:
            raise PermissionError(
                f"MemoryWriteGateway: governance denied quarantine of '{family}/{record_id}'"
            )
        target = self._quarantine_dir / f"{family}_{record_id}.json"
        async with self._mutation_lock:
            if not candidate.exists():
                return
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            await async_durable_replace(candidate, target)
            try:
                bytes_written = await asyncio.to_thread(lambda: target.stat().st_size)
                await asyncio.to_thread(
                    get_receipt_store().emit,
                    MemoryWriteReceipt(
                        receipt_id=f"memwr-{uuid.uuid4()}",
                        cause=request.cause,
                        family="_quarantine",
                        record_id=f"{family}.{record_id}",
                        bytes_written=bytes_written,
                        schema_version=SCHEMA_VERSIONS["default"],
                        governance_receipt_id=gov_receipt_id,
                        metadata={
                            "path": str(target),
                            "operation": "quarantine",
                            "source_family": family,
                            "reason": str(reason or "")[:500],
                        },
                    ),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                try:
                    await async_durable_replace(target, candidate)
                except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                    record_degradation(
                        "memory_write_gateway",
                        rollback_exc,
                        severity="critical",
                        action="memory quarantine receipt failed and rollback also failed",
                        enforce_failure_policy=False,
                        extra={"source": str(candidate), "target": str(target)},
                    )
                    raise RuntimeError(
                        "memory_quarantine_receipt_failed_and_rollback_failed:"
                        f"{exc}; rollback={rollback_exc}"
                    ) from rollback_exc
                raise RuntimeError(f"memory_quarantine_receipt_failed_rolled_back:{exc}") from exc
        logger.warning(
            "MemoryWriteGateway quarantined record %s/%s: %s",
            family,
            record_id,
            str(reason or "")[:500],
        )

    async def _authorize(
        self,
        family: str,
        request: MemoryWriteRequest,
    ) -> tuple[bool, str | None]:
        from core.governance_context import get_active_governance, require_governance

        active = get_active_governance()
        if active is not None:
            token = require_governance(
                "memory_write_gateway.write",
                strict=True,
                allowed_domains={"memory_write"},
            )
            return True, token.receipt_id
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
                    "memory_metadata": _safe_memory_metadata(request.metadata or {}),
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
                    "content_length": len(str(request.content or "")),
                    "content_sha256": hashlib.sha256(
                        str(request.content or "").encode("utf-8", errors="replace")
                    ).hexdigest(),
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
            receipt_id = decision.get("receipt_id")
            return bool(decision.get("approved")), str(receipt_id) if receipt_id else None
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            receipt_id = getattr(decision, "receipt_id", None)
            return bool(approved()), str(receipt_id) if receipt_id else None
        return bool(decision), None


def _safe_component(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if text in {".", ".."} or not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(
            f"memory {label} must be 1-160 letters, digits, dot, dash, or underscore"
        )
    return text


def _safe_memory_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sensitive_markers = (
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    )
    safe: dict[str, Any] = {}
    for key, value in list(metadata.items())[:64]:
        label = str(key)[:160]
        if any(marker in label.casefold() for marker in sensitive_markers):
            safe[label] = "[REDACTED]"
        elif isinstance(value, (bool, int, float)) or value is None:
            safe[label] = value
        elif isinstance(value, str):
            safe[label] = value[:240]
        elif isinstance(value, (list, tuple, set, frozenset)):
            safe[label] = [str(item)[:120] for item in list(value)[:16]]
        elif isinstance(value, dict):
            safe[label] = f"<mapping:{len(value)}>"
        else:
            safe[label] = f"<{type(value).__qualname__}>"
    return safe


def _read_memory_payload(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {}
    envelope = read_json_envelope(path)
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"memory envelope payload is not a mapping: {path}")
    return True, dict(payload)


async def _restore_memory_payload(
    path: Path,
    *,
    existed: bool,
    payload: dict[str, Any],
    schema_version: int,
    schema_name: str,
) -> None:
    if existed:
        from core.runtime.atomic_writer import async_atomic_write_json

        await async_atomic_write_json(
            path,
            payload,
            schema_version=schema_version,
            schema_name=schema_name,
        )
        return
    await async_durable_unlink(path, missing_ok=True)


def _find_memory_record(root: Path, record_id: str) -> tuple[str, Path] | None:
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("_"):
            continue
        candidate = family_dir / f"{record_id}.json"
        if candidate.exists():
            return family_dir.name, candidate
    return None


_global: ConcreteMemoryWriteGateway | None = None


async def _default_memory_governance_decide(**kwargs: Any) -> dict[str, Any]:
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


def get_memory_write_gateway(*, root: Path | None = None) -> ConcreteMemoryWriteGateway:
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
