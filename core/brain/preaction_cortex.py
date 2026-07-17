"""Continuous pre-action cortex loop — deliberation wrapped AROUND action.

The architecture gap this closes: Aura's overt action loop already runs
choose → execute → verify → remember, but each stage thought alone. This
organ keeps ONE cognitive thread alive across the consequential-action
cycle:

    propose → REHEARSE (latent episode: predicted effect, preconditions,
    risks) → execute → observe → RECONCILE (discrepancy-driven replanning
    episode seeded with the rehearsal's own conclusion + the observed
    failure evidence)

Continuity is carried honestly: each phase's conclusion becomes a typed
cognitive-context item (source="action_thread") seeding identifiable
workspace slots in the NEXT phase's episode — same cognitive content,
receipted and individually ablatable — and reconciliation conclusions
return to the Global Workspace through the existing GWT coupling, where
replanning competes for broadcast like every other coalition. We do not
pretend to hold worker KV across generations; the thread is the workspace
content plus receipts, which is the part that must survive anyway.

Discrepancy is OBJECTIVE: reconciliation fires only on transport failure
or a failed effect verification — never on fuzzy text similarity. Matching
predictions are recorded, not re-deliberated.

Everything is defensive and bounded: no latent service, a busy generation
gate, or a kill switch (AURA_PREACTION_RLC=0) ⇒ the action proceeds
exactly as before, with a receipt saying deliberation was skipped and why.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("Aura.PreActionCortex")

PREACTION_SCHEMA = "aura.preaction_cortex.v1"

# Domains whose side effects are consequential enough to buy deliberation.
CONSEQUENTIAL_DOMAINS = frozenset(
    {
        "external_action",
        "network_call",
        "cloud_call",
        "ci_cd",
        "self_modification",
        "environment_action",
    }
)

_MAX_THREAD_ITEMS = 3
_MAX_ITEM_CHARS = 400
_REHEARSAL_TIMEOUT_S = 45.0
_RECONCILE_TIMEOUT_S = 60.0


def _enabled() -> bool:
    return os.environ.get("AURA_PREACTION_RLC", "1") != "0"


def _latent_service() -> Any:
    try:
        from core.runtime.service_registry import get_runtime_service
        from core.service_names import ServiceNames

        return get_runtime_service(ServiceNames.LATENT_CORTEX, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


class PreActionCortexThread:
    """One cognitive thread across a single consequential action."""

    def __init__(
        self,
        *,
        domain: str,
        action_name: str,
        request_digest: str = "",
    ) -> None:
        self.domain = str(domain)
        self.action_name = str(action_name)[:120]
        self.request_digest = str(request_digest)[:64]
        self.created_at = time.time()
        self.thread_items: list[dict[str, str]] = []
        self.rehearsal: dict[str, Any] = {}
        self.reconciliation: dict[str, Any] = {}

    # ── Thread continuity ────────────────────────────────────────────────
    def _remember(self, phase: str, conclusion: str) -> None:
        text = f"[{phase}] {conclusion}".strip()[:_MAX_ITEM_CHARS]
        if not text:
            return
        self.thread_items.append({"source": "action_thread", "text": text})
        del self.thread_items[:-_MAX_THREAD_ITEMS]

    def _context(self) -> list[dict[str, str]] | None:
        return [dict(item) for item in self.thread_items] or None

    # ── Phase 1: rehearsal ───────────────────────────────────────────────
    async def rehearse(
        self,
        *,
        action_summary: str,
        expectation_objective: str,
        stakes: float = 0.7,
    ) -> dict[str, Any]:
        """Deliberate the proposed action BEFORE it runs.

        The episode's objective asks for exactly the trio the loop needs:
        predicted observable effect, preconditions that must already hold,
        and the failure modes worth watching. The conclusion seeds the
        reconciliation phase's slots.
        """
        receipt: dict[str, Any] = {
            "schema": PREACTION_SCHEMA,
            "phase": "rehearsal",
            "action_name": self.action_name,
            "domain": self.domain,
            "ran": False,
        }
        if not _enabled():
            receipt["skip_reason"] = "disabled:AURA_PREACTION_RLC=0"
            self.rehearsal = receipt
            return receipt
        service = _latent_service()
        if service is None:
            receipt["skip_reason"] = "latent_cortex_absent"
            self.rehearsal = receipt
            return receipt
        objective = (
            "Before I take this action, think it through.\n"
            f"Action: {str(action_summary)[:400]}\n"
            f"Intended outcome: {str(expectation_objective)[:400]}\n"
            "State: (1) the precise observable effect I predict, "
            "(2) the preconditions that must already hold, "
            "(3) the most likely failure mode and what it would look like."
        )
        try:
            result = await service.deep_reason(
                objective,
                stakes=stakes,
                uncertainty=0.6,
                domain="action_rehearsal",
                timeout_s=_REHEARSAL_TIMEOUT_S,
                require_full_stack=False,
                foreground_request=True,
                cognitive_context=self._context(),
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as exc:
            receipt["skip_reason"] = f"episode_failed:{type(exc).__name__}"
            self.rehearsal = receipt
            return receipt
        if not isinstance(result, dict) or not result.get("ok"):
            reason = ""
            if isinstance(result, dict):
                reason = str(result.get("reason") or "")
            receipt["skip_reason"] = f"episode_refused:{reason or 'unknown'}"
            self.rehearsal = receipt
            return receipt
        prediction = str(result.get("text") or "").strip()
        episode_receipt = result.get("receipt") or {}
        receipt.update(
            {
                "ran": True,
                "prediction": prediction[:800],
                "episode_id": str(episode_receipt.get("episode_id") or ""),
                "steps_taken": episode_receipt.get("steps_taken"),
                "honest_flags": list(episode_receipt.get("honest_flags") or []),
            }
        )
        self._remember("rehearsal", prediction)
        self.rehearsal = receipt
        return receipt

    # ── Phase 2: reconciliation ──────────────────────────────────────────
    async def reconcile(self, action_result: dict[str, Any]) -> dict[str, Any]:
        """Compare prediction with reality; replan only on OBJECTIVE failure.

        Transport failure or unverified effect after transport success are
        the discrepancy triggers. The replanning episode is seeded with the
        rehearsal's own conclusion plus the observed evidence — the same
        cognitive thread, revised by reality — and its conclusion competes
        for Global Workspace broadcast via the standard RLC → GWT coupling.
        """
        transport = bool(action_result.get("transport_succeeded"))
        verified = action_result.get("effect_verified") is True
        receipt: dict[str, Any] = {
            "schema": PREACTION_SCHEMA,
            "phase": "reconciliation",
            "action_name": self.action_name,
            "domain": self.domain,
            "ran": False,
            "discrepancy": (not transport) or (transport and not verified),
            "transport_succeeded": transport,
            "effect_verified": verified,
        }
        if not receipt["discrepancy"]:
            receipt["skip_reason"] = "prediction_confirmed"
            self.reconciliation = receipt
            return receipt
        if not _enabled():
            receipt["skip_reason"] = "disabled:AURA_PREACTION_RLC=0"
            self.reconciliation = receipt
            return receipt
        service = _latent_service()
        if service is None:
            receipt["skip_reason"] = "latent_cortex_absent"
            self.reconciliation = receipt
            return receipt
        error = str(action_result.get("error") or "")[:300]
        status = str(action_result.get("status") or "")[:80]
        evidence = (
            f"status={status or 'unknown'}"
            + (f"; error={error}" if error else "")
            + (
                "; transport succeeded but the effect was never verified"
                if transport and not verified
                else ""
            )
        )
        self._remember("observed", evidence)
        objective = (
            f"My action did not go as predicted.\n"
            f"Action: {self.action_name}\n"
            f"Observed: {evidence}\n"
            "Diagnose the first divergence between my prediction and what "
            "happened, then state the revised plan: retry as-is, retry "
            "changed (say what changes), or abandon (say why)."
        )
        try:
            result = await service.deep_reason(
                objective,
                stakes=0.8,
                uncertainty=0.7,
                domain="action_reconciliation",
                timeout_s=_RECONCILE_TIMEOUT_S,
                require_full_stack=False,
                foreground_request=True,
                cognitive_context=self._context(),
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as exc:
            receipt["skip_reason"] = f"episode_failed:{type(exc).__name__}"
            self.reconciliation = receipt
            return receipt
        if not isinstance(result, dict) or not result.get("ok"):
            reason = ""
            if isinstance(result, dict):
                reason = str(result.get("reason") or "")
            receipt["skip_reason"] = f"episode_refused:{reason or 'unknown'}"
            self.reconciliation = receipt
            return receipt
        replan = str(result.get("text") or "").strip()
        episode_receipt = result.get("receipt") or {}
        broadcast = episode_receipt.get("workspace_broadcast") or {}
        receipt.update(
            {
                "ran": True,
                "replan": replan[:800],
                "episode_id": str(episode_receipt.get("episode_id") or ""),
                "workspace_broadcast_submitted": bool(
                    broadcast.get("submitted")
                ),
                "honest_flags": list(episode_receipt.get("honest_flags") or []),
            }
        )
        self._remember("replan", replan)
        self.reconciliation = receipt
        return receipt

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": PREACTION_SCHEMA,
            "action_name": self.action_name,
            "domain": self.domain,
            "request_digest": self.request_digest,
            "rehearsal": dict(self.rehearsal),
            "reconciliation": dict(self.reconciliation),
            "thread_length": len(self.thread_items),
        }


def deliberation_worthy(domain: str) -> bool:
    """Only consequential side-effect domains buy latent deliberation."""
    return str(domain).lower() in CONSEQUENTIAL_DOMAINS


__all__ = [
    "CONSEQUENTIAL_DOMAINS",
    "PREACTION_SCHEMA",
    "PreActionCortexThread",
    "deliberation_worthy",
]
