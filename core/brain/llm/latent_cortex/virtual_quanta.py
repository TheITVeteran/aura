"""Transient virtual compute quanta for recurrence-native reasoning.

This is the concrete version of the "virtual particles" idea: an episode may
create short-lived, subject-steered reasoning probes that borrow from the same
audited compute economy as recurrence and fast weights. A quantum is not a
durable belief, not a prompt decoration, and not a hidden fallback. It must:

* name where it came from and what subject it is allowed to touch;
* charge or reserve a bounded compute budget before use;
* expire automatically after a tiny TTL;
* publish contribution evidence before it can be credited; and
* erase to an empty payload with a machine-checkable certificate.

The engine can later use these quanta to seed slots, verifier probes, retrieval
directions, or fast-weight subspaces. This module intentionally stays MLX-free
so the contract can be tested without a resident model.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.types import ComputeBudget

VIRTUAL_QUANTA_SCHEMA = "aura.latent_cortex.virtual_quanta.v1"
VIRTUAL_QUANTA_RECEIPT_SCHEMA = "aura.latent_cortex.virtual_quanta.receipt.v1"

MAX_QUANTA_PER_EPISODE = 64
MAX_STEERING_TAGS = 12
MAX_SUBJECT_CHARS = 120
MAX_PAYLOAD_CHARS = 2000
MAX_TTL_STEPS = 16
MAX_LAYER_APPS_PER_QUANTUM = 250_000


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _clean_label(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise ValueError("virtual quantum labels must be non-empty")
    return text[:limit]


def _clean_tags(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError("steering_tags must be an iterable of strings")
    tags: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = " ".join(str(raw or "").lower().split()).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag[:64])
        seen.add(tag)
        if len(tags) >= MAX_STEERING_TAGS:
            break
    return tuple(tags)


@dataclass
class VirtualComputeQuantum:
    """One transient unit of subject-steered scratch cognition."""

    quantum_id: str
    subject: str
    source: str
    steering_tags: tuple[str, ...]
    payload: str
    layer_apps_reserved: int
    ttl_steps: int
    created_step: int
    created_monotonic: float = field(default_factory=time.monotonic)
    uses: int = 0
    contribution_score: float | None = None
    erased: bool = False
    erased_monotonic: float | None = None
    _payload_digest_at_create: str = ""

    def __post_init__(self) -> None:
        if not self._payload_digest_at_create:
            self._payload_digest_at_create = hashlib.sha256(
                self.payload.encode("utf-8")
            ).hexdigest()

    @property
    def expires_step(self) -> int:
        return int(self.created_step + self.ttl_steps)

    def expired(self, *, step: int) -> bool:
        return int(step) >= self.expires_step

    def usable(self, *, step: int) -> bool:
        return not self.erased and not self.expired(step=step)

    def mark_used(self, *, step: int) -> None:
        if not self.usable(step=step):
            raise RuntimeError("virtual quantum is expired or erased")
        self.uses += 1

    def record_contribution(self, score: float) -> None:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("contribution score must be finite")
        self.contribution_score = max(-1.0, min(1.0, value))

    def erase(self) -> dict[str, Any]:
        prior_digest = hashlib.sha256(self.payload.encode("utf-8")).hexdigest()
        self.payload = ""
        self.erased = True
        self.erased_monotonic = time.monotonic()
        return {
            "quantum_id": self.quantum_id,
            "erased": True,
            "prior_payload_sha256": prior_digest,
            "payload_empty": self.payload == "",
            "erased_monotonic": round(float(self.erased_monotonic), 6),
        }

    def to_receipt(self) -> dict[str, Any]:
        return {
            "quantum_id": self.quantum_id,
            "subject": self.subject,
            "source": self.source,
            "steering_tags": list(self.steering_tags),
            "payload_sha256": self._payload_digest_at_create,
            "payload_erased": self.erased and self.payload == "",
            "layer_apps_reserved": self.layer_apps_reserved,
            "ttl_steps": self.ttl_steps,
            "created_step": self.created_step,
            "expires_step": self.expires_step,
            "uses": self.uses,
            "contribution_score": self.contribution_score,
        }


class VirtualComputeQuantaLedger:
    """Episode-scoped allocator and eraser for virtual compute quanta."""

    def __init__(
        self,
        *,
        budget: ComputeBudget,
        max_quanta: int = MAX_QUANTA_PER_EPISODE,
    ) -> None:
        if not isinstance(budget, ComputeBudget):
            raise TypeError("budget must be a ComputeBudget")
        if isinstance(max_quanta, bool) or not isinstance(max_quanta, int):
            raise TypeError("max_quanta must be an integer")
        if max_quanta <= 0:
            raise ValueError("max_quanta must be positive")
        self.budget = budget
        self.max_quanta = min(max_quanta, MAX_QUANTA_PER_EPISODE)
        self._quanta: dict[str, VirtualComputeQuantum] = {}
        self._erase_receipts: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []

    def allocate(
        self,
        *,
        subject: Any,
        source: Any,
        payload: Any,
        steering_tags: Any = (),
        layer_apps: int,
        ttl_steps: int,
        step: int = 0,
    ) -> VirtualComputeQuantum:
        if len(self._quanta) >= self.max_quanta:
            raise RuntimeError("virtual quantum episode limit exhausted")
        if isinstance(layer_apps, bool) or not isinstance(layer_apps, int):
            raise TypeError("layer_apps must be an integer")
        if layer_apps <= 0 or layer_apps > MAX_LAYER_APPS_PER_QUANTUM:
            raise ValueError("layer_apps outside virtual quantum bounds")
        if isinstance(ttl_steps, bool) or not isinstance(ttl_steps, int):
            raise TypeError("ttl_steps must be an integer")
        if ttl_steps <= 0 or ttl_steps > MAX_TTL_STEPS:
            raise ValueError("ttl_steps outside virtual quantum bounds")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")

        subject_text = _clean_label(subject, limit=MAX_SUBJECT_CHARS)
        source_text = _clean_label(source, limit=120)
        payload_text = str(payload or "")[:MAX_PAYLOAD_CHARS]
        if not payload_text.strip():
            raise ValueError("virtual quantum payload must be non-empty")
        tags = _clean_tags(steering_tags)

        self.budget.charge_layer_apps(layer_apps)
        identity = {
            "schema": VIRTUAL_QUANTA_SCHEMA,
            "subject": subject_text,
            "source": source_text,
            "steering_tags": list(tags),
            "payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
            "layer_apps": layer_apps,
            "ttl_steps": ttl_steps,
            "step": step,
            "ordinal": len(self._quanta),
        }
        quantum_id = "vq-" + _digest(identity)[:24]
        quantum = VirtualComputeQuantum(
            quantum_id=quantum_id,
            subject=subject_text,
            source=source_text,
            steering_tags=tags,
            payload=payload_text,
            layer_apps_reserved=layer_apps,
            ttl_steps=ttl_steps,
            created_step=step,
        )
        self._quanta[quantum_id] = quantum
        self._events.append({"event": "allocated", **quantum.to_receipt()})
        return quantum

    def use(self, quantum_id: str, *, step: int) -> str:
        quantum = self._quanta[str(quantum_id)]
        quantum.mark_used(step=step)
        self._events.append(
            {
                "event": "used",
                "quantum_id": quantum.quantum_id,
                "step": int(step),
                "uses": quantum.uses,
            }
        )
        return quantum.payload

    def record_contribution(self, quantum_id: str, *, score: float) -> None:
        quantum = self._quanta[str(quantum_id)]
        quantum.record_contribution(score)
        self._events.append(
            {
                "event": "contribution",
                "quantum_id": quantum.quantum_id,
                "score": quantum.contribution_score,
            }
        )

    def erase_expired(self, *, step: int) -> list[dict[str, Any]]:
        erased: list[dict[str, Any]] = []
        for quantum in self._quanta.values():
            if not quantum.erased and quantum.expired(step=step):
                receipt = quantum.erase()
                erased.append(receipt)
                self._erase_receipts.append(receipt)
                self._events.append({"event": "erased_expired", **receipt})
        return erased

    def erase_all(self) -> list[dict[str, Any]]:
        erased: list[dict[str, Any]] = []
        for quantum in self._quanta.values():
            if not quantum.erased:
                receipt = quantum.erase()
                erased.append(receipt)
                self._erase_receipts.append(receipt)
                self._events.append({"event": "erased_all", **receipt})
        return erased

    def receipt(self) -> dict[str, Any]:
        open_quanta = [
            quantum.to_receipt()
            for quantum in self._quanta.values()
            if not quantum.erased
        ]
        return {
            "schema": VIRTUAL_QUANTA_RECEIPT_SCHEMA,
            "allocated": len(self._quanta),
            "open": len(open_quanta),
            "erased": len(self._erase_receipts),
            "budget": self.budget.to_receipt(),
            "quanta": [quantum.to_receipt() for quantum in self._quanta.values()],
            "open_quanta": open_quanta,
            "erase_receipts": [dict(row) for row in self._erase_receipts],
            "events_sha256": _digest(self._events),
        }


__all__ = [
    "VIRTUAL_QUANTA_RECEIPT_SCHEMA",
    "VIRTUAL_QUANTA_SCHEMA",
    "VirtualComputeQuantum",
    "VirtualComputeQuantaLedger",
]
