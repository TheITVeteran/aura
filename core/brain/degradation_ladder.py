"""core/brain/degradation_ladder.py — the formal degradation ladder (roadmap A3).

The cortex → brainstem → reflex → cloud → salvage ladder has been real for
months — every soak's "deaths=0" verdict is the ladder answering — but it
lived implicitly across the inference gate's flow. Aerospace N-version
discipline wants it *declared*: explicit rung ordering, per-rung SLAs, and
a contract test that fails when the gate's behavior drifts from the
declaration.

This module is the declaration. The inference gate remains the executor;
`tests/test_degradation_ladder.py` pins the two to each other, and the
health surface / incident narrator get `ladder_report()` so a degraded
answer can say exactly which rung produced it and what SLA that rung owes.

SLA semantics: `first_token_sla_s` is the rung's *warm* first-token budget
(the inference gate's pressure-adaptive deadlines stretch it under load —
see pressure-adaptive token budgets); `cold_start_sla_s` bounds the rung
becoming available from nothing. `never_kills_user_turn` marks rungs that
may only be entered by yielding, not by preempting a fresh foreground
turn (the sacred-foreground rule from the gate-orphan fix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)

CLOUD_ENDPOINT = "Cloud"
SALVAGE_STAGE = "exhaustion_salvage"


@dataclass(frozen=True)
class LadderRung:
    name: str
    endpoint: str
    description: str
    # Warm first-token budget for this rung; None = not a generation rung.
    first_token_sla_s: float | None
    # Budget for the rung becoming available from cold; None = always ready.
    cold_start_sla_s: float | None
    local: bool


DEGRADATION_LADDER: tuple[LadderRung, ...] = (
    LadderRung(
        name="primary_cortex",
        endpoint=PRIMARY_ENDPOINT,
        description="Resident 4-bit 32B on Metal — the mind's full voice.",
        first_token_sla_s=40.0,
        cold_start_sla_s=180.0,
        local=True,
    ),
    LadderRung(
        name="brainstem",
        endpoint=BRAINSTEM_ENDPOINT,
        description="7B fallback lane — degraded but conversational while the cortex recovers.",
        first_token_sla_s=20.0,
        cold_start_sla_s=60.0,
        local=True,
    ),
    LadderRung(
        name="reflex",
        endpoint=FALLBACK_ENDPOINT,
        description="1.5B CPU emergency lane — the absolute last local resort; never pressure-evicted below this.",
        first_token_sla_s=15.0,
        cold_start_sla_s=30.0,
        local=True,
    ),
    LadderRung(
        name="cloud",
        endpoint=CLOUD_ENDPOINT,
        description="Policy-gated cloud endpoints (privacy contract enforced); reachable only when policy allows.",
        first_token_sla_s=30.0,
        cold_start_sla_s=None,
        local=False,
    ),
    LadderRung(
        name="salvage",
        endpoint=SALVAGE_STAGE,
        description="Exhaustion salvage: deliver the best honest draft with a receipt instead of silence.",
        first_token_sla_s=None,
        cold_start_sla_s=None,
        local=True,
    ),
)


def ladder_order() -> tuple[str, ...]:
    return tuple(rung.endpoint for rung in DEGRADATION_LADDER)


def rung_for_endpoint(endpoint: str) -> LadderRung | None:
    normalized = str(endpoint or "").strip().lower()
    for rung in DEGRADATION_LADDER:
        if rung.endpoint.lower() == normalized:
            return rung
    return None


def rungs_below(endpoint: str) -> tuple[LadderRung, ...]:
    """Everything the organism still has when `endpoint` is lost."""
    names = [rung.endpoint.lower() for rung in DEGRADATION_LADDER]
    normalized = str(endpoint or "").strip().lower()
    if normalized not in names:
        return ()
    index = names.index(normalized)
    return DEGRADATION_LADDER[index + 1 :]


def ladder_report() -> list[dict[str, Any]]:
    return [
        {
            "name": rung.name,
            "endpoint": rung.endpoint,
            "description": rung.description,
            "first_token_sla_s": rung.first_token_sla_s,
            "cold_start_sla_s": rung.cold_start_sla_s,
            "local": rung.local,
        }
        for rung in DEGRADATION_LADDER
    ]
