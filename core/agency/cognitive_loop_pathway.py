"""Wire the cognitive loop into Aura's live Will (CP244).

The conductor (CP243) runs the full loop for any query. This connects it to
agency: when the ``autonomous_research`` or ``curiosity_drive`` pathway
fires during a pulse, a hook runs the loop over a query drawn from Aura's
current context, and proposes an action based on what it found. It is her
Will invoking the loop in her actual life, instead of an operator hand-
driving a lab pipeline.

The wiring obeys the discipline that has kept the live instance safe all
along -- the same one the halting head follows:

* **Gated OFF by default.** Unless ``AURA_COGNITIVE_LOOP_PATHWAY=1``, this
  registers nothing and the live agency behaves byte-identically. The loop
  goes live only when explicitly enabled, after it has earned it -- exactly
  how retrieval earned its 0->56%.
* **Degrades honestly.** If the memory or LLM organs are not resolvable, the
  loop is not built and the hook proposes nothing; it never fabricates an
  action to look busy. A pathway hook that raises is already caught by
  agency as a degradation, so a failure here cannot break the pulse.
* **Never retains the unverified.** In her real life most queries have no
  oracle. When no verifier organ is available the loop marks its answer
  UNVERIFIED and the learner never fires -- the conductor already enforces
  this, and it is why turning the loop on cannot quietly train Aura on her
  own guesses.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from core.service_names import ServiceNames

COGNITIVE_LOOP_PATHWAY_SCHEMA = "aura.cognitive_loop_pathway.v1"
ENABLE_FLAG = "AURA_COGNITIVE_LOOP_PATHWAY"
# The pathways where running the full loop is a natural act of Will:
# researching something, or resolving a curiosity.
TARGET_PATHWAYS = ("autonomous_research", "curiosity_drive")
# The loop runs real 32B inference. A cooldown keeps it from firing on every
# agency pulse and loading a latency-sensitive live instance -- the same
# rate-limit discipline the other autonomous pathways already use.
COOLDOWN_SECONDS = float(os.environ.get("AURA_COGNITIVE_LOOP_COOLDOWN", "180"))
# Every external call is bounded: an unbounded generate() could hang the
# agency pulse hook. A whole loop (up to max_attempts generations) must also
# not run away, so the cycle carries its own ceiling.
GENERATE_TIMEOUT_S = float(os.environ.get("AURA_COGNITIVE_LOOP_GENERATE_TIMEOUT", "45"))
CYCLE_TIMEOUT_S = float(os.environ.get("AURA_COGNITIVE_LOOP_CYCLE_TIMEOUT", "150"))


def _degrade(exc: BaseException, action: str, *, severity: str = "degraded") -> None:
    """Route a swallowed exception through the canonical degradation channel.

    CLAUDE.md forbids silent catch-alls; every ``except`` here records why it
    fired, at info-level for expected backpressure (timeouts) and degraded
    otherwise, so a live failure is visible rather than hidden behind a
    returned None.
    """
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "cognitive_loop_pathway", exc, severity=severity, action=action,
            enforce_failure_policy=False,
        )
    except Exception:
        # Degradation recording itself must never raise into the pulse.
        pass


def is_enabled() -> bool:
    # ON by default now (owner's decision). Set AURA_COGNITIVE_LOOP_PATHWAY=0
    # to disable. Safe-by-construction: degrades honestly, cannot break the
    # pulse, and never retains unverified output -- so on-by-default cannot
    # corrupt learning, only spend some inference the cooldown bounds.
    return os.environ.get(ENABLE_FLAG, "1") != "0"


class _RouterDeliberator:
    """Adapts the async llm_router.generate to the loop's deliberator seam."""

    def __init__(self, router: Any) -> None:
        self._router = router

    async def deliberate(self, query: str, material: list[str]) -> str:
        blocks = []
        if material:
            blocks.append("Known context:\n" + "\n".join(f"- {m}" for m in material))
        blocks.append(query)
        blocks.append("Work through it step by step, then give your answer.")
        prompt = "\n\n".join(blocks)
        try:
            # Bounded: an unbounded generate() would hang the pulse hook.
            return await asyncio.wait_for(
                self._router.generate(prompt), timeout=GENERATE_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            _degrade(exc, "cognitive-loop deliberation timed out", severity="warning")
            return ""
        except Exception as exc:
            _degrade(exc, "cognitive-loop deliberation failed")
            return ""


def build_live_loop(container: Any = None) -> Any:
    """Build a CognitiveLoop from live organs, or None if they are missing.

    Resolves memory (retrieval) and the LLM router (deliberation) from the
    ServiceContainer. Returns None -- not a crippled loop -- when the organs
    are unavailable, so the caller proposes nothing rather than fabricating.
    """
    from core.learning.cognitive_loop import CognitiveLoop
    from core.learning.facade_retrieval import FacadeRetrieval
    from core.learning.workspace_producers import (
        RetrievalProducer,
        WorkspaceComposer,
    )

    resolve = _resolver(container)
    router = resolve(ServiceNames.LLM_ROUTER)
    if router is None or not hasattr(router, "generate"):
        return None
    facade = resolve(ServiceNames.MEMORY_FACADE)

    producers = []
    if facade is not None and hasattr(facade, "search_sync"):
        producers.append(RetrievalProducer(FacadeRetrieval(facade)))
    composer = WorkspaceComposer(producers=producers)

    try:
        return CognitiveLoop(
            composer=composer,
            deliberator=_RouterDeliberator(router),
            # No live programmatic verifier for open-ended research: answers
            # are UNVERIFIED and never retained. A real verifier organ can be
            # wired here later, and only then does learning switch on.
            verifier=None,
            max_attempts=2,
        )
    except ValueError:
        return None


def _resolver(container: Any):
    if container is not None and hasattr(container, "get"):
        return lambda name: container.get(name, default=None)
    from core.container import ServiceContainer

    return lambda name: ServiceContainer.get(name, default=None)


def _derive_query(agency: Any) -> str:
    """Draw a query from Aura's current context -- her monologue or a goal.

    The loop is data-driven: it does not need a hand-written question, it
    picks up whatever Aura is currently attending to. Returns '' when there
    is nothing to reason about, which makes the hook propose nothing.
    """
    monologue = str(getattr(agency, "_current_monologue", "") or "").strip()
    if len(monologue) > 12:
        return monologue[:400]
    goals = getattr(getattr(agency, "state", None), "goals", None) or []
    for goal in goals:
        text = str((goal or {}).get("goal") or (goal or {}).get("description") or "").strip()
        if len(text) > 8:
            return text[:400]
    return ""


async def cognitive_loop_provider(
    *, pathway: str, now: float, idle_seconds: float, agency: Any
) -> dict[str, Any] | None:
    """The agency hook: run the loop and propose an action, or nothing."""
    # Cooldown: the loop is real 32B inference, so it must not fire on every
    # pulse. Mark BEFORE running so a slow loop cannot be re-entered by the
    # next pulse while it is still working.
    marks = getattr(agency, "_cognitive_loop_last_run", None)
    if marks is None:
        marks = {}
        try:
            setattr(agency, "_cognitive_loop_last_run", marks)
        except (AttributeError, TypeError):
            pass
    if now - float(marks.get(pathway, 0.0)) < COOLDOWN_SECONDS:
        return None
    loop = build_live_loop()
    if loop is None:
        return None
    query = _derive_query(agency)
    if not query:
        return None
    marks[pathway] = now

    # Run inside the governed maintenance scope the rest of agency uses for
    # internal model work, and under a whole-cycle timeout so a stuck loop
    # cannot wedge the pulse. A failure degrades to "no proposal", never an
    # exception into the pulse.
    try:
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope(
            "cognitive_loop_pathway", domain="state_mutation"
        ):
            result = await asyncio.wait_for(loop.arun(query), timeout=CYCLE_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        _degrade(exc, f"cognitive-loop cycle timed out on {pathway}", severity="warning")
        return None
    except Exception as exc:
        _degrade(exc, f"cognitive-loop cycle failed on {pathway}")
        return None
    if result.answer is None or not str(result.answer).strip():
        return None
    # A verified conclusion is offered with more priority than an unverified
    # musing -- Aura acts on what she has checked more readily than on a hunch.
    priority = 0.55 if result.verified else 0.3
    return {
        "type": "inner_reasoning",
        "source": f"{pathway}:cognitive_loop",
        "content": str(result.answer)[:600],
        "verified": result.verified,
        "attempts": result.attempts,
        "priority": priority,
        "receipt": result.to_receipt(),
    }


def register_if_enabled(agency: Any) -> dict[str, Any]:
    """Register the loop hook on the target pathways, IF enabled.

    Returns a receipt of what was registered. When the flag is off this
    registers nothing and reports so -- the live instance is unchanged.
    """
    if not is_enabled():
        return {
            "schema": COGNITIVE_LOOP_PATHWAY_SCHEMA,
            "enabled": False,
            "registered": [],
        }
    registered = []
    for pathway in TARGET_PATHWAYS:
        try:
            agency.register_pathway_hook(pathway, cognitive_loop_provider)
            registered.append(pathway)
        except (ValueError, TypeError):
            # Unknown pathway or bad provider -> skip it, do not crash agency.
            continue
    return {
        "schema": COGNITIVE_LOOP_PATHWAY_SCHEMA,
        "enabled": True,
        "registered": registered,
    }


__all__ = [
    "COGNITIVE_LOOP_PATHWAY_SCHEMA",
    "ENABLE_FLAG",
    "TARGET_PATHWAYS",
    "build_live_loop",
    "cognitive_loop_provider",
    "is_enabled",
    "register_if_enabled",
]
