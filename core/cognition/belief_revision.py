"""core/cognition/belief_revision.py — retired duplicate belief engine.

Aura had two belief-revision engines. One of them ran.

    core/epistemics/belief_revision.py    canonical. Started during boot by
                                          BootAutonomyMixin, registered as the
                                          `belief_revision_engine` service, and
                                          referenced by the memory synthesizer.
                                          639 lines: domains, PLN evidence
                                          mass, consistency checking, logical
                                          conflict resolution, atomspace
                                          mirroring, persistence with
                                          quarantine on an unreadable store.

    core/cognition/belief_revision.py     this file. 338 lines exposing
                                          get_belief_engine(), reachable from
                                          nothing at all.

The duplicate was also less honest than it looked. Its module docstring
declared four key behaviours — evidence adjusts confidence, contradictory
evidence triggers revision, beliefs decay if unreinforced, the LLM resolves
contradictions — and only the first existed. Its class docstring went further,
claiming "contradictions are detected via semantic similarity + LLM judgment",
and no such code was anywhere in the file. An epistemic engine that cannot be
argued with is a list of things it was once told.

So the resolution is not to wire this one up. Two belief engines is the problem
the affect retirement already solved once: a second opinion presented as the
first, from a model nothing else reads. What was worth keeping was the one
capability the canonical engine genuinely lacked — decay — and that has been
implemented there, damped by the evidence mass the canonical engine already
tracks, so a conclusion drawn from twenty observations fades far more slowly
than one drawn from an offhand remark.

Worth recording separately: ``core/brain/llm/context_assembler.py`` tells the
model, in the live system prompt, that "beliefs in your context carry a
confidence, and that number is part of what you know". That instruction is
about the canonical engine. It was accurate for the wired engine and would have
been quietly wrong about this one.

Retirement follows the pattern of core/global_workspace.py and
core/affect/emotion_engine.py: keep the module importable, re-export the
canonical names, let one implementation exist.
"""

from __future__ import annotations

from typing import Any

from core.epistemics.belief_revision import (
    Belief,
    BeliefDomain,
    BeliefRevisionEngine,
    get_belief_revision_engine,
)

__all__ = [
    "Belief",
    "BeliefDomain",
    "BeliefRevisionEngine",
    "get_belief_revision_engine",
    "get_belief_engine",
]


def get_belief_engine(knowledge_graph: Any = None, brain: Any = None) -> Any:
    """The one belief engine, from the container rather than a fresh instance.

    Constructing an engine here is what created the second one. A caller that
    wants beliefs wants the registered singleton the rest of the runtime reads;
    if it is not registered they want to know that, rather than be handed a
    private engine whose beliefs nothing else shares.

    The ``knowledge_graph`` and ``brain`` arguments are accepted and ignored so
    that any legacy call site keeps working — the canonical engine resolves its
    own dependencies.
    """
    from core.container import ServiceContainer

    registered = ServiceContainer.get("belief_revision_engine", default=None)
    return registered if registered is not None else get_belief_revision_engine()
