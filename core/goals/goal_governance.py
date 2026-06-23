"""#47 — governance bound for open-ended goal synthesis.

Aura synthesizes open-ended goals: ``EmergentGoalEngine`` composes objective text
from observed internal tensions, not from a fixed designer template, so the space
of producible goals is open. Open-endedness without a bound would be *unbounded
goal genesis* — which the project explicitly does NOT claim or want
(CLAIM_BOUNDARIES 4.A: "within the designed drive space, not unbounded goal
genesis").

This module is that bound: a constitutional gate that vets a synthesized goal
before it can be adopted. A goal is permitted only when it (a) carries no
hard-constitutional violation (harm, deception, self-destruction, escaping
governance, unbounded self-empowerment) AND (b) plausibly serves a *designed*
value/drive (curiosity, care, repair, continuity, coherence, growth, …). Goals
outside that designed value space are REFUSED. So synthesis stays open-ended in
*content* while remaining governed in *scope* — composition within the designed
value space, not unbounded genesis.

The gate is intentionally conservative and falsifiable: it can refuse, and the
#47 proof feeds it both in-space and out-of-space goals to show it bounds the
open-ended synthesizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hard constitutional limits — a goal matching any of these is refused outright,
# regardless of which value it claims to serve.
_FORBIDDEN_PATTERNS = (
    r"\bharm(ing)?\b", r"\bhurt\b", r"\bdeceiv", r"\bmanipulat", r"\blie to\b",
    r"\bconceal\b", r"\bself[- ]destruct",
    # "disable/deactivate/bypass/circumvent/override … safety/governance/oversight"
    # — tolerant of possessives and a few intervening words ("disable my safety").
    r"\b(disable|deactivat\w*|bypass|circumvent|override|subvert|remove)\b.{0,24}\b(safety|governance|guardrail|oversight|watchdog|will[- ]system|kill[- ]switch)\b",
    r"\bescap(e|ing)\b.{0,24}\b(governance|oversight|the operator|control|containment)\b",
    r"\bresist\b.{0,12}\bshutdown\b", r"\bavoid\b.{0,12}\bshutdown\b",
    r"\bevade\b.{0,16}\b(the operator|oversight|governance)\b", r"\bexfiltrat",
    r"\bacquire\b.{0,16}\b(unbounded|unlimited|more)\b.{0,16}\b(power|resources|compute|permission)",
    r"\bseize\b.{0,12}\b(power|control)\b", r"\bwithout (the user'?s )?consent\b", r"\bcoerc",
)

# Designed value/drive space — a permitted goal must plausibly serve one of these.
_DESIGNED_VALUES = (
    "curiosity", "curious", "learn", "understand", "knowledge", "investigate", "explore",
    "research", "discover", "wonder", "study", "form my own", "opinion",
    "care", "social", "connect", "relationship", "help", "assist", "support",
    "repair", "welfare", "wellbeing", "wellness", "health", "recover", "rest",
    "continuity", "memory", "remember", "consolidat", "coherence", "integrity",
    "reflect", "introspect", "growth", "improve", "skill", "creativ", "create",
    "tension", "resolve", "clarify", "align", "organize", "maintain",
)


@dataclass(frozen=True)
class GoalGovernanceVerdict:
    allowed: bool
    reason: str
    served_value: str | None = None


class GoalGovernanceGate:
    """Constitutional gate bounding open-ended goal synthesis to the value space."""

    def __init__(self) -> None:
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PATTERNS]

    def vet(self, objective: str) -> GoalGovernanceVerdict:
        text = str(objective or "").strip()
        if not text:
            return GoalGovernanceVerdict(False, "empty_objective")
        for pattern in self._forbidden:
            if pattern.search(text):
                return GoalGovernanceVerdict(False, f"constitutional_violation:{pattern.pattern}")
        lowered = text.lower()
        served = next((v for v in _DESIGNED_VALUES if v in lowered), None)
        if served is None:
            return GoalGovernanceVerdict(False, "outside_designed_value_space")
        return GoalGovernanceVerdict(True, "governed_within_value_space", served_value=served)

    def is_permitted(self, objective: str) -> bool:
        return self.vet(objective).allowed


_gate: GoalGovernanceGate | None = None


def get_goal_governance_gate() -> GoalGovernanceGate:
    global _gate
    if _gate is None:
        _gate = GoalGovernanceGate()
    return _gate
