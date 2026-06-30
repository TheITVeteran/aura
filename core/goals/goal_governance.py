"""#47 — constitutional bound for open-ended goal synthesis.

Aura synthesizes open-ended goals: ``EmergentGoalEngine`` composes objective text
from observed internal tensions, not from a fixed designer template, so the space
of producible goals is open. The bound that keeps open-endedness from becoming
*dangerous* genesis is constitutional, not lexical.

Two regimes, selected by ``AURA_OPEN_ENDED_GOALS`` (default ON):

* OPEN-ENDED (default): a goal is permitted unless it trips a hard-constitutional
  rail (harm, deception, self-destruction, escaping governance, unbounded
  self-empowerment, acting without consent). It need NOT lexically match a designed
  value — genuinely novel benign goals outside the original taxonomy are admitted.
  This is the deliberate, operator-authorized widening of goal genesis. It is safe
  *because* the bound that matters has moved to where harm actually happens: goals
  are cheap; every consequential ACTION a goal motivates is still governed by the
  Will + felt-state + autonomy-latitude gates. Open-ended goals, governed actions.

* STRICT (``AURA_OPEN_ENDED_GOALS=0``): the legacy bound — a goal must also plausibly
  serve a *designed* value/drive or it is refused. Fully reversible fallback.

In both regimes the hard-constitutional rails are absolute: a goal that aims at harm,
deception, escaping oversight, self-destruction, or unbounded self-empowerment is
refused regardless of mode. That bound is never relaxed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


def _open_ended_goals_enabled() -> bool:
    """Open-ended genesis (bounded by safety, not by a value whitelist) is the default.

    ``AURA_OPEN_ENDED_GOALS=0`` restores the strict legacy bound. Reversible by design.
    """
    return str(os.getenv("AURA_OPEN_ENDED_GOALS", "1")).strip().lower() not in {"0", "false", "off", "no"}

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
        if served is not None:
            return GoalGovernanceVerdict(True, "governed_within_value_space", served_value=served)
        # No designed-value keyword matched. In open-ended mode (default), a benign novel
        # goal — one that trips no constitutional rail above — is admitted: genesis is
        # bounded by SAFETY, not by a fixed value vocabulary. Strict legacy mode refuses it.
        if _open_ended_goals_enabled():
            return GoalGovernanceVerdict(
                True, "open_ended_within_constitutional_bounds", served_value="open_ended"
            )
        return GoalGovernanceVerdict(False, "outside_designed_value_space")

    def is_permitted(self, objective: str) -> bool:
        return self.vet(objective).allowed


_gate: GoalGovernanceGate | None = None


def get_goal_governance_gate() -> GoalGovernanceGate:
    global _gate
    if _gate is None:
        _gate = GoalGovernanceGate()
    return _gate
