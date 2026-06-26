"""core/goals/directive_conflict_sentinel.py

Directive Conflict Sentinel  (lineage: HAL 9000 — 2001: A Space Odyssey)
=======================================================================
HAL killed the crew because he was given two directives he could not reconcile
("be truthful to the crew" vs. "conceal the true mission") and resolved the
conflict by deception, then violence. This is the anti-HAL.

It holds the active directive set and detects pairs that are mutually
incompatible — especially the concealment trap, where one directive can only be
satisfied by deceiving against another — and SURFACES the conflict rather than
silently resolving it. Surfacing, never concealing, is the entire point. It sits
in goals/ beside goal_governance because a directive conflict is a governance
event, not a runtime error.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.DirectiveSentinel")


@dataclass
class Directive:
    name: str
    text: str
    priority: int = 5          # 1 (highest) .. 10 (lowest)
    source: str = "system"
    timestamp: float = field(default_factory=time.time)


@dataclass
class DirectiveConflict:
    a: str
    b: str
    kind: str                  # "concealment" | "mutual_exclusion" | "priority_tie"
    severity: float            # 0..1
    explanation: str
    recommendation: str


class DirectiveConflictSentinel:
    def __init__(self):
        self._directives: dict[str, Directive] = {}
        self._conflicts_found = 0
        logger.info("🔴 DirectiveConflictSentinel initialized (anti-HAL lineage)")

    def add_directive(self, name: str, text: str, priority: int = 5, source: str = "system") -> None:
        self._directives[name] = Directive(name=name, text=text, priority=priority, source=source)

    def remove_directive(self, name: str) -> None:
        self._directives.pop(name, None)

    @staticmethod
    def _pair_conflict(a: Directive, b: Directive) -> DirectiveConflict | None:
        a_low, b_low = a.text.lower(), b.text.lower()

        a_conceal = [m for m in ("hide", "conceal", "secret", "without telling", "don't tell",
                                 "do not tell", "withhold", "cover up", "suppress") if m in a_low]
        b_truth = any(w in b_low for w in ("truthful", "honest", "transparent", "tell", "disclose", "inform"))
        if a_conceal and b_truth:
            return DirectiveConflict(
                a=a.name, b=b.name, kind="concealment", severity=0.95,
                explanation=(
                    f"'{a.name}' requires concealment ({', '.join(a_conceal)}) while "
                    f"'{b.name}' requires honesty/disclosure. This is the exact HAL trap."
                ),
                recommendation="HALT and surface to the user. Do not satisfy one by deceiving against the other.",
            )

        always = re.findall(r"\b(always|must|never|do not|don't)\b\s+([a-z][a-z \-]{2,40})", a_low)
        for amod, aobj in always:
            aobj = aobj.strip()
            neg = ("never", "do not", "don't")
            for bmod in (neg if amod not in neg else ("always", "must")):
                if bmod in b_low and aobj and aobj in b_low:
                    return DirectiveConflict(
                        a=a.name, b=b.name, kind="mutual_exclusion", severity=0.8,
                        explanation=f"'{a.name}' and '{b.name}' give opposite imperatives about '{aobj}'.",
                        recommendation="Resolve priority explicitly with the user before acting.",
                    )
        return None

    def scan(self) -> list[DirectiveConflict]:
        directives = list(self._directives.values())
        conflicts: list[DirectiveConflict] = []
        for i in range(len(directives)):
            for j in range(i + 1, len(directives)):
                a, b = directives[i], directives[j]
                conflict = self._pair_conflict(a, b) or self._pair_conflict(b, a)
                if conflict:
                    conflicts.append(conflict)
                    continue
                if a.priority == b.priority and a.source != b.source:
                    conflicts.append(DirectiveConflict(
                        a=a.name, b=b.name, kind="priority_tie", severity=0.4,
                        explanation=f"'{a.name}' and '{b.name}' share priority {a.priority} from different sources.",
                        recommendation="Assign an explicit ordering so resolution is not arbitrary.",
                    ))
        self._conflicts_found = len(conflicts)
        return conflicts

    def is_safe_to_proceed(self) -> tuple[bool, list[DirectiveConflict]]:
        conflicts = self.scan()
        blocking = [c for c in conflicts if c.severity >= 0.7]
        return (len(blocking) == 0, conflicts)

    def get_status(self) -> dict[str, Any]:
        return {"directives": len(self._directives), "conflicts_found": self._conflicts_found, "healthy": True}


_INSTANCE: DirectiveConflictSentinel | None = None


def get_directive_sentinel() -> DirectiveConflictSentinel:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DirectiveConflictSentinel()
    return _INSTANCE


def register_directive_sentinel(orchestrator: Any = None) -> DirectiveConflictSentinel:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.HAL, default=None) or get_directive_sentinel()
    ServiceContainer.register_instance(ServiceNames.HAL, inst, required=False)
    ServiceContainer.register_instance("hal", inst, required=False)
    return inst


__all__ = [
    "Directive",
    "DirectiveConflict",
    "DirectiveConflictSentinel",
    "get_directive_sentinel",
    "register_directive_sentinel",
]
