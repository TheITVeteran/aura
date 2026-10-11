"""Keeping allies: someone on her side, kept clear of what would cost her through them.

An escort, a town to guard, a crew to bring home: losing one of them costs her as a blow to herself does. What is on
her side is found by what losing it cost her (core/agency/who_is_who.py); what comes for it is found as what comes for
her is found (core/agency/what_comes_for_her.py), by how well a threat's going lines up with the way to it, picture
after picture, while it moves. A threat that comes for an ally is worth taking out by what losing the ally would cost
her, by the chance it gets there: as soon as it lines up, sooner the nearer it is, and before a threat to her that is
further off.

Nothing here knows a game, an ally or a threat.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["KeepingAllies"]

#: As for what comes for her: how fast a thing must go for its going to point anywhere, how many pictures are read, and
#: how well, on average, its going must line up with the way to an ally.
GOING = 12.0
LEAST_PICTURES = 10
LINES_UP = 0.8

#: How long, at the soonest, a threat coming for an ally takes to get there for its getting there to be all but sure;
#: further off, the chance falls as the time grows, since she, or the ally, may yet change it.
SURE_WITHIN_S = 1.0


@dataclass
class KeepingAllies:
    """For each threat and each ally, how the threat's going has lined up with the way to the ally."""

    lined: dict[tuple[int, int], deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=40)))

    def saw(self, threats: Any, allies: Any) -> None:
        """One picture: the threats on the screen and her allies."""
        allies = list(allies)
        for threat in threats:
            vx, vy = float(threat.vx), float(threat.vy)
            speed = math.hypot(vx, vy)
            if speed < GOING:
                continue
            for ally in allies:
                dx, dy = float(ally.x) - float(threat.x), float(ally.y) - float(threat.y)
                far = math.hypot(dx, dy)
                if far >= 1.0:
                    self.lined[(threat.number, ally.number)].append((vx * dx + vy * dy) / (speed * far))

    def comes_for(self, threat: Any, ally: Any) -> bool:
        """Whether this threat keeps its going pointed at this ally."""
        kept = self.lined.get((threat.number, ally.number))
        return kept is not None and len(kept) >= LEAST_PICTURES and sum(kept) / len(kept) >= LINES_UP

    @staticmethod
    def seconds_to(threat: Any, ally: Any) -> float:
        """How long the threat takes to reach the ally at its own pace."""
        speed = max(math.hypot(float(threat.vx), float(threat.vy)), 1e-6)
        return math.hypot(float(ally.x) - float(threat.x), float(ally.y) - float(threat.y)) / speed

    def threatened(self, threats: Any, allies: Any) -> list[tuple[Any, Any, float]]:
        """Each threat coming for an ally: (threat, ally, how likely it gets there), the likeliest first."""
        found = []
        for threat in threats:
            for ally in allies:
                if self.comes_for(threat, ally):
                    soon = self.seconds_to(threat, ally)
                    found.append((threat, ally, min(1.0, SURE_WITHIN_S / max(soon, 1e-6))))
        return sorted(found, key=lambda item: -item[2])

    def worth_taking_out_s(self, threat: Any, allies: Any, ally_worth_s: float) -> float:
        """What taking this threat out is worth, in seconds of play, for the allies it comes for: what losing one would
        cost her, by how likely it gets there."""
        return sum(chance * ally_worth_s for found, _ally, chance in self.threatened([threat], allies) if found is threat)
