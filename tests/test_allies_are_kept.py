"""An ally is kept clear of what comes for it (core/agency/keeping_allies.py).

A world of its own. An ally walks to and fro across the top of the field; she stands below. Threats come in from the
edges, some for her and some for the ally, and when one reaches the ally the ally is lost, which costs her as a blow
does. She can take a threat out, which takes her a moment. Played for herself, she takes out what comes for her and
the ally is lost again and again; keeping her ally, she takes out what comes for it too, as soon as it lines up.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.keeping_allies import KeepingAllies

pytestmark = pytest.mark.unit

STEP, SHOOT_S, PACE = 1 / 30, 0.6, 40.0


def _a_run(keeping: bool, seed: int, seconds: float = 120.0) -> tuple[int, int]:
    rand = random.Random(seed)
    her = SimpleNamespace(number=0, x=240.0, y=260.0, vx=0.0, vy=0.0)
    ally = SimpleNamespace(number=1, x=60.0, y=100.0, vx=30.0, vy=0.0)
    allies, threats, numbers = KeepingAllies(), [], iter(range(10, 100000))
    hit_her = lost_ally = 0
    busy, next_threat, at = 0.0, 0.0, 0.0
    while at < seconds:
        if at >= next_threat:
            edge = rand.choice(((rand.uniform(0, 480), 0.0), (0.0, rand.uniform(0, 320)), (480.0, rand.uniform(0, 320))))
            threats.append(SimpleNamespace(number=next(numbers), x=edge[0], y=edge[1], vx=0.0, vy=0.0,
                                           after=rand.choice((her, ally))))
            next_threat = at + 2.5
        ally.x += ally.vx * STEP
        if not 60.0 <= ally.x <= 420.0:
            ally.vx = -ally.vx
        for threat in threats:
            dx, dy = threat.after.x - threat.x, threat.after.y - threat.y
            far = math.hypot(dx, dy) or 1.0
            threat.vx, threat.vy = PACE * dx / far, PACE * dy / far
            threat.x, threat.y = threat.x + threat.vx * STEP, threat.y + threat.vy * STEP
        allies.saw(threats, [ally])
        if busy > 0:
            busy -= STEP
        elif threats:
            # What each threat is worth taking out: the chance it lands, on her or, keeping her ally, on the ally.
            def worth(threat):
                mine = min(1.0, 1.0 / max(allies.seconds_to(threat, her), 1e-6)) if threat.after is her else 0.0
                return mine + (allies.worth_taking_out_s(threat, [ally], 1.0) if keeping else 0.0)

            best = max(threats, key=worth)
            if worth(best) >= 0.25:
                threats.remove(best)
                busy = SHOOT_S
        for threat in list(threats):
            if math.hypot(threat.after.x - threat.x, threat.after.y - threat.y) < 10.0:
                threats.remove(threat)
                if threat.after is her:
                    hit_her += 1
                else:
                    lost_ally += 1
        at += STEP
    return hit_her, lost_ally


def test_a_threat_lining_up_on_the_ally_is_told_from_one_lining_up_on_her():
    allies = KeepingAllies()
    ally = SimpleNamespace(number=1, x=100.0, y=100.0, vx=0.0, vy=0.0)
    her = SimpleNamespace(number=0, x=400.0, y=100.0, vx=0.0, vy=0.0)
    for_ally = SimpleNamespace(number=5, x=100.0, y=300.0, vx=0.0, vy=-40.0)
    for_her = SimpleNamespace(number=6, x=400.0, y=300.0, vx=0.0, vy=-40.0)
    for _ in range(15):
        allies.saw([for_ally, for_her], [ally])
        for_ally.y -= 40.0 * STEP
        for_her.y -= 40.0 * STEP
    assert allies.comes_for(for_ally, ally)
    assert not allies.comes_for(for_her, ally)
    assert allies.worth_taking_out_s(for_ally, [ally], 30.0) > 0 == allies.worth_taking_out_s(for_her, [ally], 30.0)
    _ = her


def test_kept_her_ally_is_lost_far_less_often_and_she_is_hit_no_more():
    hit_alone = lost_alone = hit_keeping = lost_keeping = 0
    for seed in range(8):
        hit, lost = _a_run(False, seed)
        hit_alone, lost_alone = hit_alone + hit, lost_alone + lost
        hit, lost = _a_run(True, seed)
        hit_keeping, lost_keeping = hit_keeping + hit, lost_keeping + lost
    assert lost_keeping <= 0.3 * lost_alone, (lost_alone, lost_keeping)
    assert hit_keeping + lost_keeping < 0.5 * (hit_alone + lost_alone), (hit_alone, lost_alone, hit_keeping, lost_keeping)
