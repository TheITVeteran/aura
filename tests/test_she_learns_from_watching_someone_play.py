"""She watches a recording of someone playing and reads what they made their thing do, as her own keys, what each move
was for, which kinds they went for and kept clear of, and the places along their way.

Bryan, 10 October 2026: research people playing and apply their actual decision-making, physics and reasoning, binding
choices to actions and both to reasoning, strategy and knowledge.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.watching_someone_play import how_far_along, read_footage, stance_by_look

pytestmark = pytest.mark.unit

WIDE, TALL, STEP = 240, 180, 1 / 15
GREEN, RED, WHITE = (40, 220, 60), (230, 40, 40), (240, 240, 240)


def _recording(seconds: float = 24.0):
    """A player at the bottom who goes under what is green as it falls and steps away from what is red."""
    falling = []                                  # [x, y, colour, alive]
    rng = np.random.default_rng(3)
    x, frames, t, next_drop = 120.0, [], 0.0, 0.0
    while t < seconds:
        if t >= next_drop:
            colour = GREEN if len(falling) % 2 == 0 else RED
            falling.append([float(rng.integers(30, 210)), 0.0, colour, True])
            next_drop = t + 1.6
        for item in falling:
            item[1] += 70 * STEP
        live = [f for f in falling if f[3] and f[1] < TALL]
        greens = [f for f in live if f[2] == GREEN]
        reds = [f for f in live if f[2] == RED and f[1] > 90 and abs(f[0] - x) < 30]
        if reds:
            x += (-1 if reds[0][0] > x else 1) * 140 * STEP
        elif greens:
            target = min(greens, key=lambda f: -f[1])[0]
            x += max(-140 * STEP, min(140 * STEP, target - x))
        x = min(WIDE - 12, max(12, x))
        picture = np.zeros((TALL, WIDE, 3), np.uint8)
        for item in live:
            if item[2] == GREEN and abs(item[0] - x) < 14 and item[1] > 150:
                item[3] = False                    # taken
                continue
            picture[int(item[1]):int(item[1]) + 8, int(item[0]) - 4:int(item[0]) + 4] = item[2]
        picture[160:168, int(x) - 10:int(x) + 10] = WHITE
        frames.append((picture, t))
        t += STEP
    return frames


def test_the_players_wants_and_aversions_and_acts_are_read_from_a_recording():
    footage = read_footage(_recording(), ways={"left": (-140.0, 0.0), "right": (140.0, 0.0)})
    assert footage.hers is not None and footage.looks[footage.hers][0][0] > 200     # the white bar is the player's
    stances = {footage.looks[k][0]: s for k, s in footage.stances().items()}
    greens = [s for colour, s in stances.items() if colour[1] > 150 and colour[0] < 100]
    reds = [s for colour, s in stances.items() if colour[0] > 150 and colour[1] < 100]
    assert greens == ["meet"] and reds in ([], ["avoid"])
    assert {m.key for m in footage.moments} - {""} <= {"left", "right"} and footage.moments
    assert any(m.toward is not None for m in footage.moments)
    kept = footage.as_memory()
    assert stance_by_look(kept, GREEN, footage.looks[next(iter(footage.stances()))][1]) in ("meet", "avoid")
    assert footage.checkpoints and how_far_along(_recording(2.0)[-1][0], footage.checkpoints) is not None
