"""What each sound comes with is learned from when it is heard, and a warning is heeded
(core/agency/what_a_sound_comes_with.py).

A world of its own, heard and not seen. A chime comes with each point; a beep with each press of hers; notes of music
come at random; and now and then a siren sounds, and a little over a second later a strike lands where she was when it
sounded, and a crash sounds as it hits her. Nothing on the screen warns of the strike. A player who cannot hear walks
on and is struck; one who has learned what the siren means gets off the spot. And where the strike finds her wherever
she is, getting off the spot is found to do nothing, and is given up.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.what_a_sound_comes_with import GAIN, HER_OWN, LOSS, WARNING, WhatSoundsComeWith

pytestmark = pytest.mark.unit

STEP = 1 / 30
LEAD = 1.2
REACH = 30.0


def _a_world(listening: bool, seed: int, seconds: float = 150.0, *, finds_her: bool = False):
    rand = random.Random(seed)
    her = SimpleNamespace(x=240.0, y=160.0, w=16.0, h=16.0)
    sounds = WhatSoundsComeWith()
    heading, turn_at = rand.uniform(0, 2 * math.pi), 0.0
    queued: list[tuple[float, str]] = []
    strikes: list[tuple[float, float, float]] = []
    blows, next_siren, at = [], 5.0, 0.0
    while at < seconds:
        # What happens of itself: a point now and then, her presses, music.
        if rand.random() < 0.4 * STEP:
            sounds.gained(at)
            queued.append((at + 0.05, "chime"))
        if rand.random() < 1.0 * STEP:
            sounds.pressed(at)
            queued.append((at + 0.03, "beep"))
        if rand.random() < 0.7 * STEP:
            queued.append((at, rand.choice(("note a", "note b", "note c"))))
        if at >= next_siren:
            queued.append((at, "siren"))
            strikes.append((at + LEAD, her.x, her.y))
            next_siren = at + rand.uniform(4.0, 7.0)
        for when, sig in [q for q in queued if q[0] <= at]:
            sounds.heard_sound(when, sig, her)
        queued = [q for q in queued if q[0] > at]
        # Where she goes: on her way, or off a spot a warning marks.
        spots = sounds.spots_to_keep_clear_of(at) if listening else []
        if spots:
            x, y, _r, _start, _end = min(spots, key=lambda s: math.dist((s[0], s[1]), (her.x, her.y)))
            away = math.atan2(her.y - y, her.x - x) if (her.x, her.y) != (x, y) else heading
            heading = away
        elif at >= turn_at:
            heading, turn_at = rand.uniform(0, 2 * math.pi), at + 1.0
        her.x = min(470.0, max(10.0, her.x + 40.0 * math.cos(heading) * STEP))
        her.y = min(310.0, max(10.0, her.y + 40.0 * math.sin(heading) * STEP))
        for strike in [s for s in strikes if s[0] <= at]:
            strikes.remove(strike)
            where = (her.x, her.y) if finds_her else strike[1:]
            if math.dist(where, (her.x, her.y)) <= REACH:
                blows.append(at)
                sounds.blow(at, her)
                queued.append((at, "crash"))
        sounds.judge_clear(at, her)
        at += STEP
    return sounds, blows


def test_each_sound_means_what_it_comes_with_and_music_means_nothing():
    sounds, _blows = _a_world(False, 1)
    now = 150.0
    assert sounds.meaning("chime", now) == GAIN
    assert sounds.meaning("beep", now) == HER_OWN
    assert sounds.meaning("crash", now) == LOSS
    assert sounds.meaning("siren", now) == WARNING
    assert sounds._lead("siren", now) == pytest.approx(LEAD, abs=0.2)
    for note in ("note a", "note b", "note c"):
        assert sounds.meaning(note, now) is None, note


def test_a_warning_heeded_keeps_her_from_the_strike():
    deaf = heeding = 0
    for seed in range(1, 11):
        _deaf, struck_deaf = _a_world(False, seed)
        heard, struck_heeding = _a_world(True, seed)
        # After the first minute, by when the siren has been heard enough times to be known.
        deaf += sum(1 for at in struck_deaf if at > 60.0)
        heeding += sum(1 for at in struck_heeding if at > 60.0)
        # Heeded, the siren still means what it meant.
        assert heard.meaning("siren", 150.0) == WARNING, seed
    assert heeding <= 0.3 * deaf, (deaf, heeding)


@pytest.mark.parametrize("seed", range(1, 11))
def test_each_sound_is_known_for_what_it_comes_with_whatever_the_run(seed):
    sounds, _blows = _a_world(False, seed)
    found = {sig: sounds.meaning(sig, 150.0) for sig in ("chime", "beep", "crash", "siren", "note a")}
    assert found == {"chime": GAIN, "beep": HER_OWN, "crash": LOSS, "siren": WARNING, "note a": None}, found


def test_where_the_strike_finds_her_wherever_she_is_getting_off_the_spot_is_given_up():
    sounds, _blows = _a_world(True, 4, finds_her=True)
    assert not sounds.heeding_pays("siren", 150.0)
    assert not sounds.spots_to_keep_clear_of(150.0)
