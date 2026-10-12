"""The settings of her ways of playing are learned from what each paid, kind of situation by kind of situation, and
carried to the next game of the kind.

Offline 2026-10-12 a food fight was lost round after round with throws every 0.6 s and was won with throws five times a
second; the 0.2 s that won was set by hand, from watching one game. Nothing here is a game: a made-up place where one
choice of a setting pays and the others do not.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from core.agency.settings_that_paid import SETTINGS, SettingsThatPaid, kind_of_situation, setting

pytestmark = pytest.mark.unit


def _played(settings: SettingsThatPaid, pays: float, stretches: int, rng: random.Random) -> list[float]:
    """Stretches in a place where throwing every ``pays`` seconds wins most of what is fought over, and any other
    choice loses most of it."""
    chosen = []
    for _ in range(stretches):
        throw_every = settings.choose(rng)["throw every"]
        chosen.append(throw_every)
        settings.paid(gains=8 if throw_every == pays else 2, losses=2 if throw_every == pays else 8)
    return chosen


def test_the_choice_that_pays_comes_to_be_played_most_though_another_was_chosen_by_hand():
    rng = random.Random(11)
    settings = SettingsThatPaid("someone aiming from this side | it goes on without me | win | against someone")
    pays = 0.35
    assert SETTINGS["throw every"].by_hand != pays
    chosen = _played(settings, pays, 60, rng)
    assert chosen[-20:].count(pays) >= 14
    assert settings.best("throw every")[0] == pays
    assert "0.35 has gone best" in settings.says()


def test_a_stretch_in_which_too_little_happened_says_nothing_and_the_hand_choice_stands_where_nothing_was_learned():
    settings = SettingsThatPaid("a body I steer with keys | it goes on without me | score | no one against")
    settings.choose(random.Random(3))
    assert not settings.paid(gains=1, losses=0)
    assert settings.went == {}
    assert setting(SimpleNamespace(), "throw every", 0.2) == 0.2
    assert setting(SimpleNamespace(settings=settings), "duck every", 9.9) in SETTINGS["duck every"].choices


def test_what_was_learned_is_carried_to_the_next_game_of_the_kind_and_discounted(tmp_path, monkeypatch):
    import core.runtime.what_she_learned as kept

    monkeypatch.setattr(kept, "_KEPT_IN", tmp_path)
    kind = "someone aiming from this side | it goes on without me | win | against someone"
    first = SettingsThatPaid(kind)
    _played(first, 0.35, 40, random.Random(5))
    assert first.keep()
    again = SettingsThatPaid.for_the_kind(kind)
    good, bad = again.went["throw every"]["0.35"]
    original = first.went["throw every"]["0.35"]
    assert good == pytest.approx(0.5 * original[0]) and bad == pytest.approx(0.5 * original[1])
    assert SettingsThatPaid.for_the_kind("someone making something | it waits for me | make | no one against").went == {}


def test_the_kind_of_situation_is_how_she_understands_the_game_never_its_name():
    from core.agency.how_a_game_works import AIMS_FROM_HERE, AGAINST_SOMEONE, RUNS, HowItWorks

    how = HowItWorks()
    assert kind_of_situation(how) == ""
    how.me.held_to(AIMS_FROM_HERE, 0.9, "the words said to aim with the mouse")
    how.tempo.held_to(RUNS, 0.9, "it went on by itself")
    how.against.held_to(AGAINST_SOMEONE, 0.9, "you against")
    assert kind_of_situation(how) == f"{AIMS_FROM_HERE} | {RUNS} | ? | against someone"


def test_where_the_situation_is_not_understood_the_settings_chosen_by_hand_are_played():
    from core.agency.settings_that_paid import for_this_stretch
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    token = THE_GUIDE.set(None)
    try:
        settings = for_this_stretch({})
    finally:
        THE_GUIDE.reset(token)
    assert settings.kind == "" and settings.chosen == {}
    assert settings.get("throw every") == SETTINGS["throw every"].by_hand


def test_a_place_that_wants_a_setting_beyond_every_choice_is_given_choices_past_them():
    """The learner improves its own search: the best of the first choices, played enough, has choices added past it,
    and the one that pays among those comes to be played."""
    rng = random.Random(7)
    settings = SettingsThatPaid("someone aiming from this side | it goes on without me | win | against someone")
    for _ in range(160):
        every = settings.choose(rng)["throw every"]
        # The faster the throws the better, down to a tenth of a second.
        well = max(0.0, 1.0 - abs(every - 0.1) * 4)
        settings.paid(gains=round(10 * well), losses=10 - round(10 * well))
    assert min(settings.choices("throw every")) < min(SETTINGS["throw every"].choices)
    assert settings.best("throw every")[0] < 0.15


def test_choices_are_drawn_closer_round_a_best_that_lies_between_others():
    rng = random.Random(9)
    settings = SettingsThatPaid("a body I steer with keys | it goes on without me | score | no one against")
    for _ in range(80):
        every = settings.choose(rng)["duck every"]
        well = 0.9 if every == 0.5 else 0.2
        settings.paid(gains=round(10 * well), losses=10 - round(10 * well))
    added = set(settings.choices("duck every")) - set(SETTINGS["duck every"].choices)
    assert added and all(0.3 < choice < 1.0 for choice in added)
