"""The keys a game's own words name are the keys its code listens for.

Each of the 56 games' words, and the keys its program listens for, were read offline once into
docs/design-docs/what_it_takes_to_beat.json. Held against each other on 13 October 2026, the keys she took from four
games' words were not the keys their code listens for, each by a misreading of the words:

- Ben 10: Krakken Attack, "Use your 'Up' arrow key to change direction, and press the 'Space' bar to fire": all four
  arrows and no space, in a game fired with space alone. A key's name in quotes is its name.
- Cartoon Cove Mini Golf, "Enter your name.": the Enter key. A verb is not a key.
- Gumball: Blind Fooled, "...press or [S] to pick it up Press or [S] again...", its full stop lost: the up arrow. A
  press is said before the key it presses.
- Gumball: Battle Bowlers, "hit 1 to switch to anais. hit 2 to switch to gumball.": no number keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agency.the_controls_a_game_names import controls_named_in

pytestmark = pytest.mark.unit

_MAP = {g["number"]: g for g in json.loads(
    (Path(__file__).parent.parent / "docs" / "design-docs" / "what_it_takes_to_beat.json").read_text())}


def _words(number: int) -> str:
    return " ".join(_MAP[number]["said"])


def test_a_key_named_in_quotes_is_that_key_and_only_that_one():
    keys, pointer = controls_named_in(_words(33))
    assert set(keys) == {"up", "space"} == set(_MAP[33]["keys"])


def test_enter_said_as_a_verb_is_not_the_enter_key():
    keys, _pointer = controls_named_in(_words(6))
    assert "return" not in keys
    assert controls_named_in("Press Enter to start.")[0] == ["return"]
    assert "return" in controls_named_in("Type your name and hit the enter key")[0]


def test_a_way_after_a_word_for_pressing_is_not_a_key_it_presses():
    keys, _pointer = controls_named_in(_words(45))
    assert "up" not in keys and {"w", "a", "s", "d", "space"} <= set(keys)
    assert "up" in controls_named_in("Press up to jump.")[0] and "up" in controls_named_in("the up key jumps")[0]


def test_a_number_key_pressed_by_name_is_a_key():
    keys, _pointer = controls_named_in(_words(55))
    assert {"1", "2"} <= set(keys)
    assert controls_named_in("Up to 5 players can play.")[0] == ["up", "down", "left", "right", "space"]


def test_the_keys_named_in_quotes_or_by_number_in_every_game_are_keys_its_code_listens_for():
    """Every game whose words name arrows, space or enter, and whose code was read for keys: what she takes from its
    words is listened for by its code, but where the words name keys its code reading missed (a letter or an arrow given
    by its number in the program), which the words themselves say."""
    missed_by_the_code_reading = {32: {"z"}, 39: {"x"}, 45: {"w", "a", "s", "d"},
                                  55: {"up", "down", "right", "w", "a", "s", "d", "1", "2"}}
    for number, game in _MAP.items():
        if "error" in game or not game["keys"]:
            continue
        keys, _pointer = controls_named_in(_words(number))
        if keys == ["up", "down", "left", "right", "space"] and "arrow" not in _words(number).lower():
            continue                                     # no keys named: the usual ones are tried, not read
        dead = set(keys) - set(game["keys"]) - missed_by_the_code_reading.get(number, set())
        assert not dead, (number, game["game"], dead)
