"""What she says while she plays is news: an account of the game again only for something learned, a change quoted as code."""
from __future__ import annotations

import pytest

from core.agency import playing_as_it_happens as playing
from core.self_modification.code_that_looks_wrong import Edit

pytestmark = pytest.mark.unit


def test_the_account_of_a_game_is_given_again_only_for_something_learned(monkeypatch):
    """LIVE 2026-10-06 the same account came every thirty seconds, because her paddle had moved."""
    import core.agency.what_kind_of_game_this_is as kind

    accounts = iter([
        "I'm the white bar at the left, moved by down and up; the white bar across from me is the other side.",
        "I'm the white bar at the top left, moved by down and up; the white bar across from me is the other side.",
        "I'm the white bar at the bottom left, moved by down and up.",
        "I'm the white bar at the left, moved by down and up; the white thing bounces off the walls.",
    ])
    monkeypatch.setattr(kind, "in_a_sentence", lambda *_a, **_k: next(accounts))
    run = playing._Run(keys=["up", "down"], began=0.0)
    run.said.add("me")
    said: list[str] = []
    for n in range(4):
        playing._what_kind_of_game(run, said.append, None, None, None, None, at=100.0 * (n + 1))
    assert len(said) == 2 and "bounces off the walls" in said[-1]


def test_a_change_is_said_as_code_so_its_signs_are_not_formatting():
    source = "step = Math.max(a, b) * 0;"
    assert Edit(7, 25, "Math.max(a, b)").says(source) == "changed `Math.max(a, b) * 0` to `Math.max(a, b)`"
    assert Edit(0, 0, "if (y < 0) {\n  y = 0;\n}\n").says(source) == "added `if (y < 0) {\\n  y = 0;\\n}\\n`"
    assert Edit(0, 7, "").says(source) == "removed `step = `"


def test_one_way_known_and_the_other_untried_is_not_a_line():
    """LIVE 2026-10-11 a girl flying every way, only "up" yet learned, had a gauge called "the other side"."""
    from types import SimpleNamespace

    from core.agency.what_kind_of_game_this_is import held_to_one_line

    tried = {"up": 6, "down": 6, "left": 0, "right": 1}
    flying = SimpleNamespace(axes=lambda keys: (False, True), tried=lambda key: tried.get(key, 0))
    assert held_to_one_line(flying, ["up", "down", "left", "right"]) == (False, False)
    tried.update(left=6, right=6)
    assert held_to_one_line(flying, ["up", "down", "left", "right"]) == (False, True)
