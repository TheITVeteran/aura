"""An act she took on a screen has been tried there, though she could not make out what it did.

What came of a click can be past making out: a picture that changed only a little, a look that came too soon. That she
clicked is not. LIVE-like 2026-10-12 a panel's arrow, whose effect could not be told, was still "never tried" after
thirty presses, so it kept the worth of a first try above everything else on the screen and nothing else was tried.
"""
from __future__ import annotations

from core.agency.looking_ahead import worth_finding_out
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

_SCREEN = [a_click_on("Take the picture"), a_click_on("the one that stands out at 80% across, 90% down"),
           a_click_on("the one that stands out at 80% across, 30% down")]
_ARROW = a_click_on("the one that stands out at 80% across, 90% down")


def _never_tried(here: WhatWorksHere) -> list[str]:
    return [act for act in _SCREEN if not here.leads.taken_here(act)]


def test_a_click_whose_effect_could_not_be_told_is_not_new_again():
    here = WhatWorksHere()
    here.looked_at(_SCREEN)
    assert _never_tried(here) == _SCREEN
    for _ in range(3):
        here.looked_at(_SCREEN)
        here.tried(_ARROW, None)          # what it did could not be made out
    assert _ARROW not in _never_tried(here)
    told = worth_finding_out(None, None, _SCREEN, never_tried=_never_tried(here))
    assert _ARROW not in told and set(told) == set(_SCREEN) - {_ARROW}


def test_on_another_screen_it_is_new():
    here = WhatWorksHere()
    here.looked_at(_SCREEN)
    here.looked_at(_SCREEN)
    here.tried(_ARROW, None)
    elsewhere = [a_click_on("Welcome to the island"), a_click_on("next"), _ARROW]
    here.looked_at(elsewhere)
    assert not here.leads.taken_here(_ARROW)


def test_screens_with_nothing_to_read_do_not_share_what_was_taken():
    """Rooms without words are one place to her map, however many there are: a door taken in one is not taken in the
    next."""
    here = WhatWorksHere()
    wordless = [_ARROW]
    here.looked_at(wordless)
    here.looked_at(wordless)
    here.tried(_ARROW, None)
    assert not here.leads.taken_here(_ARROW)


def test_an_act_is_taken_on_the_screen_it_was_chosen_on_however_often_the_next_was_looked_at():
    """Offline 2026-10-13 a "next" clicked on one page was kept as taken on the page it led to (looked at twice before
    the click was judged), and that page's own "next" lost the worth of a first try."""
    here = WhatWorksHere()
    welcome = [a_click_on("Welcome to the island"), a_click_on("back"), a_click_on("next")]
    rules = [a_click_on("How to hold the club"), a_click_on("back"), a_click_on("next")]
    here.looked_at(welcome, "Welcome to the island back next")
    here.looked_at(rules, "How to hold the club back next")
    here.looked_at(rules, "How to hold the club back next")
    here.tried(a_click_on("next"), True, before={"clickable": welcome, "text": "Welcome to the island back next"})
    assert not here.leads.taken_here(a_click_on("next"))           # on this page, it is still to be tried
    here.looked_at(welcome, "Welcome to the island back next")
    assert here.leads.taken_here(a_click_on("next"))               # on the page it was chosen on, it was
