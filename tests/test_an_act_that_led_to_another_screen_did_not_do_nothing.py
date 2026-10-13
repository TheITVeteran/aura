"""An act is measured by a look taken at once after it. Where that look found the screen as it was, and the reading
the verdict is given then shows plainly another screen, the act led there.

Offline 2026-10-12: a story page's NEXT was pressed, the first look came before the page turned, the press was
written down as having done nothing, and on the page it led to NEXT was not tried: four keys were, one at a time.
"""
from __future__ import annotations

import pytest

from core.agency.deliberate_action import Expectation, _plainly_another_screen

pytestmark = pytest.mark.unit

STORY = ("has to clean up this huge mess before Billys dad gets home! But if she uses Grims Magic Book, "
         "it will be a snap! NEXT")
TOLD = ("Follow the arrow keys in Grims Magic Book to cast the correct clean-up spell. Be careful! "
        "A mistake could create an even bigger mess! NEXT")


def test_a_press_measured_as_nothing_that_led_to_another_screen_held():
    assert Expectation(changed=True).check(STORY, TOLD, measured_change=False).held is True


def test_one_screen_read_twice_a_letter_apart_is_still_one_screen_where_nothing_was_measured_to_change():
    one, again = "LEVEL 1 SCORE 10 LIVES 3 PLAY", "LEVEL 1 SCORE 1O LIVES 3 PLAY"
    assert not _plainly_another_screen(one, again)
    assert Expectation(changed=True).check(one, again, measured_change=False).held is False
    assert Expectation(changed=True).check(STORY, STORY, measured_change=False).held is False


def test_a_screen_with_too_few_words_to_tell_it_by_is_not_another_screen():
    assert not _plainly_another_screen("GO", "PLAY")
    assert Expectation(changed=True).check("GO", "PLAY", measured_change=False).held is False


def test_a_change_that_was_measured_still_counts_whatever_the_words_say():
    assert Expectation(changed=True).check(STORY, STORY, measured_change=True).held is True
