"""What a screen asks to have typed is typed where it points, and sent the way it says; nothing is typed where she has
nothing to give. Nothing here is a game: an oracle that answers questions, a high-score table, a riddle, a search box."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.agency.typing_what_is_asked import places_to_type, type_what_is_asked, what_to_type
from core.language.words_asked_for import ANSWER, CODE, NAME, QUESTION, words_asked_for

pytestmark = pytest.mark.unit


def test_what_each_screen_asks_for_is_read_off_its_words():
    oracle = words_asked_for('1. Try to type only "Yes" or "No" questions. 2. Don\'t forget to click "Ask." TYPE YOUR QUESTION ABOVE')
    assert oracle.kind == QUESTION and oracle.where == "above" and oracle.sent_by == "Ask"
    assert words_asked_for("GAME OVER Enter your name: SUBMIT").kind == NAME
    assert words_asked_for("Riddle me this: What has keys but no locks? Answer?").kind == ANSWER
    assert words_asked_for("Type your answer and press Enter.").sent_by_key == "return"
    assert words_asked_for("Enter the secret code").kind == CODE
    assert words_asked_for("Use the arrow keys to move. Collect the stars.") is None
    # A drawn heading read with a stray letter on its front (offline 2026-10-11 the Riddler's "FYOUR NAME HERE").
    assert words_asked_for("FYOUR NAME HERE ETRYCOAIN GOUR SECRET DENTLTY").kind == NAME
    assert words_asked_for("RANDOMIZE NAME YOUR ALIEN ATOMIC SAVE TO THE GALLERY").kind == NAME


def test_she_types_her_name_a_question_or_an_answer_she_has_and_nothing_she_does_not():
    async def knows(question):
        return "a piano" if "keys" in question else None

    assert asyncio.run(what_to_type(words_asked_for("Enter your name"))) == "Aura"
    assert asyncio.run(what_to_type(words_asked_for("Type your question above"), goal="play until you win")) == "Will I win?"
    assert asyncio.run(what_to_type(words_asked_for("Riddle me this: What has keys but no locks? Answer?"), answer=knows)) == "a piano"
    assert asyncio.run(what_to_type(words_asked_for("Enter the secret code"))) is None


def test_the_place_the_words_point_to_comes_first():
    asked = words_asked_for("TYPE YOUR QUESTION ABOVE")
    regions = [{"text": "TYPE YOUR QUESTION ABOVE", "center_x": 0.5, "center_y": 0.6, "height": 0.04, "width": 0.4}]
    (x, y), *_rest = places_to_type(asked, regions)
    assert x == pytest.approx(0.5) and y < 0.6
    # "Here" in a heading labels the box drawn under it (offline 2026-10-11 the Riddler's YOUR NAME HERE).
    asked = words_asked_for("YOUR NAME HERE")
    regions = [{"text": "YOUR NAME HERE", "center_x": 0.64, "center_y": 0.3, "height": 0.084, "width": 0.37}]
    (x, y), *_rest = places_to_type(asked, regions)
    assert x == pytest.approx(0.64) and 0.35 < y < 0.45


class _Oracle:
    """A screen with a field above its words; what is typed shows only where the field was clicked; Ask answers."""

    def __init__(self):
        self.focused = False
        self.typed = ""
        self.answered = False

    async def look(self):
        return np.zeros((10, 10, 3), np.uint8), 0.0

    def words(self, _picture):
        regions = [{"text": "TYPE YOUR QUESTION ABOVE", "center_x": 0.5, "center_y": 0.6, "height": 0.04, "width": 0.4},
                   {"text": "ASK", "center_x": 0.8, "center_y": 0.7, "height": 0.04, "width": 0.08}]
        if self.typed:
            regions.append({"text": self.typed, "center_x": 0.5, "center_y": 0.5, "height": 0.04, "width": 0.4})
        if self.answered:
            regions.append({"text": "Like, wow! Most definitely!", "center_x": 0.5, "center_y": 0.3})
        return regions

    async def click(self, x, y):
        if abs(x - 0.8) < 0.05 and abs(y - 0.7) < 0.05:
            self.answered = bool(self.typed)
            return
        self.focused = abs(x - 0.5) < 0.1 and 0.45 < y < 0.55

    async def type_text(self, text):
        if self.focused:
            self.typed += text

    async def tap(self, key):
        if key == "backspace":
            self.typed = self.typed[:-1]


def test_she_types_the_question_into_the_field_above_and_asks():
    oracle = _Oracle()
    asked = words_asked_for('Try to type only "Yes" or "No" questions. Don\'t forget to click "Ask." TYPE YOUR QUESTION ABOVE')
    done = asyncio.run(type_what_is_asked(asked, oracle.look, oracle.words, oracle, goal="play it until you win"))
    assert oracle.typed == "Will I win?" and oracle.answered, done


def test_a_waiting_screen_that_asks_for_words_has_them_typed_on_her_page(monkeypatch):
    import time

    import core.skills.screen_pursuit_as_it_happens as reflexes

    oracle = _Oracle()

    class _OnHerPage(reflexes.PlayingAsItHappens):
        async def look(self):
            return await oracle.look()

        async def click(self, x, y):
            await oracle.click(x, y)

        async def type_text(self, text):
            await oracle.type_text(text)

        async def tap(self, key):
            await oracle.tap(key)

    monkeypatch.setattr(reflexes, "_said_while_playing", lambda line: None)
    monkeypatch.setattr(reflexes, "ANSWER_SHOWS_S", 0.0)
    monkeypatch.setattr("core.perception.what_the_pixels_show.recognize_text", oracle.words)
    playing = _OnHerPage(page=None, band=(0, 0, 1, 1), goal="play it until you win", ends_at=time.monotonic() + 30)
    playing.words = ['Try to type only "Yes" or "No" questions. Click "Ask." TYPE YOUR QUESTION ABOVE']
    assert asyncio.run(playing._typed_what_it_asks())
    assert oracle.typed == "Will I win?" and oracle.answered
    # What it answered is kept as it showed it, and a question answered is the thing done once (offline 2026-10-11 a
    # fortune teller answered, she clicked on for two minutes, and what it had said was gone when she came to say it).
    assert playing.keep.get("answered")
    assert playing.run_is_over({"text": "TYPE YOUR QUESTION ABOVE"}) and playing.over_because == reflexes.MADE_NOT_WON
    # Typed once is answered (offline 2026-10-11 a name left in its box was typed into again, "AuAurara"), however the
    # ask is misread the next time.
    assert not asyncio.run(playing._typed_what_it_asks()), "what was typed is not typed again"
    playing.words = ['Try to type only "Yes" or "No" questions. Click "Ask." TYPE YOUR QUESTI0N ABOVE']
    assert not asyncio.run(playing._typed_what_it_asks()), "nor again for a misreading of the same ask"


def test_the_box_drawn_under_the_words_that_ask_is_clicked_first_in_its_middle():
    """LIVE 2026-10-09 a golf game's name box sat half a line under "Enter your name."; clicked a line and more below
    the words, and to their right, the name went nowhere."""
    import numpy as np

    from core.agency.typing_what_is_asked import places_to_type
    from core.language.words_asked_for import words_asked_for

    picture = np.full((400, 600, 3), (250, 245, 190), np.uint8)
    picture[140:160, 190:300] = (200, 60, 40)                   # the words that ask, drawn
    picture[168:198, 200:350] = (215, 205, 150)                 # the box to type in
    regions = [{"text": "Enter your name.", "x": 0.32, "y": 0.35, "width": 0.18, "height": 0.05,
                "center_x": 0.41, "center_y": 0.375}]
    first = places_to_type(words_asked_for("Enter your name."), regions, picture)[0]
    assert abs(first[0] - 275 / 600) < 0.03 and abs(first[1] - 183 / 400) < 0.02, first


def test_a_place_that_answers_what_was_typed_on_the_screen_that_asked_is_done_and_one_that_leads_on_is_not():
    """Offline 2026-10-11 a secret-identity machine answered her name and offered to try again; she pressed TRY AGAIN
    and wiped the answer, having taken it for no more than a way to start again. A golf game that took a player's name
    and went on to its course had answered nothing."""
    from core.skills.screen_pursuit_as_it_happens import _answered_since_typed

    machine = {"typed": "Aura", "typed_kind": "a name", "typed_on": "YOUR NAME HERE GO YOUR SECRET IDENTITY",
               "typed_asked_by": "YOUR NAME HERE"}
    answered = _answered_since_typed(machine, {"text": "YOUR NAME HERE Aura TRY AGAIN YOUR SECRET IDENTITY THE PUZZLE MASTER"})
    assert "PUZZLE MASTER" in answered
    assert not _answered_since_typed(machine, {"text": "YOUR NAME HERE Aura GO YOUR SECRET IDENTITY"}), "nothing new yet"
    golf = {"typed": "Aura", "typed_kind": "a name", "typed_on": "Player 1 Enter your name OK",
            "typed_asked_by": "Enter your name"}
    assert not _answered_since_typed(golf, {"text": "HOLE 1 PAR 3 STROKES 0 Click on your ball and drag back"})


def test_what_sends_what_was_typed_is_not_the_box_it_was_typed_into(monkeypatch):
    """Offline 2026-10-11 the shape "beside" a name was the box the name was in, and its GO, a line below, was never
    pressed."""
    import numpy as np

    from core.agency import typing_what_is_asked as typing

    shapes = [{"center_x": 0.63, "center_y": 0.35, "width": 0.4, "height": 0.1},
              {"center_x": 0.64, "center_y": 0.5, "width": 0.08, "height": 0.06}]
    monkeypatch.setattr("core.perception.shapes_that_look_pressable.pressable_shapes", lambda picture: shapes)
    picture = np.zeros((40, 40, 3), dtype=np.uint8)

    async def look():
        return (picture, 0.0)

    class _Hands:
        clicked: list = []

        async def click(self, x, y):
            self.clicked.append((x, y))

    hands = _Hands()
    sent = asyncio.run(typing._sent_by_the_button_beside(look, hands, (0.64, 0.36), "return"))
    assert hands.clicked == [(0.64, 0.5)] and "0.64" in sent


def test_where_the_pointer_turns_into_a_hand_beside_the_box_is_what_sends_it(monkeypatch):
    """Offline 2026-10-11 a machine's GO, bright letters drawn over others, was no shape that looked pressable, and the
    pointer turned into a hand over it."""
    import numpy as np

    from core.agency import typing_what_is_asked as typing

    monkeypatch.setattr("core.perception.shapes_that_look_pressable.pressable_shapes",
                        lambda picture: [{"center_x": 0.63, "center_y": 0.15, "width": 0.2, "height": 0.05}])
    picture = np.zeros((40, 40, 3), dtype=np.uint8)

    async def look():
        return (picture, 0.0)

    class _Hands:
        answer_the_pointer = [(0.62, 0.5)]
        clicked: list = []

        async def click(self, x, y):
            self.clicked.append((x, y))

    hands = _Hands()
    asyncio.run(typing._sent_by_the_button_beside(look, hands, (0.64, 0.36), "return"))
    assert hands.clicked == [(0.62, 0.5)]


def test_an_answer_is_told_by_words_of_the_language_not_by_misread_lettering():
    """Offline 2026-10-11 the machine answered "RABBLE ROUSER"; the words that asked had been read with lettering
    misread among them ("ETRYGOAIN", "GOUR"), and the answer was taken for the asking screen gone."""
    from core.skills.screen_pursuit_as_it_happens import _answered_since_typed

    machine = {"typed": "Aura", "typed_kind": "a name", "typed_on": "FYOUR NAME HERE ETRYGOAIN GOUR SECRET DENTIY",
               "typed_asked_by": "YOUR NAME HERE ETRYGOAIN GOUR SECRET DENTIY"}
    assert _answered_since_typed(machine, {"text": "NAME HERE … RABBLE ROUSER YOUR"})
    assert not _answered_since_typed(machine, {"text": "FYOUR NAME HERE AurA *TRYGOAIN GOUR SECRETJDENTLFY"})


def test_a_riddle_a_place_answers_itself_is_understood_and_the_players_own_answer_is_hers():
    """Offline 2026-10-11 the Riddler's machine asked "What is everything to you and nothing to anyone else? Answer?
    YOUR NAME!", a riddle that is its instruction, and she said she had no answer to give."""
    from core.agency.typing_what_is_asked import HER_NAME, the_answer_it_gives, what_to_type

    riddle = "Riddle me this: What is everything to you and nothing to anyone else? Answer? YOUR NAME!"
    assert the_answer_it_gives(riddle) == HER_NAME
    assert the_answer_it_gives("The answer is: a shadow.") == "a shadow"
    assert the_answer_it_gives("Type your answer below.") == "" and the_answer_it_gives("Give your answer and press GO.") == ""
    asked = words_asked_for(riddle)
    assert asked is not None and asyncio.run(what_to_type(asked)) == HER_NAME


def test_a_riddle_answered_on_a_screen_with_nowhere_to_type_is_told_not_typed():
    import numpy as np

    from core.agency.typing_what_is_asked import type_what_is_asked

    riddle = "Riddle me this: What is everything to you and nothing to anyone else? Answer? YOUR NAME!"
    picture = np.full((60, 80, 3), 30, dtype=np.uint8)
    said, typed = [], []

    async def look():
        return (picture, 0.0)

    class _Hands:
        async def click(self, x, y):
            pass

        async def type_text(self, text):
            typed.append(text)

        async def tap(self, key):
            pass

    done = asyncio.run(type_what_is_asked(words_asked_for(riddle), look,
                                          lambda _p: [{"text": riddle, "center_x": 0.5, "center_y": 0.4,
                                                       "x": 0.1, "y": 0.35, "width": 0.8, "height": 0.1}],
                                          _Hands(), say=said.append))
    assert done["why"] == "it answered its own question" and not typed
    assert said and "Aura" in said[0]


def test_what_it_answered_is_the_new_part_of_the_screen_and_is_said_in_its_own_capitals():
    """Offline 2026-10-11 the machine answered "HENRY THE HATE..." and her line said nothing of it: the whole screen was
    quoted, its first letter lowered ("lYOUR"), and that read as a program's name and was left unsaid."""
    from core.language.what_a_watcher_hears import in_words_not_code
    from core.skills.screen_pursuit_as_it_happens import _answered_since_typed

    machine = {"typed": "Aura", "typed_kind": "a name", "typed_on": "FYOUR NAME HERE ETRYCOAIN GOUR SECRET DENTLTY",
               "typed_asked_by": "YOUR NAME HERE ETRYCOAIN GOUR SECRET DENTLTY"}
    answered = _answered_since_typed(machine, {"text": "LYOUR NAME HERE *TRYAGAIN HENRY THE HATE YOUR SECRET DENTITY"})
    assert answered == "HENRY THE HATE"
    line = f"I gave it my name, 'Aura', and it answered: {answered}."
    assert in_words_not_code(line) == line


def test_a_label_read_cleanly_only_after_typing_is_no_answer():
    """Offline 2026-10-11 "YOUR SECRET IDENTITY", read "GOUR SECRET DENTLTY" until then, was reported as the answer."""
    from core.skills.screen_pursuit_as_it_happens import _answered_since_typed

    machine = {"typed": "Aura", "typed_kind": "a name",
               "typed_on": "Riddle me this YOUR NAME HERE GO YOUR SECRET IDENTITY FYOUR NAME HERE ETRYCOAIN",
               "typed_asked_by": "YOUR NAME HERE"}
    assert not _answered_since_typed(machine, {"text": "YOUR NAME HERE Aura TRY AGAIN YOUR SECRET IDENTITY"})
    assert _answered_since_typed(machine, {"text": "YOUR NAME HERE TRY AGAIN THE PUZZLE BARON YOUR SECRET IDENTITY"}) == \
        "THE PUZZLE BARON"


def test_a_second_numbered_ask_for_a_name_is_typed_into_and_a_misread_one_is_not(monkeypatch):
    import time

    """Offline 2026-10-11 a golf game asked the second player's name, and OK was pressed thirty times with none."""
    from core.skills import screen_pursuit_as_it_happens as reflexes

    typed = []

    async def typing(asked, *a, **k):
        typed.append(asked.asked_by)
        return {"typed": "Aura", "kind": asked.kind}

    monkeypatch.setattr("core.agency.typing_what_is_asked.type_what_is_asked", typing)
    monkeypatch.setattr(reflexes, "_said_while_playing", lambda line: None)
    playing = reflexes.PlayingAsItHappens(None, (0, 0, 1, 1), "play it", time.monotonic() + 30)
    playing.words = ["Player 1 Enter your name. OK"]
    assert asyncio.run(playing._typed_what_it_asks())
    playing.words = ["Player 1 Enter your narne. OK"]
    assert not asyncio.run(playing._typed_what_it_asks()), "the same ask, misread"
    playing.words = ["Player 2 Enter your name. OK"]
    assert asyncio.run(playing._typed_what_it_asks()), "the next player's"
    assert len(typed) == 2


def test_a_choice_that_only_lights_up_is_a_choice_made():
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

    here = WhatWorksHere()
    here.asked_for_by("Please select number of players", (a_click_on("1 Player"), a_click_on("2 Players")))
    assert here.choosing_here
    here.tried(a_click_on("1 Player"), changed=False)
    assert here.chose_here


def test_a_place_of_questions_she_has_answered_by_typing_is_not_played_with_keys_when_its_screen_moves():
    """Offline 2026-10-12 a riddler's answer screen played its animation, was taken for play, and arrows were pressed at
    it for thirty-six seconds, twice, while the answer it gave went unread."""
    from types import SimpleNamespace

    from core.skills.screen_pursuit_as_it_happens import worked_by_typing

    asks = SimpleNamespace(mechanics={"questions": object()})
    a_game = SimpleNamespace(mechanics={"avoiding": object()})
    assert worked_by_typing({"typed": "Aura"}, asks, []) == "typing what it asks"
    assert worked_by_typing({"typed": "Aura"}, a_game, []) == ""          # a game that only asked her name first
    assert worked_by_typing({}, asks, []) == ""                           # nothing typed yet
    assert worked_by_typing({"typed": "Aura"}, asks, ["space"]) == ""     # its screen names a key to press
