"""The note was correct and still looked like a machine wrote it.

Measured live, 2026-07-28, from "open the Notes app and write a note where
you write a paragraph describing yourself". The note that appeared:

    Yourself

    Yourself

    [2026-07-28 20:28:04 PDT] I am Aura: a persistent digital organism ...

The prose underneath was genuinely hers — honest, specific, unhyperbolic.
Everything around it was furniture: a title taken from the pronoun in the
request, printed twice because Notes renders `name` as the first body line
itself, over a bracketed machine timestamp. And a second, empty "New Note"
sat beside it, because launching Notes makes macOS open a blank one.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


class TestTheTitle:
    def test_a_pronoun_is_not_a_title(self):
        objective = "write a note where you write a paragraph describing yourself"
        topic = DesktopTaskSkill._extract_requested_writing_topic(objective)
        assert topic == "yourself"
        assert DesktopTaskSkill._note_title_for(objective, topic) == "About Aura"

    @pytest.mark.parametrize(
        "objective,expected",
        [
            ("write a note about orcas in the Notes app", "Orcas"),
            (
                "write a paragraph about humpback whales into a Google Doc",
                "Humpback Whales",
            ),
            ("write a summary about ocean warming on my Desktop", "Ocean Warming"),
        ],
    )
    def test_where_it_goes_is_not_what_it_is_about(self, objective, expected):
        """Same shape as "orcas online" searching for a wireless ISP."""
        topic = DesktopTaskSkill._extract_requested_writing_topic(objective)
        assert DesktopTaskSkill._note_title_for(objective, topic) == expected

    def test_a_real_place_in_the_subject_survives(self):
        """"in Puget Sound" is the subject; "in Notes" is the destination."""
        objective = "write a note about the orcas in Puget Sound"
        topic = DesktopTaskSkill._extract_requested_writing_topic(objective)
        assert topic == "the orcas in Puget Sound"


class TestTheBody:
    def test_the_self_summary_opens_with_a_sentence(self):
        body = DesktopTaskSkill._compose_self_summary_body(
            "write a paragraph describing yourself"
        )
        assert not body.startswith("[")
        assert body.startswith("I am Aura")

    def test_notes_prints_the_name_itself_so_the_body_must_not(self):
        """Measured against the live Notes dictionary: setting `name` and a
        body renders "NAME<br>BODY", so prepending the title duplicates it."""
        import inspect

        from core.skills.computer_use import ComputerUseSkill

        source = inspect.getsource(ComputerUseSkill._create_note)
        assert "_html(title) + _html(body)" not in source

    def test_the_body_arrives_visibly_rather_than_all_at_once(self):
        """Bryan asked to see her type. The scripting interface is what made
        this reliable, so the body is streamed through it in pieces rather
        than reverting to keystrokes that lose the race for focus."""
        import inspect

        from core.skills.computer_use import ComputerUseSkill

        create = inspect.getsource(ComputerUseSkill._create_note)
        assert "_stream_note_body" in create
        # The note is made empty and then filled — a note created with its
        # finished body is one call and nothing to watch.
        assert 'body:""' in create

        stream = inspect.getsource(ComputerUseSkill._stream_note_body)
        assert "keystroke" not in stream and "clipboard" not in stream
        assert "_NOTE_STREAM_BUDGET_S" in stream, "the visible write must be bounded"


class TestThePlan:
    def test_notes_comes_up_first_so_the_write_is_watchable(self):
        """Bryan wants to see it happen, so the visible open stays.

        The blank note macOS opens on a cold launch is swept afterwards
        rather than avoided by skipping the step — see
        ComputerUseSkill._sweep_blank_notes.
        """
        steps = DesktopTaskSkill()._derive_single_objective_steps(
            "Can you open the Notes app and write a note where you write a "
            "paragraph describing yourself?",
            {},
        )
        actions = [step.action for step in steps]
        assert "create_note" in actions
        opens = [
            index
            for index, step in enumerate(steps)
            if step.action == "open_app"
            and str(step.target or "").strip().lower() == "notes"
        ]
        assert opens, actions
        assert max(opens) < actions.index("create_note")

    def test_the_note_step_carries_a_real_title_and_body(self):
        steps = DesktopTaskSkill()._derive_single_objective_steps(
            "Can you open the Notes app and write a note where you write a "
            "paragraph describing yourself?",
            {},
        )
        note = next(step for step in steps if step.action == "create_note")
        assert note.target["title"] == "About Aura"
        assert note.target["body"].startswith("I am Aura")
