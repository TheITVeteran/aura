"""She should meet an app she has never seen and be able to work it.

Bryan's correction, 2026-07-28, and the right one:

    oh, i wanted aura to have general os control. thats why i didnt want a
    "Create note" function ... because she may not have access to a notes
    application. but she should generally know how to search the name of an
    application, understand what it does (by context, memory, or code), and
    interact with it. Especially something like typing

A create_note action is one app hardcoded on a machine that happens to have
Notes. macOS already publishes the general answer: every scriptable app ships
a scripting definition describing its own object model, so "make a new X and
set its Y" is derivable per app with no integration written for any of them.

These tests assert the derivation, not the apps. Notes answering note.body is
a *result*; nothing in core/ should contain that fact.
"""

from __future__ import annotations

import pytest

from core.perception.app_dictionary import (
    installed_apps,
    read_dictionary,
    resolve_app,
    text_target_for,
)
from core.skills.desktop_task import DesktopTaskSkill


class TestFindingAnApp:
    def test_she_can_enumerate_what_is_installed(self):
        apps = installed_apps()
        assert apps, "no applications found on this machine"
        assert "Notes" in apps or "TextEdit" in apps

    @pytest.mark.parametrize(
        "spoken", ["notes", "Notes", "NOTES", "the Notes app", "Notes.app"]
    )
    def test_a_spoken_name_reaches_the_installed_app(self, spoken):
        name, path = resolve_app(spoken)
        assert name == "Notes"
        assert path.endswith("Notes.app")

    def test_an_app_that_is_not_installed_is_not_guessed_at(self):
        """Acting on a guess is how you type into the wrong window."""
        assert resolve_app("DefinitelyNotInstalledXYZ") == ("", "")


class TestUnderstandingAnApp:
    def test_the_write_route_is_derived_not_declared(self):
        """The single most important assertion in this file.

        Notes answering note.body must be a derivation from the machine, not
        a fact stored in core/.
        """
        import pathlib

        target = text_target_for("Notes")
        assert target is not None
        assert (target.klass, target.text_property) == ("note", "body")

        # The module may mention Notes in prose; it must not BRANCH on it.
        source = pathlib.Path("core/perception/app_dictionary.py").read_text()
        code_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith(chr(34) * 3):
                continue
            code_lines.append(line)
        code = chr(10).join(code_lines)
        for hardcoded in ("Notes", "TextEdit", "Reminders"):
            assert f'"{hardcoded}"' not in code, (
                f"{hardcoded} is quoted in executable code — that is the "
                "per-app hardcoding this module replaces"
            )

    def test_a_different_app_derives_a_different_route(self):
        target = text_target_for("TextEdit")
        assert target is not None
        assert (target.klass, target.text_property) == ("document", "text")

    def test_an_unscriptable_app_says_so_rather_than_guessing(self):
        """Calculator has no document. That is an answer, not a failure."""
        assert text_target_for("Calculator") is None

    def test_a_name_is_not_a_body(self):
        """`type="text"` covers names, senders and paths, so accepting the
        type made a document body come back as `window.given name`."""
        target = text_target_for("Google Chrome")
        assert target is None or target.text_property not in {"given name", "name"}

    def test_she_can_say_what_an_app_is(self):
        facts = read_dictionary("Notes")
        assert facts.path
        assert facts.scriptable
        assert facts.can_be_written_to
        missing = read_dictionary("DefinitelyNotInstalledXYZ")
        assert not missing.path
        assert "not installed" in missing.unavailable_reason


class TestWritingIntoAnyApp:
    def test_the_planner_asks_the_app_instead_of_recognising_it(self):
        steps = DesktopTaskSkill()._derive_single_objective_steps(
            "Open TextEdit and write a paragraph about orcas", {}
        )
        actions = [step.action for step in steps]
        assert "write_in_app" in actions, actions
        write = next(step for step in steps if step.action == "write_in_app")
        assert write.target["app"] == "TextEdit"
        assert write.target["body"]

    def test_notes_takes_the_same_path_as_everything_else(self):
        steps = DesktopTaskSkill()._derive_single_objective_steps(
            "Can you open the Notes app and write a note where you write a "
            "paragraph describing yourself?",
            {},
        )
        write = next(step for step in steps if step.action == "write_in_app")
        assert write.target["app"] == "Notes"
        assert write.target["title"] == "About Aura"

    def test_an_unscriptable_app_falls_back_rather_than_failing(self):
        """Calculator cannot hold a paragraph, so the work becomes a file —
        which is honest, and better than typing at a window that will not
        keep focus."""
        steps = DesktopTaskSkill()._derive_single_objective_steps(
            "Open Calculator and write a paragraph about orcas", {}
        )
        actions = [step.action for step in steps]
        assert "write_in_app" not in actions
        assert "write_text_file" in actions

    def test_any_installed_app_can_be_named(self):
        """The extractor was an eleven-name table, so "open Reminders" named
        no app at all."""
        assert DesktopTaskSkill._extract_apps(
            "Open Reminders and write a note about the demo"
        ) == ["Reminders"]

    def test_a_phrase_is_still_not_an_app_name(self):
        """"in your own words" once launched Microsoft Word."""
        assert DesktopTaskSkill._extract_apps(
            "write a note in your own words about orcas"
        ) == []


class TestTheActionIsOneAction:
    def test_create_note_resolves_through_the_general_path(self):
        import inspect

        from core.skills.computer_use import ComputerUseSkill

        dispatch = inspect.getsource(ComputerUseSkill._execute_action)
        assert 'action in {"write_in_app", "create_note"}' in dispatch

    def test_both_names_are_allowed_and_retry_safe(self):
        from core.runtime.desktop_task_contract import (
            DESKTOP_TASK_ALLOWED_ACTIONS,
            DESKTOP_TASK_RETRY_SAFE_ACTIONS,
        )

        assert "write_in_app" in DESKTOP_TASK_ALLOWED_ACTIONS
        assert "create_note" in DESKTOP_TASK_ALLOWED_ACTIONS
        assert "write_in_app" in DESKTOP_TASK_RETRY_SAFE_ACTIONS
