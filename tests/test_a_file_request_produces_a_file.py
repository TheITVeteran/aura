"""Asked for a file, she made a folder — and reported success.

Live 2026-07-27. "Use your actual file tools right now: create a file on my
Desktop called aura_hello.txt containing one sentence you choose." The plan the
heuristic planner produced was a single ``create_folder`` step named "Aura
Desktop Task 1785195330". The folder was genuinely created, so the step
verified, so the task reported 1/1 steps completed, so she told the user the
objective had finished. What was actually on the Desktop was a junk folder with
a timestamp in its name, and no file.

A true receipt for the wrong action is worse than a failure. A failure gets
retried; this got believed. And the word "file" was in the request, along with
the filename and the extension.

The fix is not a special case for aura_hello.txt. Any request naming a file —
any name, any extension, any of the writable roots, or an explicit path — is
read before the folder heuristics get a vote, using the same path extractor the
effect contract and the desktop router already share, so all three agree about
what the objective is asking for.
"""
from __future__ import annotations

import json

import pytest

from core.skills.desktop_task import DesktopTaskSkill


def _plan(objective: str):
    skill = DesktopTaskSkill.__new__(DesktopTaskSkill)
    return skill._derive_steps_from_objective(objective, {})


def _target(step) -> dict:
    return json.loads(step.target) if isinstance(step.target, str) else dict(step.target)


def test_the_live_failure_now_writes_a_file() -> None:
    steps = _plan(
        "Use your actual file tools right now: create a file on my Desktop "
        "called aura_hello.txt containing one sentence you choose."
    )
    assert steps[0].action == "write_text_file"
    assert _target(steps[0])["path"] == "~/Desktop/aura_hello.txt"


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("create a file on my Desktop called notes.txt", "~/Desktop/notes.txt"),
        ("save a summary as report.md on my Desktop", "~/Desktop/report.md"),
        ("put a file named budget.csv in my Documents", "~/Documents/budget.csv"),
        ("write ~/Desktop/inner/plan.md with the plan", "~/Desktop/inner/plan.md"),
        ("make a file called script.py on the desktop", "~/Desktop/script.py"),
    ],
)
def test_any_named_file_routes_to_a_write(objective: str, expected: str) -> None:
    """No special case for one filename — the rule is "a file was named"."""
    steps = _plan(objective)
    assert steps[0].action == "write_text_file"
    assert _target(steps[0])["path"] == expected


def test_a_folder_request_still_makes_a_folder() -> None:
    """The fix must not swing the other way."""
    steps = _plan("make me a folder on the desktop for my trip photos")
    assert steps[0].action == "create_folder"


def test_the_step_declares_the_effect_it_expects() -> None:
    """An unverifiable step is how a wrong action passed as a right one."""
    step = _plan("create a file on my Desktop called notes.txt")[0]
    assert "notes.txt" in step.expect
    assert step.critical


# ── What goes in the file ─────────────────────────────────────────────────

def test_quoted_content_is_used_exactly() -> None:
    step = _plan('create a file on my Desktop called note.txt containing "Meet me at seven."')[0]
    assert _target(step)["content"].strip() == "Meet me at seven."


def test_content_left_to_her_is_hers_not_the_instruction() -> None:
    """The first attempt echoed the request into the file.

    "one sentence you choose. Actually execute it, then tell me the full path."
    is the instruction, not an answer to it.
    """
    step = _plan(
        "create a file on my Desktop called aura_hello.txt containing one "
        "sentence you choose. Actually execute it, then tell me the full path."
    )[0]
    content = _target(step)["content"]
    assert "Actually execute it" not in content
    assert "tell me the full path" not in content
    assert content.strip()


@pytest.mark.parametrize(
    "phrasing",
    [
        "containing one sentence you choose",
        "with whatever you like in it",
        "containing anything you want",
        "with a line of your choosing",
    ],
)
def test_every_way_of_leaving_it_to_her_is_recognised(phrasing: str) -> None:
    step = _plan(f"create a file on my Desktop called x.txt {phrasing}")[0]
    assert "you choose" not in _target(step)["content"]
    assert "whatever you like" not in _target(step)["content"]


def test_a_file_is_never_written_empty() -> None:
    """An empty file satisfies the letter of the request and fails it."""
    for objective in (
        "create a file on my Desktop called a.txt",
        "create a file on my Desktop called b.txt containing one sentence you choose",
    ):
        assert _target(_plan(objective)[0])["content"].strip()


def test_the_write_overwrites_rather_than_versioning() -> None:
    """Asked twice for the same path, the user means the same file."""
    assert _target(_plan("create a file on my Desktop called x.txt")[0])["overwrite"] is True
