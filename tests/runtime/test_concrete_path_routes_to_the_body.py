"""A request naming a real path cannot be answered without looking.

LIVE, 2026-08-10. "Count how many .py files are in
/Users/bryan/.aura/live-source/core/introspection, then write that number and
the file names into ~/Documents/aura_probe_count.txt. Tell me the number."

looks_like_desktop_objective returned False, so the turn never reached a tool.
She answered from nothing: 3 instead of 9, three filenames that do not exist,
and a report of a write that never happened.

"write hello into ~/Documents/x.txt" routed correctly, so the action+surface
pair works when the action verb LEADS. This sentence opens with "count how
many", and the pure read "how many .py files are in /abs/path?" failed the same
way. Worse, the compound form was being claimed by
looks_like_capability_inventory_dialogue_request — "how many ... files" read as
a question about her own inventory — which decided the turn before any other
check ran.

The path is what settles it. Nothing in the model can answer a question about
the contents of a real path, and nothing can write to one without the body.
Asked about a path she has not read, the only honest moves are to look or to
decline, and she did neither.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import looks_like_desktop_objective


@pytest.mark.parametrize(
    "message",
    [
        "Count how many .py files are in /Users/bryan/.aura/live-source/core/introspection, "
        "then write that number and the file names into ~/Documents/aura_probe_count.txt. "
        "Tell me the number.",
        "how many .py files are in /Users/bryan/.aura/live-source/core/introspection?",
        "write hello into ~/Documents/x.txt",
        "list the contents of ~/Documents",
        "what's in /etc/hosts",
        "read /Users/bryan/.aura/live-source/CLAUDE.md and summarise it",
    ],
)
def test_path_operations_reach_the_body(message: str) -> None:
    assert looks_like_desktop_objective(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what skills do you have",
        "list your capabilities",
        "what is the capital of Peru",
        "explain the input/output problem",
        "what do you think about /r/programming these days",
        "I think our http/2 support is fine",
        "tell me about your memory system",
        "",
    ],
)
def test_prose_and_capability_questions_do_not(message: str) -> None:
    """A slash in a word is not a path; a question about her is not a task."""
    assert looks_like_desktop_objective(message) is False


def test_the_path_check_runs_before_the_inventory_check() -> None:
    """Ordering is the fix — the inventory check was deciding this turn."""
    import inspect

    from core.runtime import desktop_objective_intent as module

    source = inspect.getsource(module.looks_like_desktop_objective)
    path_at = source.find("_asks_about_a_concrete_path(sanitized_text)")
    inventory_at = source.find("looks_like_capability_inventory_dialogue_request(user_message)")

    assert path_at != -1 and inventory_at != -1
    assert path_at < inventory_at


def test_a_path_alone_is_not_enough() -> None:
    """Mentioning a path in passing is not a request to operate on it."""
    from core.runtime.desktop_objective_intent import _asks_about_a_concrete_path

    assert _asks_about_a_concrete_path("i keep my notes in ~/documents these days") is False
    assert _asks_about_a_concrete_path("list the files in ~/documents") is True
