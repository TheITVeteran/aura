"""On whose behalf was the browser driven, and what counted as grounds.

Two CP126 findings about attribution.

e14d8807 — every effect site minted its own governed scope from a source
STRING. The receipt said which line of code acted; nothing said on whose
behalf. Two conversations, or a background job and a foreground one, produced
indistinguishable receipts.

412f25e9 — free-form context could replace the corpus search outright, so
whatever a caller returned became "grounded contradictions", emitted to an
external service as Aura's grounded pushback and recorded in her challenge log
as such.
"""
from __future__ import annotations

import pytest

from core.capabilities.web_interlocutor import (
    _caller_authority,
    _effect_constraints,
)

pytestmark = pytest.mark.unit


# --- effects carry the caller (e14d8807) --------------------------------


def test_an_effect_outside_a_run_names_no_principal():
    """Inventing a plausible caller would be the defect, not the fix."""
    constraints = _effect_constraints("web_interlocutor.test_site")

    assert constraints["initiating_principal"] == ""
    assert constraints["interlocutor_run_id"] == ""
    assert constraints["effect_site"] == "web_interlocutor.test_site"
    assert constraints["user_visible_browser_action"] is True


def test_an_effect_inside_a_run_carries_the_caller_and_the_run():
    with _caller_authority("bryan", "webchat-run-abc123"):
        constraints = _effect_constraints("web_interlocutor.submit")

    assert constraints["initiating_principal"] == "bryan"
    assert constraints["interlocutor_run_id"] == "webchat-run-abc123"


def test_two_runs_are_distinguishable_in_the_receipt():
    with _caller_authority("bryan", "run-one"):
        first = _effect_constraints("web_interlocutor.submit")
    with _caller_authority("bryan", "run-two"):
        second = _effect_constraints("web_interlocutor.submit")

    assert first["interlocutor_run_id"] != second["interlocutor_run_id"]


def test_the_authority_does_not_leak_past_the_run():
    with _caller_authority("bryan", "run-one"):
        pass

    assert _effect_constraints("x")["initiating_principal"] == ""


def test_nested_runs_restore_the_outer_caller():
    with _caller_authority("outer", "run-outer"):
        with _caller_authority("inner", "run-inner"):
            assert _effect_constraints("x")["initiating_principal"] == "inner"
        assert _effect_constraints("x")["initiating_principal"] == "outer"


def test_every_effect_scope_in_the_module_stamps_the_constraints():
    """A scope opened without them is one whose receipt cannot be attributed."""
    import ast
    import pathlib

    source = pathlib.Path("core/capabilities/web_interlocutor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    unstamped: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name != "local_internal_governed_scope":
            continue
        kwargs = {kw.arg for kw in node.keywords}
        if "constraints" not in kwargs:
            unstamped.append(node.lineno)

    assert not unstamped, (
        "governed scope(s) opened without caller attribution at line(s) "
        f"{unstamped}"
    )


# --- grounding says where it came from (412f25e9) -----------------------


def test_an_injected_corpus_is_recorded_as_injected():
    import inspect

    from core.capabilities.web_interlocutor import WebInterlocutorSession

    source = inspect.getsource(WebInterlocutorSession._grounded_challenge)

    assert 'grounding_source = "injected" if injected_search else "local_corpus"' in source
    assert '"grounding_source": grounding_source' in source


def test_the_contradiction_evidence_is_fenced():
    """The claim half is the remote party's text and the counter half is corpus
    text; both are data, not instructions."""
    import inspect

    from core.capabilities.web_interlocutor import WebInterlocutorSession

    source = inspect.getsource(WebInterlocutorSession._grounded_challenge)

    assert "_injection_guard(fence)" in source
    assert "_fence_safe(c.interlocutor_claim, fence)" in source
    assert "_fence_safe(c.counter_evidence, fence)" in source


# --- the subject survives the parse (312c2bbf) --------------------------


from core.capabilities.web_interlocutor import (  # noqa: E402
    _dialogue_goal_from_objective,
    _strip_leading_scaffolding,
    parse_dialogue_goal,
)


@pytest.mark.parametrize(
    "objective,must_survive",
    [
        (
            "Talk to ChatGPT about how browsers use conversation turns in real time",
            ["browsers", "conversation turns", "real time"],
        ),
        (
            "Ask Claude about running a real experiment on live data",
            ["running", "real experiment", "live data"],
        ),
        (
            "Open ChatGPT and discuss the use of a single-turn exchange as evidence",
            ["use of a single-turn exchange"],
        ),
    ],
)
def test_words_inside_the_subject_are_not_deleted(objective, must_survive):
    """"use", "run", "real", "browser", "conversation" and "turns" were deleted
    globally, so a topic containing them came out as mush and the opening
    message was composed from the mush."""
    goal = _dialogue_goal_from_objective(objective).lower()

    for fragment in must_survive:
        assert fragment in goal, f"{fragment!r} was deleted from {goal!r}"


def test_leading_scaffolding_is_still_removed():
    goal = _dialogue_goal_from_objective(
        "Can you please open ChatGPT and talk about tidal resonance"
    ).lower()

    assert goal.startswith("tidal resonance")
    assert "chatgpt" not in goal


def test_the_original_casing_is_preserved():
    goal = _dialogue_goal_from_objective(
        "Talk to ChatGPT about how NASA measures Antarctic ice loss"
    )

    assert "NASA" in goal
    assert "Antarctic" in goal


@pytest.mark.parametrize(
    "text,expected",
    [
        ("open ChatGPT and discuss tides", "discuss tides"),
        ("please use the browser to explore tides", "explore tides"),
        ("tides", "tides"),
        ("", ""),
    ],
)
def test_scaffolding_stripping_only_touches_the_front(text, expected):
    assert _strip_leading_scaffolding(text) == expected


def test_the_parse_is_reported_rather_than_silent():
    receipt = parse_dialogue_goal(
        "Can you open ChatGPT and talk about tidal resonance in shallow basins"
    )

    assert receipt["objective"].startswith("Can you open ChatGPT")
    assert "tidal resonance in shallow basins" in receipt["parsed_goal"]
    assert receipt["dropped_chars"] > 0


def test_an_untouched_objective_reports_no_rewrite():
    receipt = parse_dialogue_goal("tidal resonance in shallow basins")

    assert receipt["parsed_goal"] == "tidal resonance in shallow basins"
    assert receipt["rewritten"] is False
