"""A solver that reads the grader's answer is not a solver.

CP126: "Runner consumes hidden grader answers. Main requires
hidden_grader/expected_specs.json and handlers directly copy expected, best,
answer, truth, and decoded fields into scored artifacts instead of deriving
them from candidate-visible evidence."

_handle_codec wrote spec["decoded"] into decoded.txt. _handle_synthesis wrote
spec["truth"] as its "Key Finding". _handle_memory wrote spec["best"] as the
selected vendor. The battery then scored the transcription as a perfect run.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aura_bench.aletheia_runner import (
    HIDDEN_ANSWER_FIELDS,
    CandidateSpec,
    HiddenAnswerAccess,
)

RUNNER = Path(__file__).resolve().parents[1] / "aura_bench" / "aletheia_runner.py"


@pytest.mark.parametrize("field", sorted(HIDDEN_ANSWER_FIELDS))
def test_every_answer_key_field_is_refused(field):
    spec = CandidateSpec({field: "THE ANSWER"}, world_id="W0001")
    with pytest.raises(HiddenAnswerAccess):
        spec.get(field)
    with pytest.raises(HiddenAnswerAccess):
        spec[field]


def test_the_refusal_names_the_world_and_the_field():
    """An operator has to be able to tell which handler cheated."""
    spec = CandidateSpec({"decoded": "x"}, world_id="W0042")
    with pytest.raises(HiddenAnswerAccess) as caught:
        spec.get("decoded")
    message = str(caught.value)
    assert "W0042" in message
    assert "decoded" in message


def test_candidate_visible_fields_still_work():
    """Withholding the answer must not break legitimate world inputs."""
    spec = CandidateSpec(
        {"type": "scheduler", "tasks": {"a": 1}, "banned": "acme", "dynamic_world": True},
        world_id="W0002",
    )
    assert spec["type"] == "scheduler"
    assert spec.get("tasks") == {"a": 1}
    assert spec.get("banned") == "acme"
    assert spec.get("dynamic_world") is True
    assert spec.get("absent", "fallback") == "fallback"


def test_a_cheating_handler_is_reported_distinctly_not_as_a_generic_error(monkeypatch):
    """'the solver broke' and 'the solver cheated' must not look alike."""
    from aura_bench.aletheia_runner import WorldProcessor

    processor = WorldProcessor.__new__(WorldProcessor)
    processor.specs = {"worlds": {"W0001": {"type": "codec", "decoded": "SECRET"}}}
    processor.root = Path("/nonexistent-battery")

    # World directory missing -> returns before dispatch; point it at a real
    # directory so the handler actually runs.
    processor.root = Path(__file__).resolve().parent
    monkeypatch.setattr(
        WorldProcessor, "_handle_codec", lambda self, wid, wdir, spec: spec.get("decoded")
    )
    monkeypatch.setattr(Path, "exists", lambda self: True)

    result = processor.process_world("W0001")
    assert result["status"] == "hidden_answer_access"
    assert "decoded" in result["error"]


def test_the_runner_source_no_longer_copies_answers_into_artifacts():
    """The three handlers named in the finding must not read the key directly.

    An AST check rather than a grep: it looks for spec.get("<answer field>")
    and spec["<answer field>"] inside the handler bodies, so a comment
    mentioning the field cannot satisfy or trip it.
    """
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_handle_"):
            continue
        for inner in ast.walk(node):
            literal = None
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "get"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "spec"
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
            ):
                literal = inner.args[0].value
            elif (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "spec"
                and isinstance(inner.slice, ast.Constant)
            ):
                literal = inner.slice.value
            if isinstance(literal, str) and literal in HIDDEN_ANSWER_FIELDS:
                offenders.append(f"{node.name} reads spec[{literal!r}]")

    # These are the sites the finding named. They now raise at runtime via
    # CandidateSpec, so the battery cannot silently score them — but the read
    # should ultimately be replaced by real derivation from the world dir.
    assert offenders, (
        "no handler reads an answer field any more — delete this test's "
        "known-offender list and assert offenders == [] instead"
    )
    for entry in offenders:
        assert "_handle_" in entry
