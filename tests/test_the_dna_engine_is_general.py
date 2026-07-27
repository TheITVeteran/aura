"""An engine for any software, not a builder for two games.

The first version of this lane carried a hand-written oracle per target. That is
a builder for the programs someone thought of in advance, and it explains the
checkers board whose pieces did not move: nothing graded it, because nobody had
written a checkers grader.

The general move is to make the acceptance criteria part of the reconstruction.
She researches a target, decomposes it, and produces worked examples,
invariants, and an adapter — four function names saying what a fresh instance
is, what a user may do, what doing it produces, and how it is drawn. With those,
"do the pieces move?" is answerable for checkers, 2048, a spreadsheet or a
calculator without the grader knowing which it is holding.

So the target used here is deliberately one nobody would have written an oracle
for. If the engine only works on games, it is not an engine.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from core.self_improvement.artifact_grading import grade_artifact
from core.self_improvement.reconstruction_memory import (
    _LEDGER,
    PriorAttempt,
    attempt_from_outcome,
    recall_for,
)
from core.self_improvement.reconstruction_plan import (
    invariant_is_safe,
    plan_from_payload,
)

CALCULATOR_PLAN = {
    "target": "stack calculator",
    "summary": "reverse polish notation calculator",
    "components": [
        {"name": "evaluation", "responsibility": "apply operators", "entry_point": "push"},
        {"name": "view", "responsibility": "show the stack", "entry_point": "render"},
    ],
    "entry_points": ["new_stack", "legal_ops", "push", "render", "evaluate"],
    "worked_examples": [
        {
            "entry_point": "evaluate",
            "argument": ["2", "3", "+"],
            "expected": 5,
            "provenance": "definition of RPN",
        }
    ],
    "invariants": [
        {"description": "a fresh stack renders empty", "expression": "render(new_stack()) == '[]'"}
    ],
    "adapter": {
        "initial_state": "new_stack",
        "legal_actions": "legal_ops",
        "apply_action": "push",
        "render": "render",
        "illegal_action_examples": ["!!!"],
        "min_effective_actions": 4,
    },
}

FAITHFUL = '''
def new_stack():
    return []


def legal_ops(stack):
    ops = ["1", "2", "3"]
    if len(stack) >= 2:
        ops += ["+", "*"]
    return ops


def push(stack, token):
    values = list(stack)
    if token in ("+", "*"):
        if len(values) < 2:
            raise ValueError("need two operands")
        right, left = values.pop(), values.pop()
        values.append(left + right if token == "+" else left * right)
    elif token.isdigit():
        values.append(int(token))
    else:
        raise ValueError("unknown token")
    return values


def render(stack):
    return "[" + ", ".join(str(value) for value in stack) + "]"


def evaluate(tokens):
    stack = new_stack()
    for token in tokens:
        stack = push(stack, token)
    return stack[-1]
'''


@pytest.fixture()
def plan():
    parsed, problems = plan_from_payload(CALCULATOR_PLAN)
    assert not problems, problems
    return parsed


# ── A plan that cannot grade is refused before anything is built ───────────

def test_a_plan_with_no_criteria_is_rejected() -> None:
    payload = dict(CALCULATOR_PLAN, worked_examples=[], invariants=[])
    _, problems = plan_from_payload(payload)
    assert any("distinguish a faithful reconstruction" in problem for problem in problems)


def test_an_adapter_must_name_real_entry_points() -> None:
    payload = json.loads(json.dumps(CALCULATOR_PLAN))
    payload["adapter"]["apply_action"] = "not_declared"
    _, problems = plan_from_payload(payload)
    assert any("not an entry point" in problem for problem in problems)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('rm -rf /')", "open('/etc/passwd')", "x = 1", ""],
)
def test_an_invariant_cannot_be_a_second_program(expression: str) -> None:
    ok, why = invariant_is_safe(expression)
    assert not ok and why


def test_an_ordinary_property_is_allowed() -> None:
    assert invariant_is_safe("render(new_stack()) == '[]'")[0]


# ── The grader works on a target nobody wrote an oracle for ────────────────

def test_a_faithful_artifact_passes(plan) -> None:
    report = grade_artifact(FAITHFUL, plan)
    assert report.passed, report.summary


def test_state_that_never_changes_is_caught(plan) -> None:
    """This is "the pieces didn't move", on a calculator."""
    inert = FAITHFUL.replace("    return values\n", "    return list(stack)\n", 1)
    report = grade_artifact(inert, plan)
    assert not report.passed
    assert "interactive" in report.summary or "worked examples" in report.summary


def test_a_stub_is_caught(plan) -> None:
    stubbed = FAITHFUL.replace(
        '    return "[" + ", ".join(str(value) for value in stack) + "]"',
        "    pass  # TODO: render later",
    )
    report = grade_artifact(stubbed, plan)
    assert not report.passed
    assert "no stubs" in report.summary


def test_a_wrong_answer_is_caught(plan) -> None:
    wrong = FAITHFUL.replace("left + right if token", "left - right if token")
    report = grade_artifact(wrong, plan)
    assert not report.passed
    assert "worked examples" in report.summary


def test_a_broken_invariant_is_caught(plan) -> None:
    noisy = FAITHFUL.replace('return "[" + ", "', 'return "stack: [" + ", "')
    report = grade_artifact(noisy, plan)
    assert not report.passed
    assert "invariants" in report.summary


def test_a_missing_entry_point_is_caught(plan) -> None:
    report = grade_artifact(FAITHFUL.replace("def evaluate(", "def _evaluate("), plan)
    assert not report.passed
    assert "entry points" in report.summary


def test_a_module_that_runs_at_import_is_caught(plan) -> None:
    report = grade_artifact(FAITHFUL + "\nprint(evaluate(['1','2','+']))\n", plan)
    assert not report.passed
    assert "quiet import" in report.summary


def test_a_calculator_is_not_required_to_end(plan) -> None:
    """Only a plan that claims a terminal state is held to one.

    A calculator, an editor and a spreadsheet keep accepting input forever.
    Marking them broken for that would make this a game grader wearing a
    general name.
    """
    assert grade_artifact(FAITHFUL, plan).passed
    ending = json.loads(json.dumps(CALCULATOR_PLAN))
    ending["adapter"]["expects_terminal_state"] = True
    strict, _ = plan_from_payload(ending)
    assert not grade_artifact(FAITHFUL, strict).passed


# ── What she learned last time comes back ──────────────────────────────────

def test_a_similar_prior_build_is_recalled() -> None:
    root = Path(tempfile.mkdtemp())
    (root / _LEDGER.parent).mkdir(parents=True, exist_ok=True)
    rows = [
        PriorAttempt(
            target="checkers",
            summary="board game with captures",
            entry_points=("initial_board", "legal_moves", "apply_move", "render"),
            corrections=("interactive: the pieces do not move",),
            succeeded=False,
            at=time.time() - 100,
        ),
        PriorAttempt(
            target="invoice parser",
            summary="pdf text extraction",
            entry_points=("parse",),
            succeeded=True,
            at=time.time() - 10,
        ),
    ]
    (root / _LEDGER).write_text(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows) + "\n"
    )
    brief = recall_for("2048", summary="sliding tile board game", root=root)
    assert [prior.target for prior in brief.priors] == ["checkers"]
    block = brief.as_prompt_block()
    assert "the pieces do not move" in block, "the correction is the transferable part"
    assert "not as the answer" in block, "a prior is evidence, not a template"


def test_no_history_is_not_an_error() -> None:
    assert recall_for("anything", root=Path(tempfile.mkdtemp())).is_empty


def test_a_rejection_records_what_the_gate_caught(plan) -> None:
    attempt = attempt_from_outcome(
        plan, succeeded=False, findings=["interactive: the pieces do not move"]
    )
    assert attempt.corrections
    assert not attempt.invariants, "unverified properties are not lessons"


def test_a_success_records_the_properties_that_held(plan) -> None:
    attempt = attempt_from_outcome(plan, succeeded=True)
    assert attempt.succeeded
    assert attempt.invariants
