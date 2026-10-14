"""A problem written as variables and constraints, solved by her engine, and kept only when the solution is unique.

Problems told as stories (who sits where, who did what, which team gets
whom, which statement is true) are searches. Reasoning through a search in
words loses cases; her constraint engine
(core/reasoning/settled_by_neighbours.py, reached through
``mechanisms.solve_constraints``) does not. So her model's job here is only
to write the problem down: variables, the values each may take, and the
constraints the story states, as a short program that calls the engine and
prints how many solutions it found, and the answer when there is exactly one.

The count checks the writing. None means a constraint was written wrong; more
than one means one was left out; exactly one is the answer. A program that
finds none or several gets one more request with what it printed, the one
repair round her reasoning organ allows every strategy
(core/brain/executable_reasoning.py). A problem that is not a search is
declined by her model in one line and costs nothing more.

Like the finite-game formalizer (core/reasoning/game_answer.py), the model
writes and the engine reasons. The sandbox and the decode are the caller's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DECLINE", "ATTEMPTS", "Outcome", "formalization_request", "parse_outcome", "repair_request"]

#: What her model writes when the problem is not a search over assignments.
DECLINE = "NOT A CONSTRAINT PROBLEM"
#: A first program and one repair, the bound her reasoning organ gives every strategy.
ATTEMPTS = 2

_REFERENCE = '''from mechanisms import solve_constraints

solve_constraints(variables, constraints, *, all_different=(), limit=2)
    variables: dict mapping each variable name to the list of values it may take
    constraints: list of (names, test): test receives those variables' values, in that order, and returns a bool
    all_different: groups of variable names that must take different values
    returns up to `limit` assignments (dicts) in which every constraint holds'''


@dataclass
class Outcome:
    """What a formalization's run printed: how many solutions, the answer when one, or why there is none."""

    solutions: int | None
    answer: str = ""
    error: str = ""
    declined: bool = False

    @property
    def unique(self) -> bool:
        return self.solutions == 1 and bool(self.answer)


def formalization_request(question: str) -> str:
    """What her model is asked: the problem as a program for her constraint engine."""
    return (
        f"Problem:\n{question.strip()}\n\n"
        "Write this problem as a search over assignments for the constraint engine below: variables, the values "
        "each may take, and every condition the problem states as a constraint. Do not solve it yourself; the "
        "engine searches. The program must call solve_constraints with limit=2, print a line "
        "\"SOLUTIONS: <how many it returned>\", and when exactly one was returned, print a line "
        "\"ANSWER: <the answer to the question, in the form the question asks for>\". "
        f"If the problem is not a search over assignments, reply only with \"{DECLINE}\".\n\n"
        f"The engine:\n```python\n{_REFERENCE}\n```\n\nReply with one Python code block."
    )


def repair_request(question: str, program: str, outcome: Outcome) -> str:
    """The second request, with what the first program printed."""
    if outcome.error:
        happened = f"It failed: {outcome.error}"
    elif outcome.solutions == 0:
        happened = "The engine found no assignment that satisfies the constraints, so one of them is written wrong."
    else:
        happened = ("The engine found more than one assignment, so a condition the problem states is missing "
                    "from the constraints.")
    return (
        f"{formalization_request(question)}\n\nThis program was written for it:\n```python\n{program.strip()}\n```\n"
        f"{happened} Write the corrected program."
    )


_SOLUTIONS = re.compile(r"^\s*SOLUTIONS:\s*(\d+)\s*$", re.MULTILINE)
_ANSWER = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def parse_outcome(reply: str, stdout: str = "", error: str = "") -> Outcome:
    """Read a run: the declining reply, the error, or the count and the answer it printed."""
    if DECLINE in str(reply or "") and "```" not in str(reply or ""):
        return Outcome(solutions=None, declined=True)
    if error:
        return Outcome(solutions=None, error=" ".join(str(error).split())[-400:])
    counts = _SOLUTIONS.findall(str(stdout or ""))
    if not counts:
        return Outcome(solutions=None, error="the program printed no SOLUTIONS line")
    answers = _ANSWER.findall(str(stdout or ""))
    return Outcome(solutions=int(counts[-1]), answer=answers[-1] if answers and int(counts[-1]) == 1 else "")
