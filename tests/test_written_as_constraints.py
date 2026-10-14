"""A story written as constraints is kept only when her engine finds exactly one solution."""

from __future__ import annotations

import pytest

from core.reasoning.written_as_constraints import (
    DECLINE,
    Outcome,
    formalization_request,
    parse_outcome,
    repair_request,
)
from core.sandbox.untrusted_python import available_boundary

PROGRAM = '''from mechanisms import solve_constraints
people = ["Ann", "Bo", "Cy"]
variables = {p: [1, 2, 3] for p in people}
constraints = [(["Ann", "Bo"], lambda a, b: a < b), (["Cy"], lambda c: c == 1)]
found = solve_constraints(variables, constraints, all_different=[people], limit=2)
print(f"SOLUTIONS: {len(found)}")
if len(found) == 1:
    print("ANSWER: " + min(people, key=lambda p: found[0][p]))
'''


def test_the_count_is_read_and_only_a_unique_solution_gives_an_answer() -> None:
    assert parse_outcome("x", "SOLUTIONS: 1\nANSWER: Cy").unique
    two = parse_outcome("x", "SOLUTIONS: 2\nANSWER: Cy")
    assert two.solutions == 2 and not two.unique and two.answer == ""
    assert parse_outcome(DECLINE).declined
    assert parse_outcome("x", "", "NameError: y").error.startswith("NameError")
    assert parse_outcome("x", "nothing").error
    request = formalization_request("Who is first?")
    assert "solve_constraints" in request and DECLINE in request and "limit=2" in request
    assert "missing" in repair_request("Who is first?", PROGRAM, Outcome(solutions=2))
    assert "written wrong" in repair_request("Who is first?", PROGRAM, Outcome(solutions=0))


@pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")
@pytest.mark.asyncio
async def test_a_written_program_runs_her_engine_in_the_sandbox() -> None:
    from core.brain.symbolic_sandbox import SymbolicSandbox
    from core.learning.procedures_from_solved_examples import _mechanisms_directory, _mechanisms_prelude

    result = await SymbolicSandbox().run(PROGRAM, prelude=_mechanisms_prelude(),
                                         extra_read_paths=(_mechanisms_directory(),))
    outcome = parse_outcome(PROGRAM, result.stdout, "" if result.ok else result.stderr or "failed")
    assert outcome.unique and outcome.answer == "Cy"
