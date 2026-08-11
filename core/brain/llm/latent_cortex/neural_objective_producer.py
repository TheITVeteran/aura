"""The neural solver for public objective programs — a producer, not a critic.

Split out of objective_program_verifier.py on 2026-08-10, which had been both.

That module is listed in ``_CRITIC_SOURCE_FILES``, so the critic identity audit
reads its imports and found ``mlx.core`` arriving through
``neural_transition_tissue`` and ``systematic_neural_alu``. Every latent-cortex
turn therefore failed with::

    critic function identity is not independently proven:
    dependency_audit_failed(undeclared_internal_imports=[...]);
    function_identity_distinct=False

and the critic never attached to a single live turn.

The two requirements looked contradictory — one test demands the canonical
producer run teacher-removed neural tissue, another demands the critic be free
of the neural generator — but only because one file was doing both jobs. They
are not in tension once the jobs are separated, which is what "disjoint
authority" meant in the first place.

Worth stating plainly, because it is the substantive half: the grader used to
compute its expected value by running the same learned tissue that produced the
answer it was grading. It stayed sound only because the result was cross-checked
against an independent parser and disagreement raised — and that is the tell.
The parser was the authority. The neural execution contributed an identity
violation and no evidence. So the critic keeps the parser and the certified
typed recurrence; the neural rollin, which is a claim about what the student can
do unaided, lives here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.brain.llm.latent_cortex.neural_transition_tissue import (
    NeuralTransitionTissue,
    execute_neural_action_program,
    load_neural_transition_tissue,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (
    _BOOLEAN_OBJECTIVE_RE,
    _MODULAR_OBJECTIVE_RE,
    _boolean_expected,
    _execute_objective,
    _modular_expected,
    build_solution_receipt,
)
from core.brain.llm.latent_cortex.systematic_neural_alu import (
    SystematicNeuralALU,
    execute_systematic_neural_program,
    load_systematic_neural_alu,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (
    compile_public_transition_program,
)

__all__ = [
    "neural_compiled_transition_expected",
    "solve_objective_program_neural",
    "validate_objective_program_solution_neural",
]


@lru_cache(maxsize=1)
def _resident_neural_transition_tissue() -> NeuralTransitionTissue:
    return load_neural_transition_tissue()


@lru_cache(maxsize=1)
def _resident_systematic_neural_alu() -> SystematicNeuralALU:
    return load_systematic_neural_alu()


def neural_compiled_transition_expected(
    objective: str,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Run learned neural tissue and independently verify its public result.

    Returns ``None`` when the objective does not compile to a transition
    program. When the sealed artifact is missing the caller falls back to the
    deterministic path, because an unavailable student is not a failed one.
    """

    try:
        program = compile_public_transition_program(objective)
    except ValueError:
        return None
    if program.family == "boolean":
        try:
            tissue = _resident_neural_transition_tissue()
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return None
        execution = execute_neural_action_program(program, tissue)
        match = _BOOLEAN_OBJECTIVE_RE.match(objective)
        if match is None:
            raise RuntimeError("compiled Boolean objective lost parser agreement")
        family = "nested_boolean"
        expected = {"value": execution.terminal_state[1]}
        crosscheck, crosscheck_receipt = _boolean_expected(match)
        engine = "neural_transition_tissue.v1"
    elif program.family == "modular":
        try:
            tissue = _resident_systematic_neural_alu()
            execution = execute_systematic_neural_program(program, tissue)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return None
        match = _MODULAR_OBJECTIVE_RE.match(objective)
        if match is None:
            raise RuntimeError("compiled modular objective lost parser agreement")
        family = "modular_chain"
        expected = {"residue": execution.terminal_state[1]}
        crosscheck, crosscheck_receipt = _modular_expected(match)
        engine = "systematic_neural_alu.v1"
    else:  # pragma: no cover - the compiler's family registry is closed
        raise RuntimeError("compiled transition family is unsupported")
    if crosscheck != expected:
        raise RuntimeError("neural recurrent execution and independent parser disagree")
    return family, expected, {
        "engine": engine,
        "teacher_available": False,
        "tissue_sha256": tissue.tissue_sha256,
        "compiler": program.public_receipt(),
        "student_rollin": execution.receipt(),
        "independent_crosscheck": crosscheck_receipt,
        "independent_crosscheck_match": True,
    }


def solve_objective_program_neural(objective: str) -> tuple[str, dict[str, Any]] | None:
    """Solve a public objective with the learned tissue where one applies.

    Falls back to the deterministic solver for objectives that do not compile
    to a transition program, and for installations without the sealed neural
    artifact. The receipt shape is identical either way — ``execution.engine``
    is what says which ran.
    """

    executed = neural_compiled_transition_expected(objective)
    if executed is None:
        executed = _execute_objective(objective)
    if executed is None:
        return None
    return build_solution_receipt(objective, executed)


def validate_objective_program_solution_neural(
    value: Any,
    *,
    objective: str,
    candidate: str,
) -> dict[str, Any]:
    """Re-derive a neural solution receipt and require an exact match.

    Validation must rebuild through the SAME engine that produced the receipt.
    The deterministic validator in objective_program_verifier rebuilds through
    the certified typed recurrence, so handing it a neural receipt reports a
    reconstruction difference for a receipt that was never wrong — the engine
    fields simply differ.
    """

    rebuilt = solve_objective_program_neural(objective)
    if rebuilt is None:
        raise ValueError("objective program solution is unavailable")
    expected_candidate, expected_receipt = rebuilt
    if candidate != expected_candidate or value != expected_receipt:
        raise ValueError("objective program solution reconstruction differs")
    return expected_receipt
