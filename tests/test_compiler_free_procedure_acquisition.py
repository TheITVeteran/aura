"""No P_k anywhere in the path: the held-out family, induced and lesioned.

The external review put its flag here. Aura learns within representations and
procedures the architecture supplies; the strongest controlled evidence still
runs through a hand-written compiler that recognises the domain and supplies
the program structure —

    x --> P_k(x) --> N_theta

— and what is missing is a P that learns what program a family needs.

These tests hold the two halves that make the answer worth anything. The
positive half: a procedure is induced from twelve input/output pairs with no
family label, frozen, transferred to instances it never saw, and both lesions
collapse. The other half, which is the one that decides it: the rig refuses
when it should. A family a single primitive solves voids the run, and permuted
outputs induce nothing.
"""

from __future__ import annotations

import random

import pytest

from core.learning.compiler_free_experiment import (
    HELDOUT_FAMILY,
    TRAIN_FAMILIES,
    VERDICT_ACQUIRED,
    VERDICT_NOT_ACQUIRED,
    VERDICT_VOID,
    run_compiler_free_experiment,
)
from core.learning.procedure_induction import (
    PRIMITIVE_SET_SHA,
    PRIMITIVES,
    ProcedureInducer,
    TaskFamily,
    TaskInstance,
    accuracy,
)

pytestmark = pytest.mark.unit


def _family(fn, family_id: str = "probe") -> TaskFamily:
    def generate(rng: random.Random) -> TaskInstance:
        values = tuple(rng.randint(1, 30) for _ in range(rng.randint(3, 6)))
        return TaskInstance(inputs=(values,), output=fn(values))

    return TaskFamily(family_id, generate)


# ---------------------------------------------------------------------------
# The rig must be able to refuse
# ---------------------------------------------------------------------------


def test_permuted_outputs_induce_nothing():
    """The decisive control. A searcher that fits noise proves nothing by fitting."""
    family = _family(lambda v: sum(v) - max(v))
    inducer = ProcedureInducer(max_depth=3)
    found = 0
    for seed in range(15):
        support = family.sample(12, seed=seed)
        outputs = [item.output for item in support]
        random.Random(5000 + seed).shuffle(outputs)
        permuted = [
            TaskInstance(item.inputs, output)
            for item, output in zip(support, outputs)
        ]
        if inducer.induce(permuted).found:
            found += 1
    assert found == 0, f"induced a program from permuted outputs on {found}/15 runs"


def test_a_single_primitive_family_voids_the_run():
    """A family one primitive solves is a compiled family wearing a costume."""
    result = run_compiler_free_experiment(
        heldout=_family(lambda v: max(v), "just_the_largest"),
        null_runs=3,
        transfer_size=40,
    )
    assert result.verdict == VERDICT_VOID
    assert result.single_primitive_shortcut is True
    assert "compiled family" in result.reasons[0]


def test_an_unreachable_family_is_reported_not_forced():
    """Depth 1 cannot compose, so the held-out family must come back empty."""
    result = run_compiler_free_experiment(max_depth=1, null_runs=2, transfer_size=40)
    assert result.verdict in {VERDICT_NOT_ACQUIRED, VERDICT_VOID}
    assert result.induced is None or result.transfer_accuracy == 0.0


# ---------------------------------------------------------------------------
# ...and then succeed
# ---------------------------------------------------------------------------


def test_the_heldout_procedure_is_acquired_and_survives_its_lesions():
    result = run_compiler_free_experiment()

    assert result.verdict == VERDICT_ACQUIRED, result.reasons
    assert result.induced is not None
    assert result.single_primitive_shortcut is False
    assert result.null_found == 0

    # Frozen, then transferred to instances drawn from a different seed.
    assert result.transfer_accuracy == 1.0
    assert result.transfer_size >= 200

    # Both lesions collapse. `no_composition` is the arm that matters: it keeps
    # the primitives and removes only the composition.
    by_name = {arm.name: arm.accuracy for arm in result.lesions}
    assert by_name["no_composition"] < 0.1
    assert by_name["no_procedure"] < 0.1
    assert result.gain > 0.8


def test_the_same_inducer_solves_families_it_was_not_written_for():
    """One family would be one lucky primitive set."""
    result = run_compiler_free_experiment(null_runs=2, transfer_size=40)
    assert len(result.train_families_solved) >= 2
    assert result.heldout_family not in result.train_families_solved


def test_the_induced_program_is_a_composition_not_a_lookup():
    result = run_compiler_free_experiment(null_runs=2, transfer_size=40)
    assert result.induced is not None
    assert result.induced["depth"] >= 2, result.induced["expression"]
    assert result.induced["frozen_sha"].startswith("sha256:")


def test_transfer_instances_are_not_the_support_instances():
    """Otherwise 'transfer' is the training score under another name."""
    support = HELDOUT_FAMILY.sample(12, seed=20260816 + 500)
    fresh = HELDOUT_FAMILY.sample(200, seed=20260816 + 777)
    overlap = {(item.inputs, item.output) for item in support} & {
        (item.inputs, item.output) for item in fresh
    }
    assert not overlap


# ---------------------------------------------------------------------------
# No family-specific code exists anywhere in the path
# ---------------------------------------------------------------------------


def test_the_inducer_is_never_told_which_family_it_is_solving():
    """Structural. The signature cannot accept what it must not know."""
    import inspect

    params = set(inspect.signature(ProcedureInducer.induce).parameters)
    assert not {"family", "family_id", "kind", "domain", "task_type"} & params
    assert set(TaskInstance.__dataclass_fields__) == {"inputs", "output"}


def test_no_primitive_is_a_family_in_disguise():
    """The property is behavioural, not lexical.

    A first version of this compared names and flagged `count_of`, which is an
    ordinary count-occurrences operation. What actually matters is whether any
    single primitive *is* the answer to a family: that would make composition
    a formality and the rig a compiler with extra steps.
    """
    families = {f.family_id for f in TRAIN_FAMILIES} | {HELDOUT_FAMILY.family_id}
    assert not {p.name for p in PRIMITIVES} & families

    depth_one = ProcedureInducer(max_depth=1)
    for family in (*TRAIN_FAMILIES, HELDOUT_FAMILY):
        outcome = depth_one.induce(family.sample(12, seed=4242))
        assert not outcome.found, (
            f"{family.family_id} is solved by the single primitive "
            f"{outcome.program.describe() if outcome.program else '?'}"
        )


def test_the_primitive_set_is_pinned():
    """A set edited to fit a family changes this hash, and the result records it."""
    assert PRIMITIVE_SET_SHA.startswith("sha256:")
    result = run_compiler_free_experiment(null_runs=2, transfer_size=40)
    assert result.primitive_set_sha == PRIMITIVE_SET_SHA


def test_a_family_is_a_generator_and_never_carries_a_solver():
    for family in (*TRAIN_FAMILIES, HELDOUT_FAMILY):
        assert set(TaskFamily.__dataclass_fields__) == {"family_id", "generate"}
        instance = family.sample(1, seed=1)[0]
        assert not hasattr(instance, "family_id")


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_a_frozen_program_reruns_identically():
    result = run_compiler_free_experiment(null_runs=2, transfer_size=40)
    assert result.induced is not None
    rerun = run_compiler_free_experiment(null_runs=2, transfer_size=40)
    assert rerun.induced is not None
    assert rerun.induced["frozen_sha"] == result.induced["frozen_sha"]


def test_accuracy_is_zero_on_an_empty_set():
    inducer = ProcedureInducer(max_depth=2)
    outcome = inducer.induce(_family(lambda v: sum(v)).sample(6, seed=3))
    assert outcome.found
    assert accuracy(outcome.program, []) == 0.0


def test_an_arity_mismatch_is_refused():
    support = [
        TaskInstance(((1, 2),), 3),
        TaskInstance(((1, 2), 5), 3),
    ]
    outcome = ProcedureInducer().induce(support)
    assert not outcome.found and "arity" in outcome.refusal


# ---------------------------------------------------------------------------
# This is a rig, and it says so
# ---------------------------------------------------------------------------


def test_the_experiment_modules_are_declared_experimental_with_reasons():
    """An experiment has no production consumer, and must not pretend to.

    `make reachability` counts modules reached only by tests, because that is
    how a dead runtime protection hides behind passing tests. An experiment is
    the honest exception, so it is declared — with a reason, because an
    undeclared list is where "experimental" becomes "unfinished and forgotten".
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    declared = json.loads(
        (root / "config" / "experimental_modules.json").read_text(encoding="utf-8")
    )["modules"]

    for module in (
        "core.learning.procedure_induction",
        "core.learning.compiler_free_experiment",
    ):
        assert module in declared, f"{module} is a rig and must say so"
        assert len(declared[module]) > 40, f"{module} needs a real reason, not a label"
