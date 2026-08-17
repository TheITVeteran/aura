"""Two paths changed Aura's own source, and they disagreed.

`SafeSelfModification.validate_proposal` classified its target with
`classify_mutation_path` and refused a sealed path outright.
`self_code_improver.improve_function` — reached by the `improve_own_code`
skill, which declares `requires_approval = False` and defaults
`enact=True` — used its own containment check instead: inside the source
root, off a substring denylist, ends in `.py`.

Measured against the tier scheme, that admitted
`core/self_modification/safe_modification.py` — tier3_sealed, the file
holding the other route's refusal — for autonomous rewriting. The weaker of
the two rules won, because it belonged to the path that did not ask.

A constitution with a route around it is not a constitution. Both callers
ask `admit_mutation` now, and these tests hold them to the same answer.
"""

from __future__ import annotations

import asyncio

import pytest

from core.self_modification.mutation_constitution import (
    DEFER,
    ENACT,
    PROPOSE,
    REFUSE,
    MutationAdmission,
    admit_mutation,
)
from core.self_modification.mutation_tiers import MutationTier

SEALED = "core/self_modification/safe_modification.py"
ORDINARY = "core/brain/cognitive_engine.py"


# ── the answer is one of four, and it says which ────────────────────────────


def test_a_disposition_outside_the_four_is_refused():
    with pytest.raises(ValueError, match="disposition"):
        MutationAdmission(disposition="probably_fine", tier=MutationTier.SEALED, reason="")


def test_a_sealed_path_is_refused_however_the_turn_looks():
    """Sealed outranks owner approval through this route: its own gates are
    external review, a manual patch and a cold restart, and a running
    process cannot perform any of them on itself."""
    for owner_approved in (False, True):
        for trust in ("trusted", "unknown", "untrusted"):
            admission = admit_mutation(SEALED, owner_approved=owner_approved, turn_trust=trust)
            assert admission.disposition == REFUSE
            assert admission.tier is MutationTier.SEALED
            assert not admission.may_enact
            assert not admission.may_propose


def test_an_untrusted_turn_is_refused_on_an_ordinary_path():
    admission = admit_mutation(ORDINARY, turn_trust="untrusted")

    assert admission.disposition == REFUSE
    assert "untrusted" in admission.reason


def test_an_unknown_turn_defers_and_the_draft_survives():
    admission = admit_mutation(ORDINARY, turn_trust="unknown")

    assert admission.disposition == DEFER
    assert not admission.may_enact
    assert admission.may_propose, "a deferral must not throw the draft away"


def test_owner_approval_answers_the_unknown_turn():
    admission = admit_mutation(ORDINARY, owner_approved=True, turn_trust="unknown")

    assert admission.disposition == ENACT


def test_a_clean_turn_on_an_ordinary_path_may_enact():
    admission = admit_mutation(ORDINARY, turn_trust="trusted")

    assert admission.disposition == ENACT
    assert admission.required_gates, "an enactment with no gates is not a tier"


def test_every_answer_carries_a_receipt():
    for trust in ("trusted", "unknown", "untrusted"):
        receipt = admit_mutation(ORDINARY, turn_trust=trust).receipt
        assert receipt["schema"] == "aura.mutation_constitution.v1"
        assert receipt["tier"] and receipt["disposition"] and receipt["turn_trust"]
        assert receipt["required_gates"]


# ── neither route may be the weaker one ─────────────────────────────────────


def test_the_improver_refuses_the_sealed_path_before_spending_a_model_call():
    """The defect, run end to end.

    Refusal comes before generation on purpose: drafting a patch for a
    sealed path only produces a rewrite of Aura's own governance that
    nothing may ever apply.
    """
    from core.capabilities.self_code_improver import improve_function

    result = asyncio.run(
        improve_function(
            target_file=SEALED,
            func_name="validate_proposal",
            goal="make it faster",
            checks=[{"args": [1], "expected": 1}],
            enact=True,
        )
    )

    assert result.enacted is False
    assert result.status == f"mutation_{REFUSE}"
    assert result.mutation_admission["tier"] == "tier3_sealed"
    # Nothing was generated: the model was never asked.
    assert result.improved_source == ""
    assert result.iterations == 0


def test_both_routes_ask_the_same_module():
    """A structural check, so a future edit cannot re-open the second door."""
    import inspect

    from core.capabilities import self_code_improver
    from core.self_modification import safe_modification

    for module in (self_code_improver, safe_modification):
        source = inspect.getsource(module)
        assert "admit_mutation(" in source, (
            f"{module.__name__} changes Aura's source without asking the constitution"
        )


def test_the_improver_classifies_by_repo_relative_path():
    """`_confine_target` returns an absolute path; the tier patterns are
    repo-relative globs. Handing them an absolute path matched none of them,
    so every target graded as the default tier and a sealed file read as
    ordinary."""
    from pathlib import Path

    from core.capabilities.self_code_improver import _repo_relative, _self_code_root

    absolute = _self_code_root() / "core" / "self_modification" / "safe_modification.py"

    assert _repo_relative(Path(absolute)) == SEALED
    assert admit_mutation(_repo_relative(Path(absolute))).tier is MutationTier.SEALED
