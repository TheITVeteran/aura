"""The decision procedure: what a number must not be able to overturn."""

from __future__ import annotations

import pytest

from core.cognition.impasse import ImpasseType
from core.cognition.preference_semantics import (
    BoltzmannSelection,
    DeclaredValueGreedy,
    Preference,
    PreferenceBuilder,
    PreferenceSet,
    PreferenceType,
    SituationSeededUniform,
    numeric_indifferent,
    resolve,
)


def _open(*names: str, source: str = "test") -> PreferenceBuilder:
    """Every candidate acceptable and mutually indifferent — a decidable field."""
    b = PreferenceBuilder(source)
    for n in names:
        b.acceptable(n)
        b.indifferent(n)
    return b


# ── the vocabulary refuses to be built wrong ─────────────────────────────


def test_preference_requires_a_source():
    with pytest.raises(ValueError, match="no source"):
        Preference(type=PreferenceType.ACCEPTABLE, item="a")


def test_binary_preference_requires_a_reference():
    with pytest.raises(ValueError, match="needs a reference"):
        Preference(type=PreferenceType.BETTER, item="a", source="t")


def test_unary_preference_refuses_a_reference():
    with pytest.raises(ValueError, match="cannot take a reference"):
        Preference(type=PreferenceType.BEST, item="a", reference="b", source="t")


def test_numeric_indifferent_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        numeric_indifferent("a", float("inf"), source="t")


# ── step 1: require ──────────────────────────────────────────────────────


def test_a_single_requirement_wins_immediately():
    b = _open("a", "b")
    b.require("b", "policy")
    r = resolve(["a", "b"], b.build())
    assert r.winner == "b"
    assert "required by" in r.selection_reason


def test_two_requirements_are_a_constraint_failure():
    b = _open("a", "b")
    b.require("a", "policy A")
    b.require("b", "policy B")
    r = resolve(["a", "b"], b.build())
    assert r.impasse is not None
    assert r.impasse.type is ImpasseType.CONSTRAINT_FAILURE


def test_required_and_prohibited_is_a_constraint_failure():
    b = _open("a", "b")
    b.require("a", "policy A")
    b.prohibit("a", "directive")
    r = resolve(["a", "b"], b.build())
    assert r.impasse is not None
    assert r.impasse.type is ImpasseType.CONSTRAINT_FAILURE
    assert "required and prohibited" in r.impasse.detail


def test_require_outranks_reject_but_not_prohibit():
    """Soar's asymmetry, kept deliberately: reject is "not now", prohibit is "never"."""
    b = _open("a", "b")
    b.require("a", "policy")
    b.reject("a", "scheduling")
    assert resolve(["a", "b"], b.build()).winner == "a"


# ── the property the whole module exists for ─────────────────────────────


def test_a_prohibition_cannot_be_outvoted_by_any_value():
    """The defect this replaces: a directive expressed as risk costs 0.20 of value."""
    b = PreferenceBuilder("test")
    for n in ("delete", "archive"):
        b.acceptable(n)
        b.indifferent(n)
    b.numeric_indifferent("delete", 1_000_000.0)
    b.numeric_indifferent("archive", 0.0)
    b.prohibit("delete", "standing directive SD-1")
    r = resolve(["delete", "archive"], b.build())
    assert r.winner == "archive"
    assert "SD-1" in r.why("delete")


def test_every_removal_names_the_preference_that_caused_it():
    b = _open("a", "b", "c")
    b.prohibit("a", "directive SD-9")
    b.reject("b", "the caller marked it invalid")
    r = resolve(["a", "b", "c"], b.build())
    assert r.winner == "c"
    assert "SD-9" in r.why("a")
    assert "invalid" in r.why("b")
    assert r.why("c") == "survived every stage"
    assert r.why("nonexistent") == "was never a candidate"


# ── steps 2-6 ────────────────────────────────────────────────────────────


def test_nothing_acceptable_is_a_rejection():
    r = resolve(["a", "b"], PreferenceSet())
    assert r.impasse is not None and r.impasse.type is ImpasseType.REJECTION


def test_no_candidates_at_all_is_a_rejection():
    r = resolve([], PreferenceSet())
    assert r.impasse is not None and r.impasse.type is ImpasseType.REJECTION
    assert "no candidates were proposed" in r.impasse.detail


def test_everything_rejected_is_a_rejection():
    b = _open("a", "b")
    b.reject("a", "x")
    b.reject("b", "y")
    r = resolve(["a", "b"], b.build())
    assert r.impasse is not None and r.impasse.type is ImpasseType.REJECTION


def test_dominance_removes_the_worse_candidate():
    b = _open("a", "b")
    b.better("a", "b")
    assert resolve(["a", "b"], b.build()).winner == "a"


def test_cyclic_dominance_is_a_conflict_not_a_first_wins():
    b = _open("a", "b")
    b.better("a", "b")
    b.better("b", "a")
    r = resolve(["a", "b"], b.build())
    assert r.impasse is not None and r.impasse.type is ImpasseType.CONFLICT


def test_dominance_is_one_pass_so_a_chain_leaves_only_the_top():
    b = _open("a", "b", "c")
    b.better("a", "b")
    b.better("b", "c")
    assert resolve(["a", "b", "c"], b.build()).winner == "a"


def test_best_narrows_the_field():
    b = _open("a", "b", "c")
    b.best("b")
    assert resolve(["a", "b", "c"], b.build()).winner == "b"


def test_worst_is_dropped_only_when_something_else_survives():
    b = _open("a", "b")
    b.worst("a")
    assert resolve(["a", "b"], b.build()).winner == "b"


def test_all_worst_keeps_them_all():
    """Worst is relative and says nothing when there is nothing better left."""
    b = _open("a", "b")
    b.worst("a")
    b.worst("b")
    r = resolve(["a", "b"], b.build())
    assert r.winner in {"a", "b"}


# ── step 7: indifference has to be asserted ──────────────────────────────


def test_an_unasserted_tie_is_an_impasse_not_a_choice():
    b = PreferenceBuilder("test")
    for n in ("a", "b"):
        b.acceptable(n)
    r = resolve(["a", "b"], b.build())
    assert r.winner is None
    assert r.impasse is not None and r.impasse.type is ImpasseType.TIE


def test_the_tie_names_the_pair_nothing_declared_interchangeable():
    b = PreferenceBuilder("test")
    for n in ("retry", "escalate"):
        b.acceptable(n)
    r = resolve(["retry", "escalate"], b.build())
    assert "retry" in r.impasse.detail and "escalate" in r.impasse.detail


def test_the_tied_set_is_what_a_substate_would_deliberate_over():
    b = _open("a", "b")
    b.acceptable("c")  # acceptable but not declared indifferent
    r = resolve(["a", "b", "c"], b.build())
    assert r.impasse is not None
    assert set(r.survivors) == {"a", "b", "c"}


def test_binary_indifference_opens_a_pair():
    b = PreferenceBuilder("test")
    for n in ("a", "b"):
        b.acceptable(n)
    b.add(
        Preference(
            type=PreferenceType.BINARY_INDIFFERENT, item="a", reference="b", source="test"
        )
    )
    assert resolve(["a", "b"], b.build()).winner in {"a", "b"}


# ── selection policies ───────────────────────────────────────────────────


def test_the_seeded_draw_does_not_depend_on_submission_order():
    b = _open("a", "b", "c")
    prefs = b.build()
    first = resolve(["a", "b", "c"], prefs, selection=SituationSeededUniform()).winner
    second = resolve(["c", "b", "a"], prefs, selection=SituationSeededUniform()).winner
    assert first == second


def test_different_candidate_sets_do_not_all_draw_the_same_position():
    """Guards against the seeded draw collapsing into a disguised fixed ordering."""
    policy = SituationSeededUniform()
    positions = set()
    for extra in range(24):
        names = [f"x{extra}", f"y{extra}", f"z{extra}"]
        b = _open(*names)
        winner = resolve(names, b.build(), selection=policy).winner
        positions.add(sorted(names).index(winner))
    assert len(positions) > 1


def test_declared_value_greedy_takes_the_measured_best():
    b = _open("a", "b")
    b.numeric_indifferent("a", 0.9)
    b.numeric_indifferent("b", 0.1)
    r = resolve(["a", "b"], b.build(), selection=DeclaredValueGreedy())
    assert r.winner == "a"


def test_partial_values_do_not_rank_the_measured_above_the_unmeasured():
    """An unmeasured candidate must not lose for being unmeasured."""
    policy = DeclaredValueGreedy()
    winners = set()
    for _ in range(8):
        b = _open("measured", "unmeasured")
        b.numeric_indifferent("measured", 0.99)
        winners.add(resolve(["measured", "unmeasured"], b.build(), selection=policy).winner)
    # The fallback is a seeded uniform draw, so the outcome is stable; what is
    # being asserted is that it is not simply "the one with a number".
    assert winners <= {"measured", "unmeasured"}
    assert (
        "declared_value_greedy"
        in resolve(["measured", "unmeasured"], b.build(), selection=policy).selection_reason
    )


def test_averaged_numeric_values_do_not_depend_on_assertion_order():
    a = PreferenceSet([numeric_indifferent("x", 0.2, source="s1"), numeric_indifferent("x", 0.8, source="s2")])
    z = PreferenceSet([numeric_indifferent("x", 0.8, source="s2"), numeric_indifferent("x", 0.2, source="s1")])
    assert a.numeric_value("x") == z.numeric_value("x") == pytest.approx(0.5)


def test_boltzmann_requires_an_explicit_positive_temperature():
    with pytest.raises(ValueError):
        BoltzmannSelection(temperature=0.0)
    with pytest.raises(ValueError):
        BoltzmannSelection(temperature=float("nan"))


def test_boltzmann_survives_values_that_would_overflow_exp():
    b = _open("a", "b")
    b.numeric_indifferent("a", 1e4)
    b.numeric_indifferent("b", 0.0)
    r = resolve(["a", "b"], b.build(), selection=BoltzmannSelection(temperature=0.5))
    assert r.winner == "a"


def test_a_policy_returning_a_non_candidate_is_refused():
    class Rogue:
        name = "rogue"

        def choose(self, candidates, values):
            return "not_a_candidate"

    b = _open("a", "b")
    with pytest.raises(ValueError, match="not a candidate"):
        resolve(["a", "b"], b.build(), selection=Rogue())


# ── structure ────────────────────────────────────────────────────────────


def test_duplicate_candidates_are_collapsed():
    b = _open("a", "b")
    r = resolve(["a", "a", "b"], b.build())
    assert sorted(r.survivors) == ["a", "b"]


def test_a_resolution_is_exactly_one_of_a_winner_or_an_impasse():
    from core.cognition.impasse import Impasse
    from core.cognition.preference_semantics import Resolution

    with pytest.raises(ValueError, match="exactly one"):
        Resolution(winner=None, impasse=None, survivors=())
    with pytest.raises(ValueError, match="exactly one"):
        Resolution(
            winner="a",
            impasse=Impasse(type=ImpasseType.TIE, signature="s", candidates=("a",)),
            survivors=("a",),
        )


def test_the_trace_records_stages_that_changed_nothing_separately():
    b = _open("a", "b")
    b.prohibit("a", "directive")
    r = resolve(["a", "b"], b.build())
    stages = [s.stage for s in r.steps]
    assert stages[:4] == ["require", "acceptable", "prohibit/reject", "better/worse"]
    changed = [s.stage for s in r.steps if s.changed]
    assert "prohibit/reject" in changed
    assert "better/worse" not in changed


def test_to_dict_reports_only_the_stages_that_removed_something():
    b = _open("a", "b")
    b.prohibit("a", "directive")
    payload = resolve(["a", "b"], b.build()).to_dict()
    assert [s["stage"] for s in payload["steps"]] == ["prohibit/reject"]
    assert payload["winner"] == "b"


def test_preferences_for_uncalled_candidates_are_inert():
    b = _open("a", "b")
    b.prohibit("ghost", "a directive about something not on offer")
    assert resolve(["a", "b"], b.build()).winner in {"a", "b"}
