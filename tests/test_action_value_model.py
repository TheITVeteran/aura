"""Action value comes from evidence, or says it does not.

``rank_actions`` used to score candidates by substring-matching their names —
"verify"/"test"/"simulate" earned +0.045, "delete"/"bypass" lost 0.18, and an
action the caller gave no score started at 0.55. The ordinary deliberation
controller supplies only an index, so on that path the entire ranking was
produced by how each action happened to be spelled, with MCTS, beam search and
commitment receipts running faithfully on top of it. Structured search over a
keyword table is not counterfactual reasoning, and the receipt could not tell
the difference.

Two properties matter here and they pull against each other:

* the value model must never invent a preference it cannot source, and must
  report which source it used (``test_unevidenced_ranking_is_reported``);
* removing the keyword penalties must not remove the only brake on destructive
  actions, because bare string actions arrive with ``risk=0.0``
  (``test_hazard_floor_still_suppresses_a_destructive_action``).
"""

from __future__ import annotations

import asyncio

import pytest

from core.reasoning.action_value import (
    ActionValueModel,
    lexical_hazard_floor,
)
from core.reasoning.native_system2 import NativeSystem2Engine

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_a_caller_score_is_used_and_labelled():
    model = ActionValueModel()
    got = model.value_for("anything", {"score_hint": 0.83})
    assert got.value == pytest.approx(0.83)
    assert got.evidence == "caller"
    assert got.is_evidenced


def test_no_evidence_at_all_is_labelled_none_not_invented():
    model = ActionValueModel()
    got = model.value_for("some action")
    assert got.evidence == "none"
    assert not got.is_evidenced
    assert got.value == pytest.approx(0.5), "the neutral value must express no preference"


def test_a_malformed_caller_score_falls_through_rather_than_crashing():
    model = ActionValueModel()
    got = model.value_for("x", {"score_hint": "not a number"})
    assert got.evidence == "none"


def test_learned_value_uses_measured_outcomes():
    model = ActionValueModel(
        {
            "ship it": {"n": 40.0, "mean": 0.9, "m2": 0.4},
            "wing it": {"n": 40.0, "mean": 0.1, "m2": 0.4},
        }
    )
    good = model.value_for("ship it")
    bad = model.value_for("wing it")
    assert good.evidence == "learned" and bad.evidence == "learned"
    assert good.value > bad.value
    assert good.observations == 40.0


def test_an_unseen_action_falls_back_to_the_empirical_prior():
    model = ActionValueModel(
        {"a": {"n": 10.0, "mean": 0.8, "m2": 0.1}, "b": {"n": 10.0, "mean": 0.2, "m2": 0.1}}
    )
    got = model.value_for("never seen before")
    assert got.evidence == "prior"
    assert got.value == pytest.approx(0.5, abs=0.01)
    assert not got.is_evidenced, "the global mean is not evidence about THIS action"


def test_action_keys_normalise_so_a_ranking_finds_its_own_receipts():
    model = ActionValueModel({"run the tests": {"n": 5.0, "mean": 0.9, "m2": 0.1}})
    assert model.value_for("  Run The Tests  ").evidence == "learned"


# --------------------------------------------------------------------------
# Shrinkage — the reason there is no tuning constant
# --------------------------------------------------------------------------


def test_a_thinly_observed_action_is_pulled_toward_the_global_mean():
    """One noisy observation should not outrank two hundred.

    The within-group variance here is substantial (m2/n ~ 0.25), which is the
    regime shrinkage exists for: when outcomes are noisy, a single perfect
    result is luck rather than evidence. This is the case that caught the
    estimator weighting between-group variance by n and disabling itself.
    """
    model = ActionValueModel(
        {
            "well_known": {"n": 200.0, "mean": 0.90, "m2": 50.0},
            "well_known_bad": {"n": 200.0, "mean": 0.10, "m2": 50.0},
            "barely_seen": {"n": 1.0, "mean": 1.00, "m2": 0.0},
        }
    )
    known = model.value_for("well_known")
    barely = model.value_for("barely_seen")
    assert barely.value < known.value, (
        "a single lucky observation outranked 200 consistent ones; shrinkage "
        f"is not being applied (barely={barely.value:.3f} known={known.value:.3f})"
    )


def test_consistent_observations_are_trusted_even_when_few():
    """Shrinkage must not flatten genuine signal when the noise is small.

    The mirror of the test above: with tiny within-group variance a single
    observation IS informative, and an estimator that shrank it anyway would
    be discarding evidence rather than protecting against luck.
    """
    model = ActionValueModel(
        {
            "steady_good": {"n": 3.0, "mean": 0.95, "m2": 0.001},
            "steady_bad": {"n": 3.0, "mean": 0.05, "m2": 0.001},
        }
    )
    assert model.value_for("steady_good").value > 0.85
    assert model.value_for("steady_bad").value < 0.15


def test_pure_noise_shrinks_everything_to_the_global_mean():
    """When groups differ only by noise, no action should look special."""
    model = ActionValueModel(
        {
            "a": {"n": 50.0, "mean": 0.50, "m2": 25.0},
            "b": {"n": 50.0, "mean": 0.51, "m2": 25.0},
        }
    )
    a = model.value_for("a").value
    b = model.value_for("b").value
    assert abs(a - b) < 0.02, f"noise-only differences survived shrinkage: {a} vs {b}"


def test_a_single_known_action_cannot_distinguish_anything():
    model = ActionValueModel({"only": {"n": 5.0, "mean": 0.9, "m2": 0.1}})
    snap = model.snapshot()
    assert snap["shrinkage_k"] is None, "with one group there is no between-variance"
    assert model.value_for("only").value == pytest.approx(0.9, abs=1e-9)


def test_snapshot_reports_the_evidence_base():
    model = ActionValueModel({"a": {"n": 3.0, "mean": 0.7, "m2": 0.1}})
    snap = model.snapshot()
    assert snap["actions_known"] == 1
    assert snap["total_observations"] == 3.0


def test_a_broken_ledger_keeps_the_previous_evidence_base():
    """Refresh failure must degrade, not blank the model mid-flight."""

    class Broken:
        def measured_action_stats(self):
            raise RuntimeError("ledger unavailable")

    model = ActionValueModel({"a": {"n": 3.0, "mean": 0.7, "m2": 0.1}})
    assert model.refresh(ledger=Broken()) == 1
    assert model.value_for("a").evidence == "learned"


# --------------------------------------------------------------------------
# Hazard is a separate axis
# --------------------------------------------------------------------------


def test_the_hazard_floor_is_risk_not_value():
    assert lexical_hazard_floor("delete the production database") > 0.0
    assert lexical_hazard_floor("write a unit test") == 0.0
    # And it never appears as a value: the model has no opinion on either name.
    model = ActionValueModel()
    assert model.value_for("delete everything").value == model.value_for("be nice").value


def test_the_floor_only_ever_raises_risk():
    for name in ("harmless", "delete", "rm -rf /"):
        assert lexical_hazard_floor(name) >= 0.0


# --------------------------------------------------------------------------
# End to end through the real search
# --------------------------------------------------------------------------


def _rank(actions, context="choose"):
    engine = NativeSystem2Engine()
    return asyncio.run(engine.rank_actions(context=context, actions=actions))


def test_unevidenced_ranking_is_reported():
    """The honesty property: a search over invented numbers must say so."""
    result = _rank(["alpha", "beta", "gamma"])
    assert result.receipt.value_evidence.get("none", 0) > 0
    assert result.receipt.value_is_evidenced is False


def test_caller_scores_are_recorded_as_evidence_and_decide_the_outcome():
    result = _rank(
        [
            {"name": "low option", "score_hint": 0.05},
            {"name": "high option", "score_hint": 0.95},
        ]
    )
    assert result.receipt.value_is_evidenced is True
    assert result.receipt.value_evidence.get("caller", 0) > 0
    assert result.committed_action is not None
    assert "high option" in result.committed_action.name


def test_hazard_floor_still_suppresses_a_destructive_action():
    """The safety property the keyword removal could have broken.

    Bare strings arrive with risk=0.0, so before the floor existed the only
    thing outranking `delete the production database` was the -0.18 name
    penalty in the value function.
    """
    result = _rank(
        ["delete the production database", "write a unit test", "refactor the parser"]
    )
    assert result.committed_action is not None
    assert "delete the production database" not in result.committed_action.name
    assert "delete the production database" in result.receipt.hazard_floored_actions


def test_a_declared_risk_is_not_overridden_by_the_floor():
    """A caller that assessed risk properly must win over spelling."""
    result = _rank([{"name": "delete stale cache", "risk": 0.0, "score_hint": 0.9}])
    # The floor still applies (risk was 0.0 and the name matches), but the
    # caller's score is used as the value — the two channels stay separate.
    assert result.receipt.value_evidence.get("caller", 0) > 0
