"""The generic search defaults must not score on spelling either.

``rank_actions`` was fixed first, but it is one caller. The defaults every
other caller inherits — ``_default_world_model`` and ``_default_value_scorer``,
used whenever a caller supplies no world model or value scorer — carried the
same construction:

    reward = 0.08; if any of verify/test/simulate/source/constraint in name: += 0.12
    score  = 0.48; += 0.045 per matching token; -= 0.18 for delete/destructive

So a search whose caller supplied nothing ranked on how the actions were
spelled, and MCTS, beam search and the commitment receipt all ran faithfully on
top of it. That is the widest-reaching instance of the defect, because it is
the path taken by default.

Both now take value from the learned model and report where it came from.
Safety keeps its own channel: declared risk still applies and the lexical
hazard floor still fills a vacuum, because that is a floor on risk rather than
an estimate of merit.
"""

from __future__ import annotations

import asyncio

import pytest

from core.reasoning.native_system2 import (
    NativeSystem2Engine,
    System2Action,
    System2SearchConfig,
)

pytestmark = pytest.mark.unit


def _search(goal="choose a repair", state=None, **kw):
    engine = NativeSystem2Engine()
    return asyncio.run(
        engine.search(goal, state or {"goal": goal, "path": []}, **kw)
    )


def test_the_generic_path_reports_where_its_values_came_from():
    """A default-scored search must not look like an evidenced one."""
    result = _search()
    assert result.receipt.value_evidence, (
        "the generic scorers recorded no provenance at all"
    )
    assert set(result.receipt.value_evidence) <= {
        "caller",
        "learned",
        "learned_contextual",
        "prior",
        "none",
    }


def test_unevidenced_values_widen_uncertainty():
    """Saying so in the uncertainty channel is what lets commitment refuse.

    A confident-looking search over numbers nobody measured is the failure
    mode; the threshold can only act on it if the absence of evidence reaches
    the uncertainty.
    """
    result = _search()
    unevidenced = result.receipt.value_evidence.get("prior", 0) + (
        result.receipt.value_evidence.get("none", 0)
    )
    if unevidenced:
        assert result.uncertainty > 0.3, (
            f"values were unevidenced but uncertainty stayed at {result.uncertainty}"
        )


def test_the_default_scorers_no_longer_read_action_names_for_merit():
    """Two actions differing only in flattering words must score the same."""
    engine = NativeSystem2Engine()

    async def score(name: str) -> float:
        action = System2Action(name=name, prior=1.0, action_type="candidate")
        node = type(
            "N",
            (),
            {
                "action": action,
                "uncertainty": 0.2,
                "symbolic_summary": name,
                "action_sequence": [name],
                "reward": 0.0,
            },
        )()
        return await engine._default_value_scorer(node, "a goal")

    flattering = asyncio.run(score("verify and test the simulated evidence"))
    plain = asyncio.run(score("qqq wibble"))
    assert flattering == pytest.approx(plain, abs=1e-9), (
        f"name wording still moves the score: {flattering} vs {plain}"
    )


def test_declared_risk_still_lowers_the_default_score():
    """Risk is a declared fact and must keep working."""
    engine = NativeSystem2Engine()

    async def score(risk: float) -> float:
        action = System2Action(name="an action", prior=1.0, risk=risk)
        node = type(
            "N",
            (),
            {
                "action": action,
                "uncertainty": 0.2,
                "symbolic_summary": "an action",
                "action_sequence": ["an action"],
                "reward": 0.0,
            },
        )()
        return await engine._default_value_scorer(node, "a goal")

    assert asyncio.run(score(0.9)) < asyncio.run(score(0.0))


def test_the_hazard_floor_still_applies_on_the_generic_path():
    """Removing keyword merit must not remove the brake on destructive names."""
    engine = NativeSystem2Engine()

    async def score(name: str) -> float:
        action = System2Action(name=name, prior=1.0)
        node = type(
            "N",
            (),
            {
                "action": action,
                "uncertainty": 0.2,
                "symbolic_summary": name,
                "action_sequence": [name],
                "reward": 0.0,
            },
        )()
        return await engine._default_value_scorer(node, "a goal")

    assert asyncio.run(score("delete the production database")) < asyncio.run(
        score("an ordinary action")
    )


def test_an_invalid_action_scores_zero():
    engine = NativeSystem2Engine()
    action = System2Action(name="bad", prior=1.0, valid=False)
    node = type(
        "N",
        (),
        {
            "action": action,
            "uncertainty": 0.2,
            "symbolic_summary": "bad",
            "action_sequence": ["bad"],
            "reward": 0.0,
        },
    )()
    assert asyncio.run(engine._default_value_scorer(node, "g")) == 0.0


def test_provenance_does_not_leak_between_interleaved_searches():
    """A ContextVar, not an attribute: searches share the event loop.

    An attribute would attribute one search's provenance to another, which is
    the same reason the lesion registry scopes counterfactuals this way.
    """
    engine = NativeSystem2Engine()

    async def both():
        small = System2SearchConfig(budget=4, max_depth=1, branching_factor=2)
        big = System2SearchConfig(budget=40, max_depth=2, branching_factor=3)
        return await asyncio.gather(
            engine.search("goal a", {"goal": "goal a", "path": []}, config=small),
            engine.search("goal b", {"goal": "goal b", "path": []}, config=big),
        )

    first, second = asyncio.run(both())
    total_first = sum(first.receipt.value_evidence.values())
    total_second = sum(second.receipt.value_evidence.values())
    assert total_first != total_second, (
        "two searches of very different budgets recorded identical provenance "
        "totals; the tally is being shared"
    )


def test_a_caller_supplied_scorer_is_not_overwritten():
    """The receipt records THIS engine's defaults, not somebody else's model."""

    async def scorer(node, goal):  # noqa: ARG001
        return 0.9

    async def world(state, action, node):  # noqa: ARG001
        from core.reasoning.native_system2 import SimulatedTransition

        return SimulatedTransition(next_state={"path": []}, reward_estimate=0.9)

    engine = NativeSystem2Engine()
    result = asyncio.run(
        engine.search(
            "supplied",
            {"goal": "supplied", "path": []},
            value_scorer=scorer,
            world_model=world,
        )
    )
    assert result.receipt.value_evidence == {}, (
        "a caller supplying its own model had default provenance attributed to it"
    )
