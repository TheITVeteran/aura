"""Known mechanics can be inverted without delegating arithmetic to a model."""
from __future__ import annotations

import math

import pytest

from core.agency.solving_for_a_required_state import (
    _required_state_invariant,
    solve_for_a_required_state,
)

pytestmark = pytest.mark.unit


def test_exact_transition_solves_the_required_result():
    plan = solve_for_a_required_state(1, (0, 1, 2, 3), advance=lambda value, action: value + action,
                                     outcome=lambda value: None if value > 3 else value == 3,
                                     cost=lambda value: abs(3 - value), identity=lambda value: value, steps=2)
    assert plan.reached and plan.actions == (2,) and plan.states == (1, 3)


@pytest.mark.parametrize("goal", [-8, 8])
def test_velocity_is_part_of_the_required_result(goal):
    def advance(state, action):
        position, velocity = state
        velocity += action
        return position + velocity, velocity

    def outcome(state):
        position, velocity = state
        if abs(position) > 12:
            return None
        return position == goal and velocity == 0

    plan = solve_for_a_required_state((0, 0), (-1, 0, 1), advance=advance, outcome=outcome,
                                     cost=lambda s: abs(s[0] - goal) + abs(s[1]), identity=lambda s: s,
                                     steps=12, width=128, budget=8192)
    assert plan.reached
    # Check the returned sequence by its accumulated impulses, separately
    # from the transition callback used in the search.
    speed, place = 0, 0
    for push in plan.actions:
        speed += push
        place += speed
        assert abs(place) <= 12
    assert place == goal and speed == 0


def test_posture_and_remaining_supply_are_constraints_too():
    # A transfer must arrive upright and with enough supply for its latch.
    # The shortest spatial path tilts it and cannot count as success.
    def advance(state, action):
        place, posture, supply = state
        if action == "slide":
            return place + 1, posture + 1, supply - 1
        if action == "level":
            return place, max(0, posture - 1), supply
        return place + 1, posture, supply - 2

    def outcome(state):
        place, posture, supply = state
        if supply < 2 or place > 3 or (place == 3 and posture != 0):
            return None
        return place == 3

    plan = solve_for_a_required_state((0, 0, 8), ("slide", "level", "carry"),
                                     advance=advance, outcome=outcome,
                                     cost=lambda s: 3 - s[0] + .1 * s[1], identity=lambda s: s,
                                     steps=8, width=64)
    assert plan.reached and plan.states[-1][0:2] == (3, 0)
    assert plan.states[-1][2] >= 2
    assert all(p != 3 or angle == 0 for p, angle, _supply in plan.states)


def test_building_control_response_is_state_not_an_instantaneous_command():
    def advance(state, action):
        position, response = state
        if action == "build":
            response = min(response + 1, 3)
        else:
            response = max(0, response - 1)
        return position + response, response

    plan = solve_for_a_required_state((0, 0), ("build", "release"), advance=advance,
                                     outcome=lambda s: None if s[0] > 6 else s == (6, 0),
                                     cost=lambda s: abs(6 - s[0]) + s[1], identity=lambda s: s, steps=12)
    assert plan.reached and "release" in plan.actions
    assert plan.states[-1] == (6, 0)
    # Release continues to move while the accumulated response decays.
    assert any(action == "release" and after[0] > before[0]
               for action, before, after in zip(plan.actions, plan.states, plan.states[1:], strict=False))


def test_a_failure_is_not_erased_by_a_later_recovery():
    assert _required_state_invariant() == ()


def test_unavailable_initial_state_has_no_plan():
    assert solve_for_a_required_state(None, ("a",), advance=lambda s, a: 1,
                                      outcome=lambda s: None, cost=lambda s: 0., identity=lambda s: s, steps=2) is None


def test_a_low_cost_cannot_override_a_constraint():
    plan = solve_for_a_required_state(0, ("unsafe", "safe"),
                                     advance=lambda s, a: -100 if a == "unsafe" else s + 1,
                                     outcome=lambda s: None if s < 0 else s == 2,
                                     cost=lambda s: s, identity=lambda s: s, steps=2)
    assert plan.reached and plan.actions == ("safe", "safe")


def test_its_transition_budget_is_exact_and_exhaustion_is_not_success():
    calls = []
    def advance(state, action):
        calls.append(action)
        return state + action

    plan = solve_for_a_required_state(0, (1, 2), advance=advance, outcome=lambda s: s == 100,
                                     cost=lambda s: 100 - s, identity=lambda s: s, steps=100, budget=7)
    assert len(calls) == plan.transitions == 7
    assert not plan.reached and not plan.complete


def test_beam_pruning_does_not_claim_exhaustive_search():
    plan = solve_for_a_required_state(0, (1, 2, 3), advance=lambda s, a: s + a,
                                     outcome=lambda s: s == 7, cost=lambda s: 7 - s,
                                     identity=lambda s: s, steps=3, width=1)
    assert plan.reached and not plan.complete


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_unmeasured_cost_is_not_defaulted_to_a_passing_score(bad):
    assert solve_for_a_required_state(0, (1,), advance=lambda s, a: s + a, outcome=lambda s: s == 2,
                                      cost=lambda s: bad, identity=lambda s: s, steps=3) is None


def test_a_previously_achieved_state_requires_no_input():
    plan = solve_for_a_required_state(3, (1,), advance=lambda s, a: s + a, outcome=lambda s: s == 3,
                                     cost=lambda s: 0., identity=lambda s: s, steps=3)
    assert plan.reached and not plan.actions and plan.transitions == 0


@pytest.mark.parametrize("verdict", [None, 1, 0, "success", "unknown"])
def test_unmeasured_or_untyped_outcomes_cannot_supply_a_plan(verdict):
    assert solve_for_a_required_state(0, (1,), advance=lambda s, a: s + a,
                                      outcome=lambda s: verdict, cost=lambda s: 0.,
                                      identity=lambda s: s, steps=3) is None


@pytest.mark.parametrize("arguments", [{"steps": True}, {"steps": 2.5}, {"steps": 0},
                                      {"steps": 1025}, {"steps": 2, "width": 513},
                                      {"steps": 2, "budget": 100001}])
def test_search_allocation_has_a_bounded_contract(arguments):
    assert solve_for_a_required_state(0, (1,), advance=lambda s, a: s + a,
                                      outcome=lambda s: False, cost=lambda s: 0.,
                                      identity=lambda s: s, **arguments) is None


def test_required_state_claim_is_registered_with_its_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "known_transition_plans_preserve_failure_boundaries")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
