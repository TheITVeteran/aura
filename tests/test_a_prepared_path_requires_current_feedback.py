"""Retained planning work needs current observations and completed actions."""
import math
from dataclasses import replace

import pytest

from core.agency.following_a_prepared_path import FollowingPreparedPath, _prepared_path_invariant
from core.agency.solving_for_a_required_state import RequiredStatePlan, solve_for_a_required_state

pytestmark = pytest.mark.unit


def _prepared():
    plan = RequiredStatePlan((1, 1, 1), (0, 1, 2, 3), True, 3, True)
    return FollowingPreparedPath(plan, "epoch one")


def _feedback(follower, state=0, *, at=1., now=None, **changes):
    options = dict(model_identity="epoch one", observed_at=at, now=at if now is None else now,
                   max_age=.5, advance=lambda s, a: s + a, outcome=lambda s: s == 3)
    options.update(changes)
    return follower.feedback(state, **options)


def test_feedback_keeps_search_work_and_replays_only_the_retained_suffix():
    calls = []
    follower = _prepared()
    for state in range(3):
        def step(s, a):
            calls.append((s, a))
            return s + a
        offered = _feedback(follower, state, at=state + 1., advance=step)
        assert offered.action == 1 and offered.transitions == 3 - state
        assert follower.cursor == state, "a prediction is not an input receipt"
        assert follower.finished(offered, at=state + 1.1, succeeded=True)
    assert calls == [(0, 1), (1, 1), (2, 1), (1, 1), (2, 1), (2, 1)]
    assert _feedback(follower, 3, at=4.) is None
    assert follower.status == "observed_goal"


def test_missing_receipt_cannot_skip_an_action_or_offer_the_next_one():
    follower = _prepared()
    offered = _feedback(follower)
    assert _feedback(follower, 1, at=1.2) is None
    assert follower.cursor == 0 and follower.status == "waiting_for_input"
    assert not follower.finished(replace(offered), at=1.3, succeeded=True)
    assert follower.cursor == 0
    assert follower.finished(offered, at=1.4, succeeded=True)
    assert not follower.finished(offered, at=1.5, succeeded=True)
    assert follower.cursor == 1


@pytest.mark.parametrize("at,now", [(0., 1.), (2., 1.), (math.nan, 1.), (1., math.inf)])
def test_stale_future_and_unavailable_observations_cannot_offer_input(at, now):
    follower = _prepared()
    assert _feedback(follower, at=at, now=now) is None
    assert follower.cursor == 0 and follower.plan is not None


def test_feedback_requires_an_observation_after_the_completed_input():
    follower = _prepared()
    offered = _feedback(follower)
    assert follower.finished(offered, at=1.4, succeeded=True)
    assert _feedback(follower, 1, at=1.3, now=1.45) is None
    assert _feedback(follower, 1, at=1.4, now=1.45) is None
    assert _feedback(follower, 1, at=1.45, now=1.45) is not None


@pytest.mark.parametrize("succeeded,completed", [(False, 1.2), (1, 1.2), (True, 1.6), (True, .9)])
def test_failed_expired_or_disordered_input_discards_its_unverified_path(succeeded, completed):
    follower = _prepared()
    offered = _feedback(follower)
    assert not follower.finished(offered, at=completed, succeeded=succeeded)
    assert follower.plan is None and follower.cursor == 0
    assert _feedback(follower, at=2.) is None


def test_slow_feedback_does_not_extend_its_picture_deadline():
    follower = _prepared()
    clock = [1.]
    def slow_transition(s, a):
        clock[0] += .3
        return s + a
    offered = _feedback(follower, advance=slow_transition)
    assert offered.expires_at == 1.5
    assert not follower.is_current(offered, clock[0])
    assert not follower.finished(offered, at=clock[0], succeeded=True)


def test_a_changed_body_rule_or_goal_invalidates_an_outstanding_offer():
    follower = _prepared()
    offered = _feedback(follower)
    assert _feedback(follower, at=1.1, model_identity="epoch two") is None
    assert not follower.is_current(offered, 1.2)
    assert not follower.finished(offered, at=1.2, succeeded=True)


@pytest.mark.parametrize("blocked", [0, 1, 2])
def test_a_new_failure_on_the_remaining_path_cannot_be_recovered_into_success(blocked):
    follower = _prepared()
    offered = _feedback(follower, outcome=lambda s: None if s == blocked else s == 3)
    assert offered is None and follower.plan is None


def test_displaced_state_may_continue_only_if_the_path_still_reaches_the_goal():
    assert _feedback(_prepared(), -1) is None
    follower = _prepared()
    offered = _feedback(follower, 1)
    assert offered is not None and offered.transitions == 2


@pytest.mark.parametrize("budget", [0, True, 1, 1025])
def test_feedback_is_bounded_before_it_replays_any_transition(budget):
    def uncalled(s, a):
        pytest.fail("an unavailable replay budget must not spend transitions")
    follower = _prepared()
    assert _feedback(follower, budget=budget, advance=uncalled) is None
    assert follower.cursor == 0 and follower.plan is not None


def test_an_exhausted_path_cannot_claim_that_an_unobserved_goal_was_reached():
    follower = FollowingPreparedPath(RequiredStatePlan((), (3,), True, 0, True), "epoch one")
    assert _feedback(follower, 0) is None
    assert follower.status == "needs_preparation"


@pytest.mark.parametrize("domain", ["movement", "workflow"])
def test_preparation_and_feedback_close_the_loop_on_different_transition_types(domain):
    if domain == "movement":
        initial, target = (0, 0), (4, 0)
        def advance(s, a):
            rate = s[1] + a
            return s[0] + rate, rate
        actions = (-1, 0, 1)
        def goal(s):
            return s == target
        def cost(s):
            return abs(s[0] - 4) + abs(s[1])
    else:
        initial, target = frozenset(), frozenset(("prepared", "checked", "submitted"))
        def advance(s, a):
            if a == "submit" and "checked" not in s:
                return None
            if a == "check" and "prepared" not in s:
                return None
            return s | {dict(prepare="prepared", check="checked", submit="submitted")[a]}
        actions = ("submit", "check", "prepare")
        def goal(s):
            return None if s is None else s == target
        def cost(s):
            return len(target - s)
    plan = solve_for_a_required_state(initial, actions, advance=advance, outcome=goal,
                                     cost=cost, identity=lambda s: s, steps=12, width=32)
    assert plan is not None and plan.reached
    follower = FollowingPreparedPath(plan, domain)
    observed = initial
    for tick in range(len(plan.actions)):
        offered = follower.feedback(observed, model_identity=domain, observed_at=tick + 1., now=tick + 1.,
                                    max_age=.5, advance=advance, outcome=goal)
        assert offered is not None and follower.is_current(offered, tick + 1.05)
        observed = advance(observed, offered.action)
        assert follower.finished(offered, at=tick + 1.1, succeeded=True)
    assert observed == target
    assert follower.feedback(observed, model_identity=domain, observed_at=20., now=20.,
                             max_age=.5, advance=advance, outcome=goal) is None
    assert follower.status == "observed_goal"


def test_no_input_can_supply_a_completion_receipt():
    follower = _prepared()
    assert not follower.is_current(None, 1.)
    assert not follower.finished(None, at=1., succeeded=True)
    assert follower.cursor == 0


def test_prepared_feedback_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _prepared_path_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "prepared_paths_need_current_feedback_and_receipts")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
