"""A machine she built is tested, and where it stopped is the join to mend: watched as it plays out, the part placed
nearest where it came to rest is said, given to her plan, selected and turned before anything new is added.

The player in a video of a trap-building game tested the whole and mended the link that broke, not the start.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.mending_what_broke import a_click_to_mend, heard_the_test, the_joint_that_broke
from core.cognition.a_guide_to_a_place import Guide
from core.cognition.a_plan_to_an_end import Plan
from core.cognition.reading_the_rules import Frame
from core.perception.watching_it_happen import CELLS, Watching, _Moment
from core.runtime.skill_contract import PredicateState

pytestmark = pytest.mark.unit


def _cells(column: int, row: int) -> np.ndarray:
    cells = np.zeros((CELLS[1], CELLS[0]), dtype=bool)
    cells[row, column] = True
    return cells


def test_where_what_an_act_set_going_came_to_rest_is_the_last_place_it_changed():
    async def nothing():
        return None

    watching = Watching(take=nothing)
    watching.acted('click "TEST TRAP"', at=10.0)
    for step, column in enumerate((2, 4, 6, 8)):                       # the action goes along the machine, then stops
        watching.moments.append(_Moment(10.5 + step * 0.5, 10.0 + step * 0.5, _cells(column, 5), 0.01))
    x, y = watching.where_it_came_to_rest(10.0)
    assert abs(x - 8.5 / CELLS[0]) < 1e-6 and abs(y - 5.5 / CELLS[1]) < 1e-6
    assert watching.where_it_came_to_rest(12.0) is None              # nothing went on after then


def _built():
    receipts = {"one": SimpleNamespace(effect_bbox=(0.20, 0.40, 0.06, 0.06)),
                "two": SimpleNamespace(effect_bbox=(0.50, 0.40, 0.06, 0.06))}
    return SimpleNamespace(placement_receipts=receipts,
                           placement_states={k: PredicateState.SATISFIED for k in receipts}, turned_since_placed=4)


def test_the_part_placed_where_the_test_stopped_is_the_join_to_mend_and_her_plan_hears_it():
    can_do, said = _built(), []
    guide = Guide(place="a place of devices")
    guide.plan = Plan(end="the cage drops")
    guide.plan.took([Frame("Test the trap", act="click things", thing="TEST TRAP")])
    joint = heard_the_test(guide, can_do, {"came_to_rest_at": (0.54, 0.44)}, 'click "TEST TRAP"', said.append)
    assert joint == pytest.approx((0.53, 0.43, 0.06))
    assert guide.broke_at == joint and can_do.turned_since_placed == 0
    assert "the test got as far as the part at 53% across and 43% down" in guide.plan_again_because
    assert said == ["The test got as far as the part at 53% across and 43% down, and stopped there: that's the join "
                    "to mend."]
    # Anything other than a test of what was made says nothing of a join.
    assert heard_the_test(Guide(place="x"), _built(), {"came_to_rest_at": (0.54, 0.44)}, 'click "HELP"') is None
    assert the_joint_that_broke((0.95, 0.95), [(0.2, 0.4, 0.06, 0.06)]) is None   # nothing placed is where it stopped


def test_the_broken_part_is_selected_then_turned_before_anything_new_is_added():
    from core.agency.aiming_what_was_placed import a_turn_wanted

    guide = Guide(place="a place of devices")
    guide.broke_at = (0.53, 0.43, 0.06)
    offered = ['click "the shape at 20% across, 40% down"', 'click "the shape at 55% across, 45% down"',
               'click "ROLLERS"', 'click "TURN"']
    assert a_click_to_mend(guide, offered) == 'click "the shape at 55% across, 45% down"'
    can_do = SimpleNamespace(chain_ends_at=(0.9, 0.9), turned_since_placed=0, place_named="the end of the arrow")
    guide.plan = Plan(end="the cage drops")
    guide.plan.took([Frame("Bring the action to the cage", act="carry", thing="a roller", where="the cage")])
    from core.perception import where_the_words_point

    # The broken part's output points back, away from the cage: it is turned. Measured from the last part put down
    # (bottom right, past the cage), the same output would point on, and nothing would be turned.
    places = {"the end of the arrow": (0.40, 0.43), "the cage": (0.90, 0.43)}
    original = where_the_words_point.PLACES.where
    where_the_words_point.PLACES.where = lambda name: places.get(name) or original(name)
    try:
        assert a_turn_wanted(can_do, guide, offered) == 'click "TURN"'
        guide.broke_at = None
        can_do.chain_ends_at = (0.95, 0.43)
        assert a_turn_wanted(can_do, guide, offered) == ""
    finally:
        where_the_words_point.PLACES.where = original
