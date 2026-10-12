"""Creation completion uses delivered object effects and a later independent picture."""
import time

import numpy as np
import pytest

from core.agency.what_i_can_do_here import WhatWorksHere
from core.perception.observed_transfer import (
    ScreenReading,
    bind_transfer,
    finish_delivery,
    make_snapshot,
    verify_transfer,
)
from core.runtime.skill_contract import PredicateState
from core.skills import screen_pursuit_as_it_happens as pursuit
from core.skills.screen_step_evidence import (
    _creation_completion_invariant,
    current_work_is_measured,
)

pytestmark = pytest.mark.unit


def image(*, placed=False, source=False):
    rows, columns = np.indices((32, 32))
    pattern = np.stack(((rows * 9 + columns * 3) % 251, (rows * 3 + columns * 11) % 241,
                        ((columns // 8 + rows // 8) % 2) * 160 + 30), axis=2).astype(np.uint8)
    picture = np.full((240, 320, 3), 224, np.uint8)
    if source:
        picture[30:62, 20:52] = pattern
    if placed:
        picture[130:162, 200:232] = pattern
    return picture


def reading(picture, at, **changes):
    data = {"capture_at": at, "_capture_epoch": f"frame-{at}", "owner": "editor",
            "bounds": [30, 70, 320, 240], "settled": True, "text": "Create your own scene.",
            "shapes": [], "layout": [], **changes}
    return ScreenReading(data, picture_snapshot=make_snapshot(picture))


def placed_work():
    before = reading(image(source=True), 1, shapes=[{"text": "part", "x": 20 / 320, "y": 30 / 240,
                                                  "width": .1, "height": 32 / 240, "shape": True}])
    bound = bind_transfer('drag "part" to "canvas"', before, source="part", destination="canvas",
                          source_at=(36 / 320, 46 / 240), destination_at=(216 / 320, 146 / 240))
    bound = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    after = reading(image(placed=True), 2)
    receipt = verify_transfer(bound, after)
    assert receipt.state is PredicateState.SATISFIED
    works = WhatWorksHere()
    works.carried(receipt.move, True, receipt=receipt)
    return works, after


def run():
    reflexes = pursuit.PlayingAsItHappens(None, (0, 0, 1, 1), "Make one scene.", time.monotonic() + 60)
    reflexes.words = ["Create your own scene."]
    reflexes.for_making_since = time.monotonic() - pursuit.TRYING_OUT_S - 1
    return reflexes


def test_the_real_handover_does_not_complete_from_changed_menu_counts_or_story():
    reflexes = run()
    token = pursuit.AS_IT_HAPPENS.set(reflexes)
    try:
        works = WhatWorksHere(did_something={'click "Next background"': 500},
                              carried_to={'drag "part" to "canvas"': True})
        assert not pursuit.a_handed_over_run_is_over(reading(image(), 3), made=500, can_do=works)
        assert not reflexes.over_because
        assert _creation_completion_invariant() == ()
    finally:
        pursuit.AS_IT_HAPPENS.reset(token)


def test_a_delivered_placement_needs_a_further_independent_capture_to_complete():
    works, after = placed_work()
    reflexes = run()
    assert not reflexes.run_is_over(after, can_do=works)
    assert works.placed_count() == 0
    assert reflexes.run_is_over(reading(image(placed=True), 3), can_do=works)
    assert reflexes.over_because == pursuit.MADE_NOT_WON
    assert works.placed_count() == 1


@pytest.mark.parametrize("changes", [{"owner": "other editor"}, {"_work_epoch": "new task"},
                                     {"occluded": True}, {"settled": False}, {"capture_at": 1.5}])
def test_completion_cannot_carry_placement_credit_into_another_or_unmeasured_context(changes):
    works, _after = placed_work()
    assert not run().run_is_over(reading(image(placed=True), 3, **changes), can_do=works)
    assert works.placed_count() == 0


def test_removed_work_and_same_epoch_without_pixels_do_not_complete():
    works, after = placed_work()
    assert not current_work_is_measured(works, dict(after))
    assert not run().run_is_over(reading(image(), 3), can_do=works)
    assert works.placed_count() == 0


def test_navigation_eventually_reports_unverified_work_without_a_success():
    reflexes = run()
    reflexes.for_making_since = time.monotonic() - pursuit.TRYING_OUT_MOST_S - 1
    works = WhatWorksHere(did_something={'click "Next background"': 500})
    assert reflexes.run_is_over(reading(image(), 3), can_do=works)
    assert reflexes.over_because == pursuit.MAKING_STALLED


def test_a_thing_made_by_choosing_and_its_own_finishing_control_is_made_by_her_acts_changing_it():
    """Offline 2026-10-11 an agent generator showed the agent card it made, nothing was placed in it, and she went on
    clicking for four minutes: what is not carried into place is not measured by placement."""
    reflexes = run()
    reflexes.words = ["Create your own agent by selecting your favorite body parts. Then click GENERATE."]
    token = pursuit.AS_IT_HAPPENS.set(reflexes)
    try:
        works = WhatWorksHere(did_something={'click "GENERATE"': 1, 'click "the shape at 40% across"': 6})
        assert pursuit.a_handed_over_run_is_over(reading(image(), 3), made=pursuit.CHANGES_MADE_AT_LEAST, can_do=works)
        assert reflexes.over_because == pursuit.MADE_NOT_WON
    finally:
        pursuit.AS_IT_HAPPENS.reset(token)
