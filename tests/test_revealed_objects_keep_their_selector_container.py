"""A selector owns its measured panel, not autonomous changes in another panel."""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.what_i_can_do_here import a_click_on
from core.cognition.procedure_binding import RevealedChoices
from core.perception.drawn_objects import drawn_objects, measured_objects
from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.skills.screen_step_evidence import screen_evidence


def _outline(image, x, y, width, height):
    image[y:y + 2, x:x + width] = 190
    image[y + height - 2:y + height, x:x + width] = 190
    image[y:y + height, x:x + 2] = 190
    image[y:y + height, x + width - 2:x + width] = 190


def _sprite(image, x, y, color=(220, 130, 40)):
    image[y:y + 20, x:x + 25] = color
    image[y + 4:y + 9, x + 3:x + 20] = (70, 190, 160)


def _panels(*, selector=False):
    image = np.empty((240, 500, 3), np.uint8)
    image[:] = (20, 30, 50)
    _outline(image, 15, 15, 170, 190)
    _outline(image, 260, 15, 180, 190)
    _sprite(image, 35, 45)
    _sprite(image, 285, 45)
    writing = []
    if selector:
        # The control is represented by a same-capture measured text extent.
        image[155:167, 35:110] = (150, 150, 190)
        writing = [{"text": "Category", "x": 35 / 500, "y": 155 / 240,
                    "width": 75 / 500, "height": 12 / 240,
                    "center_x": 72.5 / 500, "center_y": 161 / 240}]
    return image, writing


def _evidence(image, writing, *, epoch, captured):
    reading = ScreenReading({
        "objects": drawn_objects(image, writing=writing, capture_epoch=epoch),
        "layout": writing, "shapes": [], "text": "Category" if writing else "",
        "surface_id": "two-panel-editor", "bounds": [0, 0, 500, 240],
        "capture_at": captured, "_capture_epoch": epoch, "settled": True,
    }, picture_snapshot=make_snapshot(image))
    # Only values whose appearance/extent actually match this private frame
    # enter the procedure inventory. No manufactured candidate witnesses.
    objects = measured_objects(reading)
    assert len(objects) == len(reading["objects"])
    clicks = tuple(a_click_on(name) for name in objects)
    if writing:
        clicks += (a_click_on("Category"),)
    return screen_evidence(reading, clicks), objects


def _counterfactual(*, selector):
    before_pixels, writing = _panels(selector=selector)
    after_pixels = before_pixels.copy()
    _sprite(after_pixels, 70, 85)  # an item revealed by the left selector
    after_pixels[49:54, 288:305] = (210, 40, 120)  # unrelated right-panel animation
    before, earlier = _evidence(before_pixels, writing, epoch="before", captured=1.)
    after, later = _evidence(after_pixels, writing, epoch="after", captured=2.)
    assert len(earlier) == 2 and len(later) == 3
    introduced = set(later) - set(earlier)
    assert len(introduced) == 1
    left_name = introduced.pop()
    right_name = next(name for name, item in later.items() if item["center_x"] > .5)
    assert earlier[right_name]["candidate_id"] != later[right_name]["candidate_id"]
    assert later[left_name]["container_bbox"] != later[right_name]["container_bbox"]
    return before, after, left_name, right_name


@pytest.mark.parametrize("opened_move", ["", 'click "Category"'])
def test_simultaneous_changes_in_independent_pixel_panels_without_a_measured_owner_abstain(opened_move):
    before, after, _, _ = _counterfactual(selector=False)
    assert RevealedChoices.from_effect(
        before, after, "open-selector", "choose-item", opened_move=opened_move,
    ) is None


def test_measured_selector_owns_only_its_panel_when_another_panel_animates():
    before, after, left_name, right_name = _counterfactual(selector=True)
    binding = RevealedChoices.from_effect(
        before, after, "open-selector", "choose-item", opened_move='click "Category"',
    )
    assert binding is not None
    assert binding.moves == frozenset({a_click_on(left_name)})
    assert binding.offers(a_click_on(left_name), after, "choose-item")
    assert not binding.offers(a_click_on(right_name), after, "choose-item")


def test_duplicate_measured_selector_locations_cannot_choose_one_of_two_changed_groups():
    before, after, _, _ = _counterfactual(selector=True)
    before["regions"].append({**before["regions"][-1], "center_x": .6, "center_y": .3})
    assert RevealedChoices.from_effect(
        before, after, "open-selector", "choose-item", opened_move='click "Category"',
    ) is None
