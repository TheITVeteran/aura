"""Anonymous pickup candidates carry current pixels, geometry and container relations."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.drawn_objects import (
    MAX_OBJECTS,
    _Drawing,
    _separate,
    drawn_objects,
    measured_objects,
)
from core.perception.observed_transfer import ScreenReading, make_snapshot


def test_registered_object_candidate_canary_requires_its_capture():
    from core.perception.drawn_objects import _object_candidates_invariant

    assert _object_candidates_invariant() == ()


def outlined(image, x, y, width, height, color=(190, 190, 190), thickness=2):
    image[y:y + thickness, x:x + width] = color
    image[y + height - thickness:y + height, x:x + width] = color
    image[y:y + height, x:x + thickness] = color
    image[y:y + height, x + width - thickness:x + width] = color


def specimen(background=(20, 30, 50), foreground=(220, 130, 40)):
    image = np.empty((240, 360, 3), np.uint8)
    image[:] = background
    outlined(image, 15, 15, 170, 190)
    for index in range(8):
        x, y = 35 + (index % 2) * 75, 40 + (index // 2) * 36
        image[y:y + 18, x:x + 25] = foreground
        image[y + 4:y + 9, x + 3:x + 20] = (70, 190, 160)
    return image


@pytest.mark.parametrize(("background", "foreground"), [
    ((245, 245, 245), (40, 70, 150)),           # document/image editor
    ((20, 30, 50), (220, 130, 40)),            # inventory panel
    ((40, 50, 180), (240, 240, 250)),          # outlined construction palette
])
def test_inner_drawings_keep_their_extents_and_container_beyond_six_controls(background, foreground):
    objects = drawn_objects(specimen(background, foreground), capture_epoch="palette-opened")
    assert len(objects) == 8, objects
    assert len({item["text"] for item in objects}) == len(objects)
    assert len({item["candidate_id"] for item in objects}) == len(objects)
    for item in objects:
        assert item["role"] == "object" and item["source"] == "measured_pixels"
        assert item["capture_epoch"] == "palette-opened"
        assert "draggable" not in item and "affordance" not in item
        x, y, w, h = (item[key] for key in ("x", "y", "width", "height"))
        assert 0 <= x < item["center_x"] < x + w <= 1
        assert 0 <= y < item["center_y"] < y + h <= 1
        cx, cy, cw, ch = item["container_bbox"]
        assert cx <= x and cy <= y and x + w <= cx + cw and y + h <= cy + ch


def test_measured_writing_and_named_controls_cannot_become_pickup_targets():
    image = specimen()
    writing = [{"text": "Templates", "x": 30 / 360, "y": 35 / 240, "width": 35 / 360, "height": 25 / 240}]
    controls = [{"text": "Run", "x": 105 / 360, "y": 35 / 240, "width": 35 / 360, "height": 25 / 240}]
    objects = drawn_objects(image, writing=writing, controls=controls)
    assert len(objects) == 6
    assert all(item["center_y"] > 65 / 240 for item in objects)
    assert all("container_bbox" in item for item in objects)


def test_an_adjacent_caption_does_not_erase_the_drawing_it_describes():
    image = np.zeros((120, 160, 3), np.uint8)
    image[20:45, 40:70] = (80, 150, 230)
    writing = [{"text": "Report", "x": 35 / 160, "y": 55 / 120, "width": 50 / 160, "height": 12 / 120}]
    assert len(drawn_objects(image, writing=writing)) == 1


def test_backgrounds_flat_fields_and_cut_off_extents_supply_no_object():
    assert drawn_objects(np.zeros((120, 160, 3), np.uint8)) == []
    giant = np.zeros((120, 160, 3), np.uint8)
    giant[10:110, 10:150] = 220
    assert drawn_objects(giant) == []
    cutoff = np.zeros_like(giant)
    cutoff[30:60, 0:30] = (60, 160, 240)
    assert drawn_objects(cutoff) == []
    assert drawn_objects(np.random.default_rng(5).integers(0, 256, giant.shape, dtype=np.uint8)) == []


def test_partial_overlap_abstains_instead_of_inventing_a_merged_object():
    assert _separate([_Drawing((10, 10, 30, 30), 100), _Drawing((25, 15, 30, 30), 100)]) == []
    assert len(_separate([_Drawing((10, 10, 30, 30), 100), _Drawing((11, 11, 29, 29), 90)])) == 1


def test_same_pixels_same_witness_new_capture_and_changed_local_pixels_change_witness():
    image = specimen()
    first = drawn_objects(image, capture_epoch="one")
    second = drawn_objects(image.copy(), capture_epoch="two")
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert all(item["capture_epoch"] == "two" for item in second)
    image[44:49, 38:55] = (210, 40, 120)
    third = drawn_objects(image, capture_epoch="three")
    assert first[0]["candidate_id"] != third[0]["candidate_id"]
    assert first[1]["candidate_id"] == third[1]["candidate_id"]


def test_object_budget_is_bounded_without_reusing_the_button_budget():
    assert len(drawn_objects(specimen(), at_most=3)) == 3
    assert drawn_objects(specimen(), at_most=0) == []
    assert len(drawn_objects(specimen(), at_most=10_000)) <= MAX_OBJECTS


def test_no_extent_is_synthesized_from_labels_or_invalid_pixels():
    text = [{"text": "Visible item", "x": .1, "y": .1, "width": .2, "height": .2}]
    assert drawn_objects(None, writing=text) == []
    assert drawn_objects("picture", writing=text) == []
    assert drawn_objects(np.full((120, 160, 3), float("nan")), writing=text) == []


def observed(image=None, *, epoch="one"):
    image = specimen() if image is None else image
    return ScreenReading({"objects": drawn_objects(image, capture_epoch=epoch), "layout": [], "shapes": [],
                          "surface_id": "editor", "bounds": [0, 0, 360, 240], "capture_at": 1.,
                          "_capture_epoch": epoch, "settled": True}, picture_snapshot=make_snapshot(image))


def test_source_inventory_recomputes_local_extent_witness_from_its_own_capture():
    reading = observed()
    assert len(measured_objects(reading)) == 8
    reading.picture_snapshot = make_snapshot(np.zeros((240, 360, 3), np.uint8))
    assert measured_objects(reading) == {}


@pytest.mark.parametrize("change", [
    {"_capture_epoch": "another"}, {"surface_id": ""}, {"bounds": None}, {"capture_at": float("nan")},
    {"settled": False}, {"occluded": True},
])
def test_incomplete_changed_or_occluded_capture_cannot_supply_sources(change):
    reading = observed()
    reading.update(change)
    assert measured_objects(reading) == {}


@pytest.mark.parametrize("change", [
    {"capture_epoch": "previous"}, {"candidate_id": "made-up"}, {"role": "button"}, {"source": "memory"},
    {"x": float("nan")}, {"width": 0.}, {"center_x": .9}, {"center_y": True},
])
def test_candidate_metadata_cannot_redirect_the_dispatch_point_or_borrow_another_extent(change):
    reading = observed()
    reading["objects"] = [{**reading["objects"][0], **change}]
    assert measured_objects(reading) == {}


def test_duplicate_dispatch_names_are_ambiguous_even_when_both_local_witnesses_are_valid():
    reading = observed()
    first, second = reading["objects"][:2]
    reading["objects"] = [first, {**second, "text": first["text"]}]
    assert measured_objects(reading) == {}


def test_live_source_survey_excludes_tabs_and_old_pressable_names_from_carries():
    from core.agency.putting_things_in_place import PuttingInPlace, what_is_carried
    from core.agency.what_i_can_do_here import a_click_on

    reading = observed()
    putting = PuttingInPlace()
    putting.told_of_carrying("Drag the component onto the highlighted destination.")
    putting.observe_objects(reading)
    objects = list(putting.manipulation_sources)
    moves = [a_click_on(name) for name in (*objects, "Templates", "Run", "the shape at 50% across, 50% down", putting.place_named)]
    carries = putting.carries(moves)
    assert carries and all(what_is_carried(move)[0] in objects for move in carries)
    assert all(what_is_carried(move)[0] not in ("Templates", "Run", "the shape at 50% across, 50% down") for move in carries)
    putting.observe_objects(ScreenReading({**reading, "objects": []}, picture_snapshot=reading.picture_snapshot))
    assert putting.carries(moves) == ()


def test_ipc_preserves_objects_their_private_witness_and_capture_owner():
    from types import SimpleNamespace

    from core.perception.eyes_of_their_own import _reading_to_send, _restore_reading

    image = specimen()
    window = SimpleNamespace(owner="Editor", number=42, bounds=(0, 0, 360, 240))
    public = {"layout": [], "capture_at": 1., "_capture_epoch": "final", "settled": True}
    sent = _reading_to_send(image, public, window=window, over=None, still=True, wait=True, looked_took=.1)
    restored = _restore_reading(sent)
    assert len(measured_objects(restored)) == 8
    assert all(item["capture_epoch"] == "final" for item in restored["objects"])
    assert "data" not in restored and not any("pixels" in item for item in restored["objects"])


@pytest.mark.asyncio
async def test_reader_features_and_snapshot_give_objects_the_same_final_frame(monkeypatch):
    import time
    from types import SimpleNamespace

    from core.perception import what_the_pixels_show, where_the_words_point

    first, last = specimen(), specimen()
    last[44:49, 38:55] = (210, 40, 120)
    captures = iter((first, last))

    async def take():
        return next(captures).copy()

    monkeypatch.setattr(where_the_words_point, "remember_the_still_picture", lambda *_, **__: None)
    looker = SimpleNamespace(last_looks=(), read=lambda picture, **_: {"layout": [], "text": ""},
                             places_still=lambda *_: True, would_not_read=set(), learn_what_was_read=lambda: None)
    reading, settled, _, _, _, _ = await what_the_pixels_show._read_until_settled(
        looker, take, True, 1., time.monotonic(), surface="editor")
    reading.update(settled=settled, bounds=[0, 0, 360, 240])
    actual = measured_objects(reading)
    assert len(actual) == 8
    expected = drawn_objects(last, capture_epoch=reading["_capture_epoch"])
    assert [item["candidate_id"] for item in actual.values()] == [item["candidate_id"] for item in expected]
    assert [item["candidate_id"] for item in expected] != [item["candidate_id"] for item in drawn_objects(first)]
