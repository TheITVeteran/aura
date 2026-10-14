"""One captured frame reaches a bounded SAM 2 session without borrowing another object's mask."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.learned_tracking import sam2_backend as backend
from core.perception.learned_tracking.contracts import ObjectSeed, TrackingFrame

pytestmark = pytest.mark.unit


def _frame(number=0, *, epoch=2, at=None, shape=(24, 32)):
    pixels = np.zeros((*shape, 3), np.uint8)
    pixels[6:14, 5:13] = (210, 80, 30)
    return TrackingFrame(pixels, 10. + number / 20 if at is None else at, epoch, number)


class _Predictor:
    """A numerical predictor double with the pinned state and public propagation API."""

    image_size = 32
    device = "cpu"
    num_maskmem = 7
    memory_temporal_stride_for_eval = 1
    max_obj_ptrs_in_encoder = 16

    def __init__(self):
        self.calls = []
        self.masks = {}
        self.scores = {}
        self.extra_frame = False
        self.wrong_frame = False
        self.wrong_object = False
        self.wrong_shape = False
        self.fail = False
        self.closed = 0

    def _add(self, state, index, number, mask):
        slot = len(state["obj_ids"])
        state["obj_ids"].append(number)
        state["obj_id_to_idx"][number] = slot
        state["obj_idx_to_id"][slot] = number
        self.masks[number] = mask.copy()
        for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj"):
            state[name][slot] = {index: {"seed": True}}
        output = {"pred_masks": mask.astype(np.float32), "object_score_logits": np.asarray([10.])}
        state["output_dict_per_obj"][slot] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        state["temp_output_dict_per_obj"][slot] = {"cond_frame_outputs": {index: output}, "non_cond_frame_outputs": {}}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.calls.append(("mask", frame_idx, obj_id))
        self._add(state, frame_idx, obj_id, mask)

    def add_new_points_or_box(self, state, frame_idx, obj_id, *, box):
        self.calls.append(("box", frame_idx, obj_id, box))
        mask = np.zeros((state["video_height"], state["video_width"]), bool)
        left, top, right, bottom = (int(v) for v in box)
        mask[top:bottom, left:right] = True
        self._add(state, frame_idx, obj_id, mask)

    def propagate_in_video(self, state, *, start_frame_idx, max_frame_num_to_track, reverse):
        index = start_frame_idx
        self.calls.append(("propagate", index, max_frame_num_to_track, reverse))
        assert max_frame_num_to_track == 0 and not reverse
        assert list(state["images"]) == [index]
        assert state["images"][index].dtype == np.float32
        assert state["num_frames"] == index + 1
        assert state["images"][index].shape == (3, self.image_size, self.image_size)
        if self.fail:
            raise RuntimeError("the numerical predictor failed")
        state["cached_features"] = {index: (state["images"][index], {})}
        masks = []
        for slot, number in enumerate(state["obj_ids"]):
            temporary = state["temp_output_dict_per_obj"][slot]["cond_frame_outputs"]
            state["output_dict_per_obj"][slot]["cond_frame_outputs"].update(temporary)
            temporary.clear()
            mask = self.masks[number]
            logits = np.where(mask, 5., -5.).astype(np.float32)
            score = self.scores.get(number, 5.)
            if index not in state["output_dict_per_obj"][slot]["cond_frame_outputs"]:
                state["output_dict_per_obj"][slot]["non_cond_frame_outputs"][index] = {
                    "pred_masks": logits, "object_score_logits": score,
                }
            state["frames_tracked_per_obj"][slot][index] = {"reverse": False}
            masks.append(logits)
        result = np.stack(masks)[:, None]
        if self.wrong_shape:
            result = result[..., :-1]
        numbers = list(state["obj_ids"])
        if self.wrong_object:
            numbers[0] += 1
        try:
            yield index + int(self.wrong_frame), numbers, result
            if self.extra_frame:
                while True:
                    yield index + 1, numbers, result
        finally:
            self.closed += 1

    def remove_object(self, state, number, *, strict, need_output):
        assert strict and not need_output
        if number not in state["obj_ids"]:
            raise RuntimeError("the object does not exist")
        before = dict(state["obj_id_to_idx"])
        numbers = [n for n in state["obj_ids"] if n != number]
        for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj",
                     "output_dict_per_obj", "temp_output_dict_per_obj"):
            state[name] = {i: state[name][before[n]] for i, n in enumerate(numbers)}
        state["obj_ids"] = numbers
        state["obj_id_to_idx"] = {n: i for i, n in enumerate(numbers)}
        state["obj_idx_to_id"] = dict(enumerate(numbers))
        self.masks.pop(number)


@pytest.fixture
def setup(monkeypatch):
    torch = SimpleNamespace(from_numpy=lambda value: value.copy(), device=lambda value: value,
                            inference_mode=nullcontext)
    monkeypatch.setattr(backend, "_torch", lambda: torch)
    model = _Predictor()
    return backend.Sam2Backend("/absent/weights.pt", predictor=model), model


def test_a_box_seeds_memory_and_only_a_later_capture_supplies_visibility(setup):
    adapter, model = setup
    first = adapter.start(_frame(), (ObjectSeed(17, box=(5., 6., 13., 14.)),))
    assert first.status == "seeded" and first.masks[0].seeded
    assert not first.masks[0].visible and first.masks[0].presence is None
    second = adapter.track(_frame(1))
    assert second.status == "observed" and second.masks[0].visible
    assert second.masks[0].number == 17 and not second.masks[0].seeded
    assert (second.epoch, second.frame_number, second.at, second.shape) == (2, 1, 10.05, (24, 32))
    assert model.calls == [("box", 0, 17, (5., 6., 13., 14.)), ("propagate", 0, 0, False),
                           ("propagate", 1, 0, False)]


def test_direct_mask_seed_cannot_borrow_upstreams_fabricated_presence(setup):
    adapter, _model = setup
    mask = np.zeros((24, 32), bool)
    mask[6:14, 5:13] = True
    first = adapter.start(_frame(), (ObjectSeed(17, mask=mask),))
    assert first.masks[0].mask.any() and first.masks[0].presence is None
    assert first.masks[0].seeded and not first.masks[0].visible


@pytest.mark.parametrize("score", (None, float("nan"), float("inf"), -5., 0., np.asarray([1., 2.])))
def test_missing_invalid_or_absent_presence_cannot_grant_visibility(setup, score):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    model.scores[17] = score
    result = adapter.track(_frame(1))
    assert result.masks[0].mask.any() and not result.masks[0].visible
    if score is None or isinstance(score, np.ndarray) or not np.isfinite(score):
        assert result.masks[0].presence is None


def test_empty_masks_are_absent_even_with_a_positive_presence_score(setup):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    model.masks[17][:] = False
    result = adapter.track(_frame(1))
    assert result.masks[0].presence > .5 and not result.masks[0].visible


def test_adding_an_object_keeps_the_other_objects_current_inference(setup):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    result = adapter.add(_frame(1), (ObjectSeed(29, box=(15, 3, 20, 8)),))
    assert result.status == "observed"
    old, new = result.masks
    assert old.number == 17 and old.visible and not old.seeded
    assert new.number == 29 and new.seeded and not new.visible and new.presence is None
    assert model.calls[-1] == ("propagate", 1, 0, False)


def test_object_ids_survive_upstreams_internal_slot_remapping(setup):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)), ObjectSeed(29, box=(15, 3, 20, 8))))
    adapter.remove(17)
    model.scores[29] = -5.
    result = adapter.track(_frame(1))
    assert [mask.number for mask in result.masks] == [29]
    assert result.masks[0].presence < .5 and not result.masks[0].visible
    assert adapter._state["obj_id_to_idx"] == {29: 0}


@pytest.mark.parametrize("frame", (_frame(0), _frame(1, at=10.), _frame(1, epoch=3), _frame(1, shape=(32, 32))))
def test_old_other_surface_or_other_geometry_captures_cannot_extend_memory(setup, frame):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    calls = len(model.calls)
    with pytest.raises(ValueError):
        adapter.track(frame)
    assert len(model.calls) == calls
    assert adapter.track(_frame(1)).masks[0].visible


def test_same_object_reseeding_and_capacity_overflow_are_rejected_before_inference(setup):
    adapter, model = setup
    adapter.max_objects = 1
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    calls = len(model.calls)
    for number in (17, 29):
        with pytest.raises(ValueError):
            adapter.add(_frame(1), (ObjectSeed(number, box=(15, 3, 20, 8)),))
    assert len(model.calls) == calls and adapter._index == 0


def test_one_thousand_frames_retain_memory_dependencies_without_raw_video_history(setup):
    adapter, _model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    for number in range(1, 1000):
        adapter.track(_frame(number))
    state = adapter._state
    assert list(state["images"]) == list(state["cached_features"]) == [999]
    assert len(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == adapter.retained_frames
    assert len(state["output_dict_per_obj"][0]["cond_frame_outputs"]) == 1
    assert len(state["frames_tracked_per_obj"][0]) == adapter.retained_frames + 1


@pytest.mark.parametrize("defect", ("wrong_frame", "wrong_object", "wrong_shape", "extra_frame", "fail"))
def test_unowned_outputs_and_inference_failures_retire_partial_model_memory(setup, defect):
    adapter, model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    setattr(model, defect, True)
    with pytest.raises(RuntimeError):
        adapter.track(_frame(1))
    assert adapter._state is None
    with pytest.raises(RuntimeError, match="not started"):
        adapter.track(_frame(2))


def test_reset_retires_the_old_generation_and_all_feature_memory(setup):
    adapter, _model = setup
    adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    old = adapter._state
    adapter.reset()
    assert adapter._state is None and adapter._last is None
    new = adapter.start(_frame(epoch=3), (ObjectSeed(29, box=(15, 3, 20, 8)),))
    assert new.epoch == 3 and adapter._state is not old
    assert adapter._state["obj_ids"] == [29]


def test_missing_weights_fail_without_downloading_or_fabricating_a_mask():
    adapter = backend.Sam2Backend("/absent/weights.pt")
    with pytest.raises(FileNotFoundError, match="not present"):
        adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))


def test_a_model_with_larger_memory_requirements_cannot_have_its_history_silently_pruned(setup):
    adapter, model = setup
    model.max_obj_ptrs_in_encoder = 64
    with pytest.raises(ValueError, match="memory dependencies"):
        adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    assert not model.calls


def test_result_pixels_do_not_alias_future_model_outputs(setup):
    adapter, model = setup
    first = adapter.start(_frame(), (ObjectSeed(17, box=(5, 6, 13, 14)),))
    retained = first.masks[0].mask.copy()
    model.masks[17][:] = False
    adapter.track(_frame(1))
    assert np.array_equal(first.masks[0].mask, retained) and not first.masks[0].mask.flags.writeable

