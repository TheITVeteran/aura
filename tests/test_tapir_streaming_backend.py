"""The causal adapter preserves captured identity without promoting hidden points."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.perception.learned_tracking.contracts import PointSeed, TrackingFrame
from core.perception.learned_tracking.tapir_backend import OnlineTapirBackend

pytestmark = pytest.mark.unit


class InventedCausalModel(torch.nn.Module):
    """A tensor model with the official causal methods and invented appearances."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.outputs = None
        self.bad_state = False
        self.fail = False

    def get_feature_grids(self, video, *, is_training, refinement_resolutions):
        assert not is_training and not torch.is_grad_enabled()
        assert tuple(video.shape[:2]) == (1, 1) and video.dtype == torch.float32
        assert refinement_resolutions == [(256, 256)]
        self.calls.append(("picture", video.detach().clone()))
        return SimpleNamespace(resolutions=((256, 256), (256, 256)), picture=video)

    def get_query_features(self, video, *, is_training, query_points, feature_grids):
        assert not is_training and feature_grids.picture is video
        self.calls.append(("seeds", query_points.detach().clone()))
        return SimpleNamespace(resolutions=feature_grids.resolutions, queries=query_points)

    def construct_initial_causal_state(self, num_points, num_resolutions):
        assert num_resolutions == 1
        self.calls.append(("reset", num_points))
        return [{"past": torch.zeros((1, num_points, 2, 4))}] * 4

    def update_query_features(self, query_features, new_query_features, idx_to_update, causal_state=None):
        query_features.queries[:, idx_to_update] = new_query_features.queries
        self.calls.append(('replace', tuple(idx_to_update)))
        if causal_state is None:
            return query_features
        updated = []
        for part in causal_state:
            past = part['past'].clone()
            past[:, idx_to_update] = 0.
            updated.append({'past': past})
        return query_features, updated

    def estimate_trajectories(self, video_size, *, is_training, feature_grids, query_features,
                              query_points_in_video, query_chunk_size, causal_context, get_causal_context):
        assert not is_training and not torch.is_grad_enabled()
        assert query_points_in_video is None and get_causal_context and query_chunk_size <= 64
        assert tuple(feature_grids.picture.shape[2:4]) == tuple(video_size)
        self.calls.append(("predict", float(causal_context[0]["past"].mean())))
        if self.fail:
            raise RuntimeError("the invented image could not be tracked")
        n = query_features.queries.shape[1]
        if self.outputs is None:
            xy = query_features.queries[..., [2, 1]][:, :, None].clone()
            xy[..., 0] += causal_context[0]["past"].mean()
            occluded = torch.full((1, n, 1), -8.)
            distant = torch.full((1, n, 1), -8.)
        else:
            xy, occluded, distant = self.outputs
        state = [{"past": part["past"] + 1.} for part in causal_context]
        return {"tracks": [xy], "occlusion": [occluded], "expected_dist": [distant],
                "causal_context": [] if self.bad_state else state}


def frame(*, number=1, at=10., epoch=7, shape=(24, 40)):
    pixels = np.zeros((*shape, 3), np.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 0, 127, 255
    return TrackingFrame(pixels=pixels, at=at, epoch=epoch, number=number)


def backend(model=None, **kwargs):
    return OnlineTapirBackend(Path("unused-invented-weights.pt"), model=model or InventedCausalModel(), **kwargs)


SEEDS = (PointSeed(number=91, x=6., y=9., object_number=12), PointSeed(number=4, x=18., y=5.))


def test_frame_layout_normalization_and_query_axis_order_are_real_tensors():
    model = InventedCausalModel()
    observer = backend(model)
    result = observer.start(frame(), SEEDS)
    picture = next(value for kind, value in model.calls if kind == "picture")
    query = next(value for kind, value in model.calls if kind == "seeds")
    assert tuple(picture.shape) == (1, 1, 24, 40, 3)
    assert picture[0, 0, 0, 0].tolist() == pytest.approx([-1., 127 / 127.5 - 1., 1.])
    assert query.tolist() == [[[0., 9., 6.], [0., 5., 18.]]]
    assert result.backend == "online_tapir" and result.shape == (24, 40)
    assert (result.epoch, result.frame_number, result.at) == (7, 1, 10.)
    assert [(point.number, point.object_number, point.x, point.y) for point in result.tracks] == [
        (91, 12, 6., 9.), (4, None, 18., 5.),
    ]
    assert all(point.visible and point.visibility > .99 for point in result.tracks)
    assert result.elapsed_s >= 0.


def test_each_later_frame_advances_the_causal_state_once_and_keeps_seed_order():
    model = InventedCausalModel()
    observer = backend(model)
    first = observer.start(frame(), SEEDS)
    second = observer.track(frame(number=2, at=10.1))
    third = observer.track(frame(number=3, at=10.2))
    assert [value for kind, value in model.calls if kind == "predict"] == [0., 1., 2.]
    assert sum(kind == "reset" for kind, _ in model.calls) == 1
    assert [point.number for point in third.tracks] == [91, 4]
    assert [result.tracks[0].x for result in (first, second, third)] == [6., 7., 8.]
    assert [second.at, third.at] == [10.1, 10.2]


def test_new_texture_can_replace_one_slot_without_erasing_other_causal_history():
    model = InventedCausalModel()
    observer = backend(model)
    observer.start(frame(), SEEDS)
    observer.track(frame(number=2, at=10.1))
    result = observer.track(frame(number=3, at=10.2), replacements={1: PointSeed(300, 20., 7.)})
    assert [point.number for point in result.tracks] == [91, 300]
    assert observer._causal[0]['past'][0, 0].eq(3.).all()
    assert observer._causal[0]['past'][0, 1].eq(1.).all()
    assert [value for kind, value in model.calls if kind == 'reset'] == [2, 1]
    assert ('replace', (1,)) in model.calls


@pytest.mark.parametrize('replacements', [{-1: PointSeed(300, 4., 5.)}, {2: PointSeed(300, 4., 5.)},
                                         {1: PointSeed(91, 4., 5.)}, {1: PointSeed(300, 45., 5.)}])
def test_invalid_query_replacement_cannot_advance_an_existing_session(replacements):
    model = InventedCausalModel()
    observer = backend(model)
    observer.start(frame(), SEEDS)
    with pytest.raises(ValueError):
        observer.track(frame(number=2, at=10.1), replacements=replacements)
    assert not any(kind == 'replace' for kind, _ in model.calls)
    with pytest.raises(RuntimeError):
        observer.track(frame(number=3, at=10.2))


def test_starting_a_new_basis_discards_old_features_and_causal_history():
    model = InventedCausalModel()
    observer = backend(model)
    observer.start(frame(), SEEDS)
    observer.track(frame(number=2, at=10.1))
    result = observer.start(frame(epoch=8, number=1, at=11., shape=(30, 50)), (PointSeed(101, 3., 4.),))
    assert [value for kind, value in model.calls if kind == "predict"] == [0., 1., 0.]
    assert result.epoch == 8 and result.shape == (30, 50)
    assert [(point.number, point.x, point.y) for point in result.tracks] == [(101, 3., 4.)]


def test_occlusion_distance_and_outside_predictions_are_not_visible_observations():
    model = InventedCausalModel()
    xy = torch.tensor([[[[4., 7.]], [[8., 9.]], [[80., 3.]], [[12., 4.]]]])
    model.outputs = (xy, torch.tensor([[[8.], [-8.], [-8.], [0.]]]),
                     torch.tensor([[[-8.], [8.], [-8.], [0.]]]))
    seeds = tuple(PointSeed(number=n, x=float(n + 1), y=2.) for n in range(4))
    result = backend(model).start(frame(), seeds)
    assert [point.visible for point in result.tracks] == [False, False, False, False]
    assert [point.x for point in result.tracks] == [4., 8., 80., 12.]
    assert result.tracks[-1].visibility == .25


@pytest.mark.parametrize("changed", [dict(epoch=8), dict(shape=(25, 40)), dict(number=1, at=10.1),
                                    dict(number=2, at=10.), dict(number=2, at=9.), dict(number=0, at=10.1)])
def test_a_changed_basis_or_repeated_capture_retires_the_session(changed):
    model = InventedCausalModel()
    observer = backend(model)
    observer.start(frame(), SEEDS)
    with pytest.raises(ValueError, match="later capture in the same generation"):
        observer.track(frame(**changed))
    assert sum(kind == "predict" for kind, _ in model.calls) == 1
    with pytest.raises(RuntimeError, match="fresh point seeds"):
        observer.track(frame(number=3, at=10.2))


def test_failed_inference_cannot_leave_a_partially_advanced_session():
    model = InventedCausalModel()
    observer = backend(model)
    observer.start(frame(), SEEDS)
    model.fail = True
    with pytest.raises(RuntimeError, match="invented image"):
        observer.track(frame(number=2, at=10.1))
    model.fail = False
    with pytest.raises(RuntimeError, match="fresh point seeds"):
        observer.track(frame(number=3, at=10.2))
    assert observer.start(frame(number=3, at=10.2), SEEDS).tracks[0].x == 6.


@pytest.mark.parametrize("broken", ["point_count", "time_axis", "nonfinite_track", "nonfinite_logit", "state"])
def test_malformed_predictions_are_errors_instead_of_fabricated_points(broken):
    model = InventedCausalModel()
    xy = torch.tensor([[[[6., 9.]], [[18., 5.]]]])
    occluded, distant = torch.full((1, 2, 1), -8.), torch.full((1, 2, 1), -8.)
    if broken == "point_count":
        xy = xy[:, :1]
    elif broken == "time_axis":
        xy = xy.repeat(1, 1, 2, 1)
    elif broken == "nonfinite_track":
        xy[0, 0, 0, 0] = float("nan")
    elif broken == "nonfinite_logit":
        distant[0, 0, 0] = float("inf")
    else:
        model.bad_state = True
    model.outputs = xy, occluded, distant
    observer = backend(model)
    with pytest.raises(ValueError, match="malformed|nonfinite"):
        observer.start(frame(), SEEDS)
    with pytest.raises(RuntimeError, match="fresh point seeds"):
        observer.track(frame(number=2, at=10.1))


@pytest.mark.parametrize("seeds", [(), (PointSeed(1, -1., 2.),), (PointSeed(1, 40., 2.),),
                                  (PointSeed(1, 2., 2.), PointSeed(1, 3., 3.))])
def test_seed_inputs_are_bounded_finite_current_points(seeds):
    model = InventedCausalModel()
    with pytest.raises(ValueError, match="queries"):
        backend(model).start(frame(), seeds)
    assert not model.calls


def test_point_capacity_and_absent_weights_do_not_load_a_fallback():
    model = InventedCausalModel()
    with pytest.raises(ValueError, match="bounded tuple"):
        backend(model, max_points=1).start(frame(), SEEDS)
    assert not model.calls
    with pytest.raises(FileNotFoundError, match="checkpoint is absent"):
        OnlineTapirBackend(Path("does-not-exist-tapir-checkpoint.pt"))


def test_close_drops_both_session_and_model():
    observer = backend()
    observer.start(frame(), SEEDS)
    observer.close()
    assert observer._model is None and observer._features is None and observer._causal is None
    with pytest.raises(RuntimeError, match="fresh point seeds"):
        observer.track(frame(number=2, at=10.1))
    with pytest.raises(RuntimeError, match="backend is closed"):
        observer.start(frame(number=2, at=10.1), SEEDS)
