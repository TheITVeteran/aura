"""Online BootsTAPIR advances its learned point state on captured RGB frames.

The causal PyTorch API is from google-deepmind/tapnet, revision
730cda1c730877cfedbe01bf87fb1cadb78a565d, tapnet/torch/tapir_model.py.
Its hidden points remain predictions and cannot supply visible observations.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from core.perception.learned_tracking.contracts import (
    PointResult,
    PointSeed,
    PointTrack,
    TrackingFrame,
    following_frame,
    validate_points,
)

__all__ = ["OnlineTapirBackend"]


def _whole_number(value: Any) -> bool:
    return type(value) is int and value >= 0


class OnlineTapirBackend:
    """One causal point session, loaded explicitly and used on its owning worker."""

    name = "online_tapir"

    def __init__(self, weights: Path, *, device: str = "cpu", model: Any = None, max_points: int = 64) -> None:
        if not _whole_number(max_points) or not 1 <= max_points <= 384:
            raise ValueError("TAPIR's point limit must be a bounded positive whole number")
        if model is None and not Path(weights).is_file():
            raise FileNotFoundError(f"TAPIR checkpoint is absent: {weights}")
        import torch

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("TAPIR supports an explicit CPU, CUDA or MPS device")
        if self._device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("the requested TAPIR MPS device is unavailable")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("the requested TAPIR CUDA device is unavailable")
        if model is None:
            from tapnet.torch.tapir_model import TAPIR

            from core.perception.learned_tracking.sampling_a_single_frame import (
                use_single_frame_sampling,
            )

            use_single_frame_sampling()
            model = TAPIR(pyramid_level=1, use_casual_conv=True)
            model.load_state_dict(torch.load(Path(weights), map_location="cpu", weights_only=True), strict=True)
        self._model = model.to(self._device).eval()
        self._max_points = max_points
        self._clear()

    def _clear(self) -> None:
        self._features: Any = None
        self._causal: Any = None
        self._seeds: tuple[PointSeed, ...] = ()
        self._shape: tuple[int, int] | None = None
        self._previous: TrackingFrame | None = None

    def _video(self, frame: TrackingFrame) -> Any:
        owned = np.array(frame.pixels, copy=True, order="C")
        return self._torch.from_numpy(owned).to(self._device, dtype=self._torch.float32)[None, None] / 127.5 - 1.

    def _on_device(self, state: Any) -> Any:
        if isinstance(state, dict):
            return {key: self._on_device(value) for key, value in state.items()}
        if isinstance(state, (tuple, list)):
            return type(state)(self._on_device(value) for value in state)
        if not self._torch.is_tensor(state):
            raise TypeError("TAPIR causal state must contain tensors")
        return state.to(device=self._device, dtype=self._torch.float32).clone()

    def _grids(self, video: Any) -> Any:
        return self._model.get_feature_grids(video, is_training=False, refinement_resolutions=[(256, 256)])

    def _predict(self, frame: TrackingFrame, grids: Any, began: float) -> PointResult:
        torch = self._torch
        trajectories = self._model.estimate_trajectories(
            self._shape, is_training=False, feature_grids=grids, query_features=self._features,
            query_points_in_video=None, query_chunk_size=min(64, self._max_points),
            causal_context=self._causal, get_causal_context=True,
        )
        xy, occluded, distant = (trajectories[key][-1] for key in ("tracks", "occlusion", "expected_dist"))
        n = len(self._seeds)
        if (tuple(xy.shape) != (1, n, 1, 2) or tuple(occluded.shape) != (1, n, 1)
                or tuple(distant.shape) != (1, n, 1)
                or not all(bool(torch.isfinite(value).all()) for value in (xy, occluded, distant))):
            raise ValueError("TAPIR returned malformed or nonfinite point predictions")
        state = trajectories["causal_context"]
        if (not isinstance(state, (tuple, list)) or len(state) != 4 * (len(self._features.resolutions) - 1)
                or any(not isinstance(part, dict) or not part for part in state)
                or any(not torch.is_tensor(value) or value.ndim != 4 or tuple(value.shape[:3]) != (1, n, 2)
                       for part in state for value in part.values())):
            raise ValueError("TAPIR returned malformed causal state")
        probability = torch.sigmoid(-occluded) * torch.sigmoid(-distant)
        values = torch.cat((xy[0, :, 0], probability[0, :, 0, None]), dim=-1).detach().cpu().numpy()
        height, width = self._shape
        tracks = tuple(PointTrack(
            number=seed.number, x=float(value[0]), y=float(value[1]), visibility=float(value[2]),
            visible=bool(value[2] > .5 and 0 <= value[0] < width and 0 <= value[1] < height),
            object_number=seed.object_number,
        ) for seed, value in zip(self._seeds, values, strict=True))
        result = PointResult(backend=self.name, epoch=frame.epoch, frame_number=frame.number, at=float(frame.at),
                             shape=self._shape, tracks=tracks, elapsed_s=time.monotonic() - began)
        self._causal = state
        self._previous = frame
        return result

    def start(self, frame: TrackingFrame, seeds: tuple[PointSeed, ...]) -> PointResult:
        """Seed only this captured frame and create fresh causal history."""
        began = time.monotonic()
        self._clear()
        if self._model is None:
            raise RuntimeError("the TAPIR backend is closed")
        validate_points(frame, seeds)
        if not isinstance(seeds, tuple) or len(seeds) > self._max_points:
            raise ValueError("TAPIR needs a bounded tuple of point seeds")
        self._shape, self._seeds = frame.shape, seeds
        try:
            with self._torch.inference_mode():
                video = self._video(frame)
                grids = self._grids(video)
                queries = self._torch.tensor([[0., seed.y, seed.x] for seed in seeds],
                                             dtype=self._torch.float32, device=self._device)[None]
                self._features = self._model.get_query_features(
                    video, is_training=False, query_points=queries, feature_grids=grids,
                )
                self._causal = self._on_device(self._model.construct_initial_causal_state(
                    len(seeds), len(self._features.resolutions) - 1,
                ))
                return self._predict(frame, grids, began)
        except BaseException:
            self._clear()
            raise

    def track(self, frame: TrackingFrame, *, replacements: dict[int, PointSeed] | None = None) -> PointResult:
        """Advance once on a later captured frame in the same picture basis."""
        began = time.monotonic()
        if self._features is None or self._causal is None:
            raise RuntimeError("TAPIR needs fresh point seeds before tracking")
        try:
            following_frame(self._previous, frame)
            with self._torch.inference_mode():
                video = self._video(frame)
                grids = self._grids(video)
                if replacements:
                    self._replace_queries(frame, video, grids, replacements)
                return self._predict(frame, grids, began)
        except BaseException:
            self._clear()
            raise

    def _replace_queries(self, frame: TrackingFrame, video: Any, grids: Any,
                         replacements: dict[int, PointSeed]) -> None:
        """Newly exposed texture gets query slots without erasing continuing point history."""
        indexes = tuple(sorted(replacements))
        if any(type(index) is not int or not 0 <= index < len(self._seeds) for index in indexes):
            raise ValueError('query replacement needs existing point slots')
        updated = tuple(replacements.get(index, seed) for index, seed in enumerate(self._seeds))
        validate_points(frame, updated)
        new = tuple(replacements[index] for index in indexes)
        queries = self._torch.tensor([[0., seed.y, seed.x] for seed in new],
                                     dtype=self._torch.float32, device=self._device)[None]
        features = self._model.get_query_features(video, is_training=False, query_points=queries, feature_grids=grids)
        self._features = self._model.update_query_features(self._features, features, indexes)
        fresh = self._on_device(self._model.construct_initial_causal_state(
            len(new), len(self._features.resolutions) - 1))
        # Upstream resets are CPU tensors. Reset only these slots on the
        # worker's device; the other points keep their causal history.
        slots = self._torch.tensor(indexes, dtype=self._torch.long, device=self._device)
        self._causal = [{key: value.clone().index_copy_(1, slots, replacement[key]) for key, value in part.items()}
                        for part, replacement in zip(self._causal, fresh, strict=True)]
        self._seeds = updated

    def close(self) -> None:
        """Drop session features and its loaded model when its worker leaves."""
        self._clear()
        self._model = None
