"""Track one fresh capture through SAM 2's memory without replaying a video.

The state schema follows facebookresearch/sam2 revision
2b90b9f5ceec907a1c18123530e92e794ad901a4. Its public video loader takes
a completed video; this adapter supplies and retains the current RGB frame.
Masks and presence scores remain model proposals. A supplied mask's presence
is fabricated by upstream and cannot grant observation or control authority.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.perception.learned_tracking.contracts import (
    MaskResult,
    MaskTrack,
    ObjectSeed,
    TrackingFrame,
    following_frame,
    validate_objects,
)

SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
TINY_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _torch() -> Any:
    import torch

    return torch


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _rgb_tensor(pixels: np.ndarray, size: int) -> Any:
    rgb = np.asarray(Image.fromarray(pixels).resize((size, size)), dtype=np.float32) / 255.
    rgb = (rgb - np.asarray((.485, .456, .406), np.float32)) / np.asarray((.229, .224, .225), np.float32)
    return _torch().from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def _seed_in_frame(seed: ObjectSeed, shape: tuple[int, int]) -> None:
    if type(seed.number) is not int or seed.number < 0:
        raise ValueError("an object needs a nonnegative integer identity")
    if (seed.box is None) == (seed.mask is None):
        raise ValueError("an object needs one box or one mask")
    if seed.mask is not None:
        if (not isinstance(seed.mask, np.ndarray) or seed.mask.dtype != np.bool_
                or seed.mask.shape != shape or not seed.mask.any()):
            raise ValueError("an object mask needs current nonempty boolean pixels")
        return
    box = seed.box
    if (len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= shape[1] or not 0 <= box[1] < box[3] <= shape[0]):
        raise ValueError("an object box must lie inside its current capture")


class Sam2Backend:
    """Own one bounded streaming session and the model that measures its masks."""

    def __init__(self, weights: str | Path, *, device: str = "cpu", config: str = TINY_CONFIG,
                 predictor: Any = None, max_objects: int = 16, retained_frames: int = 32,
                 retained_conditioning: int = 2, memory_lease: Any = None) -> None:
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("tracking device must be cpu, mps or cuda")
        if (type(max_objects) is not int or not 1 <= max_objects <= 64
                or type(retained_frames) is not int or not 1 <= retained_frames <= 128
                or type(retained_conditioning) is not int or not 1 <= retained_conditioning <= 8):
            raise ValueError("tracking memory needs bounded positive limits")
        self.weights, self.device, self.config = Path(weights), device, config
        self._predictor = predictor
        self._state: dict[str, Any] | None = None
        self._last: TrackingFrame | None = None
        self._shape: tuple[int, int] | None = None
        self._index = -1
        self.max_objects, self.retained_frames = max_objects, retained_frames
        self.retained_conditioning = retained_conditioning
        self._memory_lease = memory_lease

    def _model(self) -> Any:
        if self._predictor is None:
            if not self.weights.is_file():
                raise FileNotFoundError(f"SAM 2 weights are not present: {self.weights}")
            from core.runtime.model_lane_control import (
                require_active_synchronous_in_process_model_lane,
            )

            require_active_synchronous_in_process_model_lane(self._memory_lease)
            from sam2.build_sam import build_sam2_video_predictor

            self._predictor = build_sam2_video_predictor(
                self.config, str(self.weights), device=self.device,
                apply_postprocessing=False, vos_optimized=False,
            )
        model = self._predictor
        if (type(getattr(model, "image_size", None)) is not int or not 16 <= model.image_size <= 2048
                or not all(callable(getattr(model, method, None)) for method in
                           ("add_new_points_or_box", "add_new_mask", "propagate_in_video", "remove_object"))):
            raise RuntimeError("SAM 2 predictor does not provide the pinned streaming interface")
        memory = int(getattr(model, "num_maskmem", 7))
        stride = int(getattr(model, "memory_temporal_stride_for_eval", 1))
        pointers = int(getattr(model, "max_obj_ptrs_in_encoder", 16))
        if min(memory, stride, pointers) <= 0 or self.retained_frames < max(memory * stride + 2, pointers + 1):
            raise ValueError("retained frames cannot discard SAM 2's memory dependencies")
        return model

    def _new_state(self, shape: tuple[int, int], model: Any) -> dict[str, Any]:
        # The official loader constructs these containers before any prompt.
        # We retain absolute frame indexes and a single current input tensor.
        state = {name: {} for name in ("images", "point_inputs_per_obj", "mask_inputs_per_obj", "cached_features",
            "constants", "output_dict_per_obj", "temp_output_dict_per_obj", "frames_tracked_per_obj")}
        # MPS's asynchronous CPU offload can expose unread memory before the
        # next prompt consumes it. Bounded memory stays on its compute device.
        state.update(num_frames=0, offload_video_to_cpu=True, offload_state_to_cpu=False,
                     video_height=shape[0], video_width=shape[1], device=model.device,
                     storage_device=model.device, obj_id_to_idx=OrderedDict(),
                     obj_idx_to_id=OrderedDict(), obj_ids=[])
        return state

    def start(self, frame: TrackingFrame, seeds: tuple[ObjectSeed, ...]) -> MaskResult:
        validate_objects(frame, seeds)
        shape = frame.shape
        self._validate_seeds(seeds, shape, existing=())
        model = self._model()
        self.reset()
        self._shape, self._state = shape, self._new_state(shape, model)
        return self._advance(frame, seeds)

    def track(self, frame: TrackingFrame) -> MaskResult:
        return self._advance(frame, ())

    def add(self, frame: TrackingFrame, seeds: tuple[ObjectSeed, ...]) -> MaskResult:
        """New objects enter on a fresh capture without restarting the other objects."""
        return self._advance(frame, seeds)

    def _validate_seeds(self, seeds: tuple[ObjectSeed, ...], shape: tuple[int, int], *, existing: Any) -> None:
        seen = set(existing)
        for seed in seeds:
            _seed_in_frame(seed, shape)
            if seed.number in seen:
                raise ValueError("an existing object cannot be silently reseeded")
            seen.add(seed.number)
        if len(seen) > self.max_objects:
            raise ValueError("SAM 2 object capacity is full")

    def _advance(self, frame: TrackingFrame, seeds: tuple[ObjectSeed, ...]) -> MaskResult:
        began = time.perf_counter()
        shape = frame.shape
        if self._state is None:
            raise RuntimeError("SAM 2 tracking has not started")
        if shape != self._shape:
            raise ValueError("new capture geometry needs a new tracking session")
        if self._last is not None:
            following_frame(self._last, frame)
        if seeds:
            validate_objects(frame, seeds)
        self._validate_seeds(seeds, shape, existing=self._state["obj_ids"])
        model = self._model()
        index = self._index + 1
        owned = _rgb_tensor(frame.pixels.copy(), model.image_size)
        state = self._state
        state["images"] = {index: owned}
        state["num_frames"] = index + 1
        # All validation precedes mutation. Inference failure retires this
        # session because upstream may already have changed its memory.
        try:
            with _torch().inference_mode():
                for seed in seeds:
                    if seed.mask is not None:
                        model.add_new_mask(state, index, seed.number, seed.mask.copy())
                    else:
                        model.add_new_points_or_box(state, index, seed.number, box=tuple(seed.box))
                masks = self._current_masks(model, state, index, frame, {seed.number for seed in seeds})
            self._trim(state, index)
        except BaseException:
            self.reset()
            raise
        self._last, self._index = frame, index
        status = "seeded" if masks and all(mask.seeded for mask in masks) else "observed"
        return MaskResult("sam2", frame.epoch, frame.number, float(frame.at), shape, masks,
                          time.perf_counter() - began, status=status)

    def _current_masks(self, model: Any, state: dict[str, Any], index: int, frame: TrackingFrame,
                       seeded: set[int]) -> tuple[MaskTrack, ...]:
        if not state["obj_ids"]:
            return ()
        stream = model.propagate_in_video(state, start_frame_idx=index, max_frame_num_to_track=0, reverse=False)
        try:
            results = list(islice(stream, 2))
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        if len(results) != 1 or results[0][0] != index:
            raise RuntimeError("SAM 2 propagation must answer exactly the current frame")
        _, numbers, logits = results[0]
        arrays = _array(logits)
        shape = frame.pixels.shape[:2]
        if (list(numbers) != state["obj_ids"] or arrays.shape != (len(numbers), 1, *shape)
                or not np.isfinite(arrays).all()):
            raise RuntimeError("SAM 2 returned masks without their current object and pixel ownership")
        found = []
        for offset, number in enumerate(numbers):
            obj_idx = state["obj_id_to_idx"][number]
            outputs = state["output_dict_per_obj"][obj_idx]
            output = outputs["cond_frame_outputs"].get(index) or outputs["non_cond_frame_outputs"].get(index)
            score = None if output is None else output.get("object_score_logits")
            scores = np.asarray([]) if score is None else _array(score).reshape(-1)
            presence = None
            if len(scores) == 1 and math.isfinite(float(scores[0])):
                presence = 1. / (1. + math.exp(-max(-80., min(80., float(scores[0])))))
            mask = np.ascontiguousarray(arrays[offset, 0] > 0)
            was_seeded = number in seeded
            visible = not was_seeded and presence is not None and presence > .5 and bool(mask.any())
            found.append(MaskTrack(int(number), mask, None if was_seeded else presence, visible, was_seeded))
        return tuple(found)

    def _trim(self, state: dict[str, Any], index: int) -> None:
        floor = index - self.retained_frames + 1
        state["cached_features"] = {key: value for key, value in state["cached_features"].items() if key == index}
        for obj_idx in state["output_dict_per_obj"]:
            outputs = state["output_dict_per_obj"][obj_idx]
            cond = outputs["cond_frame_outputs"]
            latest = sorted(cond)[-self.retained_conditioning:]
            for key in list(cond):
                if key not in latest:
                    del cond[key]
            for key in list(outputs["non_cond_frame_outputs"]):
                if key < floor:
                    del outputs["non_cond_frame_outputs"][key]
            retained = set(cond) | set(outputs["non_cond_frame_outputs"])
            for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj"):
                entries = state[name][obj_idx]
                for key in list(entries):
                    if key not in retained:
                        del entries[key]

    def remove(self, number: int) -> None:
        """Object ids remain stable while upstream shifts its internal indexes."""
        if self._state is None:
            return
        try:
            self._model().remove_object(self._state, number, strict=True, need_output=False)
        except BaseException:
            self.reset()
            raise

    def reset(self) -> None:
        """Retire all frame and feature memory without unloading shared weights."""
        self._state, self._last, self._shape, self._index = None, None, None, -1

    def close(self) -> None:
        """Release all session memory and the worker's predictor reference."""
        self.reset()
        self._predictor = None

