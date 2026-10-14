"""Numerical tracking weights join the same host-memory ledger as the resident cortex."""
from __future__ import annotations

import gc
import uuid
from pathlib import Path
from typing import Any

from core.runtime.optional_compute import OptionalComputeBusy, optional_compute


def finish_device_work(device: str, *, empty_cache: bool = False) -> None:
    if device not in ('mps', 'cuda'):
        return
    import torch

    allocator = torch.mps if device == 'mps' else torch.cuda
    allocator.synchronize()
    if empty_cache:
        allocator.empty_cache()


def admit_tracking_memory(name: str, weights: Path):
    from core.runtime.model_lane_control import (
        ModelLaneControlError,
        acquire_synchronous_in_process_model_lane,
    )

    # Checkpoint bytes size the persistent weights; the extra allocation is
    # a declared estimate for inference and streaming state, not a measured peak.
    footprint = max(.5, 2. * weights.stat().st_size / 1024**3)
    try:
        return acquire_synchronous_in_process_model_lane(
            owner_id=f'visual-tracking:{name}:{uuid.uuid4()}', model_path=str(weights.resolve()),
            purpose='serve', request_gb=footprint + 1.5, priority=80, preemptible=False,
            metadata={'modality': 'tracking', 'device_memory_estimate': True},
        )
    except ModelLaneControlError as why:
        raise OptionalComputeBusy(f'tracking memory admission deferred: {why}') from why


class AdmittedTrackingBackend:
    """Memory stays counted until its actual model and device allocations are released."""

    def __init__(self, backend: Any, lease: Any, *, name: str, device: str) -> None:
        self._backend, self._lease = backend, lease
        self.name, self.device = name, device

    def __getattr__(self, key: str) -> Any:
        return getattr(self._backend, key)

    def _call(self, method: str, *args, **kwargs):
        if self._backend is None or self._lease is None or not self._lease.active:
            raise OptionalComputeBusy('tracking model no longer owns its memory fence')
        with optional_compute():
            try:
                return getattr(self._backend, method)(*args, **kwargs)
            finally:
                self._synchronize()

    def _synchronize(self) -> None:
        finish_device_work(self.device)

    def start(self, *args, **kwargs):
        return self._call('start', *args, **kwargs)

    def track(self, *args, **kwargs):
        return self._call('track', *args, **kwargs)

    def add(self, *args, **kwargs):
        return self._call('add', *args, **kwargs)

    def close(self) -> None:
        if self._lease is None:
            return
        if self._backend is not None:
            self._backend.close()
            self._backend = None
            gc.collect()
            finish_device_work(self.device, empty_cache=True)
        if not self._lease.release(reason='tracking_weights_and_state_released'):
            raise RuntimeError('tracking memory owner was not released')
        self._lease = None
