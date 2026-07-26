"""A lane refused for want of memory must retry, not stay broken.

`_rearm_runtime_failed_lane` only re-armed a lane whose failure began with
`mlx_runtime_unavailable` or `local_runtime_unavailable`. A worker spawn refused
for headroom produced a different reason and therefore parked the lane in
`failed` permanently — for a condition that is transient by definition, since
host memory frees constantly.

Live 2026-07-26:

    memory_pressure_refused_worker_spawn:model_load_headroom:23.3GB
        < required 24.0GB

Short by 0.7GB, and the lane never tried again on its own. She reported a broken
mind for a shortfall that had already passed, which is this pass's recurring
error one more time: a transient condition recorded as permanent damage.
"""
from __future__ import annotations

from pathlib import Path

from core.brain.inference_gate import _REARMABLE_LANE_FAILURE_PREFIXES

SOURCE = Path("core/brain/inference_gate.py")


def test_a_memory_refused_spawn_is_rearmable() -> None:
    reason = "memory_pressure_refused_worker_spawn:model_load_headroom:23.3GB < required 24.0GB"
    assert reason.startswith(_REARMABLE_LANE_FAILURE_PREFIXES)


def test_the_original_runtime_failures_stay_rearmable() -> None:
    for reason in ("mlx_runtime_unavailable:x", "local_runtime_unavailable"):
        assert reason.startswith(_REARMABLE_LANE_FAILURE_PREFIXES)


def test_a_genuine_fault_is_not_swept_into_the_rearm_lane() -> None:
    """Re-arming a real fault would loop forever instead of surfacing it."""
    for reason in (
        "model_weights_corrupt",
        "runtime_identity_mismatch",
        "",
        "cognitive_engine_self_process_grounding",
    ):
        assert not reason.startswith(_REARMABLE_LANE_FAILURE_PREFIXES)


def test_every_rearm_site_uses_the_shared_tuple() -> None:
    """Four call sites drifted apart before; they share one definition now."""
    src = SOURCE.read_text(encoding="utf-8")
    assert src.count("_REARMABLE_LANE_FAILURE_PREFIXES") >= 5, (
        "the constant plus each call site that decides re-armability"
    )
    assert '("mlx_runtime_unavailable", "local_runtime_unavailable")' not in src, (
        "no site may keep its own private copy of the prefix list"
    )
