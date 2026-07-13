"""Live Ray cluster proof (owner-authorized LAN run, 2026-07-13).

Boots ONE real bounded Ray instance and drives the actual RayBackend
through it within a single lifecycle (repeated init/shutdown cycles are
unstable on macOS and orphan gcs_server processes). Proves, against a
genuine cluster that unit tests cannot fake:

- tasks execute in separate worker PROCESSES (distinct PIDs)
- results round-trip correctly and in order
- the cluster reports the bounded resources we requested
- a worker exception is contained: it surfaces, the driver survives,
  and the cluster keeps serving

Resource discipline: 2 CPUs, 256 MB object store — the live 32B model
next door is never at risk. Skips when ray is not installed.
"""
from __future__ import annotations

import os

import pytest

ray = pytest.importorskip("ray")

from core.swarm.ray_backend import RayBackend  # noqa: E402


def _task_payload(value: int):
    def work():
        return {"pid": os.getpid(), "square": value * value}
    return work


async def test_ray_backend_live_cluster_end_to_end():
    """Complete live proof in one Ray lifecycle."""
    ray.init(
        num_cpus=2,
        object_store_memory=256 * 1024 * 1024,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="WARNING",
    )
    try:
        backend = RayBackend()
        assert backend.is_available(), "Ray initialized but backend inactive"

        # 1) Distribution: real parallel execution across worker processes.
        results = await backend.execute_parallel(
            [_task_payload(value) for value in range(6)])
        assert [row["square"] for row in results] == [0, 1, 4, 9, 16, 25]
        worker_pids = {row["pid"] for row in results}
        assert os.getpid() not in worker_pids  # genuinely separate processes
        assert len(worker_pids) >= 1

        # 2) The cluster reports the bounded resources we asked for.
        resources = ray.cluster_resources()
        assert resources.get("CPU", 0) == 2.0
        store = resources.get("object_store_memory", 0)
        assert 0 < store <= 300 * 1024 * 1024

        # 3) Worker-crash containment: exception surfaces, driver survives,
        #    cluster keeps serving.
        def poison():
            raise RuntimeError("deliberate worker failure")

        with pytest.raises(RuntimeError, match="deliberate worker failure"):
            await backend.execute_parallel([poison])

        recovered = await backend.execute_parallel([_task_payload(9)])
        assert recovered[0]["square"] == 81
    finally:
        ray.shutdown()
