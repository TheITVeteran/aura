"""tests/test_mlx_recycle_memory_reclaim.py
============================================
A worker respawn that REPLACES a resident-but-dead worker (lane recycle after a
steered-generation wedge, or crash recovery) must reclaim the old worker's
memory BEFORE the model-load headroom admission check — otherwise the check
sees the about-to-die worker's ~20GB still resident and refuses the spawn
(memory_pressure_refused_worker_spawn: model_load_headroom: 20.2GB < 22.0GB),
leaving the lane cold (recycled_model_lane_not_live_after_warmup → DNU FATAL).

Regression for the round-28 cortex-wedge recovery failure.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_client as mc


class _FakeOrphan:
    pid = 999_999  # never equals the real os.getpid()

    def __init__(self, events):
        self._events = events
        self.info = {
            "name": "MLXWorker-Qwen2.5-32B-Instruct-4bit",
            "cmdline": ["python", "mlx_worker", "Qwen2.5-32B-Instruct-4bit"],
        }

    def kill(self):
        self._events.append("kill")

    def wait(self, timeout=None):
        return None

    def parents(self):
        # Ownership check (81f2b64b): only orphans descended from THIS
        # process are reclaimed. This fake is ours.
        import os
        from types import SimpleNamespace

        return [SimpleNamespace(pid=os.getpid())]


def test_orphan_reclaimed_before_memory_admission(monkeypatch, resource_observer):
    # Disable the bounded reclaim re-poll so the contrived always-blocked
    # memcheck fails fast instead of spinning the full production window.
    monkeypatch.setenv("AURA_MLX_SPAWN_RECLAIM_WAIT_S", "0")
    client = mc.MLXLocalClient(model_path="mlx-community/Qwen2.5-32B-Instruct-4bit")
    try:
        events: list[str] = []
        orphan = _FakeOrphan(events)
        # The orphan scan reads the canonical observer census (71b5598f)
        # and acts through psutil.Process(pid).
        import os as _os

        from core.runtime.resource_observation import ProcessObservation

        resource_observer.configure_processes([
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=orphan.pid,
                ppid=_os.getpid(),
                create_time=1_700_000_000.0,
                status="running",
                name="MLXWorker-Qwen2.5-32B-Instruct-4bit",
                cmdline=("python", "mlx_worker", "Qwen2.5-32B-Instruct-4bit"),
                rss_bytes=1024,
                ancestor_pids=(_os.getpid(),),
            )
        ])
        monkeypatch.setattr(mc.psutil, "Process", lambda pid: orphan)

        def _fake_memcheck(model_path):
            events.append("memcheck")
            # Still "blocked" in this contrived test so the method stops here;
            # in production the kill above frees the headroom so this passes.
            return "model_load_headroom:20.2GB < required 22.0GB"

        monkeypatch.setattr(mc, "_memory_pressure_blocks_worker_spawn", _fake_memcheck)

        with pytest.raises(RuntimeError, match="memory_pressure_refused_worker_spawn"):
            client._spawn_worker_blocking()

        # The contract: reclamation runs FIRST, then the admission check
        # (which the bounded reclaim-wait may legitimately re-poll). Before
        # the fix the memcheck raised before the orphan was ever killed.
        assert events[0] == "kill"
        assert "memcheck" in events[1:]
    finally:
        client.close()
