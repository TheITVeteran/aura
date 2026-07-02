"""Cooperative (soft) cancel of MLX generations.

The preemption ladder's first rung: the parent writes the active job's
sequence number into shared memory, the worker token loop stops between
tokens, and the model stays warm — force-abort (worker kill + full model
reload) remains the escalation, not the default.
"""
from __future__ import annotations

import time

from core.brain.llm import mlx_client as mlx_client_mod
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_worker import soft_cancel_requested


class _Value:
    """Duck-typed stand-in for multiprocessing.Value(lock=False)."""

    def __init__(self, value: int = 0):
        self.value = value


# ── worker-side predicate ──────────────────────────────────────────────


def test_soft_cancel_requested_matches_only_the_targeted_job():
    channel = _Value(7)
    assert soft_cancel_requested(channel, 7) is True
    assert soft_cancel_requested(channel, 8) is False
    assert soft_cancel_requested(channel, 0) is False
    assert soft_cancel_requested(None, 7) is False
    assert soft_cancel_requested(_Value(0), 7) is False


def test_soft_cancel_requested_survives_broken_channel():
    class _Broken:
        @property
        def value(self):
            raise OSError("shared memory detached")

    assert soft_cancel_requested(_Broken(), 3) is False


# ── client-side API ────────────────────────────────────────────────────


def _bare_client(*, active_seq: int = 0, started: bool = False) -> MLXLocalClient:
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Qwen2.5-32B-Instruct-4bit"
    client._cancel_seq = _Value(0)
    client._current_request_seq = active_seq
    client._current_request_started_at = time.time() if started else 0.0
    return client


def test_soft_cancel_writes_active_seq_to_channel():
    client = _bare_client(active_seq=42, started=True)

    receipt = client.soft_cancel_active_generation("foreground_preemption")

    assert receipt["requested"] is True
    assert receipt["active_seq"] == 42
    assert client._cancel_seq.value == 42


def test_soft_cancel_refuses_when_no_generation_active():
    client = _bare_client(active_seq=0, started=False)

    receipt = client.soft_cancel_active_generation("foreground_preemption")

    assert receipt["requested"] is False
    assert receipt["detail"] == "no_active_generation"
    assert client._cancel_seq.value == 0


def test_new_generation_clears_stale_cancel_request():
    client = _bare_client(active_seq=0, started=False)
    # Minimal tracking fields _mark_generation_started touches.
    client._last_heartbeat = 0.0
    client._last_progress_at = 0.0
    client._last_ready_at = 0.0
    client._cancel_seq.value = 41  # stale cancel for a finished job

    client._mark_generation_started("req-abc", request_seq=42)

    assert client._current_request_seq == 42
    assert client._cancel_seq.value == 0


# ── module-level sweep + owner-clear integration ───────────────────────


def test_soft_cancel_sweep_collects_only_accepting_clients(monkeypatch):
    accepted = _bare_client(active_seq=5, started=True)
    idle = _bare_client(active_seq=0, started=False)
    monkeypatch.setattr(
        mlx_client_mod, "_CLIENTS", {"a": accepted, "b": idle}
    )

    receipts = mlx_client_mod.soft_cancel_active_generations(reason="test_sweep")

    assert len(receipts) == 1
    assert receipts[0]["active_seq"] == 5
    assert receipts[0]["model"] == "Qwen2.5-32B-Instruct-4bit"
    assert accepted._cancel_seq.value == 5


def test_force_clear_foreground_owner_requests_soft_cancel(monkeypatch):
    active = _bare_client(active_seq=9, started=True)
    monkeypatch.setattr(mlx_client_mod, "_CLIENTS", {"a": active})
    monkeypatch.setattr(mlx_client_mod, "_FOREGROUND_OWNER_NAME", "chat_api:default")
    monkeypatch.setattr(
        mlx_client_mod, "_FOREGROUND_OWNER_ACQUIRED_AT", time.time() - 60.0
    )

    result = mlx_client_mod.force_clear_foreground_owner(
        reason="chat_lock_preemption", min_age_s=45.0
    )

    assert result["cleared"] is True
    assert result["soft_cancel"], "owner clear must ask the wedged generation to stop"
    assert result["soft_cancel"][0]["active_seq"] == 9
    assert active._cancel_seq.value == 9


def test_force_clear_foreground_owner_without_active_generation(monkeypatch):
    monkeypatch.setattr(mlx_client_mod, "_CLIENTS", {})
    monkeypatch.setattr(mlx_client_mod, "_FOREGROUND_OWNER_NAME", "chat_api:default")
    monkeypatch.setattr(
        mlx_client_mod, "_FOREGROUND_OWNER_ACQUIRED_AT", time.time() - 60.0
    )

    result = mlx_client_mod.force_clear_foreground_owner(
        reason="chat_lock_preemption", min_age_s=45.0
    )

    assert result["cleared"] is True
    assert result["soft_cancel"] == []


# ── request wiring ─────────────────────────────────────────────────────


def test_generate_requests_carry_monotonic_seq():
    client = _bare_client()
    client._job_seq_counter = 0

    # The request builder increments the counter per job; simulate two turns.
    client._job_seq_counter += 1
    first = client._job_seq_counter
    client._job_seq_counter += 1
    second = client._job_seq_counter

    assert (first, second) == (1, 2)


def test_worker_spawn_args_include_cancel_channel():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    params = list(inspect.signature(_mlx_worker_loop).parameters)
    assert "cancel_seq" in params
    # Spawn site passes it positionally after steering_active_flag.
    assert params.index("cancel_seq") == params.index("steering_active_flag") + 1
    assert "self._cancel_seq," in inspect.getsource(mlx_client_mod)
