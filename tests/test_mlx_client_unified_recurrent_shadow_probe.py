from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from core.brain.llm import mlx_client
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    RECEIPT_SCHEMA,
    seal_shadow_probe_receipt,
)


def _bind(instance, function):
    return MethodType(function, instance)


@pytest.mark.asyncio
async def test_probe_never_spawns_a_worker_when_shadow_is_not_loaded() -> None:
    client = object.__new__(mlx_client.MLXLocalClient)
    client._closed = False
    client._unified_recurrent_shadow_status = {}

    result = await client.unified_recurrent_shadow_probe_async(
        [1],
        [2],
        max_tokens=1,
    )

    assert result["ok"] is False
    assert result["reason"] == "unified_recurrent_shadow_not_loaded"


@pytest.mark.asyncio
async def test_parent_accepts_only_a_bound_no_output_probe_receipt(monkeypatch) -> None:
    class Queue:
        job = None

        def put(self, job, *_args):
            self.job = job

    queue = Queue()
    client = object.__new__(mlx_client.MLXLocalClient)
    client._closed = False
    client._unified_recurrent_shadow_status = {
        "loaded": True,
        "serving_authority": False,
        "package_id": "fixture",
        "controller_sha256": "c" * 64,
    }
    client._unified_recurrent_shadow_probe_status = {}
    client._req_q = queue
    client._process = SimpleNamespace(is_alive=lambda: True)
    client._init_done = True
    client._warmup_in_flight = False
    client._active_generations = 0
    client._job_seq_counter = 0
    client._pending_generations = {}
    client._current_gen_future = None
    client._active_generation_started_at = 0.0

    async def acquire(_self, **_kwargs):
        return True

    async def fence(_self, _preemptible):
        return True

    async def finish(_self, request_id, _future, _watchdog, **_kwargs):
        _self._pending_generations.pop(request_id, None)
        _self._active_generations = 0

    client._acquire_request_lock = _bind(client, acquire)
    client._set_durable_lane_preemptible = _bind(client, fence)
    client._finish_generation_ownership = _bind(client, finish)
    client._release_request_lock = lambda: None
    client._authorize_job = lambda job, **_kwargs: job
    client._mark_generation_started = lambda *_args, **_kwargs: None
    client._mark_progress = lambda: None
    client.soft_cancel_active_generation = lambda *_args, **_kwargs: {}

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)

    async def await_receipt(_future, *, timeout_s):
        assert timeout_s > 0
        request = queue.job["unified_recurrent_shadow_contract"]
        receipt = seal_shadow_probe_receipt(
            {
                "schema": RECEIPT_SCHEMA,
                "request_sha256": request["request_sha256"],
                "status": "completed",
                "reason": "matched_shadow_probe_completed",
                "package_id": "fixture",
                "controller_sha256": "c" * 64,
                "family": "khop",
                "recurrence_depth": 4,
                "input_token_count": 2,
                "expected_token_count": 1,
                "max_tokens": 1,
                "base_token_count": 1,
                "base_output_sha256": "a" * 64,
                "base_exact_match": False,
                "base_stopped_on_eos": False,
                "base_latency_ms": 3,
                "shadow_token_count": 1,
                "shadow_output_sha256": "b" * 64,
                "shadow_exact_match": True,
                "shadow_stopped_on_eos": True,
                "shadow_latency_ms": 5,
                "outputs_equal": False,
                "output_exposed": False,
                "serving_authority": False,
            }
        )
        return {
            "id": queue.job["id"],
            "action": "unified_recurrent_shadow_probe",
            "status": "ok",
            "receipt": receipt,
        }

    monkeypatch.setattr(mlx_client, "_await_shared_future", await_receipt)

    result = await client.unified_recurrent_shadow_probe_async(
        [0, 201],
        [999],
        max_tokens=1,
    )

    assert result["ok"] is True
    assert result["receipt"]["shadow_exact_match"] is True
    assert "text" not in result
    assert "tokens" not in result
    assert client._unified_recurrent_shadow_probe_status == result["receipt"]
