from __future__ import annotations

import hashlib
import json
from types import MethodType, SimpleNamespace

import pytest

from core.brain.llm import mlx_client
from core.brain.llm.unified_recurrent_qualified_activation import (
    seal_qualified_activation_load_receipt,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _bind(instance, function):
    return MethodType(function, instance)


def _activation() -> dict:
    body = {
        "schema": "aura.unified_intrinsic.qualified_activation.v1",
        "package_id": "qualified-fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "controller_sha256": "c" * 64,
        "pointer_sha256": "d" * 64,
        "lifecycle_result_sha256": "e" * 64,
        "canary_plan_sha256": "f" * 64,
        "families": ["khop"],
        "task_depths": [2],
        "recurrence_depth": 4,
        "mode": "qualified_typed_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": True,
    }
    return {**body, "activation_sha256": _sha(body)}


def _client():
    class Queue:
        job = None

        def put(self, job, *_args):
            self.job = job

    queue = Queue()
    activation = _activation()
    client = object.__new__(mlx_client.MLXLocalClient)
    client._closed = False
    client._unified_recurrent_shadow_status = {
        "loaded": True,
        "package_id": "qualified-fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "controller_sha256": "c" * 64,
        "families": ["khop"],
        "task_depths": [2],
        "recurrence_depth": 4,
    }
    client._unified_recurrent_qualified_activation_status = (
        seal_qualified_activation_load_receipt(
            configured=True,
            loaded=True,
            reason="qualified_activation_loaded",
            activation=activation,
        )
    )
    client._req_q = queue
    client._process = SimpleNamespace(is_alive=lambda: True)
    client._init_done = True
    client._warmup_in_flight = False
    client._active_generations = 0
    client._job_seq_counter = 0
    client._pending_generations = {}
    client._current_gen_future = None
    client._active_generation_started_at = 0.0
    client._authorize_job = lambda job, **_kwargs: job
    client._mark_generation_started = lambda *_args, **_kwargs: None
    client._mark_progress = lambda: None
    client._release_request_lock = lambda: None
    client.soft_cancel_active_generation = lambda *_args, **_kwargs: {}

    async def acquire(_self, **_kwargs):
        return True

    async def fence(_self, _preemptible):
        return True

    async def finish(_self, request_id, _future, _watchdog, **_kwargs):
        _self._pending_generations.pop(request_id, None)
        _self._active_generations = 0

    async def reboot(_self, **_kwargs):
        return True

    client._acquire_request_lock = _bind(client, acquire)
    client._set_durable_lane_preemptible = _bind(client, fence)
    client._finish_generation_ownership = _bind(client, finish)
    client.reboot_worker = _bind(client, reboot)
    return client, queue, activation


def _authorized_receipt(request: dict, activation: dict, *, controller: str) -> dict:
    body = {
        "schema": "aura.unified_intrinsic.qualified_decode_result.v1",
        "request_sha256": request["request_sha256"],
        "package_id": request["package_id"],
        "controller_sha256": controller,
        "family": request["family"],
        "task_depth": request["task_depth"],
        "generated_token_ids": [1],
        "parsed_values": {"node": 1},
        "latency_ms": 1,
        "grammar_valid": True,
        "output_exposed": True,
        "serving_authority": True,
        "authority_source": "qualified_activation",
        "qualified_activation_sha256": activation["activation_sha256"],
    }
    return {**body, "result_sha256": _sha(body)}


def test_client_reports_exact_qualified_serving_identity() -> None:
    client, _queue, activation = _client()

    status = client.unified_recurrent_qualified_serving_status()

    assert status == {
        "active": True,
        "reason": "qualified_recurrent_serving_active",
        "package_id": "qualified-fixture",
        "controller_sha256": "c" * 64,
        "activation_sha256": activation["activation_sha256"],
    }


def test_client_status_refuses_mismatched_activation_identity() -> None:
    client, _queue, _activation_value = _client()
    client._unified_recurrent_shadow_status["controller_sha256"] = "9" * 64

    status = client.unified_recurrent_qualified_serving_status()

    assert status["active"] is False
    assert status["reason"] == "qualified_activation_shadow_identity_differs"


@pytest.mark.asyncio
async def test_client_never_dispatches_without_qualified_authority() -> None:
    client = object.__new__(mlx_client.MLXLocalClient)
    client._closed = False
    client._unified_recurrent_shadow_status = {}
    client._unified_recurrent_qualified_activation_status = {}

    result = await client.unified_recurrent_qualified_decode_async(
        [1], family="khop", task_depth=2, max_tokens=4
    )

    assert result["ok"] is False
    assert result["reason"] == "qualified_recurrent_serving_not_active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned_controller", "expected_ok"),
    [("c" * 64, True), ("9" * 64, False)],
)
async def test_parent_accepts_only_request_bound_qualified_receipt(
    monkeypatch: pytest.MonkeyPatch,
    returned_controller: str,
    expected_ok: bool,
) -> None:
    client, queue, activation = _client()

    async def await_receipt(_future, *, timeout_s):
        assert timeout_s > 0
        request = queue.job["unified_recurrent_qualified_decode_contract"]
        return {
            "id": queue.job["id"],
            "action": "unified_recurrent_qualified_decode",
            "status": "ok",
            "receipt": _authorized_receipt(
                request,
                activation,
                controller=returned_controller,
            ),
        }

    monkeypatch.setattr(mlx_client, "_await_shared_future", await_receipt)

    result = await client.unified_recurrent_qualified_decode_async(
        [1], family="khop", task_depth=2, max_tokens=4
    )

    assert result["ok"] is expected_ok
    if expected_ok:
        assert result["status"] == "completed"
    else:
        assert result["status"] == "integrity_failed"
        assert "qualified_decode_result_domain_differs" in result["reason"]
