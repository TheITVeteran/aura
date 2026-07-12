from __future__ import annotations

import asyncio
import base64
import os
import threading
from pathlib import Path

import pytest

from core.brain.llm.mlx_vision_client import DEFAULT_VISION_MODEL, MLXVisionClient
from core.runtime.model_lane_control import ModelLaneController
from core.runtime.receipts import ReceiptStore

pytestmark = pytest.mark.unit


class _CurrentProcessProbe:
    pid = os.getpid()
    name = "FakeVisionWorker"

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.alive = False


def _controller(tmp_path: Path) -> ModelLaneController:
    return ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_discovery=None,
    )


@pytest.fixture(autouse=True)
def _fixed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")


@pytest.mark.asyncio
async def test_vision_worker_commits_only_after_ready_and_releases_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    process = _CurrentProcessProbe()
    client = MLXVisionClient("/models/vision-1.5b", lane_controller=controller)

    def _spawn() -> bool:
        client._process = process
        client._init_done = True
        return True

    monkeypatch.setattr(client, "_spawn_worker_blocking", _spawn)

    assert await client.start_async() is True
    snapshot = controller.snapshot()
    assert snapshot["reserved_gb"] == 0.0
    assert len(snapshot["owners"]) == 1
    assert snapshot["owners"][0]["process"]["pid"] == os.getpid()
    assert snapshot["owners"][0]["metadata"]["modality"] == "vision"
    assert client._lane_decision.receipt_id

    await client.stop_async()

    assert process.terminated is True
    assert controller.snapshot()["owners"] == []


@pytest.mark.asyncio
async def test_vision_failed_spawn_cancels_reservation_with_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    client = MLXVisionClient("/models/vision-1.5b", lane_controller=controller)
    monkeypatch.setattr(client, "_spawn_worker_blocking", lambda: False)

    assert await client.start_async() is False

    snapshot = controller.snapshot()
    assert snapshot["owners"] == []
    assert snapshot["reserved_gb"] == 0.0
    assert snapshot["reservations"][0]["state"] == "cancelled"
    assert snapshot["reservations"][0]["terminal_receipt_id"]


@pytest.mark.asyncio
async def test_vision_start_cancellation_reaps_late_worker_and_cancels_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    process = _CurrentProcessProbe()
    entered = threading.Event()
    release = threading.Event()
    client = MLXVisionClient("/models/vision-1.5b", lane_controller=controller)

    def _spawn() -> bool:
        entered.set()
        release.wait(timeout=2.0)
        client._process = process
        client._init_done = True
        return True

    monkeypatch.setattr(client, "_spawn_worker_blocking", _spawn)
    task = asyncio.create_task(client.start_async())
    assert await asyncio.to_thread(entered.wait, 1.0) is True

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert controller.snapshot()["owners"] == []
    assert controller.snapshot()["reserved_gb"] == 0.0
    assert controller.snapshot()["reservations"][0]["state"] == "cancelled"


@pytest.mark.asyncio
async def test_vision_start_wait_cancellation_does_not_strand_start_mutex() -> None:
    client = MLXVisionClient("/models/vision-1.5b")
    assert client._start_guard.acquire(blocking=False) is True
    task = asyncio.create_task(client.start_async())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client._start_guard.release()

    assert client._start_guard.acquire(timeout=0.1) is True
    client._start_guard.release()


@pytest.mark.asyncio
async def test_describe_reads_bounded_image_off_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"real-image-bytes")
    client = MLXVisionClient()
    captured: dict[str, str] = {}

    async def _see(prompt: str, image_base64: str, **_kwargs: object) -> str:
        captured["prompt"] = prompt
        captured["payload"] = image_base64
        return "visible content"

    monkeypatch.setattr(client, "see_async", _see)

    result = await client.describe(str(image_path), prompt="inspect")

    assert result == "visible content"
    assert captured["prompt"] == "inspect"
    assert base64.b64decode(captured["payload"]) == b"real-image-bytes"
    assert client.model_path == DEFAULT_VISION_MODEL
