from __future__ import annotations

from typing import Any

import pytest

from core.brain.llm import mlx_client as mlx_module
from core.brain.llm import model_registry
from core.brain.llm.mlx_client import MLXLocalClient


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_swap_cooldown_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    primary_path = "/models/32B"
    deep_path = "/models/72B"
    sleep_calls: list[float] = []
    spawn_calls: list[str] = []

    path_map = {
        "primary": primary_path,
        "deep": deep_path,
        primary_path: primary_path,
        deep_path: deep_path,
    }
    monkeypatch.setattr(
        mlx_module.os.path,
        "realpath",
        lambda value, *args, **kwargs: path_map.get(value, value),
    )
    monkeypatch.setattr(mlx_module.time, "time", lambda: 1005.0)
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit")
    monkeypatch.setattr(model_registry, "DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit")
    monkeypatch.setattr(
        model_registry,
        "get_model_path",
        lambda model=None: "primary" if "32B" in str(model) or model is None else "deep",
    )

    async def record_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        sleep_calls.append(delay)

    async def spawn_worker(client: MLXLocalClient) -> _AliveProcess:
        spawn_calls.append(client.model_path)
        return _AliveProcess()

    async def complete_listener_handshake(client: MLXLocalClient) -> None:
        mlx_module._set_shared_future_result(client._init_future, {"status": "ok"})

    monkeypatch.setattr(mlx_module.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(MLXLocalClient, "_spawn_worker", spawn_worker)
    monkeypatch.setattr(MLXLocalClient, "_ensure_listener_task", complete_listener_handshake)

    mlx_module._GLOBAL_LAST_SWAP_TIME = 1000.0
    mlx_module._GLOBAL_LAST_HEAVY_MODEL = deep_path
    mlx_module._client_instances = {}
    mlx_module._CLIENTS.clear()

    client_72b = MLXLocalClient(model_path=deep_path)
    client_72b._process = _AliveProcess()
    client_72b._init_done = True

    client_32b = MLXLocalClient(model_path=primary_path)
    client_32b._process = None

    assert await client_32b._ensure_worker_alive()

    assert sleep_calls == [7.0]
    assert spawn_calls == [primary_path]
