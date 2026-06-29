from __future__ import annotations

from types import SimpleNamespace

from core.brain.llm import mlx_client as mlx_module
from core.brain.llm.mlx_client import MLXLocalClient


def test_mlx_client_rebases_active_generation_after_system_sleep(monkeypatch, tmp_path) -> None:
    client = MLXLocalClient(model_path=str(tmp_path / "Aura-32B"))
    client._clock_sample_wall = 1_000.0
    client._clock_sample_monotonic = 500.0
    client._current_request_started_at = 990.0
    client._current_first_token_at = 995.0
    client._last_token_progress_at = 996.0
    client._last_heartbeat = 1_599.0
    monkeypatch.setattr(
        mlx_module,
        "time",
        SimpleNamespace(time=lambda: 1_600.0, monotonic=lambda: 510.0),
    )

    gap = client._rebase_after_system_sleep()

    assert gap == 590.0
    assert client._current_request_started_at == 1_580.0
    assert client._current_first_token_at == 1_585.0
    assert client._last_token_progress_at == 1_586.0
    assert client._last_heartbeat == 1_599.0


def test_mlx_client_does_not_rebase_normal_scheduler_delay(monkeypatch, tmp_path) -> None:
    client = MLXLocalClient(model_path=str(tmp_path / "Aura-32B"))
    client._clock_sample_wall = 1_000.0
    client._clock_sample_monotonic = 500.0
    client._current_request_started_at = 999.0
    monkeypatch.setattr(
        mlx_module,
        "time",
        SimpleNamespace(time=lambda: 1_002.0, monotonic=lambda: 502.0),
    )

    assert client._rebase_after_system_sleep() == 0.0
    assert client._current_request_started_at == 999.0
