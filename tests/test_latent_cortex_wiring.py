"""Contract tests: latent-cortex runtime wiring (service, handler, economy).

No worker processes are spawned here — the worker/client IPC bodies are
exercised through the handler function and a mocked client, which is exactly
the seam the live path uses.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.worker_handler import (
    budget_from_job,
    config_from_job,
    cortex_enabled,
    handle_latent_reason,
)


# ── Worker handler ──────────────────────────────────────────────────────


def test_config_from_job_defaults_are_conservative():
    cfg = config_from_job(None)
    assert cfg.workspace.n_slots == 16
    assert cfg.recurrence.max_steps == 8
    assert cfg.branches.n_branches == 2
    assert cfg.latent_opt.enabled is False
    assert cfg.fast_weights.enabled is False
    assert cfg.validate() == []


def test_config_from_job_rejects_out_of_band_requests():
    with pytest.raises(ValueError):
        config_from_job({"n_branches": 640})
    with pytest.raises(ValueError):
        config_from_job({"max_steps": 100000})


def test_budget_from_job_caps_apply():
    budget = budget_from_job({"max_layer_apps": 10**15, "wall_clock_s": 5.0})
    assert budget.wall_clock_s == 5.0
    assert budget.remaining_layer_apps <= 500_000_000  # ABSOLUTE_MAX cap governs


def test_kill_switch_refuses_honestly(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    assert cortex_enabled() is False
    body = handle_latent_reason(
        {"prompt": "hi"}, model=None, tokenizer=None, model_path=""
    )
    assert body["status"] == "error"
    assert "latent_cortex_disabled" in body["message"]


def test_handler_requires_prompt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    body = handle_latent_reason({}, model=None, tokenizer=None, model_path="")
    assert body["status"] == "error"
    assert "requires prompt" in body["message"]


def test_handler_runs_full_episode_on_tiny_model(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())

    class StubTokenizer:
        eos_token_id = 0

        def encode(self, text):
            return [ord(c) % 128 for c in text][:16]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    body = handle_latent_reason(
        {
            "prompt": "compose the deepest thought",
            "config": {"n_slots": 4, "n_branches": 2, "max_steps": 4, "decode_max_tokens": 6},
            "budget": {"wall_clock_s": 30.0},
            "domain": "unit",
        },
        model=model,
        tokenizer=StubTokenizer(),
        model_path="",
    )
    assert body["status"] == "ok", body
    assert body["receipt"]["params_unchanged"] is True
    assert body["receipt"]["steps_taken"] >= 2
    assert body["requires_cache_clear"] is False


# ── Service economy ─────────────────────────────────────────────────────


def test_allocation_scales_with_stakes_and_uncertainty():
    svc = LatentCortexService()
    low_cfg, low_budget = svc.allocate(stakes=0.1, uncertainty=0.1)
    high_cfg, high_budget = svc.allocate(stakes=0.9, uncertainty=0.9)
    assert high_cfg["max_steps"] > low_cfg["max_steps"]
    assert high_cfg["n_branches"] > low_cfg["n_branches"]
    assert high_budget["max_layer_apps"] > low_budget["max_layer_apps"]
    assert high_budget["wall_clock_s"] > low_budget["wall_clock_s"]


def test_allocation_damped_by_body_pressure(monkeypatch):
    svc = LatentCortexService()
    monkeypatch.setattr(svc, "_body_pressure", lambda: 0.0)
    calm_cfg, calm_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    monkeypatch.setattr(svc, "_body_pressure", lambda: 1.0)
    strained_cfg, strained_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    assert strained_cfg["max_steps"] < calm_cfg["max_steps"]
    assert strained_budget["max_layer_apps"] < calm_budget["max_layer_apps"]
    assert strained_cfg["n_branches"] <= calm_cfg["n_branches"]


def test_service_kill_switch_and_status(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    svc = LatentCortexService()
    result = asyncio.run(svc.deep_reason("why?"))
    assert result["ok"] is False and "disabled" in result["reason"]
    status = svc.get_status()
    assert status["enabled"] is False
    assert status["healthy"] is True


def test_service_routes_through_client_and_records_receipt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    captured = {}

    class StubClient:
        async def latent_reason_async(self, prompt=None, **kwargs):
            captured["prompt"] = prompt
            captured["config"] = kwargs.get("config")
            captured["budget"] = kwargs.get("budget")
            return {
                "ok": True,
                "text": "the deep answer",
                "receipt": {"steps_taken": 7, "halting_reason": "converged",
                            "n_branches": 2, "episode_id": "abc"},
                "reason": "",
            }

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: StubClient())
    result = asyncio.run(svc.deep_reason("hard question", stakes=0.9, uncertainty=0.9))
    assert result["ok"] and result["text"] == "the deep answer"
    assert captured["prompt"] == "hard question"
    assert captured["config"]["n_branches"] >= 2
    assert captured["budget"]["max_layer_apps"] > 0
    assert svc.get_status()["last_receipt"]["halting_reason"] == "converged"


def test_service_reports_refusals_honestly(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class BusyClient:
        async def latent_reason_async(self, **kwargs):
            return {"ok": False, "reason": "generation_active"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: BusyClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False and result["reason"] == "generation_active"
    assert svc.get_status()["last_refusal"] == "generation_active"


def test_service_name_registered_in_spine():
    from core.service_names import ServiceNames

    assert ServiceNames.LATENT_CORTEX == "latent_cortex"
