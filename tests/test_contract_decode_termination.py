"""Contract tests: contract-aware decode termination (CP180b).

The engine must stop decoding with termination ``contract_complete`` the
moment a single FINAL_ANSWER JSON object completes (config-gated, default
off), the completion must count as a COMPLETE answer in the engine's own
failure gate and the service receipt contract, and the campaign's vanilla
path must stop streaming at the same uniform rule without consuming the
rest of the stream.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23]
CONTRACT_TEXT = 'FINAL_ANSWER: {"node": 6}'


def _tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


class _ContractAtFive:
    """Decoded text becomes a complete contract answer at the fifth token."""

    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 127 + 1 for c in str(text)][:16] or [5]

    def decode(self, ids):
        if len(ids) == 1:
            return "}"  # every piece can close an object: cheap gate fires
        return CONTRACT_TEXT if len(ids) >= 5 else "still reasoning"


def _config(decode_contract: str) -> CortexConfig:
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=3),
        recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
        branches=BranchConfig(n_branches=1),
        decode_max_tokens=48,
        decode_contract=decode_contract,
    )


def test_engine_stops_at_contract_completion_and_counts_complete():
    engine = LatentCortexEngine(
        _tiny_model(), _ContractAtFive(), config=_config("final_answer_v1")
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.decode_termination == "contract_complete"
    assert result.receipt.decode_generated_tokens == 5
    assert result.ok, result.reason  # completion, not decode_incomplete


def test_engine_contract_off_preserves_historical_behavior():
    engine = LatentCortexEngine(
        _tiny_model(), _ContractAtFive(), config=_config("none")
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.decode_termination != "contract_complete"


def test_service_receipt_contract_accepts_contract_complete():
    from core.brain.latent_cortex_service import LatentCortexService

    receipt = {
        "episode_id": "ep",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 4,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 2,
        "n_slots": 4,
        "n_branches": 1,
        "budget": {
            "max_layer_apps": 1000,
            "spent_layer_apps": 10,
            "exhausted": False,
        },
        "decode_requested_tokens": 48,
        "decode_generated_tokens": 5,
        "decode_termination": "contract_complete",
        "honest_flags": [],
    }
    config = {"n_slots": 4, "n_branches": 1}
    errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "decode_incomplete" not in errors


def test_campaign_vanilla_stream_stops_at_contract(monkeypatch):
    import mlx_lm

    from tools import run_latent_cortex_paired_campaign as runner

    consumed = {"count": 0}
    segments = [
        SimpleNamespace(text="I think ", generation_tokens=2),
        SimpleNamespace(text='FINAL_ANSWER: {"node', generation_tokens=7),
        SimpleNamespace(text='": 6}', generation_tokens=9),
        SimpleNamespace(text=" babble that must never stream", generation_tokens=15),
    ]

    def scripted_stream(model, tokenizer, prompt, max_tokens, **kwargs):
        for segment in segments:
            consumed["count"] += 1
            yield segment

    monkeypatch.setattr(mlx_lm, "stream_generate", scripted_stream)

    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * 11)
    model = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 8))
    task = SimpleNamespace(prompt="What node?", domain="mathematics")
    monkeypatch.setattr(
        runner, "_render_prompt", lambda tok, tsk: "rendered prompt"
    )
    text, layer_apps = runner._vanilla_once(
        model, tokenizer, task, max_tokens=256
    )
    assert text == 'I think FINAL_ANSWER: {"node": 6}'
    assert consumed["count"] == 3  # the babble segment never streamed
    assert layer_apps == (11 + 9) * 8  # prompt + ACTUAL generated tokens


def test_campaign_vanilla_without_contract_consumes_whole_stream(monkeypatch):
    import mlx_lm

    from tools import run_latent_cortex_paired_campaign as runner

    segments = [
        SimpleNamespace(text="no marker here ", generation_tokens=3),
        SimpleNamespace(text="just prose", generation_tokens=6),
    ]

    def scripted_stream(model, tokenizer, prompt, max_tokens, **kwargs):
        yield from segments

    monkeypatch.setattr(mlx_lm, "stream_generate", scripted_stream)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * 4)
    model = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 8))
    task = SimpleNamespace(prompt="Say something", domain="calibration")
    monkeypatch.setattr(
        runner, "_render_prompt", lambda tok, tsk: "rendered prompt"
    )
    text, layer_apps = runner._vanilla_once(
        model, tokenizer, task, max_tokens=256
    )
    assert text == "no marker here just prose"
    assert layer_apps == (4 + 6) * 8
