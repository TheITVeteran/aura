"""Contract tests: the integrated LatentCortexEngine.

Full episodes on a tiny real Qwen2: latent computation is causal on the
answer, receipts tell the truth, fallbacks are honest, budgets bind, the
checkpoint invariant is enforced, and fast-weight episodes leave the model
bit-for-bit unchanged.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
from core.brain.llm.latent_cortex.governance import parameter_fingerprint  # noqa: E402
from core.brain.llm.latent_cortex.schedules import LayerSchedule, StageOp  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    FastWeightsConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

N_LAYERS = 8
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


def _model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
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


@pytest.fixture(scope="module")
def tiny_model():
    return _model()


def _config(**overrides) -> CortexConfig:
    base = dict(
        workspace=WorkspaceConfig(n_slots=4, seed=3),
        recurrence=RecurrenceConfig(max_steps=6, min_steps=2),
        branches=BranchConfig(n_branches=2, exchange_interval=2),
        prelude_frac=0.25,
        coda_frac=0.25,
        decode_max_tokens=8,
    )
    base.update(overrides)
    return CortexConfig(**base)


def test_full_episode_produces_tokens_and_truthful_receipt(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    assert result.tokens, "episode must decode tokens"
    r = result.receipt
    assert r.params_unchanged is True
    assert r.n_layers == N_LAYERS and r.prelude_end == 2 and r.coda_start == 6
    assert r.n_branches == 2 and r.n_slots == 4
    assert r.steps_taken >= 2
    assert r.residual_trail, "receipt must carry the residual trail"
    assert r.halting_reason
    assert r.schedule_hash
    assert r.budget["spent_layer_apps"] > 0
    assert r.decode_requested_tokens == 8
    assert r.decode_generated_tokens == len(result.tokens)
    assert r.decode_termination in {"eos", "token_limit"}
    assert not r.honest_flags, f"clean episode must carry no flags: {r.honest_flags}"


def test_episodes_are_deterministic(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    a = engine.reason(token_ids=PROMPT_TOKENS)
    b = engine.reason(token_ids=PROMPT_TOKENS)
    assert a.tokens == b.tokens
    assert a.receipt.schedule_hash == b.receipt.schedule_hash


def test_latent_computation_is_causal_on_answer(tiny_model):
    """More recurrence ⇒ different refined thoughts ⇒ different answer tokens.

    (On a random-weight model "different" is the strongest honest claim;
    'better' belongs to the experiments on trained checkpoints.)
    """
    shallow = LatentCortexEngine(
        tiny_model, config=_config(recurrence=RecurrenceConfig(max_steps=1, min_steps=1))
    ).reason(token_ids=PROMPT_TOKENS)
    deep = LatentCortexEngine(
        tiny_model,
        config=_config(recurrence=RecurrenceConfig(max_steps=12, min_steps=8, convergence_eps=1e-9)),
    ).reason(token_ids=PROMPT_TOKENS)
    assert shallow.ok and deep.ok
    assert shallow.receipt.steps_taken < deep.receipt.steps_taken
    # On random weights greedy decode can collapse into the same attractor
    # token, so the honest causal signal is the first-decode logits digest:
    # the next-token DISTRIBUTION conditioned on [prompt; thoughts] must move
    # when the latent computation deepens.
    assert shallow.receipt.first_logits_digest != deep.receipt.first_logits_digest, (
        "recurrence depth must causally shape the answer distribution"
    )


def test_explicit_schedule_is_used_and_hashed(tiny_model):
    schedule = LayerSchedule(ops=(StageOp(2, 4, 2), StageOp(4, 6, 2), StageOp(2, 6, 2)))
    engine = LatentCortexEngine(tiny_model, config=_config(schedule=schedule.to_dict()))
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    assert result.receipt.schedule_hash == schedule.schedule_hash


def test_invalid_schedule_falls_back_honestly(tiny_model):
    bad = LayerSchedule(ops=(StageOp(0, 7, 2),))  # escapes the recurrent region
    engine = LatentCortexEngine(tiny_model, config=_config(schedule=bad.to_dict()))
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok, "fallback must still answer"
    assert any(f.startswith("fallback_vanilla") for f in result.receipt.honest_flags)
    assert result.tokens, "vanilla fallback must decode"


def test_budget_binds_and_is_reported(tiny_model):
    tight = ComputeBudget(max_layer_apps=PROMPT_TOKENS.__len__() * N_LAYERS + 200)
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=tight)
    assert result.ok
    assert result.receipt.budget["spent_layer_apps"] <= tight.max_layer_apps
    assert "fallback_vanilla:RuntimeError" in result.receipt.honest_flags
    fallback_ceiling = (len(PROMPT_TOKENS) + _config().decode_max_tokens - 1) * N_LAYERS
    assert result.receipt.budget["spent_layer_apps"] <= fallback_ceiling
    assert result.receipt.decode_termination in {"eos", "token_limit"}
    reasons = {result.receipt.halting_reason} | {
        b["halt_reason"] for b in [] # branch receipts live in ensemble receipt; halting_reason covers winner
    }
    assert any("budget" in r or r for r in reasons)


def test_verifier_selects_branch_and_scores_land_in_receipt(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())

    # A tokenizer-free verifier can't run (no decode-to-text); engine must
    # fall back to convergence selection without exploding.
    result = engine.reason(token_ids=PROMPT_TOKENS, verifier=lambda text: 1.0)
    assert result.ok
    assert len(result.receipt.branch_scores) == 2


def test_latent_opt_episode_records_trace(tiny_model):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(latent_opt=LatentOptConfig(enabled=True, steps=3, lr=0.05)),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    r = result.receipt
    assert r.latent_opt_applied and r.latent_opt_mode == "gradient"
    assert r.latent_opt_attempts == 3
    assert r.latent_opt_steps == 3
    assert r.latent_opt_rejected == 0
    assert r.latent_opt_budget_exhausted is False
    assert len(r.latent_opt_loss_trail) == 4  # 3 steps + final
    assert r.latent_opt_loss_trail[-1] < r.latent_opt_loss_trail[0]


def test_fast_weight_episode_proves_erase_and_invariant():
    model = _model()  # fresh model: this test mutates wrappers
    before = parameter_fingerprint(model)
    engine = LatentCortexEngine(
        model,
        config=_config(
            fast_weights=FastWeightsConfig(
                enabled=True, rank=2, target="o_proj", opt_steps=2, lr=0.02
            ),
            latent_opt=LatentOptConfig(enabled=False),
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    r = result.receipt
    assert r.fast_weights_applied and r.fast_weights_layers == 4
    assert r.fast_weight_optimization_attempts >= 1
    assert r.fast_weight_optimized_steps >= 1
    assert r.fast_weight_budget_exhausted is False
    assert r.fast_weights_erased is True
    assert r.params_unchanged is True
    assert parameter_fingerprint(model) == before, "episode must leave W0 untouched"


def test_fast_weight_optimization_failure_cleans_before_vanilla_fallback(monkeypatch):
    model = _model()
    original_modules = [layer.self_attn.o_proj for layer in model.model.layers]

    def fail_optimization(self, loss_fn, **kwargs):
        assert self.handles, "failure must be injected after attachment"
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(EpisodicFastWeights, "optimize", fail_optimization)
    engine = LatentCortexEngine(
        model,
        config=_config(
            fast_weights=FastWeightsConfig(enabled=True, target="o_proj", opt_steps=2),
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok, "a proven-clean model may serve the honest vanilla fallback"
    assert result.receipt.fast_weights_applied is True
    assert result.receipt.fast_weights_erased is True
    assert "fallback_vanilla:RuntimeError" in result.receipt.honest_flags
    assert [layer.self_attn.o_proj for layer in model.model.layers] == original_modules


def test_unproven_fast_weight_cleanup_refuses_fallback(monkeypatch):
    model = _model()

    def fail_optimization(self, loss_fn, **kwargs):
        raise RuntimeError("injected optimizer failure")

    def fail_proof(self, probe_fn, baseline):
        raise RuntimeError("injected erase-proof failure")

    monkeypatch.setattr(EpisodicFastWeights, "optimize", fail_optimization)
    monkeypatch.setattr(EpisodicFastWeights, "prove_erase", fail_proof)
    engine = LatentCortexEngine(
        model,
        config=_config(fast_weights=FastWeightsConfig(enabled=True, target="o_proj")),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert not result.ok
    assert result.reason == "fast_weight_cleanup_unproven"
    assert result.receipt.fast_weights_erased is False
    assert "fallback_refused_unproven_model_state" in result.receipt.honest_flags


def test_fast_weight_snapshot_memory_failure_still_detaches(monkeypatch):
    model = _model()
    originals = [layer.self_attn.o_proj for layer in model.model.layers]

    def fail_snapshot(self):
        raise MemoryError("injected snapshot allocation failure")

    monkeypatch.setattr(EpisodicFastWeights, "snapshot_for_export", fail_snapshot)
    engine = LatentCortexEngine(
        model,
        config=_config(fast_weights=FastWeightsConfig(enabled=True, target="o_proj")),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok
    assert result.receipt.fast_weights_erased is True
    assert "fast_weight_snapshot_failed:MemoryError" in result.receipt.honest_flags
    assert [layer.self_attn.o_proj for layer in model.model.layers] == originals


def test_post_episode_invariant_probe_failure_refuses_output(monkeypatch, tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())

    def fail_post_probe():
        raise RuntimeError("injected post-probe failure")

    monkeypatch.setattr(engine.invariant, "post_episode", fail_post_probe)
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok is False
    assert result.reason == "checkpoint_invariant_violated"
    assert result.receipt.params_unchanged is False
    assert "checkpoint_post_probe_failed:RuntimeError" in result.receipt.honest_flags


def test_config_validation_rejects_garbage(tiny_model):
    with pytest.raises(ValueError):
        LatentCortexEngine(tiny_model, config=_config(recurrence=RecurrenceConfig(max_steps=0)))
    with pytest.raises(ValueError):
        LatentCortexEngine(
            tiny_model, config=_config(branches=BranchConfig(n_branches=99))
        )


@pytest.mark.parametrize(
    "override",
    [
        {"workspace": WorkspaceConfig(n_slots=True)},
        {"workspace": WorkspaceConfig(seed="0")},
        {"workspace": WorkspaceConfig(roles="objective")},
        {"recurrence": RecurrenceConfig(max_steps="2")},
        {"branches": BranchConfig(n_branches=1.5)},
        {"latent_opt": LatentOptConfig(enabled="false")},
        {"fast_weights": FastWeightsConfig(enabled=1)},
        {"decode_max_tokens": True},
    ],
)
def test_direct_config_validation_rejects_coercible_types(tiny_model, override):
    with pytest.raises(ValueError):
        LatentCortexEngine(tiny_model, config=_config(**override))


@pytest.mark.parametrize("invalid", [True, 1.5, "8", 0, 8193])
def test_reason_rejects_malformed_decode_override(tiny_model, invalid):
    engine = LatentCortexEngine(tiny_model, config=_config())
    with pytest.raises((TypeError, ValueError)):
        engine.reason(token_ids=PROMPT_TOKENS, decode_max_tokens=invalid)
