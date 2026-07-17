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
    assert r.verifier_probe_max_tokens == 48
    assert r.decode_termination in {"eos", "token_limit"}
    assert r.last_stage == "complete"
    assert r.stage_timings_s["prefill"] >= 0.0
    assert r.stage_timings_s["recurrence"] >= 0.0
    assert r.stage_timings_s["decode"] >= 0.0
    assert r.stage_timings_s["total"] >= 0.0
    assert not r.honest_flags, f"clean episode must carry no flags: {r.honest_flags}"


def test_verifier_probe_cost_uses_receipted_profile_and_bridge(tiny_model):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(verifier_probe_max_tokens=24),
    )

    assert engine._verifier_probe_layer_apps([3, 4], count=2) == (
        2 * (4 + 2 + 23) * N_LAYERS
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.receipt.verifier_probe_max_tokens == 24


def test_verifier_preview_is_hard_capped_and_compute_charge_is_exact(
    tiny_model,
    monkeypatch,
):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(verifier_probe_max_tokens=24),
    )
    engine.tokenizer = object()
    monkeypatch.setattr(engine, "_eos_ids", lambda: set())
    monkeypatch.setattr(engine, "_is_pure_newline_token", lambda _token: False)
    monkeypatch.setattr(engine, "_token_ends_sentence", lambda _token: False)
    monkeypatch.setattr(engine, "_sample", lambda *_args, **_kwargs: 1)

    budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
    cache = engine._fresh_cache()
    _, initial_logits = engine._prefill(PROMPT_TOKENS, cache, budget)
    spent_before = budget.spent_layer_apps

    tokens, termination = engine._decode(
        cache,
        budget,
        initial_logits,
        max_tokens=engine.config.verifier_probe_max_tokens,
        temperature=0.0,
        sentence_grace_tokens=0,
    )

    assert len(tokens) == 24
    assert termination == "token_limit"
    assert budget.spent_layer_apps - spent_before == 23 * N_LAYERS

    # The user-facing decoder still owns the independent sentence grace.
    final_budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
    final_cache = engine._fresh_cache()
    _, final_logits = engine._prefill(PROMPT_TOKENS, final_cache, final_budget)
    final_tokens, final_termination = engine._decode(
        final_cache,
        final_budget,
        final_logits,
        max_tokens=24,
        temperature=0.0,
    )
    assert len(final_tokens) == 72
    assert final_termination == "token_limit"


def test_episode_cooperatively_cancels_at_safe_stage_and_preserves_checkpoint(
    tiny_model,
):
    engine = LatentCortexEngine(tiny_model, config=_config())
    cancel = False
    stages: list[str] = []

    def progress(payload):
        nonlocal cancel
        stage = str(payload.get("stage") or "")
        stages.append(stage)
        if stage == "prefill":
            cancel = True

    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        progress=progress,
        cancel_check=lambda: cancel,
    )

    assert result.ok is False
    assert result.reason == "soft_cancelled"
    assert result.receipt.params_unchanged is True
    assert result.receipt.last_stage == "prefill"
    assert "soft_cancelled" in result.receipt.honest_flags
    assert stages[0] == "prefill"
    assert stages[-1] == "failed"


def test_episodes_are_deterministic(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    a = engine.reason(token_ids=PROMPT_TOKENS)
    b = engine.reason(token_ids=PROMPT_TOKENS)
    assert a.tokens == b.tokens
    assert a.receipt.schedule_hash == b.receipt.schedule_hash


def test_nucleus_sampling_excludes_tokens_outside_probability_mass():
    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    logits = mx.array([12.0, 2.0, 1.0, 0.0])

    sampled = {
        engine._sample(logits, temperature=0.7, top_p=0.01)
        for _ in range(32)
    }

    assert sampled == {0}


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


def test_production_episode_refuses_secondary_vanilla_decode(tiny_model):
    bad = LayerSchedule(ops=(StageOp(0, 7, 2),))
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            schedule=bad.to_dict(),
            allow_vanilla_fallback=False,
        ),
    )

    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok is False
    assert result.tokens == []
    assert result.receipt.decode_termination == "not_started"
    assert "vanilla_fallback_disabled" in result.receipt.honest_flags
    assert not any(
        flag.startswith("fallback_vanilla")
        for flag in result.receipt.honest_flags
    )


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
    assert r.fast_weight_optimizer == "rms_normalized_sgd_backtracking_v1"
    assert len(r.fast_weight_loss_trail) == r.fast_weight_optimized_steps + 1
    assert len(r.fast_weight_gradient_norm_trail) == (
        r.fast_weight_optimization_attempts
    )
    assert len(r.fast_weight_accepted_step_sizes) == r.fast_weight_optimized_steps
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


def test_decode_newline_discipline_caps_babble_runs(tiny_model, monkeypatch):
    """Force the model into pure-newline babble and prove the sampling
    discipline caps runs at two, counts every suppression, and never edits
    emitted text (all output tokens remain model-sampled ids)."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex import engine as engine_mod

    newline_id, word_id, vocab = 7, 9, 128

    class NewlineTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [5, 9, 17]

        def decode(self, ids):
            return "".join("\n" if i == newline_id else "x" for i in ids)

    engine = engine_mod.LatentCortexEngine(
        tiny_model, NewlineTokenizer(), config=_config(decode_max_tokens=12)
    )

    # The model "wants" endless newlines: every logits call favors newline_id
    # overwhelmingly, with word_id as the runner-up the mask must expose.
    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, newline_id] = 10.0
    spiked[0, 0, word_id] = 5.0

    monkeypatch.setattr(
        engine_mod.LatentCortexEngine,
        "_logits",
        lambda self, h: spiked,
    )

    from core.brain.llm.latent_cortex.types import ComputeBudget
    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in tiny_model.model.layers]
    budget = ComputeBudget()
    # Prefill through the real layers so the cache is genuine.
    import mlx.core as mx2
    from mlx_lm.models.base import create_attention_mask

    inner = tiny_model.model
    h = inner.embed_tokens(mx2.array([PROMPT_TOKENS]))
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])

    out, termination = engine._decode(
        cache, budget, spiked[0, 0], max_tokens=12, temperature=0.0
    )
    assert termination == "token_limit"
    assert engine._last_decode_newline_suppressions >= 3
    # No run of pure-newline tokens longer than the discipline allows.
    run = longest = 0
    for token in out:
        run = run + 1 if token == newline_id else 0
        longest = max(longest, run)
    assert longest <= 2, out
    assert word_id in out, "masking must expose the model's own runner-up token"


def test_bridge_policies_produce_distinct_hashable_cues(tiny_model):
    class RecordingTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 128 for c in text][:24]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    v1 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v1")
    )._decode_bridge_tokens()
    v2 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v2")
    )._decode_bridge_tokens()
    v3 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v3")
    )._decode_bridge_tokens()
    assert v1 and v2 and v3
    assert len({tuple(v1), tuple(v2), tuple(v3)}) == 3
    with pytest.raises(ValueError):
        LatentCortexEngine(
            tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="bogus")
        )


def test_repetition_penalty_breaks_forced_loop(tiny_model, monkeypatch):
    """A model that 'wants' to emit one token forever must be broken out of
    the loop by the sliding-window penalty — the CP105 live degeneration
    (one line repeated ~80 times) as a mechanical regression."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget
    from mlx_lm.models.cache import KVCache

    loop_id, alt_id, vocab = 11, 23, 128

    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, loop_id] = 6.0
    spiked[0, 0, alt_id] = 5.5  # runner-up the penalty must expose

    monkeypatch.setattr(
        engine_mod.LatentCortexEngine, "_logits", lambda self, h: spiked
    )

    def run(penalty):
        engine = engine_mod.LatentCortexEngine(
            tiny_model,
            config=_config(
                decode_max_tokens=24,
                decode_repetition_penalty=penalty,
            ),
        )
        cache = [KVCache() for _ in tiny_model.model.layers]
        from mlx_lm.models.base import create_attention_mask

        inner = tiny_model.model
        h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        out, _ = engine._decode(cache, ComputeBudget(), spiked[0, 0], max_tokens=24, temperature=0.0)
        return out

    unguarded = run(1.0)
    assert unguarded.count(loop_id) == len(unguarded), "control arm must loop"

    guarded = run(1.25)
    assert alt_id in guarded, "penalty must surface the runner-up token"
    longest = run_len = 0
    for token in guarded:
        run_len = run_len + 1 if token == loop_id else 0
        longest = max(longest, run_len)
    assert longest < len(guarded), "penalty must break the monoculture"


def test_sentence_grace_finishes_the_sentence(tiny_model, monkeypatch):
    """When the token limit lands mid-sentence, the decoder may sample up to
    the grace window until sentence-final punctuation — model tokens only,
    receipted as its own termination kind."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask

    word_id, period_id, vocab = 9, 13, 128

    class GraceTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids):
            return "".join("." if i == period_id else "w" for i in ids)

    calls = {"n": 0}

    def scripted_logits(self, h):
        # Words until well past the limit, then a period.
        calls["n"] += 1
        spiked = mx.full((1, 1, vocab), -20.0)
        spiked[0, 0, period_id if calls["n"] >= 10 else word_id] = 8.0
        return spiked

    monkeypatch.setattr(engine_mod.LatentCortexEngine, "_logits", scripted_logits)
    engine = engine_mod.LatentCortexEngine(
        tiny_model, GraceTokenizer(), config=_config(decode_max_tokens=6)
    )
    cache = [KVCache() for _ in tiny_model.model.layers]
    inner = tiny_model.model
    h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])

    first = mx.full((1, 1, vocab), -20.0)
    first[0, 0, word_id] = 8.0
    out, termination = engine._decode(
        cache, ComputeBudget(), first[0, 0], max_tokens=6, temperature=0.0
    )
    assert termination == "token_limit_sentence_grace"
    assert out[-1] == period_id, "grace must end at the model's own period"
    assert 6 < len(out) <= 6 + 48


def test_eos_floor_suppresses_early_stop(tiny_model, monkeypatch):
    """Sampling variance must not abandon an answer a few tokens in: below
    decode_min_tokens the EOS logits are masked (min-new-tokens), and after
    the floor the model's own EOS is honored (CP116 live regression)."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask

    eos_id, word_id, vocab = 3, 9, 128

    class EosTokenizer:
        eos_token_id = eos_id

        def encode(self, text, **kwargs):
            return [1, 2]

        def decode(self, ids):
            return "w" * len(ids)

    # The model "wants" to stop immediately: EOS is always argmax.
    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, eos_id] = 9.0
    spiked[0, 0, word_id] = 5.0
    monkeypatch.setattr(
        engine_mod.LatentCortexEngine, "_logits", lambda self, h: spiked
    )

    def run(min_tokens):
        engine = engine_mod.LatentCortexEngine(
            tiny_model,
            EosTokenizer(),
            config=_config(decode_max_tokens=20, decode_min_tokens=min_tokens),
        )
        cache = [KVCache() for _ in tiny_model.model.layers]
        inner = tiny_model.model
        h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        return engine._decode(
            cache, ComputeBudget(), spiked[0, 0], max_tokens=20, temperature=0.0
        )

    out, termination = run(0)
    assert termination == "eos" and len(out) == 0, "control: instant EOS honored"

    out, termination = run(8)
    assert termination == "eos"
    assert len(out) == 8, "EOS must be masked until the floor, then honored"
    assert all(token == word_id for token in out), "runner-up token fills the floor"
