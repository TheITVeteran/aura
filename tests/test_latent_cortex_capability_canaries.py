"""Contract tests: in-episode fast-weight capability canaries.

The protected battery is the safety boundary between "the proxy loss went
down" and "the adapted function is safe to decode with":
- the battery is deterministic and its budget cost is declared up front;
- an identity ΔW (no accepted optimization step) is never charged for a
  measurement it cannot fail;
- a behaviorally destructive ΔW is rescaled and, if still regressing,
  erased BEFORE decode — and such an episode never exports a
  consolidation candidate;
- receipts carry the per-canary evidence either way.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.capability_canaries import (  # noqa: E402
    CapabilityCanaries,
    compare_canaries,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
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


@pytest.fixture(scope="module")
def tiny_model():
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


def _config(**overrides) -> CortexConfig:
    fast_weights = overrides.pop(
        "fast_weights",
        FastWeightsConfig(enabled=True, target="o_proj", opt_steps=2),
    )
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=7),
        recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        fast_weights=fast_weights,
        decode_max_tokens=4,
        **overrides,
    )


def _canary_logits(model, tokens: list[int]):
    inner = model.model
    h = inner.embed_tokens(mx.array([tokens]))
    for layer in inner.layers:
        h = layer(h, None, None)
    h = inner.norm(h)
    logits = inner.embed_tokens.as_linear(h)
    mx.eval(logits)
    return logits


# ── Battery construction + measurement ──────────────────────────────────


def test_synthetic_battery_is_deterministic_and_bounded():
    first = CapabilityCanaries(None, vocab_size=128)
    second = CapabilityCanaries(None, vocab_size=128)
    assert [s.name for s in first.sequences] == [s.name for s in second.sequences]
    assert [s.prompt_tokens for s in first.sequences] == [
        s.prompt_tokens for s in second.sequences
    ]
    assert len(first.sequences) == 6
    assert first.tokens_per_measurement == sum(
        s.total_tokens for s in first.sequences
    )
    for sequence in first.sequences:
        assert all(0 <= t < 128 for t in sequence.prompt_tokens)
        assert all(0 <= t < 128 for t in sequence.continuation_tokens)


def test_text_battery_covers_protected_behaviors():
    class WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [(hash(word) % 96) + 1 for word in text.split()]

    canaries = CapabilityCanaries(WordTokenizer(), vocab_size=128)
    names = {s.name for s in canaries.sequences}
    assert names == {
        "prose_coherence",
        "instruction_following",
        "tool_call_syntax",
        "identity_continuity",
        "factual_calibration",
        "basic_reasoning",
    }
    for sequence in canaries.sequences:
        assert sequence.total_tokens <= 24
        assert len(sequence.continuation_tokens) >= 1


def test_measurement_is_deterministic_and_finite(tiny_model):
    canaries = CapabilityCanaries(None, vocab_size=128)
    first = canaries.measure(lambda toks: _canary_logits(tiny_model, toks))
    second = canaries.measure(lambda toks: _canary_logits(tiny_model, toks))
    assert first == second
    assert all(value < 0.0 for value in first.values())


def test_compare_flags_only_drops_beyond_threshold():
    baseline = {"a": -1.0, "b": -2.0, "c": -3.0}
    adapted = {"a": -1.2, "b": -2.9, "c": -2.5}
    verdict = compare_canaries(baseline, adapted, max_logprob_drop=0.5)
    assert verdict["regressed"] == ["b"]
    assert verdict["max_drop"] == pytest.approx(0.9)
    improvements = {row["name"]: row["logprob_drop"] for row in verdict["items"]}
    assert improvements["c"] < 0.0  # improvements are negative drops
    with pytest.raises(ValueError):
        compare_canaries(baseline, {"a": -1.0}, max_logprob_drop=0.5)
    with pytest.raises(ValueError):
        compare_canaries(baseline, adapted, max_logprob_drop=0.0)


# ── Engine integration ──────────────────────────────────────────────────


def test_clean_fast_weight_episode_passes_canaries(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    receipt = result.receipt
    assert receipt.fast_weights_applied is True
    assert receipt.fast_weights_erased is True
    canaries = receipt.fast_weight_canaries
    assert canaries.get("decision") in {"accepted", "identity_no_check", "rescaled"}
    assert "fast_weight_canary_erased" not in receipt.honest_flags
    if canaries.get("evaluated"):
        assert canaries["items"], "evaluated canaries must carry per-item evidence"
        assert "fast_weight_canaries" in receipt.to_dict()


def test_destructive_delta_is_erased_before_decode(tiny_model, monkeypatch):
    """A ΔW that wrecks protected behavior must not survive to decode."""
    original_optimize = EpisodicFastWeights.optimize

    def destructive_optimize(self, loss_fn, **kwargs):
        original_optimize(self, loss_fn, **kwargs)
        # Simulate an optimizer that found a proxy win with catastrophic
        # side effects: blow up the delta far beyond the trained manifold.
        for handle in self.handles:
            handle.wrapper.U = handle.wrapper.U * 0.0 + 50.0
            handle.wrapper.V = handle.wrapper.V * 0.0 + 50.0
        if self.lifecycle.optimized_steps == 0:
            self.lifecycle.optimized_steps = 1
            self.lifecycle.optimization_attempts += 1
            self.lifecycle.loss_trail.extend([1.0, 0.5])

    monkeypatch.setattr(EpisodicFastWeights, "optimize", destructive_optimize)
    config = _config(
        fast_weights=FastWeightsConfig(
            enabled=True,
            target="o_proj",
            opt_steps=1,
            canary_rescale_attempts=1,
            export_candidates=True,
        )
    )
    engine = LatentCortexEngine(tiny_model, config=config)
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    receipt = result.receipt

    assert receipt.fast_weight_canaries["decision"] == "erased"
    assert receipt.fast_weight_canaries["rescales"] == 1
    assert receipt.fast_weight_canaries["behavioral_evaluated"] is False
    final_magnitude = receipt.fast_weight_canaries["delta_magnitude_history"][-1]
    assert final_magnitude["structural_regression"] is True
    assert final_magnitude["max_effective_delta_rms"] > receipt.fast_weight_canaries[
        "threshold_effective_delta_rms"
    ]
    assert "fast_weight_canary_erased" in receipt.honest_flags
    assert "fast_weight_canary_rescaled" in receipt.honest_flags
    # The episode still finishes and the erase is still proven.
    assert receipt.fast_weights_erased is True
    # A canary-erased delta is never a consolidation candidate.
    assert "fast_weight_candidate_exported" not in receipt.honest_flags


def test_rescale_ladder_recovers_marginal_delta(tiny_model, monkeypatch):
    """A marginally-regressing ΔW should survive at reduced scale."""
    calls = {"count": 0}
    original_compare = compare_canaries

    def marginal_compare(baseline, adapted, *, max_logprob_drop):
        calls["count"] += 1
        verdict = original_compare(
            baseline, adapted, max_logprob_drop=max_logprob_drop
        )
        if calls["count"] == 1:
            verdict["regressed"] = [verdict["items"][0]["name"]]
        return verdict

    import core.brain.llm.latent_cortex.engine as engine_mod

    monkeypatch.setattr(engine_mod, "compare_canaries", marginal_compare)
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    receipt = result.receipt
    if receipt.fast_weight_canaries.get("evaluated"):
        assert receipt.fast_weight_canaries["decision"] in {"rescaled", "accepted"}
        if receipt.fast_weight_canaries["decision"] == "rescaled":
            assert "fast_weight_canary_rescaled" in receipt.honest_flags
            assert "fast_weight_canary_erased" not in receipt.honest_flags
    assert receipt.fast_weights_erased is True


def test_canaries_disabled_skips_measurement_and_cost(tiny_model):
    config = _config(
        fast_weights=FastWeightsConfig(
            enabled=True, target="o_proj", opt_steps=2, canary_enabled=False
        )
    )
    engine = LatentCortexEngine(tiny_model, config=config)
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.fast_weight_canaries == {}
    assert result.receipt.fast_weights_erased is True


def test_rescale_validates_inputs(tiny_model):
    fast_weights = EpisodicFastWeights(FastWeightsConfig(enabled=True))
    with pytest.raises(RuntimeError):
        fast_weights.rescale(0.5)  # nothing attached
    fast_weights.attach(
        tiny_model.model, (2, 6), seed_stat=1.0, episode_id="canary-test"
    )
    try:
        identity_metrics = fast_weights.effective_delta_metrics()
        assert identity_metrics["finite"] is True
        assert identity_metrics["max_effective_delta_rms"] == 0.0
        with pytest.raises(ValueError):
            fast_weights.rescale(1.5)
        with pytest.raises(ValueError):
            fast_weights.rescale(0.0)
        before = float(fast_weights.handles[0].wrapper.scale)
        after = fast_weights.rescale(0.5)
        assert after == pytest.approx(before * 0.5)
        assert fast_weights.lifecycle.canary_rescales == 1
    finally:
        fast_weights.detach()


def test_config_validation_bounds_canary_settings(tiny_model):
    bad = _config(
        fast_weights=FastWeightsConfig(
            enabled=True, canary_max_logprob_drop=0.0
        )
    )
    with pytest.raises(ValueError, match="canary_max_logprob_drop"):
        LatentCortexEngine(tiny_model, config=bad)
    bad_magnitude = _config(
        fast_weights=FastWeightsConfig(
            enabled=True, canary_max_effective_delta_rms=float("inf")
        )
    )
    with pytest.raises(ValueError, match="canary_max_effective_delta_rms"):
        LatentCortexEngine(tiny_model, config=bad_magnitude)
    bad_attempts = _config(
        fast_weights=FastWeightsConfig(enabled=True, canary_rescale_attempts=99)
    )
    with pytest.raises(ValueError, match="canary_rescale_attempts"):
        LatentCortexEngine(tiny_model, config=bad_attempts)
