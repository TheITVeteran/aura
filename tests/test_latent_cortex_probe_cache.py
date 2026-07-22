"""Contract tests: decode-probe memoization (ReasonCache-style state reuse).

The optimization is only allowed to exist because it is provably safe:
- a hit returns EXACTLY what the recomputation would have produced;
- a hit charges the budget NOTHING (nothing ran) and the saving is receipted;
- any fast-weight lifecycle transition (attach/optimize/rescale/erase)
  flushes every memoized probe — a probe from a different model function is
  a lie, and the invalidation trail proves the boundary held.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.branches import BranchEnsemble  # noqa: E402
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import (  # noqa: E402
    EpisodicFastWeights,
)
from core.brain.llm.latent_cortex.probe_cache import (  # noqa: E402
    DecodeProbeCache,
    PROBE_CACHE_SCHEMA,
)
from core.brain.llm.latent_cortex.recurrence import WindowRunner  # noqa: E402
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
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=7),
        recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
        **overrides,
    )


# ── Unit: the cache itself ──────────────────────────────────────────────


def test_keys_bind_the_exact_probe_ingredients():
    cache = DecodeProbeCache()
    seed_z = mx.ones((1, 4, 8))
    z = mx.ones((1, 4, 8)) * 2
    base = cache.key(seed_z, z, [1, 2], 4)
    assert base == cache.key(seed_z, z, [1, 2], 4)
    assert base != cache.key(seed_z, z * 1.001, [1, 2], 4)
    assert base != cache.key(seed_z * 1.001, z, [1, 2], 4)
    assert base != cache.key(seed_z, z, [1, 3], 4)
    assert base != cache.key(seed_z, z, [1, 2], 5)


def test_hit_returns_copy_and_accounts_savings():
    cache = DecodeProbeCache()
    key = cache.key(mx.ones((1, 2, 4)), mx.ones((1, 2, 4)), [], 4)
    assert cache.get(key) is None
    cache.put(key, [7, 8, 9], layer_apps_spent=640)
    hit = cache.get(key)
    assert hit == [7, 8, 9]
    hit.append(999)  # mutating the returned list must not poison the store
    assert cache.get(key) == [7, 8, 9]
    receipt = cache.to_receipt()
    assert receipt["schema"] == PROBE_CACHE_SCHEMA
    assert receipt["hits"] == 2
    assert receipt["misses"] == 1
    assert receipt["layer_apps_saved"] == 1280


def test_eviction_is_bounded_fifo():
    cache = DecodeProbeCache(max_entries=2)
    keys = [
        cache.key(mx.ones((1, 1, 2)) * i, mx.ones((1, 1, 2)), [], 4)
        for i in range(1, 4)
    ]
    for index, key in enumerate(keys):
        cache.put(key, [index], layer_apps_spent=1)
    assert cache.get(keys[0]) is None  # evicted
    assert cache.get(keys[1]) == [1]
    assert cache.get(keys[2]) == [2]


def test_invalidation_flushes_and_leaves_a_trail():
    cache = DecodeProbeCache()
    key = cache.key(mx.ones((1, 1, 2)), mx.ones((1, 1, 2)), [], 4)
    cache.put(key, [1], layer_apps_spent=10)
    cache.invalidate("fast_weights_attached")
    assert cache.get(key) is None
    assert cache.to_receipt()["invalidations"] == ["fast_weights_attached"]


def test_cache_constructor_validates():
    with pytest.raises(ValueError):
        DecodeProbeCache(max_entries=0)
    cache = DecodeProbeCache()
    with pytest.raises(ValueError):
        cache.put("k", [1], layer_apps_spent=-1)


# ── Fast-weight lifecycle notifications ─────────────────────────────────


def test_every_fast_weight_transition_notifies(tiny_model):
    events: list[str] = []
    fast_weights = EpisodicFastWeights(FastWeightsConfig(enabled=True, opt_steps=1))
    fast_weights.on_function_change = events.append
    fast_weights.attach(
        tiny_model.model, (2, 6), seed_stat=1.0, episode_id="probe-cache-test"
    )
    fast_weights.rescale(0.5)
    fast_weights.detach()
    assert events == [
        "fast_weights_attached",
        "fast_weights_rescaled",
        "fast_weights_detached",
    ]


def test_listener_failure_never_breaks_the_lifecycle(tiny_model):
    def broken(_reason: str) -> None:
        raise RuntimeError("listener exploded")

    fast_weights = EpisodicFastWeights(FastWeightsConfig(enabled=True))
    fast_weights.on_function_change = broken
    fast_weights.attach(
        tiny_model.model, (2, 6), seed_stat=1.0, episode_id="probe-cache-test-2"
    )
    assert fast_weights.detach() > 0


# ── Engine integration ──────────────────────────────────────────────────


def _seed_branch(engine, budget):
    cache = engine._fresh_cache()
    runner = WindowRunner(engine.model.model, budget)
    engine._prefill(PROMPT_TOKENS, cache, budget)
    embeddings = engine.model.model.embed_tokens(mx.array([PROMPT_TOKENS]))
    ensemble = BranchEnsemble.seed(
        embeddings,
        engine.config.workspace,
        engine.config.branches,
        engine.config.recurrence,
        runner,
        cache,
        engine.prelude_end,
    )
    return ensemble.branches[0], cache, runner


def test_identical_probe_is_free_and_identical(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    budget = ComputeBudget()
    branch, cache, runner = _seed_branch(engine, budget)
    engine._episode_probe_cache = DecodeProbeCache()

    first = engine._decode_probe(branch, cache, runner, budget, max_tokens=4)
    spent_after_first = budget.spent_layer_apps
    second = engine._decode_probe(branch, cache, runner, budget, max_tokens=4)

    assert second == first
    assert budget.spent_layer_apps == spent_after_first, "a hit must cost nothing"
    receipt = engine._episode_probe_cache.to_receipt()
    assert receipt["hits"] == 1 and receipt["misses"] == 1
    assert receipt["layer_apps_saved"] > 0


def test_changed_state_misses_and_invalidation_recomputes_identically(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    budget = ComputeBudget()
    branch, cache, runner = _seed_branch(engine, budget)
    engine._episode_probe_cache = DecodeProbeCache()

    first = engine._decode_probe(branch, cache, runner, budget, max_tokens=4)
    branch.z = branch.z * 1.01
    branch.workspace.update(branch.z)
    changed = engine._decode_probe(branch, cache, runner, budget, max_tokens=4)
    assert engine._episode_probe_cache.to_receipt()["misses"] == 2

    # Invalidate (as a fast-weight transition would) and re-probe the
    # original state: recomputation must reproduce the memoized answer,
    # proving hits were never stale while the function was unchanged.
    branch.z = branch.z / 1.01
    branch.workspace.update(branch.z)
    engine._episode_probe_cache.invalidate("fast_weights_attached")
    recomputed = engine._decode_probe(branch, cache, runner, budget, max_tokens=4)
    assert recomputed == first
    assert changed is not None


class _ProbeTokenizer:
    """Minimal decode-only tokenizer so verifier probes run in tests."""

    eos_token_id = 0

    def decode(self, tokens):
        return " ".join(str(int(token)) for token in tokens)

    def encode(self, text, add_special_tokens=False):
        return [(hash(word) % 96) + 1 for word in str(text).split()]


def test_episode_ships_probe_cache_receipt(tiny_model):
    engine = LatentCortexEngine(
        tiny_model, tokenizer=_ProbeTokenizer(), config=_config()
    )
    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        budget=ComputeBudget(),
        verifier=lambda _text: 0.5,
    )
    receipt = result.receipt.probe_cache
    assert receipt.get("schema") == PROBE_CACHE_SCHEMA
    assert receipt["hits"] + receipt["misses"] >= 1
    assert "probe_cache" in result.receipt.to_dict()
    assert result.receipt.decoy_verification["selection_admitted"] is False
    assert "branch_verifier_decoy_calibration_failed" in result.receipt.honest_flags
    assert result.receipt.branch_scores != [0.5, 0.5]


def test_broken_verifier_loses_authority_without_collapsing_the_episode(tiny_model):
    engine = LatentCortexEngine(
        tiny_model, tokenizer=_ProbeTokenizer(), config=_config()
    )

    def broken_verifier(_text: str) -> float:
        raise RuntimeError("critic unavailable")

    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        budget=ComputeBudget(),
        verifier=broken_verifier,
    )

    assert result.ok
    assert result.receipt.blind_review == {}
    assert result.receipt.decoy_verification == {}
    assert result.receipt.branch_contract == []
    assert "verifier_preflight_failed:RuntimeError" in result.receipt.honest_flags
    assert not any(
        flag.startswith("fallback_vanilla") for flag in result.receipt.honest_flags
    )


def test_probe_cache_can_be_disabled(tiny_model):
    engine = LatentCortexEngine(
        tiny_model, config=_config(probe_cache_enabled=False)
    )
    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        budget=ComputeBudget(),
        verifier=lambda _text: 0.5,
    )
    assert result.receipt.probe_cache == {}
