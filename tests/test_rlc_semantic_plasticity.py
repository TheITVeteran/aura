from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.fast_weights import (  # noqa: E402
    EpisodicFastWeights,
)
from core.brain.llm.latent_cortex.semantic_plasticity import (  # noqa: E402
    build_contrastive_semantic_seeds,
)
from core.brain.llm.latent_cortex.types import FastWeightsConfig  # noqa: E402


def _model():
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=64,
            max_position_embeddings=128,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    return model


def test_contrastive_semantic_seeds_are_checkpoint_native_and_orthogonal():
    model = _model()
    seeds = build_contrastive_semantic_seeds(
        model,
        target_tokens=[3, 5, 7, 9],
        contrast_tokens=[2, 4, 6, 8],
        rank=2,
    )

    assert seeds.shape == (2, 32)
    assert float(mx.linalg.norm(seeds[0])) == pytest.approx(1.0, abs=1e-5)
    assert float(mx.linalg.norm(seeds[1])) == pytest.approx(1.0, abs=1e-5)
    assert float(mx.sum(seeds[0] * seeds[1])) == pytest.approx(0.0, abs=1e-5)


def test_verified_semantic_seed_source_is_receipted_without_changing_attach():
    model = _model()
    seeds = build_contrastive_semantic_seeds(
        model,
        target_tokens=[11, 12, 13],
        contrast_tokens=[20, 21, 22],
        rank=2,
    )
    tokens = mx.array([[1, 2, 3]])
    baseline = model(tokens)
    mx.eval(baseline)
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="semantic-seed-test",
        seed_vectors=seeds,
        seed_source="verified_semantic_contrast",
    )
    try:
        attached = model(tokens)
        mx.eval(attached)
        assert bool(mx.array_equal(attached, baseline))
        assert fast_weights.lifecycle.semantic_seeded_columns == 2
        assert fast_weights.lifecycle.retrieval_seeded_columns == 0
        assert fast_weights.lifecycle.to_receipt()["semantic_seeded_columns"] == 2
    finally:
        fast_weights.detach()


def test_semantic_seed_builder_rejects_missing_contrast():
    with pytest.raises(ValueError, match="target and contrast"):
        build_contrastive_semantic_seeds(
            _model(),
            target_tokens=[1],
            contrast_tokens=[],
            rank=1,
        )


def test_matched_arm_reseed_changes_u_but_preserves_identity():
    model = _model()
    treatment = build_contrastive_semantic_seeds(
        model,
        target_tokens=[3, 5, 7],
        contrast_tokens=[2, 4, 6],
        rank=2,
    )
    sham = build_contrastive_semantic_seeds(
        model,
        target_tokens=[31, 33, 35],
        contrast_tokens=[2, 4, 6],
        rank=2,
    )
    tokens = mx.array([[1, 2, 3]])
    baseline = model(tokens)
    mx.eval(baseline)
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="semantic-reseed-test",
        seed_vectors=treatment,
        seed_source="verified_semantic_contrast",
    )
    try:
        treatment_snapshot = fast_weights.snapshot_delta()
        fast_weights.reseed_output_subspace(
            sham,
            seed_source="verified_semantic_contrast",
        )
        sham_snapshot = fast_weights.snapshot_delta()
        assert not bool(
            mx.array_equal(treatment_snapshot[0]["U"], sham_snapshot[0]["U"])
        )
        assert bool(mx.array_equal(treatment_snapshot[0]["V"], sham_snapshot[0]["V"]))
        after = model(tokens)
        mx.eval(after)
        assert bool(mx.array_equal(after, baseline))
    finally:
        fast_weights.detach()


def test_minimum_norm_key_emits_requested_seeded_direction_and_erases():
    model = _model()
    seeds = build_contrastive_semantic_seeds(
        model,
        target_tokens=[3, 5, 7],
        contrast_tokens=[2, 4, 6],
        rank=2,
    )
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="minimum-norm-key-test",
        seed_vectors=seeds,
        seed_source="verified_semantic_contrast",
    )
    wrapper = fast_weights.handles[0].wrapper
    x = mx.arange(wrapper.V.shape[1]).astype(mx.float32) / wrapper.V.shape[1]
    wrapper.last_input_summary = x
    try:
        write = fast_weights.install_minimum_norm_keys(
            gain=0.25,
            regularization=1e-4,
        )
        coefficient = x @ wrapper.V.T
        expected = 0.25 / (2**0.5)
        assert float(coefficient[0]) == pytest.approx(expected, rel=1e-3)
        assert float(coefficient[1]) == pytest.approx(expected, rel=1e-3)
        assert write["schema"] == "aura.fast_weight_minimum_norm_write.v1"
        assert write["layers"][0]["seeded_columns"] == 2
    finally:
        fast_weights.detach()


def test_minimum_norm_key_refuses_missing_activation():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="minimum-norm-key-missing",
        seed_vectors=mx.ones((1, 32)),
        seed_source="verified_semantic_contrast",
    )
    try:
        with pytest.raises(RuntimeError, match="captured activation"):
            fast_weights.install_minimum_norm_keys(gain=0.25, regularization=1e-4)
    finally:
        fast_weights.detach()


def test_interpolated_delta_is_exact_and_zero_restores_identity():
    model = _model()
    tokens = mx.array([[1, 2, 3]])
    baseline = model(tokens)
    mx.eval(baseline)
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="interpolated-delta-test",
        seed_vectors=mx.ones((1, 32)),
        seed_source="verified_semantic_contrast",
    )
    wrapper = fast_weights.handles[0].wrapper
    initial = fast_weights.snapshot_delta()
    wrapper.V = mx.ones_like(wrapper.V) * 0.125
    candidate = fast_weights.snapshot_delta()
    try:
        fast_weights.interpolate_delta(initial, candidate, gain=0.5, reason="test_half")
        halfway = fast_weights.snapshot_delta()
        assert bool(mx.array_equal(halfway[0]["U"], candidate[0]["U"]))
        assert bool(
            mx.allclose(
                halfway[0]["V"],
                candidate[0]["V"] * 0.5,
                atol=1e-7,
                rtol=1e-7,
            )
        )
        fast_weights.interpolate_delta(initial, candidate, gain=0.0, reason="test_zero")
        restored = model(tokens)
        mx.eval(restored)
        assert bool(mx.array_equal(restored, baseline))
    finally:
        fast_weights.detach()
