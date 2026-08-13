from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.fast_weights import (  # noqa: E402
    EpisodicDeltaLinear,
    EpisodicFastWeights,
)
from core.brain.llm.latent_cortex.semantic_plasticity import (  # noqa: E402
    build_contrastive_semantic_seeds,
    build_layerwise_trajectory_directions,
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
        assert len(write["layers"][0]["input_norms"]) == 2
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


def test_layerwise_teacher_trajectory_is_distinct_and_query_scoped():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(
            enabled=True,
            rank=2,
            target="down_proj",
            max_wrapped_layers=2,
            layer_placement="late",
        )
    )
    fast_weights.attach(
        model.model,
        (1, 4),
        seed_stat=0.5,
        episode_id="trajectory-transplant-test",
        seed_vectors=mx.ones((2, 32)),
        seed_source="verified_semantic_contrast",
    )
    try:
        incumbent = mx.array([[1, 2, 3, 4]])
        teacher = mx.array([[1, 2, 9, 10]])
        incumbent_features = fast_weights.capture_output_features(
            lambda: model(incumbent),
            token_start=2,
        )
        teacher_features = fast_weights.capture_output_features(
            lambda: model(teacher),
            token_start=2,
        )
        directions = build_layerwise_trajectory_directions(
            teacher_features,
            incumbent_features,
            rank=2,
        )
        assert set(directions) == {2, 3}
        assert all(value.shape == (2, 32) for value in directions.values())
        assert not bool(mx.array_equal(directions[2], directions[3]))

        before = model(incumbent)
        mx.eval(before)
        fast_weights.reseed_output_subspace_by_layer(
            directions,
            seed_source="verified_semantic_contrast",
        )
        after = model(incumbent)
        mx.eval(after)
        assert bool(mx.array_equal(after, before))
    finally:
        fast_weights.detach()


def test_io_trajectory_capture_keeps_aligned_position_features():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=3, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="io-trajectory-capture-test",
    )
    try:
        inputs, outputs = fast_weights.capture_io_features(
            lambda: model(mx.array([[1, 2, 3, 4, 5]])),
            token_start=2,
        )
        assert set(inputs) == set(outputs) == {1}
        assert inputs[1].shape == outputs[1].shape == (3, 32)
        assert not bool(mx.array_equal(inputs[1][0], inputs[1][1]))
        assert not bool(mx.array_equal(outputs[1][0], outputs[1][1]))
    finally:
        fast_weights.detach()


def test_supervised_trajectory_map_fits_distinct_keys_and_erases():
    model = _model()
    tokens = mx.array([[1, 2, 3, 4]])
    baseline = model(tokens)
    mx.eval(baseline)
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=3, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="supervised-trajectory-map-test",
    )
    wrapper = fast_weights.handles[0].wrapper
    try:
        inputs, outputs = fast_weights.capture_io_features(
            lambda: model(tokens),
            token_start=1,
        )
        wrapper.last_input_features = inputs[1]
        corrections = {1: mx.roll(outputs[1], shift=1, axis=0) - outputs[1]}
        with mx.stream(mx.gpu):
            receipt = fast_weights.install_supervised_trajectory_map(
                inputs,
                corrections,
                gain=0.5,
                regularization=1e-4,
            )
        fast_weights.activate_adaptation_path()

        targets = corrections[1] / mx.linalg.norm(
            corrections[1], axis=1, keepdims=True
        )
        predicted = wrapper.scale * (inputs[1] @ wrapper.V.T) @ wrapper.U.T
        relative_error = float(
            mx.linalg.norm(predicted - 0.5 * targets)
            / mx.linalg.norm(0.5 * targets)
        )
        assert receipt["schema"] == "aura.fast_weight_supervised_trajectory_map.v2"
        assert receipt["key_source"] == "captured_query_activation"
        assert receipt["layers"][0]["teaching_pairs"] == 3
        assert relative_error == pytest.approx(
            receipt["layers"][0]["training_relative_error"],
            abs=1e-6,
        )
        assert relative_error < 0.1
        changed = model(tokens)
        mx.eval(changed)
        assert not bool(mx.array_equal(changed, baseline))
    finally:
        fast_weights.detach()

    restored = model(tokens)
    mx.eval(restored)
    assert bool(mx.array_equal(restored, baseline))


def test_supervised_trajectory_map_rejects_non_query_keys():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="supervised-map-query-binding",
    )
    try:
        fast_weights.handles[0].wrapper.last_input_features = mx.ones((1, 32))
        with pytest.raises(ValueError, match="captured query activations"):
            fast_weights.install_supervised_trajectory_map(
                {1: mx.zeros((1, 32))},
                {1: mx.ones((1, 32))},
                gain=0.5,
                regularization=1e-4,
            )
    finally:
        fast_weights.detach()


def test_query_gate_preserves_matching_write_and_suppresses_unrelated_context():
    layer = _model().model.layers[1].self_attn.o_proj
    wrapper = EpisodicDeltaLinear(
        layer,
        rank=1,
        scale=1.0,
        seed_stat=0.5,
        tag="query-gate-selectivity",
    )
    key = mx.concatenate([mx.ones((1,)), mx.zeros((31,))])
    wrapper.U = mx.ones_like(wrapper.U)
    wrapper.V = key[None, :]
    wrapper.identity_bypass = False
    wrapper.install_query_gate(
        key[None, :],
        threshold=0.8,
        temperature=0.02,
    )

    matching = key.reshape((1, 1, 32))
    unrelated = mx.concatenate(
        [mx.ones((1,)), mx.array([10.0]), mx.zeros((30,))]
    ).reshape((1, 1, 32))
    matching_delta = wrapper(matching) - layer(matching)
    unrelated_delta = wrapper(unrelated) - layer(unrelated)
    mx.eval(matching_delta, unrelated_delta)

    assert float(mx.linalg.norm(matching_delta)) > 5.0
    assert float(mx.linalg.norm(unrelated_delta)) < 1e-5


def test_manager_receipts_private_query_gate_commitments():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="query-gate-receipt",
    )
    try:
        wrapper = fast_weights.handles[0].wrapper
        wrapper.last_input_features = mx.stack(
            [mx.ones((32,)), mx.arange(32).astype(mx.float32)]
        )
        captured = fast_weights.captured_input_features()
        assert bool(mx.array_equal(captured[1], wrapper.last_input_features))
        wrapper.last_input_features = mx.zeros_like(wrapper.last_input_features)
        assert bool(mx.any(captured[1] != 0))
        wrapper.last_input_features = captured[1]
        receipt = fast_weights.install_captured_query_gates(
            threshold=0.8,
            temperature=0.05,
        )

        assert receipt["schema"] == "aura.fast_weight_query_gate.v1"
        assert receipt["layers"][0]["layer"] == 1
        assert receipt["layers"][0]["key_count"] == 2
        assert len(receipt["layers"][0]["keys_sha256"]) == 64
        assert wrapper.query_gate_keys.shape == (2, 32)
        assert fast_weights.effective_delta_metrics()["layers"][0][
            "query_conditioned"
        ] is True
    finally:
        fast_weights.detach()


def test_query_conditioned_candidate_cannot_export_as_unconditional_adapter(tmp_path):
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="query-gate-export-boundary",
    )
    wrapper = fast_weights.handles[0].wrapper
    wrapper.last_input_features = mx.ones((1, 32))
    fast_weights.install_captured_query_gates(threshold=0.8, temperature=0.05)
    fast_weights.snapshot_for_export()
    fast_weights.detach()
    fast_weights.lifecycle.erase_proven = True

    assert (
        fast_weights.export_candidate(
            tmp_path,
            episode_id="query-gate-export-boundary",
            evidence={},
        )
        is None
    )
    assert fast_weights.last_export_error == "query_conditioned_candidate_not_generalized"


def test_layerwise_trajectory_rejects_missing_layer():
    with pytest.raises(ValueError, match="inventories differ"):
        build_layerwise_trajectory_directions(
            {1: mx.ones((2, 8))},
            {2: mx.ones((2, 8))},
            rank=2,
        )


def test_decode_activation_capture_assigns_distinct_rank_keys():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="decode-key-test",
        seed_vectors=mx.ones((2, 32)),
        seed_source="verified_semantic_contrast",
    )
    try:
        fast_weights.capture_input_summaries(
            lambda: (model(mx.array([[1, 2, 3]])), model(mx.array([[1, 2, 4]])))
        )
        commitments = fast_weights.input_feature_commitments()
        wrapper = fast_weights.handles[0].wrapper
        assert set(commitments) == {1}
        assert len(commitments[1]) == 64
        assert wrapper.last_input_features.shape == (2, 32)
        assert not bool(
            mx.array_equal(
                wrapper.last_input_features[0],
                wrapper.last_input_features[1],
            )
        )
        fast_weights.install_minimum_norm_keys(gain=0.5, regularization=1e-4)
        assert not bool(mx.array_equal(wrapper.V[0], wrapper.V[1]))
    finally:
        fast_weights.detach()


def test_position_capture_preserves_distinct_slot_activations():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="position-feature-test",
    )
    try:
        _output, features = fast_weights.capture_input_position_features(
            lambda: model(mx.array([[1, 2, 3, 4]])),
            max_features=4,
        )
        assert set(features) == {1}
        assert features[1].shape == (4, 32)
        assert not bool(mx.array_equal(features[1][0], features[1][1]))
    finally:
        fast_weights.detach()


def test_position_capture_rejects_incomplete_inventory_and_resets_flag():
    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="position-feature-short-test",
    )
    try:
        with pytest.raises(RuntimeError, match="declared inventory"):
            fast_weights.capture_input_position_features(
                lambda: model(mx.array([[1, 2]])),
                max_features=3,
            )
        assert fast_weights.handles[0].wrapper.capture_input_positions is False
    finally:
        fast_weights.detach()


def test_position_capture_honors_recurrence_slot_span() -> None:
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="position-feature-scope-test",
    )
    tokens = mx.array([[1, 2, 3, 4, 5]])
    try:
        _output, all_features = fast_weights.capture_input_position_features(
            lambda: model(tokens),
            max_features=5,
        )
        with recurrence_adapter_scope(start=3, stop=5):
            _output, features = fast_weights.capture_input_position_features(
                lambda: model(tokens),
                max_features=2,
            )
        assert bool(mx.allclose(features[1], all_features[1][3:5]))
        assert not bool(mx.allclose(features[1], all_features[1][:2]))
    finally:
        fast_weights.detach()


def test_position_capture_filters_unscoped_prefill_from_decode_phase() -> None:
    from core.brain.llm.latent_cortex.recurrence_adapter import coda_adapter_scope

    model = _model()
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, max_wrapped_layers=1)
    )
    fast_weights.attach(
        model.model,
        (1, 2),
        seed_stat=0.5,
        episode_id="decode-phase-position-feature-test",
    )
    prefill = mx.array([[1, 2, 3, 4]])
    decode = mx.array([[17, 19]])

    def decode_only():
        with coda_adapter_scope():
            return model(decode)

    def prefill_then_decode():
        model(prefill)
        return decode_only()

    try:
        _output, expected = fast_weights.capture_input_position_features(
            decode_only,
            max_features=2,
            phase="decode",
        )
        _output, observed = fast_weights.capture_input_position_features(
            prefill_then_decode,
            max_features=2,
            phase="decode",
        )
        assert bool(mx.array_equal(observed[1], expected[1]))
        assert fast_weights.handles[0].wrapper.input_position_phase is None
        with pytest.raises(ValueError, match="phase is unsupported"):
            fast_weights.capture_input_position_features(
                decode_only,
                max_features=2,
                phase="dream",
            )
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
