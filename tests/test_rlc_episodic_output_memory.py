from __future__ import annotations

import copy

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.episodic_output_memory import (  # noqa: E402
    EpisodicOutputMemory,
    build_output_memory_experiment_receipt,
    validate_output_memory_experiment_receipt,
)
from core.brain.llm.latent_cortex.fast_weight_learning import (  # noqa: E402
    token_sequence_sha256,
)


def test_exact_ordered_keys_raise_only_the_committed_targets():
    keys = mx.array([[1.0, 0.0], [0.0, 1.0]])
    memory = EpisodicOutputMemory(keys, [2, 3], margin=4.0)
    memory.reset(gain=1.0)
    logits = mx.zeros((1, 1, 5))

    first = memory.apply(mx.array([[[1.0, 0.0]]]), logits)
    second = memory.apply(mx.array([[[0.0, 1.0]]]), logits)

    assert int(mx.argmax(first[0, -1])) == 2
    assert int(mx.argmax(second[0, -1])) == 3
    assert memory.matches == 2
    assert memory.cursor == 2


def test_nonmatching_state_cannot_advance_or_change_logits():
    memory = EpisodicOutputMemory(mx.array([[1.0, 0.0]]), [2])
    memory.reset(gain=1.0)
    logits = mx.array([[[0.0, 1.0, -1.0]]])
    result = memory.apply(mx.array([[[0.0, 1.0]]]), logits)

    assert bool(mx.array_equal(result, logits))
    assert memory.cursor == 0
    assert memory.misses == 1


def test_zero_gain_is_exact_identity_and_erase_is_terminal():
    memory = EpisodicOutputMemory(mx.array([[1.0, 0.0]]), [2])
    memory.reset(gain=0.0)
    logits = mx.array([[[0.0, 1.0, -1.0]]])
    assert bool(mx.array_equal(memory.apply(mx.array([[[1.0, 0.0]]]), logits), logits))
    before = memory.receipt()
    assert before["targets_sha256"] and before["keys_sha256"]
    memory.erase()
    assert memory.receipt() == {
        "schema": "aura.rlc.episodic_output_memory.v1",
        "erased": True,
        "key_count": 0,
        "target_count": 0,
    }
    with pytest.raises(RuntimeError, match="erased"):
        memory.reset(gain=1.0)


def test_experiment_requires_treatment_to_beat_baseline_and_sham():
    def rows(arm: str, scores: list[float]):
        return [
            {
                "arm": arm,
                "gain": gain,
                "score": score,
                "probe_tokens_sha256": token_sequence_sha256([index]),
                "probe_token_count": 1,
                "matches": index,
                "misses": 0,
                "minimum_similarity": 1.0,
            }
            for index, (gain, score) in enumerate(
                zip((0.0, 0.5, 1.0), scores, strict=True)
            )
        ]

    identity = {
        "schema": "aura.rlc.episodic_output_memory.v1",
        "erased": False,
        "keys_sha256": "a" * 64,
        "targets_sha256": "b" * 64,
        "key_count": 1,
        "target_count": 1,
        "hidden_width": 2,
        "similarity_floor": 0.995,
        "margin": 8.0,
        "gain": 0.0,
        "matches": 0,
        "misses": 0,
        "minimum_similarity": 1.0,
    }
    sham_identity = {**identity, "targets_sha256": "c" * 64}
    accepted = build_output_memory_experiment_receipt(
        baseline_score=0.25,
        treatment_identity=identity,
        sham_identity=sham_identity,
        treatment_rows=rows("treatment", [0.25, 0.5, 1.0]),
        sham_rows=rows("sham", [0.25, 0.1, 0.0]),
        erase_proven=True,
    )
    assert accepted["accepted"] is True
    assert accepted["selected_treatment_gain"] == 1.0
    assert validate_output_memory_experiment_receipt(accepted) == accepted

    rejected = build_output_memory_experiment_receipt(
        baseline_score=0.25,
        treatment_identity=identity,
        sham_identity=sham_identity,
        treatment_rows=rows("treatment", [0.25, 0.5, 0.6]),
        sham_rows=rows("sham", [0.25, 0.7, 0.1]),
        erase_proven=True,
    )
    assert rejected["accepted"] is False

    tampered = copy.deepcopy(accepted)
    tampered["selected_treatment_score"] = 0.0
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_output_memory_experiment_receipt(tampered)


def test_engine_output_boundary_applies_only_an_explicit_active_memory():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(model)
    hidden = mx.random.normal((1, 1, 32))
    baseline = engine._logits(hidden)
    target = int(mx.argmin(baseline[0, -1]))
    key = mx.expand_dims(engine._last_output_hidden[0, -1], axis=0)
    memory = EpisodicOutputMemory(key, [target], margin=8.0)
    memory.reset(gain=1.0)

    engine._active_output_memory = memory
    steered = engine._logits(hidden)
    engine._active_output_memory = None

    assert int(mx.argmax(steered[0, -1])) == target
    assert memory.matches == 1
    assert bool(mx.array_equal(engine._logits(hidden), baseline))


def test_diagnostic_flag_is_default_off_and_wire_explicit():
    from core.brain.llm.latent_cortex.worker_handler import config_from_job

    assert config_from_job({}).fast_weights.output_memory_diagnostic_enabled is False
    enabled = config_from_job({"fast_weights_output_memory_diagnostic": True})
    assert enabled.fast_weights.output_memory_diagnostic_enabled is True
    explicit = config_from_job(
        {
            "fast_weights_layer_placement": "late",
            "fast_weights_canary_generated": False,
        }
    )
    assert explicit.fast_weights.layer_placement == "late"
    assert explicit.fast_weights.canary_generated_enabled is False
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"fast_weights_output_memory_diagnostic": 1})
