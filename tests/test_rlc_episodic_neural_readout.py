from __future__ import annotations

import copy

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.episodic_neural_readout import (  # noqa: E402
    EpisodicNeuralReadout,
    build_neural_readout_experiment_receipt,
    validate_neural_readout_experiment_receipt,
)
from core.brain.llm.latent_cortex.fast_weight_learning import (  # noqa: E402
    token_sequence_sha256,
)


def test_low_rank_hidden_state_map_learns_targets_without_a_cursor() -> None:
    keys = np.eye(3, dtype=np.float32)
    readout = EpisodicNeuralReadout(
        keys,
        [4, 3, 2],
        [6.0, 6.0, 6.0],
        max_rank=3,
        ridge=1e-8,
    )
    readout.reset(gain=1.0)
    logits = mx.zeros((1, 1, 6))

    third = readout.apply(mx.array([[[0.0, 0.0, 1.0]]]), logits)
    first = readout.apply(mx.array([[[1.0, 0.0, 0.0]]]), logits)

    assert int(mx.argmax(third[0, -1])) == 2
    assert int(mx.argmax(first[0, -1])) == 4
    assert readout.applications == 2
    assert "cursor" not in vars(readout)


def test_zero_gain_is_identity_and_erase_drops_private_tensors() -> None:
    readout = EpisodicNeuralReadout(
        np.eye(2, dtype=np.float32),
        [2, 3],
        [4.0, 4.0],
        max_rank=2,
    )
    readout.reset(gain=0.0)
    logits = mx.zeros((1, 1, 5))
    assert bool(
        mx.array_equal(
            readout.apply(mx.array([[[1.0, 0.0]]]), logits),
            logits,
        )
    )
    identity = readout.receipt()
    assert identity["effective_rank"] == 2
    assert identity["weights_sha256"]
    readout.erase()
    assert readout.weights is None
    assert readout.receipt() == {
        "schema": "aura.rlc.episodic_neural_readout.v1",
        "erased": True,
        "sample_count": 0,
        "token_count": 0,
    }
    with pytest.raises(RuntimeError, match="erased"):
        readout.reset(gain=1.0)


def _identity(target: str, weight: str) -> dict[str, object]:
    return {
        "schema": "aura.rlc.episodic_neural_readout.v1",
        "erased": False,
        "keys_sha256": "a" * 64,
        "weights_sha256": weight * 64,
        "targets_sha256": target * 64,
        "sample_count": 2,
        "hidden_width": 3,
        "token_count": 2,
        "effective_rank": 2,
        "ridge": 1e-4,
        "margin": 4.0,
        "gain": 0.0,
        "applications": 0,
    }


def _rows(arm: str, *, verified_at: float | None) -> list[dict[str, object]]:
    return [
        {
            "arm": arm,
            "gain": gain,
            "score": 1.0 if gain == verified_at else 0.25,
            "probe_tokens_sha256": token_sequence_sha256([index]),
            "probe_token_count": 1,
            "target_replayed_exactly": gain == verified_at,
            "task_verified": arm == "treatment" and gain == verified_at,
            "applications": index,
        }
        for index, gain in enumerate((0.0, 0.5, 1.0, 2.0))
    ]


def test_experiment_requires_exact_verified_replay_and_matched_sham() -> None:
    receipt = build_neural_readout_experiment_receipt(
        treatment_identity=_identity("b", "c"),
        sham_identity=_identity("d", "e"),
        treatment_rows=_rows("treatment", verified_at=1.0),
        sham_rows=_rows("sham", verified_at=1.0),
        erase_proven=True,
    )
    assert receipt["accepted"] is True
    assert receipt["capability_claim_authority"] is False
    assert validate_neural_readout_experiment_receipt(receipt) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["accepted"] = False
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_neural_readout_experiment_receipt(tampered)


def test_engine_rejects_two_simultaneous_output_plasticity_mechanisms() -> None:
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.episodic_output_memory import (
        EpisodicOutputMemory,
    )

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(model)
    hidden = mx.random.normal((1, 1, 16))
    engine._active_output_memory = EpisodicOutputMemory(
        mx.expand_dims(hidden[0, -1], axis=0),
        [1],
    )
    engine._active_neural_readout = EpisodicNeuralReadout(
        np.asarray(hidden[0], dtype=np.float32),
        [2],
        [4.0],
    )
    with pytest.raises(RuntimeError, match="multiple output-boundary"):
        engine._logits(hidden)
