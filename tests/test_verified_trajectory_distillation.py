from __future__ import annotations

import numpy as np
import pytest

from core.learning.verified_trajectory_distillation import (
    DISTILLATION_SCHEMA,
    fit_verified_trajectory_factors,
    fit_verified_trajectory_inventory,
    install_verified_trajectory_inventory,
)


def _low_rank_problem(*, seed: int = 17):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(12, 9))
    left = rng.normal(size=(9, 3))
    right = rng.normal(size=(3, 7))
    corrections = inputs @ left @ right
    return inputs, corrections


def test_verified_trajectory_fit_recovers_low_rank_transition() -> None:
    inputs, corrections = _low_rank_problem()
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.3.self_attn.o_proj",
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
    )

    predicted = 20.0 * (inputs @ fitted.lora_a) @ fitted.lora_b

    assert np.allclose(predicted, 0.75 * corrections, rtol=2e-4, atol=2e-4)
    assert fitted.lora_a.shape == (9, 3)
    assert fitted.lora_b.shape == (3, 7)
    assert fitted.receipt["schema"] == DISTILLATION_SCHEMA
    assert fitted.receipt["effective_rank"] == 3
    assert fitted.receipt["training_relative_error"] < 1e-4
    assert len(fitted.receipt["receipt_sha256"]) == 64


def test_verified_trajectory_fit_normalizes_teacher_magnitude() -> None:
    inputs, corrections = _low_rank_problem()
    corrections[0] *= 1000.0
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.3.self_attn.o_proj",
        rank=8,
        regularization=1e-5,
        gain=0.25,
        adapter_scale=20.0,
    )

    target = corrections / np.linalg.norm(corrections, axis=1, keepdims=True)
    predicted = 20.0 * (inputs @ fitted.lora_a) @ fitted.lora_b

    assert fitted.receipt["corrections_normalized"] is True
    assert fitted.receipt["correction_norm_max"] > 100 * fitted.receipt["correction_norm_min"]
    assert np.linalg.norm(predicted - 0.25 * target) < np.linalg.norm(0.25 * target)


def test_verified_trajectory_inventory_rejects_partial_pair_counts() -> None:
    inputs, corrections = _low_rank_problem()

    with pytest.raises(ValueError, match="unequal pair counts"):
        fit_verified_trajectory_inventory(
            {
                "model.layers.3.self_attn.o_proj": (inputs, corrections),
                "model.layers.4.self_attn.o_proj": (inputs[:-1], corrections[:-1]),
            },
            rank=3,
            regularization=1e-5,
            gain=0.25,
            adapter_scale=20.0,
        )


def test_verified_trajectory_inventory_installs_exact_named_site() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    class _Attention:
        pass

    class _Layer:
        def __init__(self) -> None:
            self.self_attn = _Attention()
            self.self_attn.o_proj = ScopedLoRALinear.from_base(
                nn.Linear(9, 7),
                r=3,
                scale=20.0,
                block_index=0,
                site="model.layers.0.self_attn.o_proj",
            )

    class _Inner:
        def __init__(self) -> None:
            self.layers = [_Layer()]

    class _Model:
        def __init__(self) -> None:
            self.model = _Inner()

    inputs, corrections = _low_rank_problem()
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.0.self_attn.o_proj",
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
    )
    model = _Model()

    receipt = install_verified_trajectory_inventory(
        model,
        {fitted.site: fitted},
        expected_sites=[fitted.site],
    )

    projection = model.model.layers[0].self_attn.o_proj
    mx.eval(projection.lora_a, projection.lora_b)
    assert np.allclose(np.asarray(projection.lora_a), fitted.lora_a)
    assert np.allclose(np.asarray(projection.lora_b), fitted.lora_b)
    assert receipt["sites"] == [fitted.site]
    assert len(receipt["receipt_sha256"]) == 64


@pytest.mark.parametrize(
    ("inputs", "corrections", "message"),
    [
        (np.ones((1, 3)), np.ones((1, 4)), "at least two"),
        (np.ones((2, 3)), np.ones((3, 4)), "pair counts differ"),
        (np.ones((2, 3)), np.zeros((2, 4)), "collapsed row"),
    ],
)
def test_verified_trajectory_fit_rejects_invalid_evidence(
    inputs: np.ndarray,
    corrections: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_verified_trajectory_factors(
            inputs,
            corrections,
            site="model.layers.3.self_attn.o_proj",
            rank=2,
            regularization=1e-5,
            gain=0.25,
            adapter_scale=20.0,
        )
