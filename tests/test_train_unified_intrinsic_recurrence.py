"""Operational contracts for the resumable unified recurrence trainer."""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedTrainingBundle,
    _attach_window_adapters,
    _canonical_sha256,
    _residual_hidden_size,
    _restore_checkpoint,
    _save_checkpoint,
    _trainable,
)


def _model() -> Model:
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10_000.0,
        )
    )
    model.freeze()
    mx.eval(model.parameters())
    return model


def _bundle() -> tuple[UnifiedTrainingBundle, dict]:
    model = _model()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    wiring = _attach_window_adapters(
        model,
        spec,
        rank=2,
        targets=("o_proj",),
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            minimum_iterations=1,
        )
    )
    return UnifiedTrainingBundle(model, controller), wiring


def test_trainer_adapts_window_but_never_coda_or_readout() -> None:
    bundle, wiring = _bundle()
    assert wiring["window"] == [2, 4]
    assert wiring["coda_adapted"] is False
    assert wiring["readout_adapted"] is False
    for index, layer in enumerate(bundle.model.model.layers):
        wrapped = isinstance(layer.self_attn.o_proj, ScopedLoRALinear)
        assert wrapped is (2 <= index < 4)


def test_hidden_size_comes_from_residual_space_not_packed_embeddings() -> None:
    model = _model()
    model.model.embed_tokens.weight = mx.zeros((64, 4))
    assert _residual_hidden_size(model) == 32


def test_checkpoint_roundtrip_restores_exact_trainable_state(tmp_path: Path) -> None:
    bundle, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(bundle.trainable_parameters())
    mx.eval(optimizer.state)
    identity_body = {"schema": "test", "depths": (1, 2, 4)}
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }
    before = {name: value + 0 for name, value in _trainable(bundle).items()}
    mx.eval(before)
    history = [{"step": 3, "depth_helps": False}]
    _save_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        step=3,
        history=history,
        identity=identity,
    )

    bundle.controller.correction_b = mx.ones_like(bundle.controller.correction_b)
    mx.eval(bundle.parameters())
    step, restored_history = _restore_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        identity,
    )
    assert step == 3
    assert restored_history == history
    after = _trainable(bundle)
    assert set(after) == set(before)
    assert all(bool(mx.array_equal(after[name], value)) for name, value in before.items())


def test_checkpoint_refuses_a_different_campaign_identity(tmp_path: Path) -> None:
    bundle, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(bundle.trainable_parameters())
    identity_body = {"schema": "test", "depths": (1, 2, 4)}
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }
    _save_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        step=1,
        history=[],
        identity=identity,
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        _restore_checkpoint(
            tmp_path,
            bundle,
            optimizer,
            {
                "schema": "test",
                "depths": (1, 2, 8),
                "identity_sha256": "b" * 64,
            },
        )
