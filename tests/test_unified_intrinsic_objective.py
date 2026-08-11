"""The unified recurrence learns semantics without moving its readout."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    unified_answer_trajectory,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
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


def _spec() -> UnifiedIntrinsicTrainingSpec:
    return UnifiedIntrinsicTrainingSpec(
        prelude_end=2,
        coda_start=4,
        train_depths=(1, 2, 3),
        heldout_depths=(5, 8),
    )


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            minimum_iterations=1,
        )
    )


TOKENS = mx.array([[2, 7, 11, 13]])
ANSWERS = mx.array([[17, 19]])


def test_depth_split_is_strict_and_extrapolating() -> None:
    spec = _spec()
    assert not (set(spec.train_depths) & set(spec.heldout_depths))
    assert min(spec.heldout_depths) > max(spec.train_depths)
    with pytest.raises(ValueError, match="disjoint"):
        UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (2, 4))


def test_every_recurrent_state_is_decoded_through_the_same_readout() -> None:
    states, losses = unified_answer_trajectory(
        _model(),
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        _controller(),
    )
    assert len(states) == len(losses) == 3
    assert all(value.shape == () for value in losses)
    assert all(float(value.item()) > 0.0 for value in losses)


def test_optimizer_moves_controller_but_not_frozen_readout() -> None:
    model = _model()
    controller = _controller()
    spec = _spec()
    before_readout = readout_fingerprint(model, spec.coda_start)
    before_controller = controller.parameter_sha256()

    def objective(candidate: UnifiedRecurrentController):
        loss, _telemetry = unified_intrinsic_training_loss(
            model,
            TOKENS,
            ANSWERS,
            candidate,
            spec,
            readout_sha256=before_readout,
        )
        return loss

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(flat["correction_b"]))) > 0.0
    assert float(mx.max(mx.abs(flat["halt_state_weight"]))) > 0.0
    assert float(mx.abs(flat["transport_bias"])) > 0.0
    assert float(mx.abs(flat["transport_decay_logit"])) > 0.0
    assert float(mx.max(mx.abs(flat["transport_state_weight"]))) > 0.0
    assert float(mx.max(mx.abs(flat["transport_motion_weight"]))) > 0.0
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(controller, gradients)
    mx.eval(controller.parameters(), optimizer.state)

    assert controller.parameter_sha256() != before_controller
    assert readout_fingerprint(model, spec.coda_start) == before_readout


def test_training_receipt_keeps_heldout_depths_unopened() -> None:
    loss, receipt = unified_intrinsic_training_loss(
        _model(),
        TOKENS,
        ANSWERS,
        _controller(),
        _spec(),
    )
    assert float(loss.item()) == pytest.approx(receipt["total"])
    assert set(receipt["per_depth"]) == {"T1", "T2", "T3"}
    assert receipt["heldout_depths_unopened"] == [5, 8]
    assert receipt["readout_frozen_by_training_contract"] is True
