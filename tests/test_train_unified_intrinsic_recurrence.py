"""Operational contracts for the resumable unified recurrence trainer."""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
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
    _clip_gradient_norm,
    _deterministic_student_mix,
    _evaluate,
    _generate_student_rollin,
    _model_identity,
    _optimization_phase,
    _phase_gradients,
    _residual_hidden_size,
    _restore_checkpoint,
    _save_checkpoint,
    _student_rollin_probability,
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
        depth_basis_size=3,
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
    assert wiring["continuous_depth_operator_count"] == 2
    assert wiring["continuous_depth_basis_size"] == 3
    for index, layer in enumerate(bundle.model.model.layers):
        wrapped = isinstance(layer.self_attn.o_proj, ScopedLoRALinear)
        assert wrapped is (2 <= index < 4)


def test_hidden_size_comes_from_residual_space_not_packed_embeddings() -> None:
    model = _model()
    model.model.embed_tokens.weight = mx.zeros((64, 4))
    assert _residual_hidden_size(model) == 32


def test_model_identity_hashes_weight_content_not_only_path(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"hidden_size": 4}\n')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = _model_identity(str(tmp_path))
    weights.write_bytes(b"other")
    second = _model_identity(str(tmp_path))
    assert first["canonical_path"] == second["canonical_path"]
    assert first["weights"][0]["size"] == second["weights"][0]["size"]
    assert first["weights"][0]["sha256"] != second["weights"][0]["sha256"]
    assert first["identity_sha256"] != second["identity_sha256"]


def test_two_phase_gradient_partition_keeps_anchor_stationary() -> None:
    gradients = {
        "model": {
            "layer": {
                "lora_a": mx.ones((2, 2)),
                "continuous_depth_b": [mx.ones((2, 2))],
            }
        },
        "controller": {"transport_bias": mx.ones(())},
    }
    semantic = dict(tree_flatten(_phase_gradients(gradients, "semantic_anchor")))
    recurrent = dict(tree_flatten(_phase_gradients(gradients, "recurrence")))
    assert bool(mx.all(semantic["model.layer.lora_a"] == 1))
    assert bool(mx.all(semantic["model.layer.continuous_depth_b.0"] == 0))
    assert bool(mx.all(semantic["controller.transport_bias"] == 0))
    assert bool(mx.all(recurrent["model.layer.lora_a"] == 0))
    assert bool(mx.all(recurrent["model.layer.continuous_depth_b.0"] == 1))
    assert bool(mx.all(recurrent["controller.transport_bias"] == 1))
    assert _optimization_phase(39, 40) == "semantic_anchor"
    assert _optimization_phase(40, 40) == "recurrence"


def test_student_rollin_mix_is_deterministic_and_never_relabels() -> None:
    answer = mx.array([[2, 3, 4, 5]])
    generated = mx.array([[7, 8, 9, 10]])
    first, selected = _deterministic_student_mix(
        answer,
        generated,
        probability=1.0,
        seed=19,
    )
    replay, replay_selected = _deterministic_student_mix(
        answer,
        generated,
        probability=1.0,
        seed=19,
    )
    assert selected == replay_selected == (0, 1, 2)
    assert first.tolist() == replay.tolist() == [[7, 8, 9, 5]]
    teacher, no_positions = _deterministic_student_mix(
        answer,
        generated,
        probability=0.0,
        seed=19,
    )
    assert no_positions == ()
    assert teacher.tolist() == answer.tolist()


def test_student_rollin_generation_is_answer_aligned() -> None:
    bundle, _wiring = _bundle()
    with recurrence_adapter_scope(start=None, stop=None):
        generated = _generate_student_rollin(
            bundle,
            mx.array([[2, 7, 11]]),
            mx.array([[13, 17, 19, 23]]),
            UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8)).plan_at(2),
            eos_token_id=None,
        )
    assert generated.shape == (1, 4)
    assert generated.dtype == mx.array([[1]]).dtype


def test_student_rollin_schedule_and_gradient_trust_bound() -> None:
    assert _student_rollin_probability(
        4,
        semantic_warmup_steps=4,
        max_steps=9,
        initial=0.1,
        final=0.5,
    ) == pytest.approx(0.1)
    assert _student_rollin_probability(
        8,
        semantic_warmup_steps=4,
        max_steps=9,
        initial=0.1,
        final=0.5,
    ) == pytest.approx(0.5)
    gradients = {"large": mx.array([3.0, 4.0]), "small": mx.array([0.0])}
    clipped, before = _clip_gradient_norm(gradients, 1.0)
    mx.eval(clipped, before)
    assert float(before.item()) == pytest.approx(5.0)
    after = mx.sqrt(sum(mx.sum(value**2) for value in clipped.values()))
    assert float(after.item()) == pytest.approx(1.0)


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


def test_named_best_checkpoint_does_not_overwrite_latest(tmp_path: Path) -> None:
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
        step=3,
        history=[],
        identity=identity,
    )
    _save_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        step=2,
        history=[],
        identity=identity,
        stem="checkpoint_best_trained",
    )
    assert (tmp_path / "checkpoint_latest.json").is_file()
    assert (tmp_path / "checkpoint_best_trained.json").is_file()
    step, _history = _restore_checkpoint(tmp_path, bundle, optimizer, identity)
    assert step == 3


def test_evaluation_separates_trained_from_heldout_depth_gains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    values = iter((1.0, 0.9, 1.2, 1.4))

    def fake_trajectory(*_args, **_kwargs):
        return [], [mx.array(next(values))]

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.unified_answer_trajectory",
        fake_trajectory,
    )
    tokenizer = type("Tokenizer", (), {})()
    task = type("Task", (), {"prompt": "p", "answer": "a"})()
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]]), mx.array([[2]])),
    )
    envelope = type("Envelope", (), {"reclaim": lambda *_args, **_kwargs: None})()
    report = _evaluate(
        bundle,
        tokenizer,
        [task],
        spec,
        "",
        spec.depths,
        envelope=envelope,
    )
    assert report["trained_depth_helps"] is True
    assert report["heldout_depth_helps"] is False
