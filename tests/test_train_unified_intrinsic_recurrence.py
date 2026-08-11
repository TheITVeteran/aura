"""Operational contracts for the resumable unified recurrence trainer."""

from __future__ import annotations

import json
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
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    TRAINING_SOURCE_FILES,
    UnifiedTrainingBundle,
    _answer_binding_loss,
    _answer_role_place_targets,
    _attach_window_adapters,
    _canonical_sha256,
    _clip_gradient_groups,
    _clip_gradient_norm,
    _configure_window_tissue,
    _deterministic_student_mix,
    _evaluate,
    _generate_student_rollin,
    _ground_state_value_embeddings,
    _initial_rollin_totals,
    _invocation_stop_step,
    _load_latest_checkpoint,
    _model_identity,
    _optimization_phase,
    _phase_gradients,
    _residual_hidden_size,
    _restore_checkpoint,
    _restore_rollin_totals,
    _rollin_report,
    _save_checkpoint,
    _semantic_execution_depth,
    _student_rollin_probability,
    _trainable,
    _training_halt_reason,
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


def test_controller_only_tissue_leaves_every_model_projection_frozen() -> None:
    model = _model()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))

    wiring = _configure_window_tissue(
        model,
        spec,
        mode="controller_only",
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
    trainable = _trainable(UnifiedTrainingBundle(model, controller))

    assert wiring == {
        "window_tissue_mode": "controller_only",
        "window": [2, 4],
        "adapted_sites": [],
        "adapted_projection_count": 0,
        "continuous_depth_operator_count": 0,
        "continuous_depth_basis_size": 0,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": False,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_bridge": "typed_recurrent_controller_only",
    }
    assert trainable
    assert all(name.startswith("controller.") for name in trainable)
    for layer in model.model.layers:
        assert not isinstance(layer.self_attn.o_proj, ScopedLoRALinear)


def test_invocation_boundary_is_operational_and_resumable() -> None:
    assert _invocation_stop_step(0, 73, None) == 73
    assert _invocation_stop_step(0, 73, 3) == 3
    assert _invocation_stop_step(3, 73, 3) == 6
    assert _invocation_stop_step(72, 73, 3) == 73
    assert (
        _training_halt_reason(step=3, max_steps=73, invocation_stop_step=3)
        == "invocation_step_limit"
    )
    assert (
        _training_halt_reason(step=3, max_steps=73, invocation_stop_step=73)
        == "wall_clock"
    )
    assert (
        _training_halt_reason(step=73, max_steps=73, invocation_stop_step=73)
        == "max_steps"
    )
    with pytest.raises(ValueError, match="must be positive"):
        _invocation_stop_step(0, 73, 0)


def test_rollin_telemetry_round_trips_and_rejects_invalid_state() -> None:
    totals = _initial_rollin_totals()
    totals["examples"] = 7
    totals["max_preclip_gradient_norm"] = 2.5
    totals["max_preclip_gradient_norms"] = {"recurrent_controller": 2.5}
    restored = _restore_rollin_totals({"rollin_totals": totals})
    assert restored == totals
    assert restored is not totals
    assert restored["max_preclip_gradient_norms"] is not totals[
        "max_preclip_gradient_norms"
    ]
    totals["last_probability"] = float("nan")
    with pytest.raises(RuntimeError, match="probability differs"):
        _restore_rollin_totals({"rollin_totals": totals})


def test_rollin_report_is_an_immutable_historical_snapshot() -> None:
    totals = _initial_rollin_totals()
    totals["generated_positions"] = 4
    totals["generated_matches"] = 3
    totals["max_preclip_gradient_norms"] = {"state_answer_bridge": 2.0}
    report = _rollin_report(
        totals,
        initial_probability=0.25,
        final_probability=0.75,
    )
    totals["max_preclip_gradient_norms"]["state_answer_bridge"] = 9.0
    assert report["max_preclip_gradient_norms"] == {"state_answer_bridge": 2.0}
    assert report["generated_match_rate"] == pytest.approx(0.75)


def test_campaign_identity_binds_curriculum_and_state_schema_sources() -> None:
    assert "core/learning/recurrence_curriculum.py" in TRAINING_SOURCE_FILES
    assert "core/learning/recurrent_state_schema.py" in TRAINING_SOURCE_FILES
    assert "core/learning/recurrent_literal_grounding.py" in TRAINING_SOURCE_FILES
    assert "core/learning/recurrent_opcode_grounding.py" in TRAINING_SOURCE_FILES


def test_hidden_size_comes_from_residual_space_not_packed_embeddings() -> None:
    model = _model()
    model.model.embed_tokens.weight = mx.zeros((64, 4))
    assert _residual_hidden_size(model) == 32


def test_state_codebook_is_grounded_in_frozen_model_representations() -> None:
    model = _model()
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            literal_digit_token_ids=tuple(range(10)),
        )
    )
    before = controller.parameter_sha256()

    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [1 + (sum(text.encode("ascii")) % 62)]

    digest = _ground_state_value_embeddings(
        model,
        Tokenizer(),
        controller,
        prelude_end=2,
    )
    assert len(digest) == 64
    assert controller.parameter_sha256() != before
    assert controller.state_value_embeddings.shape == (5, 33, 32)
    assert controller.action_value_embeddings.shape == (8, 33, 32)
    assert controller.literal_value_embeddings.shape == (33, 32)


def test_answer_binding_targets_identify_register_roles_and_digit_places() -> None:
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=99,
        family_markers=(("khop", (70,)), ("modular", (71,)), ("register_trace", (72,))),
        syntax=(
            ("close", (6,)),
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_head", (3,)),
            ("register_mid_r1", (4,)),
            ("register_mid_r2", (5,)),
        ),
    )
    answer = mx.array([[3, 11, 12, 4, 13, 5, 14, 15, 6, 99]])
    roles, places = _answer_role_place_targets(
        "register_trace",
        answer,
        contract,
    )
    assert roles.tolist() == [[0, 2, 2, 0, 3, 0, 4, 4, 0, 0]]
    assert places.tolist() == [[0, 1, 2, 0, 2, 0, 1, 2, 0, 0]]

    role_logits = mx.zeros((1, 10, 6))
    place_logits = mx.zeros((1, 10, 3))
    loss = _answer_binding_loss(role_logits, place_logits, roles, places)
    assert float(loss.item()) > 0.0


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


def test_phase_partition_preserves_shared_t1_and_trains_depth_bridge() -> None:
    gradients = {
        "model": {
            "layer": {
                "lora_a": mx.ones((2, 2)),
                "continuous_depth_b": [mx.ones((2, 2))],
            }
        },
        "controller": {
            "answer_output": mx.ones((2, 2)),
            "transport_bias": mx.ones(()),
        },
    }
    semantic = dict(tree_flatten(_phase_gradients(gradients, "semantic_anchor")))
    answer_bridge = dict(tree_flatten(_phase_gradients(gradients, "answer_bridge")))
    state = dict(tree_flatten(_phase_gradients(gradients, "state_transition")))
    recurrent = dict(tree_flatten(_phase_gradients(gradients, "recurrence")))
    assert bool(mx.all(semantic["model.layer.lora_a"] == 1))
    assert bool(mx.all(semantic["model.layer.continuous_depth_b.0"] == 0))
    assert bool(mx.all(semantic["controller.answer_output"] == 0))
    assert bool(mx.all(semantic["controller.transport_bias"] == 0))
    assert bool(mx.all(answer_bridge["model.layer.lora_a"] == 0))
    assert bool(mx.all(answer_bridge["model.layer.continuous_depth_b.0"] == 0))
    assert bool(mx.all(answer_bridge["controller.answer_output"] == 1))
    assert bool(mx.all(answer_bridge["controller.transport_bias"] == 0))
    assert bool(mx.all(state["model.layer.lora_a"] == 0))
    assert bool(mx.all(state["model.layer.continuous_depth_b.0"] == 1))
    assert bool(mx.all(state["controller.answer_output"] == 0))
    assert bool(mx.all(state["controller.transport_bias"] == 1))
    assert bool(mx.all(recurrent["model.layer.lora_a"] == 0))
    assert bool(mx.all(recurrent["model.layer.continuous_depth_b.0"] == 1))
    assert bool(mx.all(recurrent["controller.answer_output"] == 0))
    assert bool(mx.all(recurrent["controller.transport_bias"] == 1))
    assert _optimization_phase(39, 40) == "semantic_anchor"
    assert _optimization_phase(40, 40) == "recurrence"
    assert _optimization_phase(19, 40, 20) == "state_transition"
    assert _optimization_phase(20, 40, 20) == "semantic_anchor"
    assert _optimization_phase(59, 40, 20) == "semantic_anchor"
    assert _optimization_phase(60, 40, 20) == "recurrence"
    assert _optimization_phase(59, 40, 20, 30) == "semantic_anchor"
    assert _optimization_phase(60, 40, 20, 30) == "answer_bridge"
    assert _optimization_phase(89, 40, 20, 30) == "answer_bridge"
    assert _optimization_phase(90, 40, 20, 30) == "recurrence"


def test_semantic_supervision_runs_at_the_tasks_public_execution_depth() -> None:
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2, 4), (8, 16))
    assert _semantic_execution_depth(1, spec) == 1
    assert _semantic_execution_depth(4, spec) == 4
    with pytest.raises(ValueError, match="outside the trained recurrence horizon"):
        _semantic_execution_depth(8, spec)


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


def test_student_rollin_preserves_grammar_while_exposing_wrong_digits() -> None:
    answer = mx.array([[101, 2, 102, 3, 103, 4]])
    generated = mx.array([[999, 8, 7, 9, 6, 5]])

    effective, selected = _deterministic_student_mix(
        answer,
        generated,
        probability=1.0,
        seed=23,
        interchangeable_token_ids=frozenset(range(10)),
    )

    assert selected == (1, 3)
    assert effective.tolist() == [[101, 8, 102, 9, 103, 4]]


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


def test_gradient_trust_bound_does_not_starve_independent_mechanisms() -> None:
    gradients = {
        "model": {"layer": {"lora_a": mx.array([3.0, 4.0])}},
        "controller": {
            "state_transition_output": mx.array([0.0, 12.0]),
            "state_value_embeddings": mx.array([0.0, 8.0]),
            "action_output": mx.array([0.0, 6.0]),
            "opcode_copy_logit": mx.array(2.0),
            "action_value_embeddings": mx.array([0.0, 7.0]),
            "transport_bias": mx.array([0.3, 0.4]),
        },
    }
    clipped, global_before, groups = _clip_gradient_groups(gradients, 1.0)
    flat = dict(tree_flatten(clipped))
    mx.eval(clipped, global_before, *groups.values())
    assert float(global_before.item()) > 15.0
    assert set(groups) == {
        "scoped_transformer_bridge",
        "typed_state_transition",
        "typed_state_codebook",
        "typed_action_transition",
        "typed_action_codebook",
        "recurrent_controller",
    }
    assert float(mx.linalg.norm(flat["model.layer.lora_a"]).item()) == pytest.approx(
        1.0
    )
    assert float(
        mx.linalg.norm(flat["controller.state_transition_output"]).item()
    ) == pytest.approx(1.0)
    assert float(
        mx.linalg.norm(flat["controller.state_value_embeddings"]).item()
    ) == pytest.approx(1.0)
    action_transition_norm = mx.sqrt(
        mx.sum(flat["controller.action_output"] ** 2)
        + mx.sum(flat["controller.opcode_copy_logit"] ** 2)
    )
    assert float(action_transition_norm.item()) == pytest.approx(1.0)
    assert float(mx.abs(flat["controller.opcode_copy_logit"]).item()) > 0.0
    assert float(
        mx.linalg.norm(flat["controller.action_value_embeddings"]).item()
    ) == pytest.approx(1.0)
    assert float(
        mx.linalg.norm(flat["controller.transport_bias"]).item()
    ) == pytest.approx(0.5)


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
    training_state = {"rollin_totals": _initial_rollin_totals()}
    training_state["rollin_totals"]["examples"] = 3
    _save_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        step=3,
        history=history,
        identity=identity,
        training_state=training_state,
    )

    bundle.controller.correction_b = mx.ones_like(bundle.controller.correction_b)
    mx.eval(bundle.parameters())
    step, restored_history, restored_training_state = _restore_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        identity,
    )
    assert step == 3
    assert restored_history == history
    assert restored_training_state == training_state
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


def test_latest_checkpoint_uses_immutable_generation_over_compatibility_mirror(
    tmp_path: Path,
) -> None:
    bundle, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(bundle.trainable_parameters())
    identity_body = {"schema": "test", "depths": (1, 2, 4)}
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }
    before = {name: value for name, value in _trainable(bundle).items()}
    _save_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        step=3,
        history=[{"step": 3}],
        identity=identity,
    )

    pointer = json.loads((tmp_path / "checkpoint_latest_pointer.json").read_text())
    assert pointer["step"] == 3
    loaded = _load_latest_checkpoint(tmp_path, required=True)
    assert loaded is not None
    generation_receipt, generation_weights = loaded
    assert generation_receipt["checkpoint_generation_schema"].endswith(".v2")
    assert generation_weights.parent.name in pointer["checkpoint"]

    # A crash or writer failure in the compatibility mirror cannot strand the
    # authoritative immutable generation.
    mirror = tmp_path / "checkpoint_latest.safetensors"
    mirror.unlink()
    mirror.write_bytes(b"torn compatibility mirror")
    (tmp_path / "checkpoint_latest.json").write_text("{}", encoding="utf-8")
    bundle.controller.correction_b = mx.ones_like(bundle.controller.correction_b)
    mx.eval(bundle.parameters())
    step, history, training_state = _restore_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        identity,
        required=True,
    )
    assert step == 3
    assert history == [{"step": 3}]
    assert training_state == {}
    after = _trainable(bundle)
    assert all(bool(mx.array_equal(after[name], value)) for name, value in before.items())


def test_resume_requires_a_complete_checkpoint(tmp_path: Path) -> None:
    bundle, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    identity_body = {"schema": "test", "depths": (1, 2, 4)}
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }
    with pytest.raises(RuntimeError, match="resume checkpoint is unavailable"):
        _restore_checkpoint(
            tmp_path,
            bundle,
            optimizer,
            identity,
            required=True,
        )
    (tmp_path / "checkpoint_latest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy checkpoint is incomplete"):
        _restore_checkpoint(
            tmp_path,
            bundle,
            optimizer,
            identity,
            required=True,
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
    step, _history, _training_state = _restore_checkpoint(
        tmp_path, bundle, optimizer, identity
    )
    assert step == 3


def test_evaluation_separates_trained_from_heldout_depth_gains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    values = iter((1.0, 0.9, 1.2, 1.4))

    def fake_trajectory(*_args, **_kwargs):
        return [], [], [mx.array(next(values))], []

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.unified_answer_and_recurrent_trajectory",
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
