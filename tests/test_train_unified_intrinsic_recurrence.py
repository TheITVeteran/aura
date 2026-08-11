"""Operational contracts for the resumable unified recurrence trainer."""

from __future__ import annotations

import json
import os
import threading
import time
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
    _atomic_canonical_json,
    _attach_window_adapters,
    _await_resource_guard,
    _canonical_sha256,
    _clip_gradient_groups,
    _clip_gradient_norm,
    _configure_window_tissue,
    _deterministic_student_mix,
    _evaluate,
    _freeze_dataset,
    _generate_student_rollin,
    _ground_state_value_embeddings,
    _initial_rollin_totals,
    _invocation_stop_step,
    _load_frozen_dataset,
    _load_latest_checkpoint,
    _model_identity,
    _model_lane_purpose,
    _optimization_phase,
    _phase_gradients,
    _residual_hidden_size,
    _resolve_recurrent_window,
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


def test_model_lane_envelope_tracks_the_trainable_tissue_class() -> None:
    assert _model_lane_purpose("controller_only") == "train_frozen_controller"
    assert _model_lane_purpose("scoped_lora") == "train"
    with pytest.raises(ValueError, match="window tissue mode"):
        _model_lane_purpose("unknown")


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
    assert "core/learning/intrinsic_recurrence.py" in TRAINING_SOURCE_FILES
    assert "core/learning/protected_memory.py" in TRAINING_SOURCE_FILES
    assert "core/runtime/model_lane_control.py" in TRAINING_SOURCE_FILES
    assert "tools/evaluate_unified_intrinsic_checkpoint.py" in TRAINING_SOURCE_FILES
    assert "tools/evaluate_unified_intrinsic_decoding.py" in TRAINING_SOURCE_FILES
    assert "tools/train_intrinsic_recurrence.py" in TRAINING_SOURCE_FILES
    assert "tools/unified_intrinsic_resident_identity.py" in TRAINING_SOURCE_FILES
    assert "requirements_lock.txt" in TRAINING_SOURCE_FILES


def test_training_receipt_writer_uses_canonical_json(tmp_path: Path) -> None:
    target = tmp_path / "training_receipt.json"
    _atomic_canonical_json(target, {"z": 2, "a": 1})
    assert target.read_bytes() == b'{"a":1,"z":2}\n'


def test_hidden_size_comes_from_residual_space_not_packed_embeddings() -> None:
    model = _model()
    model.model.embed_tokens.weight = mx.zeros((64, 4))
    assert _residual_hidden_size(model) == 32


def test_fractional_window_resolves_across_checkpoint_depths(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"num_hidden_layers": 64}),
        encoding="utf-8",
    )
    prelude, coda, receipt = _resolve_recurrent_window(
        str(model),
        prelude_end=None,
        coda_start=None,
        prelude_fraction=0.25,
        coda_fraction=0.25,
    )
    assert (prelude, coda) == (16, 48)
    assert receipt["mode"] == "fractional"
    assert len(receipt["contract_sha256"]) == 64

    explicit = _resolve_recurrent_window(
        str(model),
        prelude_end=12,
        coda_start=50,
        prelude_fraction=0.25,
        coda_fraction=0.25,
    )
    assert explicit[:2] == (12, 50)
    assert explicit[2]["mode"] == "explicit"
    with pytest.raises(ValueError, match="requires both boundaries"):
        _resolve_recurrent_window(
            str(model),
            prelude_end=12,
            coda_start=None,
            prelude_fraction=0.25,
            coda_fraction=0.25,
        )


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

    receipt = _ground_state_value_embeddings(
        model,
        Tokenizer(),
        controller,
        prelude_end=2,
    )
    assert len(receipt["sha256"]) == 64
    assert receipt["label_count"] == 462
    assert receipt["forward_batches"] < receipt["label_count"] // 8
    assert receipt["batch_size"] == 32
    assert controller.parameter_sha256() != before
    assert controller.state_value_embeddings.shape == (5, 33, 32)
    assert controller.action_value_embeddings.shape == (8, 33, 32)
    assert controller.literal_value_embeddings.shape == (33, 32)


def test_batched_state_codebook_matches_single_label_grounding() -> None:
    model = _model()

    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            width = 1 + len(text) % 3
            return [
                1 + (sum(text.encode("ascii")) + index) % 62
                for index in range(width)
            ]

    def controller() -> UnifiedRecurrentController:
        return UnifiedRecurrentController(
            UnifiedRecurrenceConfig(
                hidden_size=32,
                correction_rank=4,
                initialization_seed=19,
                literal_digit_token_ids=tuple(range(10)),
            )
        )

    serial = controller()
    batched = controller()
    repeated = controller()
    serial_receipt = _ground_state_value_embeddings(
        model,
        Tokenizer(),
        serial,
        prelude_end=2,
        batch_size=1,
    )
    batched_receipt = _ground_state_value_embeddings(
        model,
        Tokenizer(),
        batched,
        prelude_end=2,
        batch_size=64,
    )
    repeated_receipt = _ground_state_value_embeddings(
        model,
        Tokenizer(),
        repeated,
        prelude_end=2,
        batch_size=64,
    )

    assert batched_receipt["sha256"] == repeated_receipt["sha256"]
    assert serial_receipt["label_count"] == batched_receipt["label_count"]
    assert batched_receipt["forward_batches"] < serial_receipt["forward_batches"]

    def assert_numerically_equivalent(left: object, right: object) -> None:
        left_flat = left.reshape(-1).astype(mx.float32)
        right_flat = right.reshape(-1).astype(mx.float32)
        cosine = mx.sum(left_flat * right_flat) / (
            mx.linalg.norm(left_flat) * mx.linalg.norm(right_flat)
        )
        assert float(cosine.item()) > 0.99999
        assert float(mx.max(mx.abs(left_flat - right_flat)).item()) < 0.02

    assert_numerically_equivalent(
        serial.state_value_embeddings,
        batched.state_value_embeddings,
    )
    assert_numerically_equivalent(
        serial.action_value_embeddings,
        batched.action_value_embeddings,
    )
    assert_numerically_equivalent(
        serial.literal_value_embeddings,
        batched.literal_value_embeddings,
    )


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
    (tmp_path / "tokenizer.json").write_text('{"version": 1}\n')
    (tmp_path / "tokenizer_config.json").write_text('{"eos": 2}\n')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = _model_identity(str(tmp_path))
    weights.write_bytes(b"other")
    second = _model_identity(str(tmp_path))
    assert first["canonical_path"] == second["canonical_path"]
    assert first["weights"][0]["size"] == second["weights"][0]["size"]
    assert first["weights"][0]["sha256"] != second["weights"][0]["sha256"]
    assert first["identity_sha256"] != second["identity_sha256"]
    (tmp_path / "tokenizer.json").write_text('{"version": 2}\n')
    third = _model_identity(str(tmp_path))
    assert second["weights"] == third["weights"]
    assert second["behavior_sha256"] != third["behavior_sha256"]
    assert second["identity_sha256"] != third["identity_sha256"]


def test_dataset_freeze_binds_private_traces_and_refuses_drift(tmp_path: Path) -> None:
    from core.learning.recurrence_curriculum import task_battery

    train = task_battery(("khop",), (1,), 2, seed=101)
    holdout = task_battery(("khop",), (1,), 1, seed=202)
    first = _freeze_dataset(tmp_path, train, holdout)
    second = _freeze_dataset(tmp_path, train, holdout)
    assert first == second
    assert first["train_count"] == 2
    assert first["holdout_count"] == 1
    assert first["partition_overlap"] == 0
    payload = json.loads((tmp_path / "dataset.json").read_text(encoding="ascii"))
    assert payload["train"][0]["transition_trace"] is not None
    assert payload["train"][0]["transition_program"] is not None
    restored_train, restored_holdout = _load_frozen_dataset(tmp_path / "dataset.json")
    assert restored_train == train
    assert restored_holdout == holdout

    dataset = tmp_path / "dataset.json"
    dataset.chmod(0o600)
    dataset.write_text("{}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="source_dataset_unreadable"):
        _freeze_dataset(tmp_path, train, holdout)


def test_resource_guard_blocks_until_external_exact_pid_ack(tmp_path: Path) -> None:
    from core.runtime.resource_stage_guard import (
        publish_armed_ack,
        read_ready_marker,
    )

    marker = tmp_path / "resource-stage.json"

    def acknowledge() -> None:
        deadline = time.monotonic() + 2.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        _payload, marker_raw = read_ready_marker(
            marker,
            expected_target_pid=os.getpid(),
        )
        publish_armed_ack(
            marker,
            marker_raw=marker_raw,
            target_pid=os.getpid(),
            sentinel_pid=os.getpid(),
            startup_lethal_mb=100.0,
            steady_lethal_mb=80.0,
        )

    worker = threading.Thread(target=acknowledge, daemon=True)
    worker.start()
    receipt = _await_resource_guard(
        marker,
        trainer_sha256="a" * 64,
        startup_lethal_mb=100.0,
        steady_lethal_mb=80.0,
        timeout_s=2.0,
    )
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert receipt["marker"]["target_pid"] == os.getpid()
    assert receipt["ack"]["target_pid"] == os.getpid()


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


def test_optional_resume_starts_fresh_only_when_no_checkpoint_exists(
    tmp_path: Path,
) -> None:
    bundle, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    identity_body = {"schema": "test", "depths": (1, 2, 4)}
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }

    assert _restore_checkpoint(
        tmp_path,
        bundle,
        optimizer,
        identity,
        required=False,
    ) == (0, [], {})
    with pytest.raises(RuntimeError, match="checkpoint is unavailable"):
        _restore_checkpoint(
            tmp_path,
            bundle,
            optimizer,
            identity,
            required=True,
        )


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
    assert generation_receipt["checkpoint_generation_schema"].endswith(".v3")
    assert generation_weights.parent.name in pointer["checkpoint"]
    assert generation_weights.parent.stat().st_mode & 0o222 == 0
    assert generation_weights.stat().st_mode & 0o222 == 0

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
    assert (tmp_path / "checkpoint_best_trained_pointer.json").is_file()
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
