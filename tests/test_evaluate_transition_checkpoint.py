from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

import tools.evaluate_transition_checkpoint as evaluator
from core.learning.unified_intrinsic_recurrence import (
    TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES,
    TRANSITION_MEMORY_PARAMETER_NAMES,
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
    TRANSITION_REPLAY_PARAMETER_NAMES,
    TRANSITION_TAPE_READER_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit: set[str] | None = None,
    state_slots: int = 5,
) -> tuple[UnifiedRecurrentController, Path]:
    source = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=8,
            state_slots=state_slots,
            minimum_iterations=1,
            initialization_seed=498,
        )
    )
    source.transition_tape_output = mx.ones_like(source.transition_tape_output)
    source.transition_processor_opcode_interaction_down = mx.ones_like(
        source.transition_processor_opcode_interaction_down
    )
    source.action_value_embeddings = mx.full_like(
        source.action_value_embeddings,
        0.375,
    )
    names = (
        set(TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES)
        | set(TRANSITION_MEMORY_PARAMETER_NAMES)
        | set(TRANSITION_PROCESSOR_PARAMETER_NAMES)
        | set(TRANSITION_TAPE_READER_PARAMETER_NAMES)
        | set(TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES)
        | set(TRANSITION_REPLAY_PARAMETER_NAMES)
    ) - set(omit or ())
    path = tmp_path / "checkpoint.safetensors"
    mx.save_safetensors(
        str(path),
        {f"bundle.controller.{name}": getattr(source, name) for name in names},
    )
    monkeypatch.setattr(
        evaluator,
        "_load_latest_checkpoint",
        lambda *_args, **_kwargs: (
            {
                "checkpoint_sha256": "a" * 64,
                "receipt_sha256": "b" * 64,
                "step": 503,
                "identity": {
                    "controller_rank": 8,
                    "init_seed": 498,
                    "state_slots": state_slots,
                    "identity_sha256": "c" * 64,
                },
            },
            path,
        ),
    )
    return source, path


def test_checkpoint_evaluator_loads_complete_deployed_transition_tissue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _path = _checkpoint(tmp_path, monkeypatch)

    loaded, receipt = evaluator._load_controller(tmp_path)

    mx.eval(
        loaded.transition_tape_output,
        loaded.transition_processor_opcode_interaction_down,
        loaded.action_value_embeddings,
    )
    assert bool(mx.allclose(loaded.transition_tape_output, source.transition_tape_output))
    assert bool(
        mx.allclose(
            loaded.transition_processor_opcode_interaction_down,
            source.transition_processor_opcode_interaction_down,
        )
    )
    assert bool(
        mx.allclose(
            loaded.action_value_embeddings,
            source.action_value_embeddings,
        )
    )
    assert receipt["zero_attached_extensions"] == []
    assert "transition_tape_output" in receipt["loaded_transition_tensor_names"]
    assert (
        "transition_processor_opcode_interaction_down"
        in receipt["loaded_transition_tensor_names"]
    )
    assert "action_value_embeddings" in receipt["loaded_transition_tensor_names"]


def test_checkpoint_evaluator_reconstructs_semantic_state_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _path = _checkpoint(tmp_path, monkeypatch, state_slots=11)

    loaded, _receipt = evaluator._load_controller(tmp_path)

    assert loaded.config.state_slots == 11
    assert loaded.state_value_embeddings.shape == source.state_value_embeddings.shape


def test_checkpoint_evaluator_aggregates_recovery_terminal_and_registers() -> None:
    base = {
        "active_state_exact_accuracy": 0.5,
        "active_value_exact_accuracy": 0.5,
        "active_trajectory_exact": False,
        "final_active_state_exact": True,
        "first_error_fraction": 0.5,
        "first_error_step": 2,
        "recovery_observable": True,
        "recovered_after_first_error": True,
        "sustained_recovery_after_first_error": False,
        "conditional_transition_counts": {
            "correct_after_correct": 1,
            "correct_predecessors": 2,
            "correct_after_wrong": 1,
            "wrong_predecessors": 1,
        },
        "terminal_stability_observable": True,
        "terminal_correct_stable": True,
        "terminal_self_stable": True,
        "per_register_accuracy": {"pc": 1.0, "value0": 0.5, "done": 1.0},
    }

    report = evaluator._aggregate([base, {**base, "recovered_after_first_error": False}])

    assert report["final_active_state_exact"] == 1.0
    assert report["recovered_after_first_error"] == 0.5
    assert report["sustained_recovery_after_first_error"] == 0.0
    assert report["terminal_correct_stability"] == 1.0
    assert report["per_register_accuracy"]["value0"] == 0.5


def test_checkpoint_evaluator_rejects_missing_action_codebook_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _path = _checkpoint(
        tmp_path,
        monkeypatch,
        omit={"action_value_embeddings"},
    )

    with pytest.raises(RuntimeError, match="action_value_embeddings"):
        evaluator._load_controller(tmp_path)


def test_checkpoint_evaluator_zero_attaches_complete_older_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = set(TRANSITION_TAPE_READER_PARAMETER_NAMES) | {
        "transition_processor_opcode_interaction_up",
        "transition_processor_opcode_interaction_down",
        "transition_processor_opcode_hidden",
    } | set(TRANSITION_REPLAY_PARAMETER_NAMES)
    _source, _path = _checkpoint(tmp_path, monkeypatch, omit=missing)

    loaded, receipt = evaluator._load_controller(tmp_path)

    mx.eval(
        loaded.transition_tape_output,
        loaded.transition_processor_opcode_interaction_down,
        loaded.transition_processor_opcode_hidden,
    )
    assert bool(mx.all(loaded.transition_tape_output == 0))
    assert bool(mx.all(loaded.transition_processor_opcode_interaction_down == 0))
    assert bool(mx.all(loaded.transition_processor_opcode_hidden == 0))
    assert set(receipt["zero_attached_extensions"]) == missing


def test_checkpoint_evaluator_rejects_partial_extension_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _path = _checkpoint(
        tmp_path,
        monkeypatch,
        omit={"transition_tape_output"},
    )

    with pytest.raises(RuntimeError, match="extension inventory is partial"):
        evaluator._load_controller(tmp_path)
