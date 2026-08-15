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
    TRANSITION_TAPE_READER_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit: set[str] | None = None,
) -> tuple[UnifiedRecurrentController, Path]:
    source = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=8,
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
    }
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
