"""Operational contracts for the resumable unified recurrence trainer."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    ACTION_LITERAL_BINDING_PARAMETER_NAMES,
    ACTION_WORKSPACE_PARAMETER_NAMES,
    CAUSAL_ACTION_PARAMETER_NAMES,
    FAMILY_ACTION_PARAMETER_NAMES,
    INITIAL_STATE_PARAMETER_NAMES,
    PROCESS_READER_PARAMETER_NAMES,
    TRANSITION_MEMORY_PARAMETER_NAMES,
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
    TRANSITION_REPLAY_PARAMETER_NAMES,
    TRANSITION_TAPE_READER_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    TRAINING_SOURCE_FILES,
    UnifiedTrainingBundle,
    _answer_binding_loss,
    _answer_bridge_process_preflight,
    _answer_bridge_task,
    _answer_bridge_teacher_policy,
    _answer_role_place_targets,
    _atomic_canonical_json,
    _attach_window_adapters,
    _await_resource_guard,
    _bootstrap_bundle_from_checkpoint,
    _bootstrap_numeric_observation_extension,
    _cached_answer_binding_loss,
    _canonical_sha256,
    _clip_gradient_groups,
    _clip_gradient_norm,
    _combine_process_gradient_trees,
    _configure_window_tissue,
    _deterministic_student_mix,
    _direct_transition_curriculum_window,
    _dual_ridge_residual_readout,
    _evaluate,
    _evaluate_answer_bridge_admission,
    _evaluate_answer_bridge_diagnostic,
    _evaluate_depth,
    _evaluate_process_admission,
    _freeze_dataset,
    _fresh_public_transition_acquisition,
    _generate_student_rollin,
    _gradient_conflict_diagnostics,
    _ground_state_value_embeddings,
    _initial_rollin_totals,
    _invocation_stop_step,
    _load_frozen_dataset,
    _load_latest_checkpoint,
    _masked_process_decisions,
    _mean_gradient_trees,
    _merge_bootstrap_action_literal_binding_extension,
    _merge_bootstrap_action_workspace_extension,
    _merge_bootstrap_causal_action_extension,
    _merge_bootstrap_codebook_extension,
    _merge_bootstrap_family_action_extension,
    _merge_bootstrap_initial_state_extension,
    _merge_bootstrap_process_reader_extension,
    _merge_bootstrap_scoped_lora_target_extension,
    _merge_bootstrap_transition_memory_extension,
    _merge_bootstrap_transition_opcode_expert_extension,
    _merge_bootstrap_transition_processor_extension,
    _merge_bootstrap_transition_replay_extension,
    _merge_bootstrap_transition_tape_reader_extension,
    _model_identity,
    _model_lane_purpose,
    _optimization_phase,
    _ownership_optimizer,
    _phase_gradients,
    _phase_schedule,
    _process_component_gradients,
    _process_family_training_batch,
    _process_training_policy,
    _rbf_residual_readout,
    _recurrent_training_task,
    _residual_hidden_size,
    _resolve_recurrent_window,
    _restore_checkpoint,
    _restore_rollin_totals,
    _rollin_report,
    _save_checkpoint,
    _semantic_execution_depth,
    _set_ownership_optimizer_rates,
    _streamed_recurrent_objective_gradients,
    _student_rollin_probability,
    _trainable,
    _training_halt_reason,
    _training_verdict,
)


def _codebook_extension_values() -> tuple[dict, dict]:
    parent_action = mx.zeros((8, 33, 2), dtype=mx.float32)
    child_action = mx.concatenate(
        (
            mx.concatenate(
                (
                    parent_action[0:1, :9],
                    mx.ones((1, 7, 2), dtype=mx.float32),
                    parent_action[0:1, 16:],
                ),
                axis=1,
            ),
            parent_action[1:],
        ),
        axis=0,
    )
    shared = {
        "controller.action_slot_embeddings": mx.zeros((8, 2)),
        "controller.literal_value_embeddings": mx.zeros((33, 2)),
        "controller.state_slot_embeddings": mx.zeros((5, 2)),
        "controller.state_value_embeddings": mx.zeros((5, 33, 2)),
        "controller.correction_a": mx.ones((2, 2)),
    }
    return (
        {**shared, "controller.action_value_embeddings": parent_action},
        {**shared, "controller.action_value_embeddings": child_action},
    )


def test_bootstrap_codebook_extension_replaces_only_new_opcode_rows() -> None:
    parent, child = _codebook_extension_values()
    migrated, receipt = _merge_bootstrap_codebook_extension(
        parent,
        child,
        mismatches=["state_codebook_sha256"],
        parent_identity={"state_codebook_sha256": "parent"},
        child_identity={"state_codebook_sha256": "child"},
    )

    assert receipt is not None
    assert receipt["value_start_inclusive"] == 9
    assert receipt["value_stop_exclusive"] == 31
    assert bool(
        mx.array_equal(
            migrated["controller.action_value_embeddings"][0, 9:31],
            child["controller.action_value_embeddings"][0, 9:31],
        )
    )
    assert bool(
        mx.array_equal(
            migrated["controller.action_value_embeddings"][0, :9],
            parent["controller.action_value_embeddings"][0, :9],
        )
    )
    assert bool(
        mx.array_equal(
            migrated["controller.correction_a"],
            parent["controller.correction_a"],
        )
    )


def test_bootstrap_codebook_extension_ignores_non_extension_child_drift() -> None:
    parent, child = _codebook_extension_values()
    changed = child["controller.action_value_embeddings"] + 0
    changed[0, 8] = 1
    child["controller.action_value_embeddings"] = changed

    migrated, receipt = _merge_bootstrap_codebook_extension(
        parent,
        child,
        mismatches=["state_codebook_sha256"],
        parent_identity={"state_codebook_sha256": "parent"},
        child_identity={"state_codebook_sha256": "child"},
    )

    assert receipt is not None
    assert bool(
        mx.array_equal(
            migrated["controller.action_value_embeddings"][0, 8],
            parent["controller.action_value_embeddings"][0, 8],
        )
    )


def test_bootstrap_codebook_extension_rejects_any_other_topology_drift() -> None:
    parent, child = _codebook_extension_values()
    with pytest.raises(RuntimeError, match="controller_rank"):
        _merge_bootstrap_codebook_extension(
            parent,
            child,
            mismatches=["state_codebook_sha256", "controller_rank"],
            parent_identity={"state_codebook_sha256": "parent"},
            child_identity={"state_codebook_sha256": "child"},
        )


def test_bootstrap_numeric_observation_extension_is_explicit_and_bounded() -> None:
    literal = {
        "schema": "aura.recurrent_literal_grounding.v1",
        "digit_token_ids": list(range(10)),
        "max_value": 32,
        "contract_sha256": "parent",
    }
    numeric = {
        **literal,
        "max_value": 960,
        "contract_sha256": "child",
        "encoding": "direct_category_then_ordered_radix_pair",
        "radix": 31,
    }

    receipt = _bootstrap_numeric_observation_extension(
        {"literal_observation_contract": literal},
        {"literal_observation_contract": literal, "numeric_observation_contract": numeric},
    )

    assert receipt == {
        "schema": "aura.unified_intrinsic.numeric_observation_extension.v1",
        "parent_max_value": 32,
        "child_max_value": 960,
        "digit_token_ids_preserved": True,
        "encoding": "direct_category_then_ordered_radix_pair",
        "radix": 31,
        "tensor_inventory_changed": False,
        "newly_observable_values": [33, 960],
    }

    with pytest.raises(RuntimeError, match="numeric observation extension"):
        _bootstrap_numeric_observation_extension(
            {"literal_observation_contract": literal},
            {
                "numeric_observation_contract": {
                    **numeric,
                    "digit_token_ids": list(range(1, 11)),
                }
            },
        )


def test_bootstrap_process_reader_extension_adds_only_new_reader_tensors() -> None:
    parent = {"controller.answer_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in PROCESS_READER_PARAMETER_NAMES
        },
    }

    migrated, receipt = _merge_bootstrap_process_reader_extension(parent, child)

    assert receipt is not None
    assert receipt["parent_tensor_inventory_preserved"] is True
    assert migrated["controller.answer_output"] is parent["controller.answer_output"]
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)


def test_bootstrap_process_reader_extension_rejects_partial_inventory() -> None:
    parent = {"controller.answer_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        "controller.process_reader_1_query": mx.ones((3, 2), dtype=mx.float32),
    }

    with pytest.raises(RuntimeError, match="tensor inventory differs"):
        _merge_bootstrap_process_reader_extension(parent, child)


def test_bootstrap_action_workspace_extension_is_exact_and_audited() -> None:
    parent = {"controller.action_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in ACTION_WORKSPACE_PARAMETER_NAMES
            if name != "action_workspace_output"
        },
        "controller.action_workspace_output": mx.zeros((2, 2), dtype=mx.float32),
    }

    migrated, receipt = _merge_bootstrap_action_workspace_extension(parent, child)

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["parent_tensor_inventory_preserved"] is True
    assert migrated["controller.action_output"] is parent["controller.action_output"]
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)


def test_bootstrap_transition_memory_extension_is_exact_and_audited() -> None:
    parent = {"controller.state_transition_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in TRANSITION_MEMORY_PARAMETER_NAMES
            if name != "transition_memory_output"
        },
        "controller.transition_memory_output": mx.zeros(
            (2, 2), dtype=mx.float32
        ),
    }

    migrated, receipt = _merge_bootstrap_transition_memory_extension(parent, child)

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["parent_tensor_inventory_preserved"] is True
    assert receipt["future_action_visible"] is False
    assert migrated["controller.state_transition_output"] is parent[
        "controller.state_transition_output"
    ]
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)


def test_bootstrap_transition_memory_rejects_partial_or_active_extension() -> None:
    child = {
        "controller.state_transition_output": mx.ones((2, 3), dtype=mx.float32),
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in TRANSITION_MEMORY_PARAMETER_NAMES
        },
    }
    partial_parent = {
        "controller.state_transition_output": child[
            "controller.state_transition_output"
        ],
        "controller.transition_memory_input": child[
            "controller.transition_memory_input"
        ],
    }
    with pytest.raises(RuntimeError, match="transition-memory inventory differs"):
        _merge_bootstrap_transition_memory_extension(partial_parent, child)

    parent = {
        "controller.state_transition_output": child[
            "controller.state_transition_output"
        ]
    }
    with pytest.raises(RuntimeError, match="transition memory is not a no-op"):
        _merge_bootstrap_transition_memory_extension(parent, child)


def test_bootstrap_transition_tape_reader_is_exact_and_audited() -> None:
    parent = {"controller.transition_memory_output": mx.ones((2, 3))}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in TRANSITION_TAPE_READER_PARAMETER_NAMES
            if name != "transition_tape_output"
        },
        "controller.transition_tape_output": mx.zeros((2, 2)),
    }

    migrated, receipt = _merge_bootstrap_transition_tape_reader_extension(
        parent, child
    )

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["current_prefix_retained_before_query"] is True
    assert receipt["future_action_visible"] is False
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)

    partial_parent = {
        **parent,
        "controller.transition_tape_key": child["controller.transition_tape_key"],
    }
    with pytest.raises(RuntimeError, match="transition-tape inventory differs"):
        _merge_bootstrap_transition_tape_reader_extension(partial_parent, child)

    active = dict(child)
    active["controller.transition_tape_output"] = mx.ones((2, 2))
    with pytest.raises(RuntimeError, match="tape reader is not a no-op"):
        _merge_bootstrap_transition_tape_reader_extension(parent, active)


def test_bootstrap_transition_processor_extension_is_exact_and_audited() -> None:
    parent = {"controller.state_transition_output": mx.ones((2, 3))}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in TRANSITION_PROCESSOR_PARAMETER_NAMES
            if name
            not in {
                "transition_processor_output",
                "transition_processor_state_cross_projection",
            }
        },
        "controller.transition_processor_state_cross_projection": mx.zeros((2, 2)),
        "controller.transition_processor_output": mx.zeros((2, 2)),
    }

    migrated, receipt = _merge_bootstrap_transition_processor_extension(
        parent, child
    )

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["category_identity"] == "exact_one_hot_or_deterministic_fourier"
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)

    incremental_parent = dict(child)
    del incremental_parent["controller.transition_processor_state_cross_projection"]
    incremental_parent["controller.transition_processor_output"] = mx.ones((2, 2))
    incremental, incremental_receipt = _merge_bootstrap_transition_processor_extension(
        incremental_parent,
        child,
    )
    assert incremental_receipt is not None
    assert incremental_receipt["new_tensor_names"] == [
        "controller.transition_processor_state_cross_projection"
    ]
    assert bool(
        mx.array_equal(
            incremental["controller.transition_processor_output"],
            incremental_parent["controller.transition_processor_output"],
        )
    )

    active_cross = dict(child)
    active_cross["controller.transition_processor_state_cross_projection"] = mx.ones(
        (2, 2)
    )
    with pytest.raises(RuntimeError, match="cross-register tissue is not a no-op"):
        _merge_bootstrap_transition_processor_extension(
            incremental_parent,
            active_cross,
        )


def test_bootstrap_transition_processor_rejects_partial_or_active_extension() -> None:
    child = {
        "controller.state_transition_output": mx.ones((2, 3)),
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in TRANSITION_PROCESSOR_PARAMETER_NAMES
        },
    }
    partial_parent = {
        "controller.state_transition_output": child[
            "controller.state_transition_output"
        ],
        "controller.transition_processor_action_left": child[
            "controller.transition_processor_action_left"
        ],
    }
    with pytest.raises(RuntimeError, match="transition-processor inventory differs"):
        _merge_bootstrap_transition_processor_extension(partial_parent, child)

    parent = {
        "controller.state_transition_output": child[
            "controller.state_transition_output"
        ]
    }
    with pytest.raises(RuntimeError, match="transition processor is not a no-op"):
        _merge_bootstrap_transition_processor_extension(parent, child)


def test_bootstrap_transition_opcode_experts_are_exact_and_audited() -> None:
    parent = {"controller.transition_processor_output": mx.ones((2, 3))}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.zeros((4, 2, 3), dtype=mx.float32)
            for name in TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES
        },
    }

    migrated, receipt = _merge_bootstrap_transition_opcode_expert_extension(
        parent, child
    )

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)

    incremental_parent = dict(child)
    del incremental_parent["controller.transition_processor_opcode_hidden"]
    incremental, incremental_receipt = (
        _merge_bootstrap_transition_opcode_expert_extension(
            incremental_parent,
            child,
        )
    )
    assert incremental_receipt is not None
    assert incremental_receipt["matched_capacity_control"] == (
        "uniform_public_opcode_router"
    )
    assert set(incremental) == set(child)

    active = dict(child)
    active["controller.transition_processor_opcode_hidden"] = mx.ones((4, 2, 3))
    with pytest.raises(RuntimeError, match="expert is not a no-op"):
        _merge_bootstrap_transition_opcode_expert_extension(parent, active)

    random_basis = dict(child)
    random_basis["controller.transition_processor_opcode_interaction_up"] = (
        mx.random.normal((4, 2, 3), key=mx.random.key(498))
    )
    migrated_basis, basis_receipt = (
        _merge_bootstrap_transition_opcode_expert_extension(parent, random_basis)
    )
    assert basis_receipt is not None
    assert set(migrated_basis) == set(random_basis)

    behavior_changing = dict(random_basis)
    behavior_changing["controller.transition_processor_opcode_interaction_down"] = (
        mx.ones((4, 2, 3), dtype=mx.float32)
    )
    with pytest.raises(RuntimeError, match="expert is not a no-op"):
        _merge_bootstrap_transition_opcode_expert_extension(
            parent,
            behavior_changing,
        )


def test_bootstrap_transition_replay_is_exact_and_audited() -> None:
    parent = {"controller.transition_processor_output": mx.ones((2, 3))}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.zeros((2, 3), dtype=mx.float32)
            for name in TRANSITION_REPLAY_PARAMETER_NAMES
        },
    }

    migrated, receipt = _merge_bootstrap_transition_replay_extension(parent, child)

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["private_transition_trace_visible"] is False
    assert set(migrated) == set(child)

    partial = dict(parent)
    partial["controller.transition_replay_key"] = child[
        "controller.transition_replay_key"
    ]
    with pytest.raises(RuntimeError, match="transition-replay inventory differs"):
        _merge_bootstrap_transition_replay_extension(partial, child)

    active = dict(child)
    active["controller.transition_replay_output"] = mx.ones((2, 3))
    with pytest.raises(RuntimeError, match="replay is not a no-op"):
        _merge_bootstrap_transition_replay_extension(parent, active)


def test_bootstrap_scoped_lora_query_extension_is_exact_and_audited() -> None:
    parent = {
        "model.model.layers.7.self_attn.o_proj.lora_a": mx.ones((3, 2)),
        "model.model.layers.7.self_attn.o_proj.lora_b": mx.zeros((2, 3)),
    }
    child = {
        **parent,
        "model.model.layers.7.self_attn.q_proj.lora_a": mx.ones((3, 2)),
        "model.model.layers.7.self_attn.q_proj.lora_b": mx.zeros((2, 3)),
        "model.model.layers.7.self_attn.q_proj.continuous_depth_a.0": mx.ones((3, 2)),
        "model.model.layers.7.self_attn.q_proj.continuous_depth_b.0": mx.zeros((2, 3)),
    }

    migrated, receipt, remaining = _merge_bootstrap_scoped_lora_target_extension(
        parent,
        child,
        parent_identity={"lora_targets": ["o_proj", "v_proj"]},
        child_identity={"lora_targets": ["q_proj", "o_proj", "v_proj"]},
        mismatches=["lora_targets"],
    )

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["added_targets"] == ["q_proj"]
    assert set(migrated) == set(child)
    assert remaining == []


def test_bootstrap_scoped_lora_query_extension_rejects_active_or_broad_change() -> None:
    parent = {
        "model.model.layers.7.self_attn.o_proj.lora_a": mx.ones((3, 2)),
        "model.model.layers.7.self_attn.o_proj.lora_b": mx.zeros((2, 3)),
    }
    active = {
        **parent,
        "model.model.layers.7.self_attn.q_proj.lora_a": mx.ones((3, 2)),
        "model.model.layers.7.self_attn.q_proj.lora_b": mx.ones((2, 3)),
    }
    with pytest.raises(RuntimeError, match="LoRA target is not a no-op"):
        _merge_bootstrap_scoped_lora_target_extension(
            parent,
            active,
            parent_identity={"lora_targets": ["o_proj", "v_proj"]},
            child_identity={"lora_targets": ["q_proj", "o_proj", "v_proj"]},
            mismatches=["lora_targets"],
        )

    with pytest.raises(RuntimeError, match="LoRA targets differ"):
        _merge_bootstrap_scoped_lora_target_extension(
            parent,
            active,
            parent_identity={"lora_targets": ["o_proj", "v_proj"]},
            child_identity={"lora_targets": ["q_proj", "k_proj", "o_proj", "v_proj"]},
            mismatches=["lora_targets"],
        )


def test_bootstrap_action_workspace_rejects_partial_or_active_extension() -> None:
    child = {
        "controller.action_output": mx.ones((2, 3), dtype=mx.float32),
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in ACTION_WORKSPACE_PARAMETER_NAMES
        },
    }
    partial_parent = {
        "controller.action_output": child["controller.action_output"],
        "controller.action_workspace_seed": child["controller.action_workspace_seed"],
    }
    with pytest.raises(RuntimeError, match="action-workspace inventory differs"):
        _merge_bootstrap_action_workspace_extension(partial_parent, child)

    parent = {"controller.action_output": child["controller.action_output"]}
    with pytest.raises(RuntimeError, match="action workspace is not a no-op"):
        _merge_bootstrap_action_workspace_extension(parent, child)


def test_bootstrap_causal_action_extension_is_exact_and_audited() -> None:
    parent = {"controller.action_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in CAUSAL_ACTION_PARAMETER_NAMES
            if name != "action_causal_output"
        },
        "controller.action_causal_output": mx.zeros((2, 2), dtype=mx.float32),
    }

    migrated, receipt = _merge_bootstrap_causal_action_extension(parent, child)

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["future_field_teacher_leakage"] is False
    assert migrated["controller.action_output"] is parent["controller.action_output"]
    assert set(migrated) == set(child)
    assert set(receipt["new_tensor_names"]) == set(child) - set(parent)


def test_bootstrap_causal_action_rejects_partial_or_active_extension() -> None:
    child = {
        "controller.action_output": mx.ones((2, 3), dtype=mx.float32),
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in CAUSAL_ACTION_PARAMETER_NAMES
        },
    }
    partial_parent = {
        "controller.action_output": child["controller.action_output"],
        "controller.action_causal_value_embeddings": child[
            "controller.action_causal_value_embeddings"
        ],
    }
    with pytest.raises(RuntimeError, match="causal-action inventory differs"):
        _merge_bootstrap_causal_action_extension(partial_parent, child)

    parent = {"controller.action_output": child["controller.action_output"]}
    with pytest.raises(RuntimeError, match="not a no-op"):
        _merge_bootstrap_causal_action_extension(parent, child)


def test_bootstrap_family_action_extension_is_exact_and_rejects_active_output() -> None:
    parent = {"controller.action_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.zeros((7, 8, 2, 33), dtype=mx.float32)
            for name in FAMILY_ACTION_PARAMETER_NAMES
        },
    }
    migrated, receipt = _merge_bootstrap_family_action_extension(parent, child)
    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["private_transition_program_visible"] is False
    assert set(migrated) == set(child)

    active = {
        **child,
        "controller.action_family_output": mx.ones((7, 8, 2, 33)),
    }
    with pytest.raises(RuntimeError, match="experts are not a no-op"):
        _merge_bootstrap_family_action_extension(parent, active)


def test_bootstrap_family_action_extension_can_add_one_new_additive_tensor() -> None:
    bias_name = "controller.action_family_bias"
    child = {
        f"controller.{name}": mx.zeros((1,), dtype=mx.float32)
        for name in FAMILY_ACTION_PARAMETER_NAMES
    }
    parent = {name: value for name, value in child.items() if name != bias_name}
    migrated, receipt = _merge_bootstrap_family_action_extension(
        parent,
        child,
    )
    assert set(migrated) == set(child)
    assert receipt is not None
    assert receipt["new_tensor_names"] == [bias_name]


def test_bootstrap_action_literal_binding_is_exact_and_rejects_active_output() -> None:
    parent = {"controller.action_output": mx.ones((2, 3), dtype=mx.float32)}
    child = {
        **parent,
        **{
            f"controller.{name}": mx.ones((2, 2), dtype=mx.float32)
            for name in ACTION_LITERAL_BINDING_PARAMETER_NAMES
            if name != "action_literal_binding_output"
        },
        "controller.action_literal_binding_output": mx.zeros(
            (8, 6), dtype=mx.float32
        ),
        "controller.action_literal_binding_family_output": mx.zeros(
            (7, 8, 6), dtype=mx.float32
        ),
    }
    migrated, receipt = _merge_bootstrap_action_literal_binding_extension(parent, child)
    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    assert receipt["private_transition_program_visible"] is False
    assert set(migrated) == set(child)

    partial = {
        **parent,
        "controller.action_literal_binding_query": child[
            "controller.action_literal_binding_query"
        ],
    }
    with pytest.raises(RuntimeError, match="action-literal-binding inventory differs"):
        _merge_bootstrap_action_literal_binding_extension(partial, child)

    active = {
        **child,
        "controller.action_literal_binding_family_output": mx.ones((7, 8, 6)),
    }
    with pytest.raises(RuntimeError, match="binding is not a no-op"):
        _merge_bootstrap_action_literal_binding_extension(parent, active)

    active_global = {
        **child,
        "controller.action_literal_binding_output": mx.ones((8, 6)),
    }
    with pytest.raises(RuntimeError, match="binding is not a no-op"):
        _merge_bootstrap_action_literal_binding_extension(parent, active_global)

    legacy_parent = {
        key: value
        for key, value in child.items()
        if key != "controller.action_literal_binding_family_output"
    }
    legacy_parent["controller.action_literal_binding_output"] = mx.ones((8, 6))
    migrated_legacy, legacy_receipt = (
        _merge_bootstrap_action_literal_binding_extension(legacy_parent, child)
    )
    assert legacy_receipt is not None
    assert bool(
        mx.array_equal(
            migrated_legacy["controller.action_literal_binding_output"],
            legacy_parent["controller.action_literal_binding_output"],
        )
    )


def test_bootstrap_initial_state_extension_copies_legacy_transition_exactly() -> None:
    sources = {
        "initial_state_query": "state_transition_query",
        "initial_state_key": "state_transition_key",
        "initial_state_value": "state_transition_value",
        "initial_state_output": "state_transition_output",
        "initial_state_bias": "state_transition_bias",
        "initial_state_literal_copy_logit": "state_literal_copy_logit",
    }
    parent = {
        f"controller.{source}": mx.full((2, 2), index + 1.0)
        for index, source in enumerate(sources.values())
    }
    child = {
        **parent,
        **{f"controller.{name}": mx.zeros((2, 2)) for name in INITIAL_STATE_PARAMETER_NAMES},
    }

    migrated, receipt = _merge_bootstrap_initial_state_extension(parent, child)

    assert receipt is not None
    assert receipt["behavior_before_training_preserved"] is True
    for name, source in sources.items():
        assert bool(
            mx.array_equal(
                migrated[f"controller.{name}"],
                parent[f"controller.{source}"],
            )
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
        "state_transition_trains_shared_process_parser": False,
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


def test_only_fresh_controller_with_public_direct_actions_skips_process_bootstrap() -> None:
    assert _fresh_public_transition_acquisition(
        window_tissue_mode="controller_only",
        public_action_program=True,
        direct_transition_processor=True,
    )
    assert not _fresh_public_transition_acquisition(
        window_tissue_mode="scoped_lora",
        public_action_program=True,
        direct_transition_processor=True,
    )
    assert not _fresh_public_transition_acquisition(
        window_tissue_mode="controller_only",
        public_action_program=False,
        direct_transition_processor=True,
    )
    assert not _fresh_public_transition_acquisition(
        window_tissue_mode="controller_only",
        public_action_program=True,
        direct_transition_processor=False,
    )


def test_streamed_recurrent_gradients_equal_monolithic_objective() -> None:
    from core.learning.recurrence_curriculum import task_battery

    original, _wiring = _bundle()
    bundle = UnifiedTrainingBundle(
        original.model,
        UnifiedRecurrentController(
            UnifiedRecurrenceConfig(
                hidden_size=32,
                correction_rank=4,
                literal_digit_token_ids=tuple(range(10)),
            )
        ),
    )
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2, 4), (8, 16))
    task = task_battery(("khop",), (4,), 1, seed=20260812)[0]
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    answer = mx.array([[4, 5]], dtype=mx.int32)
    readout = "0" * 64

    def monolithic(candidate: UnifiedTrainingBundle) -> object:
        return unified_intrinsic_training_loss(
            candidate.model,
            prompt,
            answer,
            candidate.controller,
            spec,
            readout_sha256=readout,
            decoder_input_tokens=answer,
            transition_trace=task.transition_trace,
            transition_program=task.transition_program,
        )[0]

    expected_loss, expected_gradients = nn.value_and_grad(bundle, monolithic)(bundle)
    mx.eval(expected_loss, expected_gradients)
    reclaims: list[bool] = []
    envelope = type(
        "Envelope",
        (),
        {"reclaim": lambda _self, *_args, **kwargs: reclaims.append(kwargs.get("force") is True)},
    )()
    observed_loss, observed_gradients = _streamed_recurrent_objective_gradients(
        bundle,
        prompt,
        answer,
        spec,
        readout_sha256=readout,
        decoder_input_tokens=answer,
        transition_trace=task.transition_trace,
        transition_program=task.transition_program,
        state_teacher_forcing_probability=0.0,
        envelope=envelope,
    )
    mx.eval(observed_loss, observed_gradients)

    assert float(observed_loss.item()) == pytest.approx(float(expected_loss.item()), abs=1e-5)
    expected = dict(tree_flatten(expected_gradients))
    observed = dict(tree_flatten(observed_gradients))
    assert observed.keys() == expected.keys()
    for name in expected:
        matches = mx.allclose(observed[name], expected[name], rtol=2e-4, atol=2e-5)
        assert matches.item(), name
    assert reclaims == [True] * len(spec.train_depths)


def test_invocation_boundary_is_operational_and_resumable() -> None:
    assert _invocation_stop_step(0, 73, None) == 73
    assert _invocation_stop_step(0, 73, 3) == 3
    assert _invocation_stop_step(3, 73, 3) == 6
    assert _invocation_stop_step(72, 73, 3) == 73
    assert (
        _training_halt_reason(step=3, max_steps=73, invocation_stop_step=3)
        == "invocation_step_limit"
    )
    assert _training_halt_reason(step=3, max_steps=73, invocation_stop_step=73) == "wall_clock"
    assert _training_halt_reason(step=73, max_steps=73, invocation_stop_step=73) == "max_steps"
    with pytest.raises(ValueError, match="must be positive"):
        _invocation_stop_step(0, 73, 0)


def test_training_verdict_never_promotes_an_incomplete_invocation() -> None:
    final = {"heldout_depth_helps": True, "trained_depth_helps": True}

    assert (
        _training_verdict(
            complete=False,
            answer_bridge_admission=None,
            process_admission=None,
            final=final,
        )
        == "incomplete_checkpoint"
    )
    assert (
        _training_verdict(
            complete=True,
            answer_bridge_admission={"admitted": False},
            process_admission=None,
            final=final,
        )
        == "answer_bridge_not_admitted"
    )
    assert (
        _training_verdict(
            complete=True,
            answer_bridge_admission={"admitted": True},
            process_admission={"admitted": True},
            final=final,
        )
        == "heldout_depth_gain"
    )
    assert (
        _training_verdict(
            complete=True,
            answer_bridge_admission=None,
            process_admission={"admitted": False},
            final=final,
        )
        == "autonomous_process_not_admitted"
    )


def test_phase_schedule_allows_only_bootstrapped_bridge_only_adaptation(tmp_path) -> None:
    bootstrap = tmp_path / "parent"
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=0,
        answer_bridge_steps=84,
        max_steps=84,
        bootstrap_output_dir=bootstrap,
    )

    assert schedule == {
        "schema": "aura.unified_intrinsic.phase_schedule.v1",
        "mode": "bootstrap_answer_bridge_only",
        "semantic_anchor_steps": 0,
        "state_transition_steps": 0,
        "answer_bridge_steps": 84,
        "recurrence_steps": 0,
        "max_steps": 84,
        "bootstrap_required": True,
    }
    with pytest.raises(ValueError, match="bootstrapped answer-bridge"):
        _phase_schedule(
            semantic_warmup_steps=0,
            state_warmup_steps=0,
            answer_bridge_steps=84,
            max_steps=84,
            bootstrap_output_dir=None,
        )
    with pytest.raises(ValueError, match="exceed maximum"):
        _phase_schedule(
            semantic_warmup_steps=0,
            state_warmup_steps=1,
            answer_bridge_steps=84,
            max_steps=84,
            bootstrap_output_dir=bootstrap,
        )


def test_phase_schedule_allows_explicit_process_only_acquisition() -> None:
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=280,
        answer_bridge_steps=0,
        max_steps=280,
        bootstrap_output_dir=None,
        process_only=True,
    )

    assert schedule == {
        "schema": "aura.unified_intrinsic.phase_schedule.v1",
        "mode": "process_acquisition_only",
        "semantic_anchor_steps": 0,
        "state_transition_steps": 280,
        "answer_bridge_steps": 0,
        "recurrence_steps": 0,
        "max_steps": 280,
        "bootstrap_required": False,
    }
    with pytest.raises(ValueError, match="process-only acquisition"):
        _phase_schedule(
            semantic_warmup_steps=1,
            state_warmup_steps=279,
            answer_bridge_steps=0,
            max_steps=280,
            bootstrap_output_dir=None,
            process_only=True,
        )


def test_phase_schedule_allows_only_explicit_bootstrapped_process_acquisition(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "parent"
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=112,
        answer_bridge_steps=0,
        max_steps=112,
        bootstrap_output_dir=bootstrap,
        process_only=True,
        process_bootstrap=True,
    )

    assert schedule["mode"] == "bootstrap_process_acquisition_only"
    assert schedule["bootstrap_required"] is True
    with pytest.raises(ValueError, match="process-only acquisition"):
        _phase_schedule(
            semantic_warmup_steps=0,
            state_warmup_steps=112,
            answer_bridge_steps=0,
            max_steps=112,
            bootstrap_output_dir=None,
            process_only=True,
            process_bootstrap=True,
        )


def test_phase_schedule_admits_bootstrapped_factorized_process_completion(
    tmp_path: Path,
) -> None:
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=256,
        answer_bridge_steps=0,
        max_steps=256,
        bootstrap_output_dir=tmp_path / "parent",
        process_only=True,
        process_bootstrap=True,
    )

    assert schedule == {
        "schema": "aura.unified_intrinsic.phase_schedule.v1",
        "mode": "bootstrap_process_acquisition_only",
        "semantic_anchor_steps": 0,
        "state_transition_steps": 256,
        "answer_bridge_steps": 0,
        "recurrence_steps": 0,
        "max_steps": 256,
        "bootstrap_required": True,
    }


def test_factorized_process_curriculum_owns_each_stage_and_removes_teacher() -> None:
    expected = {
        0: ("initializer", 1.0),
        34: ("initializer", 1.0),
        35: ("action", 1.0),
        139: ("action", 1.0),
        140: ("transition", 1.0),
        244: ("transition", 1.0),
        245: ("joint", 1.0),
        279: ("joint", 0.0),
    }
    for step, (component, teacher_probability) in expected.items():
        policy = _process_training_policy(step, 280, "factorized")
        assert policy["component"] == component
        assert policy["teacher_forcing_probability"] == pytest.approx(teacher_probability)
        assert 0.0 < policy["stage_progress"] <= 1.0

    assert _process_training_policy(4, 8, "joint") == {
        "component": "joint",
        "teacher_forcing_probability": 1.0,
        "stage_progress": 0.625,
    }
    assert _process_training_policy(4, 8, "action_workspace") == {
        "component": "action_workspace",
        "teacher_forcing_probability": 1.0,
        "stage_progress": 0.625,
    }
    assert _process_training_policy(6, 8, "action_workspace") == {
        "component": "action_workspace",
        "teacher_forcing_probability": 0.5,
        "stage_progress": 0.875,
    }
    assert _process_training_policy(7, 8, "action_workspace") == {
        "component": "action_workspace",
        "teacher_forcing_probability": 0.0,
        "stage_progress": 1.0,
    }
    assert _process_training_policy(3, 8, "transition_only") == {
        "component": "transition",
        "teacher_forcing_probability": pytest.approx(4 / 7),
        "stage_progress": 0.5,
    }
    assert _process_training_policy(0, 8, "transition_only")[
        "teacher_forcing_probability"
    ] == pytest.approx(1.0)
    assert _process_training_policy(7, 8, "transition_only")[
        "teacher_forcing_probability"
    ] == pytest.approx(0.0)
    held = [
        _process_training_policy(
            step,
            8,
            "transition_only",
            teacher_hold_fraction=0.375,
        )["teacher_forcing_probability"]
        for step in range(8)
    ]
    assert held[:3] == [1.0, 1.0, 1.0]
    assert held[3] == 1.0
    assert held[-1] == 0.0
    assert held[3:] == sorted(held[3:], reverse=True)
    assert _process_training_policy(
        3,
        8,
        "transition_only",
        initial_teacher_probability=0.8,
        final_teacher_probability=0.2,
    )["teacher_forcing_probability"] == pytest.approx(0.8 - 3 * 0.6 / 7)
    with pytest.raises(ValueError, match="teacher-forcing schedule"):
        _process_training_policy(
            0,
            8,
            "transition_only",
            initial_teacher_probability=0.2,
            final_teacher_probability=0.8,
        )
    with pytest.raises(ValueError, match="teacher-forcing schedule"):
        _process_training_policy(
            0,
            8,
            "transition_only",
            teacher_hold_fraction=1.0,
        )
    with pytest.raises(ValueError, match="too short"):
        _process_training_policy(0, 7, "factorized")


def test_direct_transition_curriculum_reaches_every_scale_then_closed_loop() -> None:
    stages = [
        _direct_transition_curriculum_window(step, 20, 10, mode="progressive")
        for step in range(20)
    ]

    assert {row["stage"] for row in stages[:3]} == {"verified_window_1"}
    assert {row["stage"] for row in stages[3:6]} == {"verified_window_2"}
    assert {row["stage"] for row in stages[6:9]} == {"verified_window_4"}
    assert {row["stage"] for row in stages[9:14]} == {"closed_loop"}
    assert {row["stage"] for row in stages[14:17]} == {"controlled_recovery"}
    assert {row["stage"] for row in stages[17:]} == {"closed_loop"}
    assert all(row["complete_public_prefix_visible"] for row in stages)
    assert any(row["training_only_midtrace_initial_state"] for row in stages[:9])
    assert all(row["transition_start"] == 0 for row in stages[9:])
    assert all(row["transition_count"] == 10 for row in stages[9:])
    assert all(row["corrupt_transition"] is not None for row in stages[14:17])
    assert all(
        row["corrupt_state_mode"] == "coherent_trace_state"
        for row in stages[14:17]
    )
    assert all(row["corrupt_state_slot"] is None for row in stages[14:17])
    assert all(row["corrupt_state_offset"] is not None for row in stages[14:17])
    assert all(row["corrupt_transition"] is None for row in stages[17:])
    assert _direct_transition_curriculum_window(
        3, 8, 6, mode="closed_loop"
    ) == {
        "stage": "closed_loop",
        "transition_start": 0,
        "transition_count": 6,
        "training_only_midtrace_initial_state": False,
        "complete_public_prefix_visible": True,
        "corrupt_transition": None,
        "corrupt_state_mode": None,
        "corrupt_state_slot": None,
        "corrupt_state_offset": None,
    }


def test_dual_ridge_residual_readout_writes_exact_training_decision() -> None:
    import numpy as np

    features = np.asarray(
        [[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    base = np.zeros((4, 3), dtype=np.float32)
    labels = np.asarray([0, 0, 2, 2], dtype=np.int64)
    weight, bias, report = _dual_ridge_residual_readout(
        features,
        base,
        labels,
        regularization=1e-4,
        margin=8.0,
    )
    fitted = base + features @ weight + bias
    assert np.array_equal(np.argmax(fitted, axis=1), labels)
    assert report["after_accuracy"] == 1.0


def test_rbf_residual_readout_separates_nonlinear_training_cells() -> None:
    import numpy as np

    features = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    base = np.zeros((4, 2), dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    parameters, report = _rbf_residual_readout(
        features,
        base,
        labels,
        capacity=8,
        regularization=1e-4,
        margin=8.0,
    )
    normalized = (features - parameters["mean"]) * parameters["inv_scale"]
    distance = np.mean(
        np.square(normalized[:, None, :] - parameters["prototypes"][None, :, :]),
        axis=-1,
    )
    kernel = (
        np.exp(-parameters["gamma"] * distance)
        * parameters["mask"][None, :]
    )
    fitted = base + kernel @ parameters["coefficients"]
    assert np.array_equal(np.argmax(fitted, axis=1), labels)
    assert report["after_accuracy"] == 1.0


def test_process_component_gradients_prevent_cross_role_rewrites() -> None:
    gradients = {
        "model": {
            "layer": {"lora_a": mx.ones((2, 2))},
            "block": {
                "self_attn": {"q_proj": {"lora_b": mx.ones((2, 2))}}
            },
        },
        "controller": {
            "initial_state_output": mx.ones((2, 2)),
            "action_output": mx.ones((2, 2)),
            "action_workspace_output": mx.ones((2, 2)),
            "action_causal_output": mx.ones((2, 2)),
            "state_transition_output": mx.ones((2, 2)),
            "transition_memory_output": mx.ones((2, 2)),
            "transition_processor_output": mx.ones((2, 2)),
            "transition_processor_opcode_output": mx.ones((2, 2)),
            "answer_output": mx.ones((2, 2)),
        },
    }
    expected = {
        "initializer": {
            "model.block.self_attn.q_proj.lora_b",
            "model.layer.lora_a",
            "controller.initial_state_output",
        },
        "action": {
            "model.block.self_attn.q_proj.lora_b",
            "model.layer.lora_a",
            "controller.action_output",
            "controller.action_workspace_output",
            "controller.action_causal_output",
        },
        "action_workspace": {
            "model.block.self_attn.q_proj.lora_b",
            "model.layer.lora_a",
            "controller.action_workspace_output",
            "controller.action_causal_output",
        },
        "transition": {
            "model.block.self_attn.q_proj.lora_b",
            "model.layer.lora_a",
            "controller.state_transition_output",
            "controller.transition_memory_output",
            "controller.transition_processor_output",
            "controller.transition_processor_opcode_output",
        },
        "joint": {
            "model.block.self_attn.q_proj.lora_b",
            "model.layer.lora_a",
            "controller.initial_state_output",
            "controller.action_output",
            "controller.action_workspace_output",
            "controller.action_causal_output",
            "controller.state_transition_output",
            "controller.transition_memory_output",
            "controller.transition_processor_output",
            "controller.transition_processor_opcode_output",
        },
    }
    for component, live_names in expected.items():
        masked = dict(tree_flatten(_process_component_gradients(gradients, component)))
        assert {name for name, value in masked.items() if bool(mx.any(value != 0))} == live_names

def test_bridge_only_preflight_requires_exact_autonomous_process(tmp_path) -> None:
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=0,
        answer_bridge_steps=84,
        max_steps=84,
        bootstrap_output_dir=tmp_path / "parent",
    )
    diagnostic_body = {
        "schema": "aura.unified_intrinsic.answer_bridge_diagnostic.v1",
        "tasks": 7,
        "autonomous_process_exact": 6,
        "oracle_exact": 7,
        "autonomous_exact": 6,
        "sham_exact": 0,
        "diagnosis": "recurrent_process_limited",
    }
    diagnostic = {
        **diagnostic_body,
        "diagnostic_sha256": _canonical_sha256(diagnostic_body),
    }

    refused = _answer_bridge_process_preflight(
        diagnostic,
        identity_sha256="b" * 64,
        phase_schedule=schedule,
        start_step=0,
    )

    assert refused["admitted"] is False
    assert refused["optimizer_steps_executed"] == 0
    assert refused["reason"] == "autonomous_process_not_exact_train_process_before_bridge"
    diagnostic_body["autonomous_process_exact"] = 7
    diagnostic = {
        **diagnostic_body,
        "diagnostic_sha256": _canonical_sha256(diagnostic_body),
    }
    admitted = _answer_bridge_process_preflight(
        diagnostic,
        identity_sha256="b" * 64,
        phase_schedule=schedule,
        start_step=0,
    )
    assert admitted["admitted"] is True
    assert admitted["reason"] == "autonomous_process_exact"


def test_bridge_process_preflight_rejects_non_bridge_schedule() -> None:
    with pytest.raises(ValueError, match="bridge-only schedule"):
        _answer_bridge_process_preflight(
            {
                "tasks": 1,
                "autonomous_process_exact": 1,
                "diagnostic_sha256": "a" * 64,
            },
            identity_sha256="b" * 64,
            phase_schedule={"mode": "recurrent_training"},
            start_step=0,
        )


def test_bridge_process_preflight_rejects_resealed_diagnostic(tmp_path) -> None:
    schedule = _phase_schedule(
        semantic_warmup_steps=0,
        state_warmup_steps=0,
        answer_bridge_steps=1,
        max_steps=1,
        bootstrap_output_dir=tmp_path / "parent",
    )
    with pytest.raises(ValueError, match="diagnostic was resealed"):
        _answer_bridge_process_preflight(
            {
                "tasks": 1,
                "autonomous_process_exact": 1,
                "diagnostic_sha256": "a" * 64,
            },
            identity_sha256="b" * 64,
            phase_schedule=schedule,
            start_step=0,
        )


def test_rollin_telemetry_round_trips_and_rejects_invalid_state() -> None:
    totals = _initial_rollin_totals()
    totals["examples"] = 7
    totals["max_preclip_gradient_norm"] = 2.5
    totals["max_preclip_gradient_norms"] = {"recurrent_controller": 2.5}
    totals["last_process_component"] = "action_workspace"
    restored = _restore_rollin_totals({"rollin_totals": totals})
    assert restored == totals
    assert restored is not totals
    assert restored["max_preclip_gradient_norms"] is not totals["max_preclip_gradient_norms"]
    totals["last_probability"] = float("nan")
    with pytest.raises(RuntimeError, match="probability differs"):
        _restore_rollin_totals({"rollin_totals": totals})

    totals["last_probability"] = None
    totals["last_process_component"] = "unregistered_component"
    with pytest.raises(RuntimeError, match="process component differs"):
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
    assert "core/learning/frontier_process_supervision.py" in TRAINING_SOURCE_FILES
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
            return [1 + (sum(text.encode("ascii")) + index) % 62 for index in range(width)]

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
        family_markers=(
            ("khop", (70,)),
            ("modular", (71,)),
            ("register_trace", (72,)),
        ),
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


def test_answer_binding_targets_leave_general_answer_schemas_semantic_only() -> None:
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

    assert (
        _answer_role_place_targets(
            "frontier_scientific_inference",
            mx.array([[21, 22, 23, 99]]),
            contract,
        )
        is None
    )


def test_answer_bridge_schedule_covers_each_family_before_repeating() -> None:
    tasks = [
        type(
            "Task",
            (),
            {
                "family": family,
                "depth": depth,
                "task_id": f"{family}-{depth}",
            },
        )()
        for family in ("modular", "khop", "register_trace")
        for depth in (1, 2, 4)
    ]
    assert [_answer_bridge_task(tasks, index).family for index in range(3)] == [
        "khop",
        "modular",
        "register_trace",
    ]
    assert {
        (_answer_bridge_task(tasks, index).family, _answer_bridge_task(tasks, index).depth)
        for index in range(9)
    } == {
        (family, depth) for family in ("khop", "modular", "register_trace") for depth in (1, 2, 4)
    }


def test_answer_bridge_schedule_covers_every_example_before_repetition() -> None:
    tasks = [
        type(
            "Task",
            (),
            {
                "family": family,
                "depth": depth,
                "task_id": f"{family}-{depth}-{example}",
            },
        )()
        for family in ("modular", "khop", "register_trace")
        for depth in (1, 2, 4)
        for example in range(8)
    ]

    scheduled = [_answer_bridge_task(tasks, index) for index in range(len(tasks))]

    assert len({task.task_id for task in scheduled}) == len(tasks)
    assert _answer_bridge_task(tasks, len(tasks)).task_id == scheduled[0].task_id


def test_recurrent_schedule_uses_max_depth_in_memory_cost_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        type(
            "Task",
            (),
            {
                "family": family,
                "task_id": task_id,
                "depth": depth,
            },
        )()
        for family, task_id, depth in (
            ("khop", "shallow", 2),
            ("register_trace", "large", 4),
            ("khop", "medium", 4),
            ("modular", "small", 4),
        )
    ]
    lengths = {
        "shallow": (2, 1),
        "large": (20, 5),
        "medium": (10, 2),
        "small": (6, 2),
    }
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda _tokenizer, task, _bridge: (
            mx.zeros((1, lengths[task.task_id][0]), dtype=mx.int32),
            mx.zeros((1, lengths[task.task_id][1]), dtype=mx.int32),
        ),
    )

    scheduled = [_recurrent_training_task(tasks, object(), "", index).task_id for index in range(4)]

    assert scheduled == ["small", "medium", "large", "small"]


def test_broad_recurrent_schedule_covers_every_natural_process_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        type(
            "Task",
            (),
            {"family": family, "task_id": task_id, "depth": depth},
        )()
        for family, task_id, depth in (
            ("frontier_calibration", "calibration", 3),
            ("frontier_scientific_inference", "science", 4),
            ("frontier_coding", "coding", 12),
        )
    ]
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda _tokenizer, task, _bridge: (
            mx.zeros((1, task.depth), dtype=mx.int32),
            mx.zeros((1, 1), dtype=mx.int32),
        ),
    )
    scheduled = [
        _recurrent_training_task(
            tasks,
            object(),
            "",
            index,
            cover_all_cells=True,
        ).task_id
        for index in range(3)
    ]
    assert scheduled == ["calibration", "science", "coding"]


def test_process_family_training_batch_cooptimizes_siblings_before_repetition() -> None:
    tasks = [
        type(
            "Task",
            (),
            {"family": family, "task_id": f"{family}-{index}"},
        )()
        for family in ("frontier_coding", "frontier_mathematics")
        for index in range(4)
    ]

    batches = [_process_family_training_batch(tasks, update, 2) for update in range(4)]

    assert [[task.task_id for task in batch] for batch in batches] == [
        ["frontier_coding-0", "frontier_coding-1"],
        ["frontier_mathematics-0", "frontier_mathematics-1"],
        ["frontier_coding-2", "frontier_coding-3"],
        ["frontier_mathematics-2", "frontier_mathematics-3"],
    ]
    assert all(len({task.family for task in batch}) == 1 for batch in batches)


def test_process_family_training_batch_balances_shared_tissue() -> None:
    tasks = [
        type("Task", (), {"family": family, "task_id": f"{family}-{index}"})()
        for family in ("coding", "science", "planning")
        for index in range(2)
    ]

    batches = [
        _process_family_training_batch(
            tasks,
            update,
            3,
            mode="balanced_families",
        )
        for update in range(2)
    ]

    assert [[task.task_id for task in batch] for batch in batches] == [
        ["coding-0", "planning-0", "science-0"],
        ["coding-1", "planning-1", "science-1"],
    ]
    assert all(len({task.family for task in batch}) == 3 for batch in batches)


def test_mean_gradient_trees_is_equal_weight_and_topology_bound() -> None:
    first = {"controller": {"weight": mx.array([1.0, 3.0])}}
    second = {"controller": {"weight": mx.array([3.0, 7.0])}}

    averaged = _mean_gradient_trees([first, second])

    assert averaged["controller"]["weight"].tolist() == [2.0, 5.0]
    with pytest.raises(ValueError, match="topology"):
        _mean_gradient_trees([first, {"controller": {"bias": mx.array([1.0])}}])


def test_cached_answer_binding_loss_trains_without_model_execution() -> None:
    bundle, _wiring = _bundle()
    controller = bundle.controller
    answer = mx.random.normal((1, 5, controller.config.hidden_size))
    state = mx.random.normal((1, controller.config.state_slots, controller.config.hidden_size))
    probabilities = controller.exact_probabilities(
        tuple((3, 12, 7, 1, 0)[: controller.config.state_slots]),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    targets = (
        mx.array([[0, 2, 2, 0, 0]], dtype=mx.int32),
        mx.array([[0, 1, 2, 0, 0]], dtype=mx.int32),
    )
    features = tuple(mx.stop_gradient(value) for value in (answer, state, probabilities))
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(bundle.trainable_parameters())
    before = float(_cached_answer_binding_loss(bundle, features, targets).item())
    for _ in range(20):
        loss, gradients = nn.value_and_grad(bundle, _cached_answer_binding_loss)(
            bundle,
            features,
            targets,
        )
        gradients = _phase_gradients(gradients, "answer_bridge")
        optimizer.update(bundle, gradients)
        mx.eval(bundle.parameters(), optimizer.state)
    after = float(_cached_answer_binding_loss(bundle, features, targets).item())
    assert after < before


def test_answer_bridge_admission_requires_exact_autonomous_emission_per_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    tasks = [
        type(
            "Task",
            (),
            {
                "family": family,
                "task_id": f"{family}-{depth}",
                "depth": depth,
            },
        )()
        for family in ("khop", "modular", "register_trace")
        for depth in (1, 2, 4)
    ]
    answers = {
        family: mx.array([[index + 10, 63]], dtype=mx.int32)
        for index, family in enumerate(("khop", "modular", "register_trace"))
    }
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda _tokenizer, task, _bridge: (mx.array([[1]]), answers[task.family]),
    )
    pointer_policies: list[bool] = []

    def exact_rollin(
        _bundle: object,
        _prompt: object,
        answer: object,
        _plan: object,
        **kwargs: object,
    ) -> object:
        pointer_policies.append(bool(kwargs["answer_digit_pointer_enabled"]))
        return answer

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._generate_student_rollin",
        exact_rollin,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        lambda _task, _depth, _capture: {
            "process_exact": True,
            "evidence_sha256": "a" * 64,
        },
    )
    tokenizer = type("Tokenizer", (), {"eos_token_id": 63})()
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=63,
        family_markers=(("khop", (1,)), ("modular", (2,)), ("register_trace", (3,))),
        syntax=(
            ("close", (4,)),
            ("khop", (5,)),
            ("modular", (6,)),
            ("register_head", (7,)),
            ("register_mid_r1", (8,)),
            ("register_mid_r2", (9,)),
        ),
    )

    report = _evaluate_answer_bridge_admission(
        bundle,
        tokenizer,
        tasks,
        spec,
        "",
        contract,
        answer_digit_pointer_enabled=False,
    )

    assert report["admitted"] is True
    assert report["exact_accuracy"] == 1.0
    assert report["schema"] == "aura.unified_intrinsic.answer_bridge_admission.v6"
    assert report["process_tape_enabled"] is True
    assert report["answer_digit_pointer_enabled"] is False
    assert pointer_policies == [False] * 9
    assert report["cells"] == 9
    assert report["tasks"] == 9
    assert report["answer_exact"] == 9
    assert report["process_exact"] == 9
    assert {(row["family"], row["task_depth"]) for row in report["rows"]} == {
        (family, depth) for family in ("khop", "modular", "register_trace") for depth in (1, 2, 4)
    }

    calls = 0

    def corrupt_final_cell(
        _bundle: UnifiedTrainingBundle,
        _prompt: object,
        answer: object,
        _plan: object,
        **_kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls < 9:
            return answer
        values = answer.tolist()
        values[0][0] = (int(values[0][0]) + 1) % 64
        return mx.array(values, dtype=answer.dtype)

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._generate_student_rollin",
        corrupt_final_cell,
    )
    rejected = _evaluate_answer_bridge_admission(
        bundle,
        tokenizer,
        tasks,
        spec,
        "",
        contract,
    )
    assert rejected["admitted"] is False
    assert rejected["exact"] == 8
    failed = [row for row in rejected["rows"] if not row["exact"]]
    assert len(failed) == 1
    assert failed[0]["mismatches"] == [
        {
            "position": 0,
            "expected_token_id": 12,
            "generated_token_id": 13,
        }
    ]


def test_answer_bridge_admission_evaluates_every_unseen_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1,), (4, 8))
    tasks = [
        type(
            "Task",
            (),
            {
                "family": "khop",
                "task_id": f"khop-1-{index}",
                "depth": 1,
            },
        )()
        for index in range(3)
    ]
    answer = mx.array([[10, 63]], dtype=mx.int32)
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda _tokenizer, _task, _bridge: (mx.array([[1]]), answer),
    )
    calls = 0

    def corrupt_last_holdout(
        _bundle: UnifiedTrainingBundle,
        _prompt: object,
        expected: object,
        _plan: object,
        **_kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls < 3:
            return expected
        return mx.array([[11, 63]], dtype=expected.dtype)

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._generate_student_rollin",
        corrupt_last_holdout,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        lambda _task, _depth, _capture: {
            "process_exact": True,
            "evidence_sha256": "b" * 64,
        },
    )
    tokenizer = type("Tokenizer", (), {"eos_token_id": 63})()
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=63,
        family_markers=(
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_trace", (3,)),
        ),
        syntax=(
            ("close", (4,)),
            ("khop", (5,)),
            ("modular", (6,)),
            ("register_head", (7,)),
            ("register_mid_r1", (8,)),
            ("register_mid_r2", (9,)),
        ),
    )

    report = _evaluate_answer_bridge_admission(
        bundle,
        tokenizer,
        tasks,
        spec,
        "",
        contract,
    )

    assert calls == 3
    assert report["cells"] == 1
    assert report["tasks"] == 3
    assert report["exact"] == 2
    assert report["admitted"] is False


def test_answer_bridge_admission_rejects_correct_text_from_wrong_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1,), (4, 8))
    task = type(
        "Task",
        (),
        {"family": "khop", "task_id": "khop-process-failure", "depth": 1},
    )()
    answer = mx.array([[10, 63]], dtype=mx.int32)
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]]), answer),
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._generate_student_rollin",
        lambda _bundle, _prompt, expected, _plan, **_kwargs: expected,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        lambda _task, _depth, _capture: {
            "process_exact": False,
            "evidence_sha256": "c" * 64,
        },
    )
    tokenizer = type("Tokenizer", (), {"eos_token_id": 63})()
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=63,
        family_markers=(
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_trace", (3,)),
        ),
        syntax=(
            ("close", (4,)),
            ("khop", (5,)),
            ("modular", (6,)),
            ("register_head", (7,)),
            ("register_mid_r1", (8,)),
            ("register_mid_r2", (9,)),
        ),
    )

    report = _evaluate_answer_bridge_admission(bundle, tokenizer, [task], spec, "", contract)

    assert report["answer_exact"] == 1
    assert report["process_exact"] == 0
    assert report["exact"] == 0
    assert report["admitted"] is False


def test_process_admission_requires_every_unseen_task_without_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    tasks = [
        type(
            "Task",
            (),
            {
                "family": "frontier_mathematics",
                "task_id": f"process-{index}",
                "depth": 2,
            },
        )()
        for index in range(2)
    ]
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]], dtype=mx.int32), mx.array([[2]], dtype=mx.int32)),
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._capture_autonomous_process",
        lambda *_args: {},
    )
    calls = 0

    def process_evidence(_task: object, _depth: int, _capture: object) -> dict:
        nonlocal calls
        calls += 1
        exact = calls == 1
        return {
            "process_exact": exact,
            "evidence_sha256": ("a" if exact else "b") * 64,
        }

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        process_evidence,
    )

    report = _evaluate_process_admission(bundle, object(), tasks, spec, "")

    assert calls == 2
    assert report["teacher_available"] is False
    assert report["transition_processor_available"] is True
    assert report["transition_processor_lesioned"] is False
    assert report["transition_action_history_available"] is True
    assert report["transition_action_history_lesioned"] is False
    assert report["tasks"] == 2
    assert report["process_exact"] == 1
    assert report["exact_accuracy"] == 0.5
    assert report["admitted"] is False
    assert len(report["admission_sha256"]) == 64


def test_process_admission_propagates_processor_and_history_lesions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    task = type(
        "Task",
        (),
        {
            "family": "frontier_mathematics",
            "task_id": "history-lesion",
            "depth": 2,
            "prompt": "Compute the public program.",
        },
    )()
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]], dtype=mx.int32), mx.array([[2]], dtype=mx.int32)),
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._public_actions_for_task",
        lambda *_args: ((0, 1, 32, 32, 32, 32, 32, 1),) * 2,
    )
    observed: list[dict[str, object]] = []

    def capture(*_args, **kwargs):
        observed.append(kwargs)
        return {}

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._capture_autonomous_process",
        capture,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        lambda *_args: {"process_exact": False, "evidence_sha256": "a" * 64},
    )

    report = _evaluate_process_admission(
        bundle,
        object(),
        [task],
        spec,
        "",
        public_action_program=True,
        transition_processor_lesion=True,
        transition_history_lesion=True,
    )

    assert observed == [
        {
            "public_action_values": ((0, 1, 32, 32, 32, 32, 32, 1),) * 2,
            "microcode_lesion": True,
            "transition_processor_lesion": True,
            "transition_processor_mode": "authoritative",
            "transition_copy_prior_logit_bias": 2.0,
            "transition_opcode_expert_routing": "opcode",
            "transition_replay_mode": "disabled",
            "transition_history_lesion": True,
        }
    ]
    assert report["transition_processor_available"] is False
    assert report["transition_processor_lesioned"] is True
    assert report["transition_processor_mode"] == "authoritative"
    assert report["transition_opcode_expert_routing"] == "opcode"
    assert report["transition_action_history_available"] is False
    assert report["transition_action_history_lesioned"] is True


def test_process_evidence_ignores_unrequired_post_completion_action_rows() -> None:
    logits = (
        mx.array([[[0.0, 5.0], [4.0, 0.0]]]),
        mx.array([[[8.0, 0.0], [0.0, 8.0]]]),
    )

    evidence = _masked_process_decisions(
        logits,
        ((1, 0), (1, 0)),
        ((True, True), (False, False)),
    )

    assert evidence["correct"] == 2
    assert evidence["required"] == 2
    assert evidence["required_steps"] == 1
    assert evidence["exact_steps"] == 1
    assert evidence["exact"] is True


def test_answer_bridge_diagnostic_classifies_process_and_reader_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1,), (4, 8))
    task = type(
        "Task",
        (),
        {
            "family": "khop",
            "task_id": "khop-diagnostic",
            "depth": 1,
            "transition_trace": object(),
            "transition_program": object(),
        },
    )()
    answer = mx.array([[10, 63]], dtype=mx.int32)
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]]), answer),
    )
    target = type(
        "Targets",
        (),
        {
            "initial_values": (0,),
            "values": ((0,),),
        },
    )()
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.state_targets_from_trace",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.action_targets_from_program",
        lambda *_args: type("Actions", (), {"values": ((0,),)})(),
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._process_evidence_from_capture",
        lambda *_args: {"process_exact": False, "evidence_sha256": "d" * 64},
    )
    calls = 0

    def diagnostic_rollin(
        _bundle: object,
        _prompt: object,
        expected: object,
        _plan: object,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if kwargs.get("state_teacher_forcing_probability") == 1.0:
            return expected
        return mx.array([[11, 63]], dtype=expected.dtype)

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._generate_student_rollin",
        diagnostic_rollin,
    )
    tokenizer = type("Tokenizer", (), {"eos_token_id": 63})()
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=63,
        family_markers=(
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_trace", (3,)),
        ),
        syntax=(
            ("close", (4,)),
            ("khop", (5,)),
            ("modular", (6,)),
            ("register_head", (7,)),
            ("register_mid_r1", (8,)),
            ("register_mid_r2", (9,)),
        ),
    )

    report = _evaluate_answer_bridge_diagnostic(bundle, tokenizer, [task], spec, "", contract)

    assert calls == 3
    assert report["oracle_exact"] == 1
    assert report["autonomous_process_exact"] == 0
    assert report["diagnosis"] == "recurrent_process_limited"


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
            "process_reader_1_output": mx.ones((2, 2)),
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
    assert bool(mx.all(semantic["controller.process_reader_1_output"] == 0))
    assert bool(mx.all(semantic["controller.transport_bias"] == 0))
    assert bool(mx.all(answer_bridge["model.layer.lora_a"] == 0))
    assert bool(mx.all(answer_bridge["model.layer.continuous_depth_b.0"] == 0))
    assert bool(mx.all(answer_bridge["controller.answer_output"] == 1))
    assert bool(mx.all(answer_bridge["controller.process_reader_1_output"] == 1))
    assert bool(mx.all(answer_bridge["controller.transport_bias"] == 0))
    assert bool(mx.all(state["model.layer.lora_a"] == 1))
    assert bool(mx.all(state["model.layer.continuous_depth_b.0"] == 1))
    assert bool(mx.all(state["controller.answer_output"] == 0))
    assert bool(mx.all(state["controller.process_reader_1_output"] == 0))
    assert bool(mx.all(state["controller.transport_bias"] == 1))
    assert bool(mx.all(recurrent["model.layer.lora_a"] == 0))
    assert bool(mx.all(recurrent["model.layer.continuous_depth_b.0"] == 1))
    assert bool(mx.all(recurrent["controller.answer_output"] == 0))
    assert bool(mx.all(recurrent["controller.process_reader_1_output"] == 0))
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


def test_answer_bridge_teacher_policy_ends_unassisted_and_blocks_bad_process() -> None:
    mapping = _answer_bridge_teacher_policy(
        11,
        bridge_start=10,
        bridge_steps=10,
        autonomous_tail_steps=4,
        process_exact=False,
    )
    tail_start = _answer_bridge_teacher_policy(
        16,
        bridge_start=10,
        bridge_steps=10,
        autonomous_tail_steps=4,
        process_exact=True,
    )
    tail_bad = _answer_bridge_teacher_policy(
        18,
        bridge_start=10,
        bridge_steps=10,
        autonomous_tail_steps=4,
        process_exact=False,
    )
    terminal = _answer_bridge_teacher_policy(
        19,
        bridge_start=10,
        bridge_steps=10,
        autonomous_tail_steps=4,
        process_exact=True,
    )

    assert mapping == {
        "state_teacher_forcing_probability": 1.0,
        "autonomous_tail": False,
        "update_admitted": True,
    }
    assert tail_start["state_teacher_forcing_probability"] == 1.0
    assert tail_start["autonomous_tail"] is True
    assert tail_start["update_admitted"] is True
    assert tail_bad["state_teacher_forcing_probability"] == pytest.approx(1 / 3)
    assert tail_bad["update_admitted"] is False
    assert terminal["state_teacher_forcing_probability"] == 0.0
    assert terminal["update_admitted"] is True


def test_gradient_trust_bound_does_not_starve_independent_mechanisms() -> None:
    gradients = {
        "model": {
            "layer": {"lora_a": mx.array([3.0, 4.0])},
            "block": {
                "self_attn": {"q_proj": {"lora_b": mx.array([6.0, 8.0])}}
            },
        },
        "controller": {
            "state_transition_output": mx.array([0.0, 12.0]),
            "transition_memory_output": mx.array([0.0, 9.0]),
            "transition_tape_output": mx.array([0.0, 8.0]),
            "transition_processor_output": mx.array([0.0, 11.0]),
            "transition_processor_opcode_output": mx.array([0.0, 10.0]),
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
        "scoped_transformer_query",
        "scoped_transformer_bridge",
        "typed_state_transition",
        "typed_state_codebook",
        "typed_action_transition",
        "typed_action_codebook",
        "recurrent_controller",
    }
    assert float(mx.linalg.norm(flat["model.layer.lora_a"]).item()) == pytest.approx(1.0)
    assert float(
        mx.linalg.norm(flat["model.block.self_attn.q_proj.lora_b"]).item()
    ) == pytest.approx(1.0)
    assert float(
        mx.sqrt(
            mx.sum(flat["controller.state_transition_output"] ** 2)
            + mx.sum(flat["controller.transition_memory_output"] ** 2)
            + mx.sum(flat["controller.transition_tape_output"] ** 2)
            + mx.sum(flat["controller.transition_processor_output"] ** 2)
            + mx.sum(flat["controller.transition_processor_opcode_output"] ** 2)
        ).item()
    ) == pytest.approx(1.0)
    assert float(mx.linalg.norm(flat["controller.state_value_embeddings"]).item()) == pytest.approx(
        1.0
    )
    action_transition_norm = mx.sqrt(
        mx.sum(flat["controller.action_output"] ** 2)
        + mx.sum(flat["controller.opcode_copy_logit"] ** 2)
    )
    assert float(action_transition_norm.item()) == pytest.approx(1.0)
    assert float(mx.abs(flat["controller.opcode_copy_logit"]).item()) > 0.0
    assert float(
        mx.linalg.norm(flat["controller.action_value_embeddings"]).item()
    ) == pytest.approx(1.0)
    assert float(mx.linalg.norm(flat["controller.transport_bias"]).item()) == pytest.approx(0.5)


def test_gradient_conflict_diagnostics_report_negative_and_unmeasured_pairs() -> None:
    aligned = {
        "controller": {
            "transition_processor_output": mx.array([1.0, 0.0]),
            "transport_bias": mx.array([5.0]),
        }
    }
    opposed = {
        "controller": {
            "transition_processor_output": mx.array([-1.0, 0.0]),
            "transport_bias": mx.array([5.0]),
        }
    }
    empty = {
        "controller": {
            "transition_processor_output": mx.array([0.0, 0.0]),
            "transport_bias": mx.array([5.0]),
        }
    }

    report = _gradient_conflict_diagnostics(
        [aligned, opposed, empty],
        ["math", "coding", "premise"],
        ownership_group="typed_state_transition",
    )

    assert report["parameter_count"] == 1
    assert report["measured_pairs"] == 1
    assert report["negative_pairs"] == 1
    assert report["minimum_cosine"] == pytest.approx(-1.0)
    assert report["mean_cosine"] == pytest.approx(-1.0)
    assert sum(pair["measured"] for pair in report["pairs"]) == 1


def test_pcgrad_projects_only_owned_negative_conflicts() -> None:
    left = {
        "controller": {
            "transition_processor_output": mx.array([1.0, 0.0]),
            "transport_bias": mx.array([2.0]),
        }
    }
    right = {
        "controller": {
            "transition_processor_output": mx.array([-1.0, 1.0]),
            "transport_bias": mx.array([4.0]),
        }
    }

    combined, receipt = _combine_process_gradient_trees(
        [left, right],
        ["math", "coding"],
        mode="pcgrad",
        ownership_group="typed_state_transition",
    )
    flat = dict(tree_flatten(combined))
    mx.eval(*flat.values())

    assert receipt["projection_count"] == 2
    assert flat["controller.transition_processor_output"].tolist() == pytest.approx(
        [0.25, 0.75]
    )
    assert flat["controller.transport_bias"].tolist() == pytest.approx([3.0])


def test_mean_process_gradient_combiner_is_exact_mean() -> None:
    left = {"controller": {"transition_processor_output": mx.array([1.0])}}
    right = {"controller": {"transition_processor_output": mx.array([3.0])}}

    combined, receipt = _combine_process_gradient_trees(
        [left, right],
        ["left", "right"],
        mode="mean",
        ownership_group="typed_state_transition",
    )
    flat = dict(tree_flatten(combined))
    mx.eval(*flat.values())

    assert receipt["projection_count"] == 0
    assert flat["controller.transition_processor_output"].tolist() == pytest.approx(
        [2.0]
    )


def test_balanced_mean_equalizes_owned_family_norms_only() -> None:
    left = {
        "controller": {
            "transition_processor_output": mx.array([1.0, 0.0]),
            "transport_bias": mx.array([2.0]),
        }
    }
    right = {
        "controller": {
            "transition_processor_output": mx.array([0.0, 3.0]),
            "transport_bias": mx.array([4.0]),
        }
    }

    combined, receipt = _combine_process_gradient_trees(
        [left, right],
        ["calibration", "coding"],
        mode="balanced_mean",
        ownership_group="typed_state_transition",
    )
    flat = dict(tree_flatten(combined))
    mx.eval(*flat.values())

    assert receipt["owned_norms"] == pytest.approx(
        {"calibration": 1.0, "coding": 3.0}
    )
    assert receipt["owned_scales"] == pytest.approx(
        {"calibration": 2.0, "coding": 2.0 / 3.0}
    )
    assert flat["controller.transition_processor_output"].tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert flat["controller.transport_bias"].tolist() == pytest.approx([3.0])


def test_ownership_optimizer_preserves_rate_ratio_after_adam_normalization() -> None:
    parameters = {
        "model": {
            "layers": [
                {
                    "self_attn": {
                        "q_proj": {"lora_b": mx.array([1.0])},
                        "o_proj": {"lora_a": mx.array([1.0])},
                    }
                }
            ],
        },
        "controller": {"action_workspace_output": mx.array([1.0])},
    }
    gradients = {
        "model": {
            "layers": [
                {
                    "self_attn": {
                        "q_proj": {"lora_b": mx.array([50.0])},
                        "o_proj": {"lora_a": mx.array([50.0])},
                    }
                }
            ],
        },
        "controller": {"action_workspace_output": mx.array([50.0])},
    }
    optimizer = _ownership_optimizer(
        0.01,
        transformer_rate_scale=0.1,
        query_rate_scale=0.01,
    )
    updated = optimizer.apply_gradients(gradients, parameters)
    mx.eval(updated, optimizer.state)
    flat = dict(tree_flatten(updated))
    tissue_delta = 1.0 - float(flat["model.layers.0.self_attn.o_proj.lora_a"].item())
    query_delta = 1.0 - float(flat["model.layers.0.self_attn.q_proj.lora_b"].item())
    controller_delta = 1.0 - float(flat["controller.action_workspace_output"].item())

    assert tissue_delta > 0.0
    assert controller_delta > tissue_delta
    assert tissue_delta / controller_delta == pytest.approx(0.1, rel=1e-4)
    assert query_delta / controller_delta == pytest.approx(0.01, rel=1e-4)

    _set_ownership_optimizer_rates(
        optimizer,
        0.02,
        transformer_rate_scale=0.25,
        query_rate_scale=0.05,
    )
    assert float(optimizer.optimizers[0].learning_rate.item()) == pytest.approx(0.001)
    assert float(optimizer.optimizers[1].learning_rate.item()) == pytest.approx(0.005)
    assert float(optimizer.optimizers[2].learning_rate.item()) == pytest.approx(0.02)

    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        _ownership_optimizer(0.01, transformer_rate_scale=1.1)


def test_ownership_optimizer_initializes_disjoint_state_from_parameter_topology() -> None:
    parameters = {
        "model": {
            "layers": [
                {
                    "self_attn": {
                        "o_proj": {"lora_a": mx.array([1.0])},
                    }
                }
            ],
        },
        "controller": {
            "transition_processor_opcode_hidden": [
                mx.array([1.0]),
                mx.array([2.0]),
            ],
            "transition_processor_output": mx.array([3.0]),
        },
    }
    gradients = tree_unflatten(
        [(name, mx.ones_like(value)) for name, value in tree_flatten(parameters)]
    )
    optimizer = _ownership_optimizer(0.01, transformer_rate_scale=0.1)

    updated = optimizer.apply_gradients(gradients, parameters)
    mx.eval(updated, optimizer.state)
    updated_again = optimizer.apply_gradients(gradients, updated)
    mx.eval(updated_again, optimizer.state)

    assert {name for name, _value in tree_flatten(updated)} == {
        name for name, _value in tree_flatten(parameters)
    }


def test_checkpoint_roundtrip_restores_exact_trainable_state(tmp_path: Path) -> None:
    bundle, _wiring = _bundle()
    optimizer = _ownership_optimizer(0.01, transformer_rate_scale=0.1)
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


def test_bootstrap_imports_only_compatible_tissue_into_a_new_campaign(
    tmp_path: Path,
) -> None:
    parent, _wiring = _bundle()
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(parent.trainable_parameters())
    compatibility = {
        name: f"value-{name}"
        for name in (
            "model",
            "runtime",
            "tokenizer",
            "spec",
            "window_geometry",
            "families",
            "task_depths",
            "init_seed",
            "bridge",
            "window_tissue_mode",
            "lora_rank",
            "controller_rank",
            "state_weight",
            "stutter_weight",
            "state_codebook_sha256",
            "literal_observation_contract",
            "opcode_observation_contract",
            "answer_emission_contract",
            "depth_basis_size",
            "lora_targets",
            "readout_sha256",
        )
    }
    compatibility["spec"] = {
        "prelude_end": 2,
        "coda_start": 4,
        "train_depths": [1, 2, 4],
        "heldout_depths": [8, 16],
    }
    compatibility["model"] = {
        "canonical_path": "/old/location",
        "config_sha256": "c" * 64,
        "weights": [{"name": "model.safetensors", "sha256": "d" * 64, "size": 8}],
    }
    parent_identity_body = {"schema": "test", **compatibility}
    parent_identity = {
        **parent_identity_body,
        "identity_sha256": _canonical_sha256(parent_identity_body),
    }
    expected = {name: value + 0 for name, value in _trainable(parent).items()}
    mx.eval(expected)
    _save_checkpoint(
        tmp_path,
        parent,
        optimizer,
        step=73,
        history=[{"step": 73}],
        identity=parent_identity,
    )
    child, _child_wiring = _bundle()

    receipt = _bootstrap_bundle_from_checkpoint(
        tmp_path,
        "checkpoint_latest",
        child,
        expected_identity={
            **compatibility,
            "model": {
                "canonical_path": "/new/location",
                "config_sha256": "c" * 64,
                "weights": [
                    {
                        "name": "model.safetensors",
                        "sha256": "d" * 64,
                        "size_bytes": 8,
                    }
                ],
            },
            "families": ["fresh", "broad"],
            "task_depths": [3, 5, 7],
            "spec": {
                "prelude_end": 2,
                "coda_start": 4,
                "train_depths": (3, 5, 7),
                "heldout_depths": (9, 11),
            },
            "dataset": "fresh",
        },
    )

    imported = _trainable(child)
    assert all(bool(mx.array_equal(imported[name], value)) for name, value in expected.items())
    assert receipt["parent_step"] == 73
    assert receipt["optimizer_inherited"] is False
    assert receipt["history_inherited"] is False
    assert receipt["dataset_inherited"] is False
    assert receipt["dataset_transfer"] == "explicit_new_campaign"
    assert receipt["tensor_shapes_verified"] is True
    assert receipt["tensor_dtypes_verified"] is True

    with pytest.raises(RuntimeError, match="topology differs: controller_rank"):
        _bootstrap_bundle_from_checkpoint(
            tmp_path,
            "checkpoint_latest",
            child,
            expected_identity={**compatibility, "controller_rank": "different"},
        )

    wrong_shape, _wrong_shape_wiring = _bundle()
    wrong_shape.controller.correction_a = wrong_shape.controller.correction_a[:-1]
    with pytest.raises(RuntimeError, match="tensor topology differs"):
        _bootstrap_bundle_from_checkpoint(
            tmp_path,
            "checkpoint_latest",
            wrong_shape,
            expected_identity=compatibility,
        )


def test_bootstrap_extends_legacy_parent_with_reader_action_workspace_and_transition_memory(
    tmp_path: Path,
) -> None:
    parent, _wiring = _bundle()
    for name in PROCESS_READER_PARAMETER_NAMES:
        delattr(parent.controller, name)
    for name in ACTION_WORKSPACE_PARAMETER_NAMES:
        delattr(parent.controller, name)
    for name in TRANSITION_MEMORY_PARAMETER_NAMES:
        delattr(parent.controller, name)
    for name in TRANSITION_TAPE_READER_PARAMETER_NAMES:
        delattr(parent.controller, name)
    for name in TRANSITION_PROCESSOR_PARAMETER_NAMES:
        delattr(parent.controller, name)
    parent_values = {name: value + 0 for name, value in _trainable(parent).items()}
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(parent.trainable_parameters())
    compatibility = {
        name: f"value-{name}"
        for name in (
            "bridge",
            "window_tissue_mode",
            "lora_rank",
            "controller_rank",
            "state_codebook_sha256",
            "literal_observation_contract",
            "opcode_observation_contract",
            "answer_emission_contract",
            "depth_basis_size",
            "lora_targets",
            "readout_sha256",
        )
    }
    compatibility["model"] = {
        "config_sha256": "c" * 64,
        "weights": [{"name": "model.safetensors", "sha256": "d" * 64, "size": 8}],
    }
    compatibility["spec"] = {"prelude_end": 2, "coda_start": 4}
    identity = {
        "schema": "test",
        **compatibility,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    _save_checkpoint(
        tmp_path,
        parent,
        optimizer,
        step=41,
        history=[],
        identity=identity,
    )
    child, _child_wiring = _bundle()

    receipt = _bootstrap_bundle_from_checkpoint(
        tmp_path,
        "checkpoint_latest",
        child,
        expected_identity=compatibility,
    )

    imported = _trainable(child)
    assert all(bool(mx.array_equal(imported[name], value)) for name, value in parent_values.items())
    extension = receipt["process_reader_extension"]
    assert extension["parent_tensor_inventory_preserved"] is True
    assert set(extension["new_tensor_names"]) == {
        f"controller.{name}" for name in PROCESS_READER_PARAMETER_NAMES
    }
    action_extension = receipt["action_workspace_extension"]
    assert action_extension["parent_tensor_inventory_preserved"] is True
    assert action_extension["behavior_before_training_preserved"] is True
    assert set(action_extension["new_tensor_names"]) == {
        f"controller.{name}" for name in ACTION_WORKSPACE_PARAMETER_NAMES
    }
    transition_extension = receipt["transition_memory_extension"]
    assert transition_extension["parent_tensor_inventory_preserved"] is True
    assert transition_extension["behavior_before_training_preserved"] is True
    assert set(transition_extension["new_tensor_names"]) == {
        f"controller.{name}" for name in TRANSITION_MEMORY_PARAMETER_NAMES
    }
    tape_extension = receipt["transition_tape_reader_extension"]
    assert tape_extension["parent_tensor_inventory_preserved"] is True
    assert tape_extension["behavior_before_training_preserved"] is True
    assert set(tape_extension["new_tensor_names"]) == {
        f"controller.{name}" for name in TRANSITION_TAPE_READER_PARAMETER_NAMES
    }
    processor_extension = receipt["transition_processor_extension"]
    assert processor_extension["parent_tensor_inventory_preserved"] is True
    assert processor_extension["behavior_before_training_preserved"] is True
    assert set(processor_extension["new_tensor_names"]) == {
        f"controller.{name}" for name in TRANSITION_PROCESSOR_PARAMETER_NAMES
    }


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
    step, _history, _training_state = _restore_checkpoint(tmp_path, bundle, optimizer, identity)
    assert step == 3


def test_evaluation_separates_trained_from_heldout_depth_gains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    values = iter((1.0, 0.9, 1.2, 1.4))

    final_only_calls: list[bool] = []

    def fake_trajectory(*_args, **kwargs):
        final_only_calls.append(kwargs.get("final_answer_only") is True)
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
    reclaim_calls: list[bool] = []
    envelope = type(
        "Envelope",
        (),
        {
            "reclaim": lambda _self, *_args, **kwargs: reclaim_calls.append(
                kwargs.get("force") is True
            )
        },
    )()
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
    assert report["process_by_family_at_max_depth"] == {"depth": 8, "families": {}}
    assert final_only_calls == [True] * len(spec.depths)
    assert reclaim_calls == [True] * len(spec.depths)


def test_evaluation_propagates_public_action_program_to_every_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    spec = UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (4, 8))
    observed: list[tuple[int, bool]] = []

    def fake_depth(
        _bundle_value,
        _prompt,
        _answer,
        _task,
        _spec_value,
        depth,
        *,
        public_action_program=False,
        transition_processor_mode="authoritative",
        transition_copy_prior_logit_bias=2.0,
        transition_opcode_expert_routing="opcode",
        transition_replay_mode="disabled",
        direct_transition_processor=False,
    ):
        assert transition_processor_mode == "authoritative"
        assert transition_copy_prior_logit_bias == 2.0
        assert transition_opcode_expert_routing == "opcode"
        assert transition_replay_mode == "disabled"
        assert direct_transition_processor is False
        observed.append((depth, public_action_program))
        return {"loss": float(depth)}

    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence._evaluate_depth",
        fake_depth,
    )
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.encode_example",
        lambda *_args: (mx.array([[1]]), mx.array([[2]])),
    )
    envelope = type("Envelope", (), {"reclaim": lambda *_args, **_kwargs: None})()
    report = _evaluate(
        bundle,
        type("Tokenizer", (), {})(),
        [type("Task", (), {"prompt": "p", "answer": "a"})()],
        spec,
        "",
        spec.depths,
        envelope=envelope,
        public_action_program=True,
    )

    assert observed == [(depth, True) for depth in spec.depths]
    assert report["ce"] == {f"T{depth}": float(depth) for depth in spec.depths}


def test_direct_transition_interim_evaluation_never_runs_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _wiring = _bundle()
    bundle.controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            state_slots=11,
            minimum_iterations=1,
        )
    )
    task = frontier_process_task_battery(
        ("coding",),
        (1,),
        1,
        seed=2026081515,
    )[0]
    monkeypatch.setattr(
        "tools.train_unified_intrinsic_recurrence.unified_answer_and_recurrent_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct transition interim evaluation loaded the model path")
        ),
    )

    report = _evaluate_depth(
        bundle,
        mx.array([[1]]),
        mx.array([[2]]),
        task,
        UnifiedIntrinsicTrainingSpec(2, 4, (1,), (3,)),
        3,
        public_action_program=True,
        transition_processor_mode="masked_copy_write",
        transition_copy_prior_logit_bias=0.01,
        direct_transition_processor=True,
    )

    assert math.isfinite(report["loss"])
    assert report["initial_state_accuracy"] == 1.0
    assert report["action_accuracy"] == 1.0
    assert "active_state_exact_accuracy" in report
