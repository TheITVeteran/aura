"""Evidence contracts for the native one-step transfer discriminator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.recurrent_transition_core import (  # noqa: E402
    RecurrentTransitionCore,
    RecurrentTransitionCoreConfig,
)
from core.learning.recurrence_curriculum import nested_boolean  # noqa: E402
from tools.run_native_transition_discriminator import (  # noqa: E402
    _admission_gates,
    _aggregate,
    _atomic_save_core,
    _mint_splits,
    _parameter_fingerprint,
    _task_commitment,
)


def _evaluation(
    *, state_exact: bool, target: bool, action_exact: bool, loss: float
):
    expected_state = (1, 1, 0)
    predicted_state = (1, 1 if target else 0, 0 if state_exact else 1)
    expected_action = (1, 1, 1)
    predicted_action = (1, 1, 1 if action_exact else 0)
    return SimpleNamespace(
        loss=loss,
        state_exact=state_exact,
        state_exact_fields=sum(
            left == right
            for left, right in zip(predicted_state, expected_state, strict=True)
        ),
        state_field_count=3,
        predicted_state=predicted_state,
        expected_state=expected_state,
        action_exact=action_exact,
        action_exact_fields=sum(
            left == right
            for left, right in zip(predicted_action, expected_action, strict=True)
        ),
        action_field_count=3,
    )


def test_native_discriminator_splits_are_globally_disjoint():
    splits = _mint_splits(
        depths=(1, 2),
        train_per_cell=2,
        development_per_cell=1,
        holdout_per_cell=1,
        seed=20260810185,
    )
    for left in range(3):
        for right in range(left + 1, 3):
            assert {task.task_id for task in splits[left]}.isdisjoint(
                {task.task_id for task in splits[right]}
            )
            assert {task.prompt for task in splits[left]}.isdisjoint(
                {task.prompt for task in splits[right]}
            )


def test_full_split_fits_finite_depth_one_support_and_refuses_overflow():
    splits = _mint_splits(
        depths=(1, 2, 3, 4),
        train_per_cell=8,
        development_per_cell=2,
        holdout_per_cell=4,
        seed=20260810185,
    )
    assert [len(split) for split in splits] == [64, 16, 32]

    with pytest.raises(ValueError, match="14-program support"):
        _mint_splits(
            depths=(1, 2),
            train_per_cell=9,
            development_per_cell=2,
            holdout_per_cell=4,
            seed=20260810185,
        )


def test_public_manifest_commits_program_without_private_states_or_actions():
    task = nested_boolean(2, 41)
    program = task.transition_program
    assert program is not None
    commitment = _task_commitment(task, split="holdout")
    encoded = json.dumps(commitment, sort_keys=True)

    assert "states" not in commitment["program"]["state_trace"]
    assert "actions" not in commitment["program"]
    assert "program_sha256" in commitment["program"]
    assert json.dumps([list(state) for state in program.state_trace.states]) not in encoded
    assert json.dumps([list(action) for action in program.actions]) not in encoded


def test_native_aggregate_keeps_state_action_and_target_metrics_separate():
    first = nested_boolean(1, 51)
    second = nested_boolean(1, 52)
    report = _aggregate(
        [
            (
                first,
                _evaluation(
                    state_exact=True,
                    target=True,
                    action_exact=False,
                    loss=0.1,
                ),
            ),
            (
                second,
                _evaluation(
                    state_exact=False,
                    target=False,
                    action_exact=True,
                    loss=0.9,
                ),
            ),
        ]
    )
    aggregate = report["aggregate"]
    assert aggregate["state_exact_accuracy"] == 0.5
    assert aggregate["target_field_accuracy"] == 0.5
    assert aggregate["action_exact_accuracy"] == 0.5
    assert aggregate["state_field_accuracy"] == 4 / 6
    assert aggregate["action_field_accuracy"] == 5 / 6
    assert aggregate["mean_loss"] == 0.5


def test_admission_rejects_loss_only_improvement_and_weak_cells():
    baseline = {
        "aggregate": {
            "mean_loss": 2.0,
            "state_exact_accuracy": 0.0,
            "state_field_accuracy": 0.50,
            "target_field_accuracy": 0.50,
            "action_field_accuracy": 0.50,
            "action_exact_accuracy": 0.10,
        },
        "cells": {"boolean:d1": {"state_exact_accuracy": 0.0}},
    }
    treatment = {
        "aggregate": {
            "mean_loss": 0.2,
            "state_exact_accuracy": 0.80,
            "state_field_accuracy": 0.95,
            "target_field_accuracy": 0.90,
            "action_field_accuracy": 0.90,
            "action_exact_accuracy": 0.70,
        },
        "cells": {"boolean:d1": {"state_exact_accuracy": 0.75}},
    }
    gates = _admission_gates(
        baseline=baseline,
        treatment=treatment,
        development_improved=True,
        base_checkpoint_immutable=True,
        source_published=True,
        splits_disjoint=True,
        core_changed=True,
    )
    assert all(gates.values())

    loss_only = dict(treatment)
    loss_only["aggregate"] = {
        **treatment["aggregate"],
        "state_exact_accuracy": 0.10,
    }
    rejected = _admission_gates(
        baseline=baseline,
        treatment=loss_only,
        development_improved=True,
        base_checkpoint_immutable=True,
        source_published=True,
        splits_disjoint=True,
        core_changed=True,
    )
    assert rejected["holdout_loss_lower"] is True
    assert rejected["holdout_state_exact_accuracy_at_least_0_75"] is False
    assert not all(rejected.values())

    weak_cell = dict(treatment)
    weak_cell["cells"] = {"boolean:d1": {"state_exact_accuracy": 0.49}}
    rejected = _admission_gates(
        baseline=baseline,
        treatment=weak_cell,
        development_improved=True,
        base_checkpoint_immutable=True,
        source_published=True,
        splits_disjoint=True,
        core_changed=True,
    )
    assert rejected["every_cell_state_exact_accuracy_at_least_0_50"] is False


def test_native_core_checkpoint_is_atomic_private_and_replayable(tmp_path):
    config = RecurrentTransitionCoreConfig(
        hidden_size=32,
        bottleneck_size=16,
        attention_heads=4,
    )
    core = RecurrentTransitionCore(config)
    mx.eval(core.parameters())
    initial = _parameter_fingerprint(core)
    core.delta_up.weight = mx.ones_like(core.delta_up.weight) * 0.01
    mx.eval(core.parameters())
    changed = _parameter_fingerprint(core)
    assert changed != initial

    path = tmp_path / "best_core.safetensors"
    artifact = _atomic_save_core(path, core)
    restored = RecurrentTransitionCore(config)
    restored.load_weights(list(mx.load(str(path)).items()), strict=True)
    mx.eval(restored.parameters())

    assert artifact["mode"] == "0600"
    assert path.stat().st_mode & 0o777 == 0o600
    assert _parameter_fingerprint(restored) == changed
    assert not list(tmp_path.glob("*.tmp.safetensors"))
