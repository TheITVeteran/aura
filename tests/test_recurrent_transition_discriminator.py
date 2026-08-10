"""Evidence and split contracts for the one-step recurrent discriminator."""

from __future__ import annotations

import json
from types import SimpleNamespace

from core.learning.recurrence_curriculum import nested_boolean
from tools.run_recurrent_transition_discriminator import (
    _admission_gates,
    _aggregate_evaluations,
    _mint_splits,
    _task_commitment,
)


def _evaluation(*, exact: bool, target: bool, loss: float):
    expected = (1, 1, 0)
    predicted = (1, 1 if target else 0, 0 if exact else 1)
    exact_fields = sum(left == right for left, right in zip(predicted, expected, strict=True))
    return SimpleNamespace(
        exact=exact,
        exact_fields=exact_fields,
        field_count=3,
        predicted=predicted,
        expected=expected,
        loss=loss,
    )


def test_three_way_split_is_globally_disjoint():
    train, development, holdout = _mint_splits(
        families=("boolean", "modular"),
        depths=(1, 2),
        train_per_cell=2,
        development_per_cell=1,
        holdout_per_cell=1,
        seed=20260810180,
    )
    splits = [train, development, holdout]
    for left in range(3):
        for right in range(left + 1, 3):
            assert {task.task_id for task in splits[left]}.isdisjoint(
                {task.task_id for task in splits[right]}
            )
            assert {task.prompt for task in splits[left]}.isdisjoint(
                {task.prompt for task in splits[right]}
            )


def test_public_task_commitment_contains_no_private_state_values():
    task = nested_boolean(2, 41)
    trace = task.transition_trace
    assert trace is not None
    commitment = _task_commitment(task, split="holdout")
    encoded = json.dumps(commitment, sort_keys=True)

    assert "states" not in commitment["trace"]
    assert "trace_sha256" in commitment["trace"]
    assert json.dumps([list(state) for state in trace.states]) not in encoded


def test_aggregate_keeps_exact_field_and_target_metrics_separate():
    first = nested_boolean(1, 51)
    second = nested_boolean(1, 52)
    report = _aggregate_evaluations(
        [
            (first, _evaluation(exact=True, target=True, loss=0.1)),
            (second, _evaluation(exact=False, target=False, loss=0.9)),
        ]
    )
    aggregate = report["aggregate"]
    assert aggregate["exact_accuracy"] == 0.5
    assert aggregate["target_field_accuracy"] == 0.5
    assert aggregate["field_accuracy"] == 4 / 6
    assert aggregate["mean_loss"] == 0.5


def test_admission_requires_large_heldout_gain_and_every_cell_nonregression():
    baseline = {
        "aggregate": {
            "mean_loss": 2.0,
            "exact_accuracy": 0.25,
            "field_accuracy": 0.60,
            "target_field_accuracy": 0.50,
        },
        "cells": {"boolean:d1": {"exact_accuracy": 0.25}},
    }
    treatment = {
        "aggregate": {
            "mean_loss": 0.3,
            "exact_accuracy": 0.80,
            "field_accuracy": 0.90,
            "target_field_accuracy": 0.90,
        },
        "cells": {"boolean:d1": {"exact_accuracy": 0.75}},
    }
    gates = _admission_gates(
        baseline=baseline,
        treatment=treatment,
        development_improved=True,
        base_checkpoint_immutable=True,
        source_published=True,
        splits_disjoint=True,
        adapter_changed=True,
    )
    assert all(gates.values())

    regressed = dict(treatment)
    regressed["cells"] = {"boolean:d1": {"exact_accuracy": 0.0}}
    rejected = _admission_gates(
        baseline=baseline,
        treatment=regressed,
        development_improved=True,
        base_checkpoint_immutable=True,
        source_published=True,
        splits_disjoint=True,
        adapter_changed=True,
    )
    assert rejected["every_cell_exact_nonregression"] is False
    assert not all(rejected.values())
