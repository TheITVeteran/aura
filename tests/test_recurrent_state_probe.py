from __future__ import annotations

import numpy as np
import pytest

from core.learning.recurrent_state_probe import (
    RECURRENT_STATE_PROBE_SCHEMA,
    StateProbeObservation,
    evaluate_recurrent_state_information,
)


def _synthetic_observations(
    *,
    prefix: str,
    count: int,
    seed: int,
    informative_step: int,
) -> list[StateProbeObservation]:
    rng = np.random.default_rng(seed)
    rows: list[StateProbeObservation] = []
    for task_index in range(count):
        value = task_index % 3
        for step in (0, 1):
            features = rng.normal(0.0, 0.2, size=(2, 4))
            features[0, 0] = float(step) * 3.0
            if step == informative_step:
                features[1, value] += 5.0
            rows.append(
                StateProbeObservation(
                    task_id=f"{prefix}-{task_index}",
                    family="synthetic",
                    program_depth=1,
                    recurrence_step=step,
                    field_names=("pc", "value", "done"),
                    labels=(step, value, step),
                    features=features,
                )
            )
    return rows


def test_state_probe_detects_recoverability_improvement_over_controls() -> None:
    training = _synthetic_observations(
        prefix="train",
        count=90,
        seed=1,
        informative_step=1,
    )
    validation = _synthetic_observations(
        prefix="validation",
        count=45,
        seed=2,
        informative_step=1,
    )

    report = evaluate_recurrent_state_information(training, validation)
    target = report["families"]["synthetic"]["fields"]["value"]

    assert report["schema"] == RECURRENT_STATE_PROBE_SCHEMA
    assert report["task_ids_disjoint"] is True
    assert target["disposition"] == "recoverability_improves"
    assert target["terminal"]["accuracy"] > 0.95
    assert target["terminal_effect_over_controls"] > 0.5
    assert target["terminal_minus_initial_accuracy"] > 0.4
    assert len(report["receipt_sha256"]) == 64


def test_state_probe_detects_recurrent_information_erosion() -> None:
    training = _synthetic_observations(
        prefix="train",
        count=90,
        seed=3,
        informative_step=0,
    )
    validation = _synthetic_observations(
        prefix="validation",
        count=45,
        seed=4,
        informative_step=0,
    )

    report = evaluate_recurrent_state_information(training, validation)
    target = report["families"]["synthetic"]["fields"]["value"]

    assert target["disposition"] == "recoverability_erodes"
    assert target["initial"]["accuracy"] > 0.95
    assert target["terminal_minus_initial_accuracy"] < -0.4


def test_state_probe_refuses_task_identity_overlap() -> None:
    training = _synthetic_observations(
        prefix="shared",
        count=8,
        seed=5,
        informative_step=1,
    )
    validation = _synthetic_observations(
        prefix="shared",
        count=8,
        seed=6,
        informative_step=1,
    )

    with pytest.raises(ValueError, match="overlap"):
        evaluate_recurrent_state_information(training, validation)


def test_state_probe_observation_rejects_boolean_labels() -> None:
    with pytest.raises(ValueError, match="observation"):
        StateProbeObservation(
            task_id="task",
            family="synthetic",
            program_depth=1,
            recurrence_step=0,
            field_names=("pc", "value", "done"),
            labels=(0, True, 0),
            features=np.zeros((2, 2)),
        )
