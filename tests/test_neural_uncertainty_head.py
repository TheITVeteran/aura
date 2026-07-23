"""Contracts for the hidden-state correctness and entropy head."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.neural_uncertainty import (
    HiddenStateCorrectnessExample,
    NeuralUncertaintyHead,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _examples(
    split: str,
    *,
    count: int = 64,
    width: int = 12,
    separable: bool = True,
) -> list[HiddenStateCorrectnessExample]:
    rng = np.random.default_rng(11 if split == "train" else 29)
    examples = []
    for index in range(count):
        correct = index % 2 == 0
        centre = 2.5 if correct else -2.5
        hidden = rng.normal(0.0, 0.25, size=width)
        hidden[0] = centre + float(rng.normal(0.0, 0.15)) if separable else 0.0
        if not separable:
            hidden[1:] = 0.0
        examples.append(
            HiddenStateCorrectnessExample(
                example_id=f"{split}-example-{index}",
                task_id=f"{split}-task-{index % 8}",
                hidden_state=tuple(float(value) for value in hidden),
                correct=correct,
                state_sha256=_digest(f"{split}:state:{index}"),
                outcome_receipt_sha256=_digest(f"{split}:outcome:{index}"),
                outcome_verifier_id="independent-exact-grader-v1",
            )
        )
    return examples


def _fit() -> NeuralUncertaintyHead:
    return NeuralUncertaintyHead.fit(
        _examples("train"),
        _examples("calibration"),
        hidden_width=8,
        seed=7,
        steps=350,
    )


def test_hidden_state_head_fits_task_disjoint_objective_outcomes():
    head = _fit()
    assert head.calibrated is True
    assert head.manifest()["metrics"]["auc"] >= 0.99
    assert head.manifest()["metrics"]["balanced_accuracy"] >= 0.95
    assert head.manifest()["train_tasks_sha256"] != (
        head.manifest()["calibration_tasks_sha256"]
    )

    correct = head.estimate([3.0] + [0.0] * 11)
    incorrect = head.estimate([-3.0] + [0.0] * 11)
    assert correct["correctness_probability"] > 0.8
    assert incorrect["correctness_probability"] < 0.2
    assert correct["predictive_entropy"] < 0.8
    assert incorrect["predictive_entropy"] < 0.8


def test_hidden_state_lesion_changes_prediction_causally():
    head = _fit()
    positive = np.asarray([3.0] + [0.0] * 11)
    lesioned = positive.copy()
    lesioned[0] = -3.0
    assert head.probability(positive) - head.probability(lesioned) > 0.6


def test_weak_head_remains_unadmitted_and_cannot_load(tmp_path: Path):
    head = NeuralUncertaintyHead.fit(
        _examples("train", separable=False),
        _examples("calibration", separable=False),
        hidden_width=4,
        seed=3,
        steps=100,
    )
    assert head.calibrated is False
    assert "auc_below_limit" in head.manifest()["failure_reasons"]
    path = tmp_path / "weak.json"
    digest = head.save(path)
    with pytest.raises(ValueError, match="failed calibration"):
        NeuralUncertaintyHead.load(path, expected_sha256=digest)


def test_training_and_calibration_tasks_must_be_disjoint():
    train = _examples("train")
    calibration = _examples("calibration")
    calibration[0] = HiddenStateCorrectnessExample(
        example_id=calibration[0].example_id,
        task_id=train[0].task_id,
        hidden_state=calibration[0].hidden_state,
        correct=calibration[0].correct,
        state_sha256=calibration[0].state_sha256,
        outcome_receipt_sha256=calibration[0].outcome_receipt_sha256,
        outcome_verifier_id=calibration[0].outcome_verifier_id,
    )
    with pytest.raises(ValueError, match="splits overlap"):
        NeuralUncertaintyHead.fit(train, calibration)


def test_duplicate_state_or_outcome_evidence_is_rejected():
    examples = _examples("train")
    first = examples[0]
    second = examples[1]
    examples[1] = HiddenStateCorrectnessExample(
        example_id=second.example_id,
        task_id=second.task_id,
        hidden_state=second.hidden_state,
        correct=second.correct,
        state_sha256=first.state_sha256,
        outcome_receipt_sha256=second.outcome_receipt_sha256,
        outcome_verifier_id=second.outcome_verifier_id,
    )
    with pytest.raises(ValueError, match="duplicate evidence"):
        NeuralUncertaintyHead.fit(examples, _examples("calibration"))


def test_pinned_artifact_round_trip_and_rehashed_tamper_rejection(tmp_path: Path):
    head = _fit()
    path = tmp_path / "head.json"
    digest = head.save(path)
    loaded = NeuralUncertaintyHead.load(path, expected_sha256=digest)
    assert loaded.probability([3.0] + [0.0] * 11) == pytest.approx(
        head.probability([3.0] + [0.0] * 11)
    )

    payload = json.loads(path.read_text())
    payload["manifest"]["metrics"]["auc"] = 0.1
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    forged_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="admission verdict"):
        NeuralUncertaintyHead.load(path, expected_sha256=forged_digest)


def test_pinned_loader_rejects_changed_bytes_and_symlink(tmp_path: Path):
    head = _fit()
    target = tmp_path / "head.json"
    digest = head.save(target)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        NeuralUncertaintyHead.load(target, expected_sha256=digest)

    digest = head.save(target)
    link = tmp_path / "head-link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        NeuralUncertaintyHead.load(link, expected_sha256=digest)


def test_example_requires_independently_committed_outcome():
    with pytest.raises(ValueError, match="outcome_receipt_sha256"):
        HiddenStateCorrectnessExample(
            example_id="example",
            task_id="task",
            hidden_state=(0.0, 1.0),
            correct=True,
            state_sha256=_digest("state"),
            outcome_receipt_sha256="",
            outcome_verifier_id="grader",
        )
