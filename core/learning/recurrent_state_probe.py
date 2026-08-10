"""Task-disjoint probes for structured state in the live recurrent workspace."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

RECURRENT_STATE_PROBE_SCHEMA = "aura.recurrent_state_information_probe.v1"


@dataclass(frozen=True, slots=True)
class StateProbeObservation:
    """One private tensor paired with an exact program state."""

    task_id: str
    family: str
    program_depth: int
    recurrence_step: int
    field_names: tuple[str, ...]
    labels: tuple[int, ...]
    features: Any

    def __post_init__(self) -> None:
        shape = getattr(self.features, "shape", None)
        if (
            not self.task_id
            or not self.family
            or type(self.program_depth) is not int
            or self.program_depth < 1
            or type(self.recurrence_step) is not int
            or not 0 <= self.recurrence_step <= self.program_depth
            or len(self.field_names) < 3
            or len(set(self.field_names)) != len(self.field_names)
            or len(self.labels) != len(self.field_names)
            or any(type(value) is not int for value in self.labels)
            or not isinstance(shape, tuple)
            or len(shape) < 1
            or math.prod(shape) < 1
        ):
            raise ValueError("state probe observation is invalid")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _fit_predictions(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    *,
    classes: Any,
    regularization: float,
) -> Any:
    import numpy as np

    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("state probe feature matrices are incompatible")
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    fitted_x = (train_x - mean) / scale
    heldout_x = (validation_x - mean) / scale
    fitted_x = np.concatenate(
        [fitted_x, np.ones((fitted_x.shape[0], 1), dtype=np.float64)], axis=1
    )
    heldout_x = np.concatenate(
        [heldout_x, np.ones((heldout_x.shape[0], 1), dtype=np.float64)], axis=1
    )
    targets = (train_y[:, None] == classes[None, :]).astype(np.float64)
    width = max(1, fitted_x.shape[1])
    gram = (fitted_x @ fitted_x.T) / width
    ridge = float(regularization) * max(float(np.trace(gram)) / len(gram), 1e-9)
    dual = np.linalg.solve(gram + ridge * np.eye(len(gram)), targets)
    scores = ((heldout_x @ fitted_x.T) / width) @ dual
    return classes[np.argmax(scores, axis=1)]


def _accuracy(predicted: Any, expected: Any) -> float:
    import numpy as np

    return float(np.mean(np.asarray(predicted) == np.asarray(expected)))


def evaluate_recurrent_state_information(
    training: Sequence[StateProbeObservation],
    validation: Sequence[StateProbeObservation],
    *,
    regularization: float = 1.0,
    null_seed: int = 20260810177,
    information_margin: float = 0.05,
) -> dict[str, Any]:
    """Fit linear state decoders and classify the recurrent failure mode."""

    import numpy as np

    if (
        not training
        or not validation
        or isinstance(regularization, bool)
        or not isinstance(regularization, (int, float))
        or not math.isfinite(float(regularization))
        or float(regularization) <= 0.0
        or type(null_seed) is not int
        or null_seed < 0
        or isinstance(information_margin, bool)
        or not isinstance(information_margin, (int, float))
        or not 0.0 <= float(information_margin) <= 1.0
    ):
        raise ValueError("state probe configuration is invalid")
    train_ids = {row.task_id for row in training}
    validation_ids = {row.task_id for row in validation}
    if train_ids & validation_ids:
        raise ValueError("state probe task identities overlap")
    families = sorted({row.family for row in training} | {row.family for row in validation})
    if not families:
        raise ValueError("state probe has no families")
    rng = np.random.default_rng(null_seed)
    family_reports: dict[str, Any] = {}
    for family in families:
        train_rows = [row for row in training if row.family == family]
        validation_rows = [row for row in validation if row.family == family]
        if not train_rows or not validation_rows:
            raise ValueError("state probe family is missing from one split")
        field_names = train_rows[0].field_names
        if any(row.field_names != field_names for row in (*train_rows, *validation_rows)):
            raise ValueError("state probe family field schemas differ")
        train_x = np.stack(
            [np.asarray(row.features, dtype=np.float64).reshape(-1) for row in train_rows]
        )
        validation_x = np.stack(
            [np.asarray(row.features, dtype=np.float64).reshape(-1) for row in validation_rows]
        )
        if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(validation_x)):
            raise ValueError("state probe features must be finite")
        field_reports: dict[str, Any] = {}
        for field_index, field_name in enumerate(field_names):
            train_y = np.asarray(
                [row.labels[field_index] for row in train_rows], dtype=np.int64
            )
            validation_y = np.asarray(
                [row.labels[field_index] for row in validation_rows], dtype=np.int64
            )
            classes = np.unique(np.concatenate([train_y, validation_y]))
            predictions = _fit_predictions(
                train_x,
                train_y,
                validation_x,
                classes=classes,
                regularization=float(regularization),
            )
            shuffled_y = rng.permutation(train_y)
            null_predictions = _fit_predictions(
                train_x,
                shuffled_y,
                validation_x,
                classes=classes,
                regularization=float(regularization),
            )
            majority_classes, majority_counts = np.unique(train_y, return_counts=True)
            majority_label = int(majority_classes[np.argmax(majority_counts)])

            def metrics(
                indices: Sequence[int],
                *,
                expected_all: Any = validation_y,
                predicted_all: Any = predictions,
                null_all: Any = null_predictions,
                majority: int = majority_label,
            ) -> dict[str, Any]:
                selected = np.asarray(tuple(indices), dtype=np.int64)
                if selected.size == 0:
                    return {"n": 0, "accuracy": None, "baseline": None, "null": None}
                expected = expected_all[selected]
                return {
                    "n": int(selected.size),
                    "accuracy": _accuracy(predicted_all[selected], expected),
                    "baseline": _accuracy(
                        np.full(selected.shape, majority, dtype=np.int64), expected
                    ),
                    "null": _accuracy(null_all[selected], expected),
                }

            by_step = {
                str(step): metrics(
                    [
                        index
                        for index, row in enumerate(validation_rows)
                        if row.recurrence_step == step
                    ]
                )
                for step in sorted({row.recurrence_step for row in validation_rows})
            }
            initial = metrics(
                [index for index, row in enumerate(validation_rows) if row.recurrence_step == 0]
            )
            terminal = metrics(
                [
                    index
                    for index, row in enumerate(validation_rows)
                    if row.recurrence_step == row.program_depth
                ]
            )
            initial_effect = float(initial["accuracy"]) - max(
                float(initial["baseline"]), float(initial["null"])
            )
            terminal_effect = float(terminal["accuracy"]) - max(
                float(terminal["baseline"]), float(terminal["null"])
            )
            change = float(terminal["accuracy"]) - float(initial["accuracy"])
            if max(initial_effect, terminal_effect) < float(information_margin):
                disposition = "not_linearly_recoverable"
            elif change >= float(information_margin):
                disposition = "recoverability_improves"
            elif change <= -float(information_margin):
                disposition = "recoverability_erodes"
            else:
                disposition = "recoverability_preserved"
            field_reports[field_name] = {
                "classes": [int(value) for value in classes],
                "majority_label": majority_label,
                "initial": initial,
                "terminal": terminal,
                "by_recurrence_step": by_step,
                "initial_effect_over_controls": initial_effect,
                "terminal_effect_over_controls": terminal_effect,
                "terminal_minus_initial_accuracy": change,
                "disposition": disposition,
            }
        target_field = field_names[1]
        family_reports[family] = {
            "field_names": list(field_names),
            "training_observations": len(train_rows),
            "validation_observations": len(validation_rows),
            "task_ids_disjoint": True,
            "target_field": target_field,
            "target_disposition": field_reports[target_field]["disposition"],
            "fields": field_reports,
        }
    body = {
        "schema": RECURRENT_STATE_PROBE_SCHEMA,
        "configuration": {
            "regularization": float(regularization),
            "null_seed": null_seed,
            "information_margin": float(information_margin),
        },
        "training_task_count": len(train_ids),
        "validation_task_count": len(validation_ids),
        "task_ids_disjoint": True,
        "families": family_reports,
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


__all__ = [
    "RECURRENT_STATE_PROBE_SCHEMA",
    "StateProbeObservation",
    "evaluate_recurrent_state_information",
]
