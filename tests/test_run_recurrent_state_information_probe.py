from __future__ import annotations

import json

import numpy as np

from core.learning.recurrence_curriculum import TASK_GENERATORS
from core.learning.recurrent_state_probe import StateProbeObservation
from tools.run_recurrent_state_information_probe import (
    PRIVATE_ARTIFACT_SCHEMA,
    _task_commitment,
    _write_private_artifact,
    _write_receipt,
)


def _observation(task_id: str, *, step: int) -> StateProbeObservation:
    return StateProbeObservation(
        task_id=task_id,
        family="modular",
        program_depth=1,
        recurrence_step=step,
        field_names=("pc", "residue", "done"),
        labels=(step, 7 + step, step),
        features=np.full((2, 3), step + 0.5, dtype=np.float32),
    )


def test_private_state_artifact_is_owner_only_and_tensor_bound(tmp_path) -> None:
    path = tmp_path / "states.npz"

    receipt = _write_private_artifact(
        path,
        training=[_observation("training", step=0)],
        validation=[_observation("validation", step=1)],
    )

    assert receipt["schema"] == PRIVATE_ARTIFACT_SCHEMA
    assert receipt["feature_shape"] == [2, 6]
    assert receipt["label_shape"] == [2, 3]
    assert path.stat().st_mode & 0o777 == 0o600
    with np.load(path, allow_pickle=False) as payload:
        assert payload["features"].shape == (2, 6)
        assert payload["labels"].tolist() == [[0, 7, 0], [1, 8, 1]]
        metadata = json.loads(payload["metadata"].tobytes())
    assert metadata["task_ids"] == ["training", "validation"]


def test_public_task_commitment_does_not_expose_private_states() -> None:
    task = TASK_GENERATORS["modular"](3, 17)

    commitment = _task_commitment(task)

    assert commitment["task_id"] == task.task_id
    assert "states" not in commitment["transition_trace"]
    assert commitment["transition_trace"]["trace_sha256"] == (
        task.transition_trace.trace_sha256
    )


def test_state_probe_receipt_is_canonical_and_self_bound(tmp_path) -> None:
    path = tmp_path / "receipt.json"

    receipt = _write_receipt(path, {"schema": "aura.test.v1", "admitted": True})

    assert path.read_bytes() == json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert path.stat().st_mode & 0o777 == 0o600
    assert len(receipt["receipt_sha256"]) == 64
