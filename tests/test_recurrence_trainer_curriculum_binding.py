"""Non-MLX tests for the trainer's frozen curriculum source boundary."""

from __future__ import annotations

import hashlib

import pytest

from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from tools.recurrence_native_train_v2 import (
    MAX_CURRICULUM_SOURCE_BYTES,
    TASK_GENERATOR_SOURCE,
    _build_parser,
    _capture_frozen_curriculum,
    _execute_frozen_curriculum,
)


def test_default_curriculum_is_broad_and_hash_bound_to_exact_source_bytes():
    captured = _capture_frozen_curriculum()
    args = _build_parser().parse_args(["--model", "/model", "--out-dir", "/out"])
    assert args.families == RECURRENCE_TRAINING_FAMILIES
    assert captured.families == RECURRENCE_TRAINING_FAMILIES
    assert len(captured.families) == 12
    assert captured.binding == {
        "path": "core/learning/recurrence_curriculum.py",
        "sha256": hashlib.sha256(captured.source_bytes).hexdigest(),
        "size_bytes": len(captured.source_bytes),
    }
    assert captured.source_bytes == TASK_GENERATOR_SOURCE.read_bytes()


def test_frozen_curriculum_executes_captured_bytes_not_later_disk_contents():
    source = TASK_GENERATOR_SOURCE.read_bytes()
    marker = b'CURRICULUM_VERSION = "2026.07.18.1"'
    replacement = b'CURRICULUM_VERSION = "2026.07.18.9"'
    assert marker in source and len(marker) == len(replacement)
    frozen = _execute_frozen_curriculum(
        source.replace(marker, replacement, 1),
        origin=TASK_GENERATOR_SOURCE,
    )
    current = _capture_frozen_curriculum()
    frozen_task = frozen.task_battery(("khop",), (2,), 1, seed=1777)[0]
    current_task = current.task_battery(("khop",), (2,), 1, seed=1777)[0]
    assert frozen.binding["sha256"] != current.binding["sha256"]
    assert frozen_task != current_task


@pytest.mark.parametrize("source", (b"", b"x" * (MAX_CURRICULUM_SOURCE_BYTES + 1)))
def test_frozen_curriculum_rejects_unbounded_source(source: bytes):
    with pytest.raises(RuntimeError, match="source size"):
        _execute_frozen_curriculum(source, origin=TASK_GENERATOR_SOURCE)
