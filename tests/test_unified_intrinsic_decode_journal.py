from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.canonical_json import canonical_json_bytes
from tools.unified_intrinsic_decode_journal import (
    DecodeProgressError,
    DecodeProgressJournal,
)


def _candidate(*, task: str = "task-a", arm: str = "trained_t4") -> dict[str, object]:
    return {
        "task_id": task,
        "family": "khop",
        "task_depth": 1,
        "prompt_sha256": "a" * 64,
        "arm": arm,
        "decoded": "7",
        "expected": "7",
        "token_ids": [22],
        "stopped_on_eos": True,
        "response_sha256": "b" * 64,
    }


def test_candidate_can_be_replayed_only_under_the_same_identity(tmp_path: Path) -> None:
    journal = DecodeProgressJournal(tmp_path / "progress", {"seed": 7, "depth": 4})
    candidate = _candidate()
    journal.commit(0, candidate)

    replay = DecodeProgressJournal(tmp_path / "progress", {"seed": 7, "depth": 4})

    assert replay.load(
        0,
        task_id="task-a",
        arm="trained_t4",
        prompt_sha256="a" * 64,
    ) == candidate
    with pytest.raises(DecodeProgressError, match="binding differs"):
        replay.load(
            0,
            task_id="task-a",
            arm="untrained_t4",
            prompt_sha256="a" * 64,
        )


def test_different_experiment_cannot_reuse_existing_manifest(tmp_path: Path) -> None:
    DecodeProgressJournal(tmp_path / "progress", {"seed": 7})

    with pytest.raises(DecodeProgressError, match="already differs"):
        DecodeProgressJournal(tmp_path / "progress", {"seed": 8})


def test_tampered_candidate_is_refused_even_with_parseable_json(tmp_path: Path) -> None:
    root = tmp_path / "progress"
    journal = DecodeProgressJournal(root, {"seed": 7})
    journal.commit(0, _candidate())
    path = root / "candidate-00000000.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["candidate"]["decoded"] = "8"
    path.chmod(0o600)
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(DecodeProgressError, match="hash differs"):
        journal.load(
            0,
            task_id="task-a",
            arm="trained_t4",
            prompt_sha256="a" * 64,
        )


def test_existing_sequence_cannot_be_replaced(tmp_path: Path) -> None:
    journal = DecodeProgressJournal(tmp_path / "progress", {"seed": 7})
    journal.commit(0, _candidate())

    with pytest.raises(DecodeProgressError, match="already differs"):
        journal.commit(0, _candidate(task="task-b"))


def test_non_finite_candidate_is_refused_through_stable_error(tmp_path: Path) -> None:
    root = tmp_path / "progress"
    journal = DecodeProgressJournal(root, {"seed": 7})
    journal.commit(0, _candidate())
    path = root / "candidate-00000000.json"
    path.chmod(0o600)
    path.write_text(
        '{"candidate":{"decoded":NaN},"record_sha256":"invalid"}\n',
        encoding="ascii",
    )

    with pytest.raises(DecodeProgressError, match="unreadable"):
        journal.load(
            0,
            task_id="task-a",
            arm="trained_t4",
            prompt_sha256="a" * 64,
        )


def test_symlinked_root_and_candidate_are_refused(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked_root = tmp_path / "linked-progress"
    linked_root.symlink_to(external, target_is_directory=True)
    with pytest.raises(DecodeProgressError, match="root must not be a symlink"):
        DecodeProgressJournal(linked_root, {"seed": 7})

    root = tmp_path / "progress"
    journal = DecodeProgressJournal(root, {"seed": 7})
    target = tmp_path / "candidate.json"
    target.write_bytes(canonical_json_bytes({"record_sha256": "invalid"}) + b"\n")
    (root / "candidate-00000000.json").symlink_to(target)

    with pytest.raises(DecodeProgressError, match="not a regular file"):
        journal.load(
            0,
            task_id="task-a",
            arm="trained_t4",
            prompt_sha256="a" * 64,
        )
