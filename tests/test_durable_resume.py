"""Resume: checkpoints nothing could find.

Two modules had the same shape of gap. core/runtime/durable_workflow.py could
save and load a checkpoint BY ID but had no way to discover unfinished work —
and knowing the id is exactly what a crash destroys, which is why its resume()
had zero call sites. The research orchestrator documented session resume and
wrote checkpoints for it, but nothing ever scanned for them.

Recovery has to start from "what was I doing?".
"""
from __future__ import annotations

import json

import pytest

from core.runtime.durable_workflow import (
    WorkflowCheckpoint,
    WorkflowStatus,
    WorkflowStore,
)

pytestmark = pytest.mark.unit


# ── workflow store: discovering work still owed ────────────────────────────


def _store(tmp_path, entries):
    store = WorkflowStore(root=tmp_path)
    for workflow_id, status in entries:
        store.save(WorkflowCheckpoint(workflow_id=workflow_id,
                                      objective="o", status=status))
    return store


def test_unfinished_finds_running_and_pending_work(tmp_path):
    store = _store(tmp_path, [
        ("running", WorkflowStatus.RUNNING),
        ("pending", WorkflowStatus.PENDING),
        ("done", WorkflowStatus.COMPLETED),
    ])

    assert sorted(c.workflow_id for c in store.unfinished()) == ["pending", "running"]


def test_work_paused_for_approval_is_still_owed(tmp_path):
    """It is waiting on a human, not finished. A restart must not lose it."""
    store = _store(tmp_path, [("waiting", WorkflowStatus.PAUSED_FOR_APPROVAL)])

    assert [c.workflow_id for c in store.unfinished()] == ["waiting"]


@pytest.mark.parametrize("terminal", [
    WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED,
])
def test_terminal_workflows_are_not_resumed(tmp_path, terminal):
    store = _store(tmp_path, [("t", terminal)])

    assert store.unfinished() == []


def test_a_corrupt_checkpoint_does_not_hide_the_others(tmp_path):
    """One bad file must not make every other resumable workflow invisible."""
    store = _store(tmp_path, [("good", WorkflowStatus.RUNNING)])
    (tmp_path / "corrupt.json").write_text("{ not json")
    (tmp_path / "empty.json").write_text("")

    assert [c.workflow_id for c in store.unfinished()] == ["good"]


def test_an_empty_store_is_not_an_error(tmp_path):
    assert WorkflowStore(root=tmp_path).unfinished() == []


# ── research sessions: the same gap, the same fix ──────────────────────────


def _orchestrator(tmp_path):
    from core.autonomy.autonomous_research_orchestrator import (
        AutonomousResearchOrchestrator,
    )

    orchestrator = AutonomousResearchOrchestrator.__new__(
        AutonomousResearchOrchestrator
    )
    orchestrator._sessions_dir = tmp_path
    return orchestrator


def _session(tmp_path, session_id, phase, *, title="An Item"):
    (tmp_path / f"{session_id}.json").write_text(json.dumps({
        "phase": phase,
        "result": {"item_title": title, "started_at": 1000.0},
    }))


def test_unfinished_sessions_finds_interrupted_engagements(tmp_path):
    _session(tmp_path, "s1", "comprehended")
    _session(tmp_path, "s2", "complete")

    found = _orchestrator(tmp_path).unfinished_sessions()

    assert [s["session_id"] for s in found] == ["s1"]
    assert found[0]["resumable"] is True
    assert found[0]["item_title"] == "An Item"


def test_a_failed_persist_is_resumable_not_terminal(tmp_path):
    """The research WAS done and only the commit failed — the most worthwhile
    thing to retry, and the easiest to lose."""
    _session(tmp_path, "s1", "persist_failed")

    found = _orchestrator(tmp_path).unfinished_sessions()

    assert len(found) == 1
    assert found[0]["resumable"] is True


def test_only_complete_counts_as_finished(tmp_path):
    for i, phase in enumerate(
        ["fetched", "comprehended", "reflected", "gated", "persisted"]
    ):
        _session(tmp_path, f"s{i}", phase)
    _session(tmp_path, "done", "complete")

    found = _orchestrator(tmp_path).unfinished_sessions()

    assert len(found) == 5
    assert all(s["resumable"] for s in found)
    assert "done" not in {s["session_id"] for s in found}


def test_a_corrupt_session_does_not_hide_the_others(tmp_path):
    _session(tmp_path, "good", "gated")
    (tmp_path / "corrupt.json").write_text("{{{")
    (tmp_path / "wrongshape.json").write_text('["not", "a", "dict"]')

    found = _orchestrator(tmp_path).unfinished_sessions()

    assert [s["session_id"] for s in found] == ["good"]


def test_scanning_a_missing_directory_is_not_an_error(tmp_path):
    orchestrator = _orchestrator(tmp_path / "does-not-exist")

    assert orchestrator.unfinished_sessions() == []
