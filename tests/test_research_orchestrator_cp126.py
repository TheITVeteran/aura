"""Autonomous research: failed depth and failed persistence became completed
progress, and one corrupt byte erased the history."""
from __future__ import annotations

import asyncio
import json

import pytest

import core.autonomy.autonomous_research_orchestrator as aro

pytestmark = pytest.mark.unit


# ── the one-engagement contract must be enforced, not documented ───────────


def test_run_once_is_serialized():
    """run_once and _engage had no lock, so concurrent callers could select the
    SAME item and race the shared scheduler, progress log, cache, and memory."""
    orchestrator = aro.AutonomousResearchOrchestrator.__new__(
        aro.AutonomousResearchOrchestrator
    )
    orchestrator._engagement_lock = asyncio.Lock()
    picked = []
    concurrent = {"max": 0, "now": 0}

    class _Scheduler:
        def pick_next(self):
            picked.append(1)
            return object()

    orchestrator._scheduler = _Scheduler()

    async def _engage(decision):
        concurrent["now"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["now"])
        await asyncio.sleep(0.01)
        concurrent["now"] -= 1
        return "done"

    orchestrator._engage = _engage

    async def _run():
        await asyncio.gather(*(orchestrator.run_once() for _ in range(5)))

    asyncio.run(_run())

    assert concurrent["max"] == 1, "engagements must not overlap"
    assert len(picked) == 5


# ── a corrupt read must not destroy the history ────────────────────────────


def test_corrupt_progress_is_quarantined_not_overwritten(tmp_path, monkeypatch):
    """A failed LOAD produced a fresh empty ProgressLog, which the save then
    wrote to the canonical path — one unreadable byte silently replaced the
    entire durable research history."""
    progress = tmp_path / "progress.json"
    progress.write_text("{ this is not json")
    monkeypatch.setattr(
        "core.autonomy.content_progress_tracker._default_progress_path",
        lambda: progress,
    )

    quarantined = aro._quarantine_corrupt_progress()

    assert quarantined is not None
    assert quarantined.exists(), "the prior history must still exist on disk"
    assert quarantined.read_text() == "{ this is not json"
    assert not progress.exists()


def test_quarantine_never_raises_on_a_failure_path(tmp_path, monkeypatch):
    """It runs while already handling corruption; it must not turn a
    recoverable problem into a crash."""
    monkeypatch.setattr(
        "core.autonomy.content_progress_tracker._default_progress_path",
        lambda: tmp_path / "does-not-exist.json",
    )

    assert aro._quarantine_corrupt_progress() is None


def test_quarantine_survives_an_unmovable_file(monkeypatch):
    def _boom():
        raise RuntimeError("path resolution exploded")

    monkeypatch.setattr(
        "core.autonomy.content_progress_tracker._default_progress_path", _boom
    )

    assert aro._quarantine_corrupt_progress() is None


# ── results must describe what actually happened ───────────────────────────


def test_engagement_result_reports_persistence_and_withholding():
    """A rejected receipt was copied into the result and then ignored, so
    progress, scheduler outcome, completion and session phase all said
    'completed' even when the memory commit had entirely failed."""
    result = aro.EngagementResult(item_title="x", started_at=0.0)

    assert result.persisted is True
    assert result.epistemic_content_withheld is False

    result.persisted = False
    result.epistemic_content_withheld = True
    payload = result.to_dict()

    assert payload["persisted"] is False
    assert payload["epistemic_content_withheld"] is True


def test_result_dict_round_trips_as_json():
    """Session checkpoints are written as JSON; new fields must survive."""
    result = aro.EngagementResult(item_title="x", started_at=0.0)
    result.persisted = False

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["persisted"] is False
