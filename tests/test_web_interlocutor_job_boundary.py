"""Background dialogue jobs share one visible browser and one user's transcripts.

Three CP126 findings in WebInterlocutorJobManager.

6633b065 — the docstring said jobs "are not parallel foreground typists" and
nothing enforced it. Two jobs could be active, each with its own task, both
navigating, focusing, pasting and submitting into the same visible browser.

d0b864a8 — start, status and cancel took no caller identity, so any caller
holding the singleton and a job id could read a full conversation transcript or
cancel somebody else's work, and listing handed out every transcript at once.

83e11ce9 — jobs, tasks, transcripts and errors were retained forever.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.capabilities.web_interlocutor import (
    WebInterlocutorJob,
    WebInterlocutorJobManager,
    _job_owner,
)

pytestmark = pytest.mark.unit


class _Session:
    """A session that records when it holds the browser and for how long."""

    log: list[tuple[str, float]] = []

    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, **kwargs):
        objective = kwargs.get("objective", "")
        _Session.log.append(("enter", time.monotonic(), objective))
        await asyncio.sleep(0.05)
        _Session.log.append(("exit", time.monotonic(), objective))
        return {"ok": True, "objective": objective}


@pytest.fixture(autouse=True)
def _clear_log():
    _Session.log = []
    yield
    _Session.log = []


# --- one typist at a time (6633b065) ------------------------------------


@pytest.mark.asyncio
async def test_two_jobs_never_drive_the_browser_at_once():
    manager = WebInterlocutorJobManager(max_jobs=2)

    first = manager.start(objective="one", session_factory=_Session)
    second = manager.start(objective="two", session_factory=_Session)
    assert first["ok"] and second["ok"]

    await asyncio.gather(*manager._tasks.values(), return_exceptions=True)

    spans = [entry for entry in _Session.log if entry[0] in {"enter", "exit"}]
    depth = 0
    for kind, _ts, _obj in spans:
        depth += 1 if kind == "enter" else -1
        assert depth <= 1, "two jobs held the visible browser simultaneously"


@pytest.mark.asyncio
async def test_both_jobs_still_run():
    """Serializing must not drop the second job."""
    manager = WebInterlocutorJobManager(max_jobs=2)
    manager.start(objective="one", session_factory=_Session)
    manager.start(objective="two", session_factory=_Session)

    await asyncio.gather(*manager._tasks.values(), return_exceptions=True)

    finished = {entry[2] for entry in _Session.log if entry[0] == "exit"}
    assert finished == {"one", "two"}


# --- a job belongs to whoever asked for it (d0b864a8) -------------------


def _job(manager, owner, status="completed"):
    job = WebInterlocutorJob(
        job_id=f"job-{owner}-{status}",
        status=status,
        objective="private objective",
        started_at=time.time(),
        updated_at=time.time(),
        owner=owner,
        result={"transcript": "everything that was said"},
    )
    manager._jobs[job.job_id] = job
    return job


def test_another_principals_job_is_not_readable():
    manager = WebInterlocutorJobManager()
    job = _job(manager, "someone_else")

    result = manager.status(job.job_id, context={"principal": "me"})

    assert result["ok"] is False
    assert result["status"] == "not_found"


def test_another_principals_job_cannot_be_cancelled():
    manager = WebInterlocutorJobManager()
    job = _job(manager, "someone_else", status="running")

    result = manager.cancel(job.job_id, context={"principal": "me"})

    assert result["ok"] is False


def test_your_own_job_is_readable():
    manager = WebInterlocutorJobManager()
    job = _job(manager, "me")

    result = manager.status(job.job_id, context={"principal": "me"})

    assert result["ok"] is True
    assert result["job"]["result"]["transcript"] == "everything that was said"


def test_listing_does_not_hand_out_transcripts():
    manager = WebInterlocutorJobManager()
    _job(manager, "me")

    listed = manager.status(context={"principal": "me"})

    assert listed["jobs"]
    assert "result" not in listed["jobs"][0]
    assert "everything that was said" not in str(listed)


def test_listing_only_shows_your_own_jobs():
    manager = WebInterlocutorJobManager()
    _job(manager, "me")
    _job(manager, "someone_else")

    listed = manager.status(context={"principal": "me"})

    assert [job["job_id"] for job in listed["jobs"]] == ["job-me-completed"]


def test_the_owner_is_read_from_the_context():
    assert _job_owner({"authenticated_principal": "bryan"}) == "bryan"
    assert _job_owner({"principal": "bryan"}) == "bryan"
    assert _job_owner({}) == "local_owner"
    assert _job_owner(None) == "local_owner"
    assert _job_owner({"principal": "   "}) == "local_owner"


def test_a_job_payload_never_carries_the_owner_string():
    """The record identifies whose job it is; the payload should not echo it."""
    manager = WebInterlocutorJobManager()
    job = _job(manager, "me")

    assert "owner" not in job.to_dict()
    assert "owner" not in job.summary()


# --- finished work does not accumulate forever (83e11ce9) ---------------


def test_expired_jobs_are_dropped():
    from core.capabilities.web_interlocutor import _JOB_RESULT_TTL_S

    manager = WebInterlocutorJobManager()
    job = _job(manager, "me")
    job.updated_at = time.time() - (_JOB_RESULT_TTL_S + 1.0)

    manager._retire_finished_jobs()

    assert job.job_id not in manager._jobs


def test_retention_is_capped_by_count():
    from core.capabilities.web_interlocutor import _MAX_RETAINED_FINISHED_JOBS

    manager = WebInterlocutorJobManager()
    for index in range(_MAX_RETAINED_FINISHED_JOBS * 3):
        job = WebInterlocutorJob(
            job_id=f"job-{index}",
            status="completed",
            objective="o",
            started_at=time.time(),
            updated_at=time.time() + index,
            owner="me",
        )
        manager._jobs[job.job_id] = job

    manager._retire_finished_jobs()

    assert len(manager._jobs) == _MAX_RETAINED_FINISHED_JOBS
    # The most recent survive.
    assert "job-59" in manager._jobs


def test_running_jobs_are_never_evicted():
    manager = WebInterlocutorJobManager()
    running = WebInterlocutorJob(
        job_id="live",
        status="running",
        objective="o",
        started_at=0.0,
        updated_at=0.0,
        owner="me",
    )
    manager._jobs["live"] = running

    manager._retire_finished_jobs()

    assert "live" in manager._jobs
