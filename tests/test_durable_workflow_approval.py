"""The approval pause used to be a one-way door.

The module docstring promised a workflow paused for approval ran "until
resume() is called". resume() called run(), run() reached the same
human_approval step and paused again. There was nowhere to record that a human
had said yes, so a workflow that asked permission could never receive it.

Every existing test asserted the pause happened and that the pause was
discoverable. None asserted the workflow could ever proceed — the absence of a
capability reported as a passing test.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.durable_workflow import (
    APPROVALS_KEY,
    DurableWorkflowEngine,
    RetryPolicy,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStore,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def engine(tmp_path):
    return DurableWorkflowEngine(store=WorkflowStore(root=tmp_path))


def _steps(recorder):
    return [
        WorkflowStep(step_id="x", name="x", apply=lambda o: recorder.append("x") or "X"),
        WorkflowStep(
            step_id="y", name="y",
            apply=lambda o: recorder.append("y") or "Y",
            human_approval=True,
        ),
        WorkflowStep(step_id="z", name="z", apply=lambda o: recorder.append("z") or "Z"),
    ]


# ── the defect ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_approved_workflow_actually_proceeds(engine):
    """The whole point. This could not happen before."""
    ran = []
    checkpoint = await engine.run("flow", _steps(ran), workflow_id="w")
    assert checkpoint.status is WorkflowStatus.PAUSED_FOR_APPROVAL

    await engine.approve("w", "y", approver="Bryan")
    resumed = await engine.resume("w", _steps(ran))

    assert resumed.status is WorkflowStatus.COMPLETED
    assert ran == ["x", "y", "z"]


@pytest.mark.asyncio
async def test_resuming_without_a_decision_pauses_again_rather_than_running(engine):
    """Re-entry is not consent."""
    ran = []
    await engine.run("flow", _steps(ran), workflow_id="w")

    resumed = await engine.resume("w", _steps(ran))

    assert resumed.status is WorkflowStatus.PAUSED_FOR_APPROVAL
    assert "y" not in ran


@pytest.mark.asyncio
async def test_the_approved_step_does_not_re_run_the_completed_prefix(engine):
    ran = []
    await engine.run("flow", _steps(ran), workflow_id="w")
    await engine.approve("w", "y", approver="Bryan")

    await engine.resume("w", _steps(ran))

    assert ran.count("x") == 1


# ── denial ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_denial_cancels_rather_than_deadlocks(engine):
    ran = []
    await engine.run("flow", _steps(ran), workflow_id="w")

    await engine.deny("w", "y", approver="Bryan", note="not on a Friday")
    resumed = await engine.resume("w", _steps(ran))

    assert resumed.status is WorkflowStatus.CANCELED
    assert "y" not in ran
    assert "z" not in ran


@pytest.mark.asyncio
async def test_a_denial_records_who_refused_and_why(engine):
    await engine.run("flow", _steps([]), workflow_id="w")
    await engine.deny("w", "y", approver="Bryan", note="not on a Friday")

    resumed = await engine.resume("w", _steps([]))

    assert "Bryan" in resumed.failure_reason
    assert "not on a Friday" in resumed.failure_reason


@pytest.mark.asyncio
async def test_a_cancelled_workflow_is_not_owed_more_work(engine):
    await engine.run("flow", _steps([]), workflow_id="w")
    await engine.deny("w", "y", approver="Bryan")
    await engine.resume("w", _steps([]))

    assert engine.store.unfinished() == []


# ── the decision carries content ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_approval_can_carry_a_value_to_the_step(engine):
    """'Yes, but with this budget' is most of what approvals are for."""
    seen = {}

    steps = [
        WorkflowStep(
            step_id="spend", name="spend",
            apply=lambda o: seen.update(o.get(APPROVALS_KEY, {})) or "done",
            human_approval=True,
        ),
    ]
    await engine.run("flow", steps, workflow_id="w")
    await engine.approve("w", "spend", approver="Bryan", value={"budget_usd": 25})

    await engine.resume("w", steps)

    assert seen["spend"]["value"] == {"budget_usd": 25}
    assert seen["spend"]["approver"] == "Bryan"


@pytest.mark.asyncio
async def test_the_decision_survives_a_new_engine_process(tmp_path):
    """An approval that lives only in a caller's memory never happened."""
    ran = []
    first = DurableWorkflowEngine(store=WorkflowStore(root=tmp_path))
    await first.run("flow", _steps(ran), workflow_id="w")
    await first.approve("w", "y", approver="Bryan")

    second = DurableWorkflowEngine(store=WorkflowStore(root=tmp_path))
    resumed = await second.resume("w", _steps(ran))

    assert resumed.status is WorkflowStatus.COMPLETED


# ── discovering what is waiting ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_approvals_are_discoverable_without_knowing_the_id(engine):
    await engine.run("flow", _steps([]), workflow_id="w")

    assert engine.pending_approvals() == [("w", "y")]


@pytest.mark.asyncio
async def test_nothing_is_pending_once_the_decision_lands(engine):
    await engine.run("flow", _steps([]), workflow_id="w")
    await engine.approve("w", "y", approver="Bryan")
    await engine.resume("w", _steps([]))

    assert engine.pending_approvals() == []


@pytest.mark.asyncio
async def test_approving_an_unknown_workflow_is_an_error_not_a_silent_no_op(engine):
    with pytest.raises(LookupError):
        await engine.approve("nope", "y", approver="Bryan")


# ── retries ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_step_retries_up_to_its_policy(engine):
    attempts = {"n": 0}

    def flaky(outs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    checkpoint = await engine.run("flow", [
        WorkflowStep(
            step_id="f", name="f", apply=flaky,
            retry=RetryPolicy(max_attempts=3),
        ),
    ])

    assert checkpoint.status is WorkflowStatus.COMPLETED
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_a_step_without_a_policy_is_attempted_once(engine):
    attempts = {"n": 0}

    def always_fails(outs):
        attempts["n"] += 1
        raise RuntimeError("nope")

    checkpoint = await engine.run(
        "flow", [WorkflowStep(step_id="f", name="f", apply=always_fails)]
    )

    assert checkpoint.status is WorkflowStatus.FAILED
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_rollback_runs_only_after_retries_are_exhausted(engine):
    rolled = []

    def always_fails(outs):
        raise RuntimeError("nope")

    await engine.run("flow", [
        WorkflowStep(
            step_id="f", name="f", apply=always_fails,
            rollback=lambda o: rolled.append("rb"),
            retry=RetryPolicy(max_attempts=3),
        ),
    ])

    assert rolled == ["rb"]


def test_backoff_grows_and_never_precedes_the_first_attempt():
    policy = RetryPolicy(max_attempts=4, backoff_seconds=1.0, multiplier=2.0)

    assert policy.delay_before(1) == 0.0
    assert policy.delay_before(2) == 1.0
    assert policy.delay_before(3) == 2.0
    assert policy.delay_before(4) == 4.0


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": 0},
    {"backoff_seconds": -1},
    {"multiplier": 0.5},
])
def test_incoherent_retry_policies_are_refused(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


# ── history and forking ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_revision_is_kept_not_just_the_latest(engine):
    await engine.run("flow", [
        WorkflowStep(step_id="a", name="a", apply=lambda o: "A"),
        WorkflowStep(step_id="b", name="b", apply=lambda o: "B"),
    ], workflow_id="w")

    history = engine.store.history("w")

    assert len(history) > 2
    assert [c.revision for c in history] == sorted(c.revision for c in history)


@pytest.mark.asyncio
async def test_a_past_revision_loads_as_it_stood(engine):
    await engine.run("flow", [
        WorkflowStep(step_id="a", name="a", apply=lambda o: "A"),
        WorkflowStep(step_id="b", name="b", apply=lambda o: "B"),
    ], workflow_id="w")

    early = engine.store.load_revision("w", 2)

    assert early is not None
    assert "b" not in early.completed_steps


@pytest.mark.asyncio
async def test_forking_branches_without_disturbing_the_original(engine):
    ran = []
    await engine.run("flow", [
        WorkflowStep(step_id="a", name="a", apply=lambda o: ran.append("a") or "A"),
        WorkflowStep(step_id="b", name="b", apply=lambda o: ran.append("b") or "B"),
    ], workflow_id="w")

    forked = await engine.fork("w", at_revision=2, new_workflow_id="w2")

    assert forked.forked_from == "w"
    assert forked.forked_at_revision == 2
    original = engine.store.load("w")
    assert original.status is WorkflowStatus.COMPLETED
    assert original.workflow_id == "w"


@pytest.mark.asyncio
async def test_a_fork_re_runs_only_what_the_branch_point_had_not_committed(engine):
    ran = []

    def steps():
        return [
            WorkflowStep(step_id="a", name="a", apply=lambda o: ran.append("a") or "A"),
            WorkflowStep(step_id="b", name="b", apply=lambda o: ran.append("b") or "B"),
        ]

    await engine.run("flow", steps(), workflow_id="w")
    ran.clear()

    forked = await engine.fork("w", at_revision=2, new_workflow_id="w2")
    await engine.resume(forked.workflow_id, steps())

    assert "a" not in ran  # committed before the branch point
    assert "b" in ran      # not yet committed there


@pytest.mark.asyncio
async def test_forking_from_a_revision_that_does_not_exist_says_what_does(engine):
    await engine.run("flow", [
        WorkflowStep(step_id="a", name="a", apply=lambda o: "A"),
    ], workflow_id="w")

    with pytest.raises(LookupError, match="have"):
        await engine.fork("w", at_revision=999)


# ── the event loop ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpointing_does_not_block_the_event_loop(engine):
    """run() fsynced synchronously per step — the pattern that wedged the live
    loop for 20 minutes. A concurrent task must keep getting scheduled."""
    ticks = {"n": 0}

    async def ticker():
        try:
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(ticker())
    await engine.run("flow", [
        WorkflowStep(step_id=f"s{i}", name="s", apply=lambda o: "v")
        for i in range(5)
    ])
    task.cancel()
    await task

    assert ticks["n"] > 0
