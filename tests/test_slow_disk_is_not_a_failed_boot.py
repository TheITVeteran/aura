"""A slow disk must not be indistinguishable from a broken one.

How long opening a database takes is a property of the disk and of how much
history is already in it, not of whether the runtime is healthy. Two startup
steps were awaited on hardcoded budgets — 15s for the state repository, 10s for
the database coordinator — with no operator knob, and a TimeoutError there
propagates straight out of boot.

Observed live 2026-07-26 on a busy host:

    core/orchestrator/boot.py:478
    await asyncio.wait_for(self.state_repo.initialize(), timeout=15.0)
    TimeoutError

Aura did not come up at all. That is the worst available failure: not a degraded
answer, not a slow answer — no runtime.

The budget is now tunable and generous, one timeout buys a second attempt at
double the budget, and only then is the failure fatal — recorded with the
elapsed time so the cause is attributable instead of a bare traceback.
"""
from __future__ import annotations

import asyncio

import pytest

from core.orchestrator.boot import _await_startup_io


def _slow_start(delay: float):
    async def _start():
        await asyncio.sleep(delay)

    return _start


@pytest.mark.asyncio
async def test_a_fast_step_just_succeeds() -> None:
    await _await_startup_io(
        _slow_start(0.0), what="state repository", env_var="X_UNSET", default_s=5.0
    )


@pytest.mark.asyncio
async def test_a_step_slower_than_the_budget_gets_a_second_chance() -> None:
    """The first attempt times out; the retry at double the budget succeeds."""
    calls = {"n": 0}

    async def _start():
        calls["n"] += 1
        # Too slow for the 5s first attempt, fine for the 10s retry.
        await asyncio.sleep(0.0 if calls["n"] > 1 else 10.0)

    await _await_startup_io(
        _start, what="state repository", env_var="X_UNSET", default_s=0.05
    )
    assert calls["n"] == 2, "a slow first attempt must be retried, not fatal"


@pytest.mark.asyncio
async def test_a_genuinely_stuck_step_still_fails_the_boot() -> None:
    """Tolerance is not blindness — twice over budget is a real failure."""
    with pytest.raises(TimeoutError):
        await _await_startup_io(
            _slow_start(30.0),
            what="state repository",
            env_var="X_UNSET",
            default_s=0.05,
        )


@pytest.mark.asyncio
async def test_the_budget_is_operator_tunable(monkeypatch) -> None:
    monkeypatch.setenv("AURA_STATE_REPO_INIT_TIMEOUT_S", "0.05")
    with pytest.raises(TimeoutError):
        await _await_startup_io(
            _slow_start(30.0),
            what="state repository",
            env_var="AURA_STATE_REPO_INIT_TIMEOUT_S",
            default_s=600.0,
        )


@pytest.mark.asyncio
async def test_a_junk_budget_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("AURA_STATE_REPO_INIT_TIMEOUT_S", "not-a-number")
    await _await_startup_io(
        _slow_start(0.0),
        what="state repository",
        env_var="AURA_STATE_REPO_INIT_TIMEOUT_S",
        default_s=5.0,
    )


def test_both_startup_steps_use_the_tolerant_path() -> None:
    """Neither hardcoded budget may come back."""
    from pathlib import Path

    src = Path("core/orchestrator/boot.py").read_text(encoding="utf-8")
    # Look at call sites, not the helper's docstring (which quotes the old one).
    assert "await asyncio.wait_for(self.state_repo.initialize()" not in src
    assert "await asyncio.wait_for(db_coord.start()" not in src
    assert "_await_startup_io(\n                        self.state_repo.initialize," in src
    assert "_await_startup_io(\n                    db_coord.start," in src
    assert "AURA_STATE_REPO_INIT_TIMEOUT_S" in src
    assert "AURA_DB_COORDINATOR_START_TIMEOUT_S" in src
