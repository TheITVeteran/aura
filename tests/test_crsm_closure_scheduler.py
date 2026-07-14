"""Contract tests for the autonomous CRSM→LoRA loop closer.

The scheduler mutates weights on the resident 32B, so every admission gate
must hold and no gate may be skippable: default-OFF, kill switch, loop must
actually be open, deep-idle only, real RAM headroom, Will approval, single
flight. The happy path must run exactly the monitor's next_action command
and record closure only on a clean run. Nothing here spawns real training —
the monitor and the subprocess gateway are doubled.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.learning import crsm_closure_scheduler as mod
from core.learning.crsm_closure_scheduler import (
    CRSMClosureScheduler,
    reset_crsm_closure_scheduler_for_test,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_crsm_closure_scheduler_for_test()
    yield
    reset_crsm_closure_scheduler_for_test()


class _FakeMonitor:
    def __init__(self, state, *, close_on_consume=True):
        self._state = dict(state)
        self.close_on_consume = close_on_consume
        self.next_action_calls = 0

    def loop_state(self):
        return dict(self._state)

    def next_action(self, *a, **k):
        self.next_action_calls += 1
        return {"command": ["python", "training/train_and_fuse.py", "--crsm-delta"]}

    def consume(self):
        if self.close_on_consume:
            self._state = {"state": "closed", "reason": "trained in", "unconsumed": 0}


def _install(monkeypatch, scheduler, monitor, *, ram_gb=64.0, will_ok=True,
             idle_ok=True, rc=0):
    monkeypatch.setattr(mod, "get_crsm_loop_monitor", lambda: monitor, raising=False)
    # Patch the lazy import target too.
    import core.consciousness.crsm_loop_monitor as clm
    monkeypatch.setattr(clm, "get_crsm_loop_monitor", lambda: monitor, raising=False)

    monkeypatch.setattr(scheduler, "_ram_admits",
                        lambda: (ram_gb >= scheduler.min_free_gb,
                                 f"free_ram:{ram_gb}GB" if ram_gb >= scheduler.min_free_gb
                                 else f"insufficient_free_ram:{ram_gb}"))
    monkeypatch.setattr(scheduler, "_idle_allows", lambda: idle_ok)
    monkeypatch.setattr(scheduler, "_will_approval", lambda ctx: (will_ok, "ok" if will_ok else "denied"))

    async def _fake_train(command):
        if rc == 0:
            monitor.consume()
        return {"returncode": rc, "stdout": "", "stderr": "boom" if rc else ""}

    monkeypatch.setattr(scheduler, "_run_training", _fake_train)
    # Governed scope is a no-op context in the hermetic test.
    import core.governance_context as gc

    class _NullScope:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(gc, "local_internal_governed_scope",
                        lambda *a, **k: _NullScope(), raising=False)


@pytest.mark.asyncio
async def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_CRSM_AUTOCLOSE", raising=False)
    sch = CRSMClosureScheduler()
    await sch.start()
    assert sch._task is None, "closer must not run unless explicitly enabled"


@pytest.mark.asyncio
async def test_run_now_blocked_when_disabled(monkeypatch):
    monkeypatch.delenv("AURA_CRSM_AUTOCLOSE", raising=False)
    sch = CRSMClosureScheduler()
    out = await sch.run_closure_now()
    assert out["status"] == "blocked"
    assert "disabled_by_env" in out["reasons"]


@pytest.mark.asyncio
async def test_noop_when_loop_not_open(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    monitor = _FakeMonitor({"state": "closed", "unconsumed": 0})
    _install(monkeypatch, sch, monitor)
    out = await sch.run_closure_now()
    assert out["status"] == "noop"
    assert monitor.next_action_calls == 0


@pytest.mark.asyncio
async def test_deferred_on_insufficient_ram(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    sch.min_free_gb = 40.0
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33})
    _install(monkeypatch, sch, monitor, ram_gb=12.0)  # below floor
    out = await sch.run_closure_now()
    assert out["status"] == "deferred"
    assert monitor.next_action_calls == 0  # never started training
    assert monitor.loop_state()["state"] == "open"  # captures preserved


@pytest.mark.asyncio
async def test_will_denial_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33})
    _install(monkeypatch, sch, monitor, will_ok=False)
    out = await sch.run_closure_now()
    assert out["status"] == "will_declined"
    assert monitor.loop_state()["state"] == "open"


@pytest.mark.asyncio
async def test_happy_path_closes_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33})
    _install(monkeypatch, sch, monitor)
    out = await sch.run_closure_now(reason="unit")
    assert out["status"] == "closed"
    assert out["loop"]["state"] == "closed"
    assert monitor.next_action_calls == 1


@pytest.mark.asyncio
async def test_train_failure_keeps_loop_open(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33}, close_on_consume=False)
    _install(monkeypatch, sch, monitor, rc=2)
    out = await sch.run_closure_now()
    assert out["status"] == "train_failed"
    assert out["returncode"] == 2
    assert monitor.loop_state()["state"] == "open"  # honest: still open


@pytest.mark.asyncio
async def test_scheduled_idle_respects_cooldown_and_idle(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    sch.cooldown_s = 10_000.0
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33})

    # idle gate closed -> no closure even with an open loop
    _install(monkeypatch, sch, monitor, idle_ok=False)
    await sch._maybe_close()
    assert monitor.next_action_calls == 0

    # idle opens, but a fresh attempt marker enforces cooldown
    _install(monkeypatch, sch, monitor, idle_ok=True)
    import json as _json
    sch._state_path.write_text(_json.dumps({"last_attempt_at": __import__("time").time()}))
    await sch._maybe_close()
    assert monitor.next_action_calls == 0


@pytest.mark.asyncio
async def test_single_flight_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    sch._running_cycle = True
    out = await sch.run_closure_now()
    assert out["status"] == "blocked"
    assert "closure_already_running" in out["reasons"]


def test_get_status_reports_enabled_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "1")
    sch = CRSMClosureScheduler()
    sch._state_path = tmp_path / "s.json"
    monitor = _FakeMonitor({"state": "open", "unconsumed": 33})
    _install(monkeypatch, sch, monitor)
    status = sch.get_status()
    assert status["enabled"] is True
    assert status["loop"]["state"] == "open"
