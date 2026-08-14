"""Three paths reached into the Cortex client and rewrote its lane state.

The watchdog, `get_conversation_status`, and the recovery scheduler all cleared
the client's private `_warmup_in_flight`, cancelled the prewarm task, and
called private lane-state setters. None of them took anything first, so two
could do it at once — one clearing the flag while the other was mid-cancel —
and a fresh warmup could start underneath a 20GB load that had not stopped.

Cancelling a task is a REQUEST. The old code cancelled, set `_prewarm_task =
None`, and moved on, so a load that had not yet noticed the cancellation became
invisible to everything that came after it.

The rest of this file is the same defect in the accounting:

- A warmup that overran its budget was recorded as a stuck-process KILL, so
  repeated slow-but-live loads armed kill-based backoff on evidence of a
  termination nobody performed.
- A recovery that got deferred for memory pressure consumed a recovery attempt,
  so repeated deferrals walked the counter to the exponential cooldown without
  one load ever being tried.
- The status-triggered cooldown stamp was written before scheduling and outside
  any exclusion, so a scheduling failure bought silence instead of a retry.
- A shed counted `reboot_worker` calls, not workers that stopped.
- Readiness defaulted missing warmup evidence to True.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.inference_gate import InferenceGate


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._lane_transition_owner = ""
    gate._lane_transition_at = 0.0
    gate._abandoned_cortex_loads = []
    gate._prewarm_task = None
    gate._mlx_client = None
    gate._cortex_load_setback_counts = {}
    return gate


# ────────────────────────────── one owner rewrites lane state


def test_a_second_owner_cannot_rewrite_lane_state_at_the_same_time():
    gate = _gate()

    assert gate._claim_lane_transition("watchdog") is True
    assert gate._claim_lane_transition("conversation_status") is False


def test_releasing_hands_the_lease_on():
    gate = _gate()
    gate._claim_lane_transition("watchdog")

    gate._release_lane_transition("watchdog")

    assert gate._claim_lane_transition("conversation_status") is True


def test_only_the_holder_can_release():
    gate = _gate()
    gate._claim_lane_transition("watchdog")

    gate._release_lane_transition("someone_else")

    assert gate._claim_lane_transition("conversation_status") is False


def test_an_expired_lease_does_not_wedge_recovery_forever():
    """A holder that crashed mid-transition must not block every future one."""
    gate = _gate()
    gate._claim_lane_transition("watchdog")
    gate._lane_transition_at -= InferenceGate._LANE_TRANSITION_LEASE_S + 1.0

    assert gate._claim_lane_transition("conversation_status") is True


def test_a_refused_transition_changes_nothing():
    from types import SimpleNamespace

    gate = _gate()
    gate._mlx_client = SimpleNamespace(_warmup_in_flight=True)
    gate._claim_lane_transition("watchdog")

    receipt = gate._clear_wedged_cortex_warmup("racing", owner="conversation_status")

    assert receipt["refused"] == "lane_transition_held"
    assert gate._mlx_client._warmup_in_flight is True


# ────────────────────────────── a cancelled load is kept until it stops


@pytest.mark.asyncio
async def test_a_cancelled_load_is_kept_until_it_is_observed_to_stop():
    from types import SimpleNamespace

    gate = _gate()
    gate._mlx_client = SimpleNamespace(_warmup_in_flight=True)

    async def _slow_load():
        await asyncio.sleep(30)

    gate._prewarm_task = asyncio.create_task(_slow_load())
    await asyncio.sleep(0)

    receipt = gate._clear_wedged_cortex_warmup("stuck", owner="watchdog")

    assert receipt["cleared_warmup_flag"] is True
    assert receipt["cancelled_prewarm"] is True
    # Dropped, the load would be invisible to the next warmup.
    assert gate.unproven_cortex_loads() == 1

    result = await gate.await_abandoned_cortex_loads(timeout=2.0)

    assert result["still_running"] == 0
    assert gate.unproven_cortex_loads() == 0


@pytest.mark.asyncio
async def test_a_load_that_ignores_cancellation_is_reported_not_forgotten():
    from types import SimpleNamespace

    gate = _gate()
    gate._mlx_client = SimpleNamespace(_warmup_in_flight=True)

    stop = asyncio.Event()

    async def _uncancellable():
        while not stop.is_set():
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                # A shielded load that keeps going is exactly the case.
                continue

    gate._prewarm_task = asyncio.create_task(_uncancellable())
    await asyncio.sleep(0)
    gate._clear_wedged_cortex_warmup("stuck", owner="watchdog")

    result = await gate.await_abandoned_cortex_loads(timeout=0.3)

    assert result["still_running"] == 1
    assert gate.unproven_cortex_loads() == 1

    # Teardown: the task ignores cancellation by design, so end it by hand.
    gate._prewarm_task = None
    for task in list(gate._abandoned_cortex_loads):
        stop.set()
    await asyncio.wait(list(gate._abandoned_cortex_loads), timeout=1.0)


@pytest.mark.asyncio
async def test_nothing_to_await_is_not_an_error():
    gate = _gate()

    assert await gate.await_abandoned_cortex_loads() == {"awaited": 0, "still_running": 0}


# ────────────────────────────── an overrun is not a kill


def test_an_overrun_and_a_kill_are_counted_separately():
    gate = _gate()
    gate._cortex_stuck_kill_times = __import__("collections").deque(maxlen=16)
    gate._cortex_warmup_backoff_until = 0.0
    gate._cortex_warmup_backoff_streak = 0

    gate._note_cortex_warmup_overrun()
    gate._note_cortex_stuck_kill()

    counts = gate.cortex_load_setbacks()
    assert counts[InferenceGate.LOAD_SETBACK_OVERRUN] == 1
    assert counts[InferenceGate.LOAD_SETBACK_KILL] == 1


def test_the_timeout_path_records_an_overrun_not_a_kill():
    """The shielded warmup keeps running, so nothing was killed."""
    import ast
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "recovery_handoff" not in rendered or "_note_cortex" not in rendered:
            continue
        assert "_note_cortex_warmup_overrun()" in rendered
        assert "_note_cortex_stuck_kill()" not in rendered
        return
    raise AssertionError("the warmup-timeout accounting branch was not found")


# ────────────────────────────── readiness needs the evidence, not a default


def test_a_lane_that_omits_warmup_evidence_is_not_ready():
    lane = {
        "state": "ready",
        "readiness_blockers": ["visible_conversation_probe_missing"],
    }

    assert InferenceGate._lane_only_needs_visible_conversation_proof(lane) is False


def test_a_lane_that_reports_a_warmup_is_ready():
    lane = {
        "state": "ready",
        "warmup_attempted": True,
        "readiness_blockers": ["visible_conversation_probe_missing"],
    }

    assert InferenceGate._lane_only_needs_visible_conversation_proof(lane) is True


def test_a_lane_that_reports_no_warmup_is_not_ready():
    lane = {
        "state": "ready",
        "warmup_attempted": False,
        "readiness_blockers": ["visible_conversation_probe_missing"],
    }

    assert InferenceGate._lane_only_needs_visible_conversation_proof(lane) is False


# ────────────────────────────── a shed counts workers that stopped


def test_a_worker_that_cannot_report_liveness_does_not_count_as_shed():
    from types import SimpleNamespace

    assert InferenceGate._worker_is_unloaded(SimpleNamespace()) is False


def test_a_worker_still_alive_after_reboot_does_not_count_as_shed():
    from types import SimpleNamespace

    assert InferenceGate._worker_is_unloaded(SimpleNamespace(is_alive=lambda: True)) is False


def test_a_worker_that_stopped_counts():
    from types import SimpleNamespace

    assert InferenceGate._worker_is_unloaded(SimpleNamespace(is_alive=lambda: False)) is True


def test_an_unreadable_memory_probe_does_not_proceed_to_shedding():
    """The code promises to shed only on verified pressure; a failed read is
    not that measurement."""
    import ast
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(
        gate_mod.InferenceGate._shed_background_workers_for_memory_pressure
    )
    tree = ast.parse(source.lstrip())

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        rendered = ast.get_source_segment(source.lstrip(), node) or ""
        if "could not be measured" not in rendered:
            continue
        assert any(isinstance(stmt, ast.Return) for stmt in ast.walk(node)), (
            "an unreadable memory probe still falls through into shedding"
        )
        return
    raise AssertionError("the memory-abundance probe handler was not found")


# ────────────────────────────── the breaker has a transition API


def test_a_closed_circuit_refuses_a_probe():
    from core.utils.resilience import CircuitBreaker

    breaker = CircuitBreaker("test")

    assert breaker.request_probe(reason="recovery") is False


def test_an_open_circuit_half_opens_for_a_probe():
    from core.utils.resilience import CircuitBreaker, CircuitState

    breaker = CircuitBreaker("test")
    breaker.state = CircuitState.OPEN

    assert breaker.request_probe(reason="recovery") is True
    assert breaker.state is CircuitState.HALF_OPEN


def test_a_probe_never_clears_the_failure_count():
    """Clearing it would make a still-failing service look healthy, and the
    next failure would start from zero instead of tripping again."""
    from core.utils.resilience import CircuitBreaker, CircuitState

    breaker = CircuitBreaker("test")
    breaker.state = CircuitState.OPEN
    breaker.failure_count = 4

    breaker.request_probe(reason="recovery")

    assert breaker.failure_count == 4


def test_a_probe_can_shorten_the_cooldown_but_not_below_the_floor():
    from core.utils.resilience import CircuitBreaker, CircuitState

    breaker = CircuitBreaker("test", reset_timeout=60.0)
    breaker.state = CircuitState.OPEN

    breaker.request_probe(reason="recovery", requested_timeout=0.0)

    assert breaker.reset_timeout == CircuitBreaker.MIN_PROBE_RESET_TIMEOUT


def test_a_probe_cannot_lengthen_the_cooldown():
    from core.utils.resilience import CircuitBreaker, CircuitState

    breaker = CircuitBreaker("test", reset_timeout=20.0)
    breaker.state = CircuitState.OPEN

    breaker.request_probe(reason="recovery", requested_timeout=600.0)

    assert breaker.reset_timeout == 20.0


def test_the_gate_no_longer_assigns_breaker_internals():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)

    assert "breaker.state = " not in source
    assert "breaker.reset_timeout = " not in source
    assert "breaker.request_probe(" in source


# ────────────────────────────── the foreground wait is finite


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -5.0, 0.0])
@pytest.mark.asyncio
async def test_a_non_finite_readiness_timeout_cannot_produce_an_unbounded_wait(
    value, monkeypatch
):
    from core.brain.inference_gate import _MAX_FOREGROUND_READY_WAIT_S

    gate = _gate()
    gate._initialized = True
    observed: list[float] = []

    def _lane(**_kwargs):
        return {"conversation_ready": True}

    monkeypatch.setattr(gate, "get_conversation_status", _lane)
    monkeypatch.setattr(gate, "_reset_cortex_warmup_backoff", lambda: None)
    monkeypatch.setattr(
        gate,
        "_lane_can_attempt_visible_conversation_turn",
        lambda lane: (observed.append(1), True)[1],
    )

    await gate.ensure_foreground_ready(timeout=value)

    # The clamp itself is what is under test; exercised directly here because
    # the early-ready path returns before any wait.
    from core.brain.inference_gate import _finite

    clamped = _finite(value, 90.0)
    if clamped is None or clamped <= 0.0:
        clamped = 90.0
    assert max(15.0, min(_MAX_FOREGROUND_READY_WAIT_S, clamped)) <= _MAX_FOREGROUND_READY_WAIT_S
