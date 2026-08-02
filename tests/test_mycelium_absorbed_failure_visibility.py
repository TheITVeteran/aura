"""An absorbed failure that nobody collects must not stay invisible.

CP126 34f01634: a priority rooted flow swallows the exception raised inside it,
so the caller's ``async with`` completes normally and a failed action reads as a
finished one. The failure was recoverable only by a caller that remembered to
call ``raise_for_status()`` — and remembering is not a safeguard.

Absorption is still allowed (it is a deliberate failsafe for priority roots).
What changed is that an absorption nobody ever asked about is now reported,
with the call site that opened the flow.
"""
from __future__ import annotations

import pytest

import core.mycelium as module
from core.mycelium import MycelialNetwork

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def network():
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    yield net
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False


class _Reports(list):
    """Only the unclaimed-absorption reports.

    The rooted-flow failure itself also records a degradation under
    ``mycelium``; that one is expected and is not what these tests are about.
    """

    def append_if_relevant(self, args, kwargs):
        if args and args[0] == "mycelium.rooted_flow":
            self.append({"args": args, **kwargs})


@pytest.fixture()
def degradations(monkeypatch):
    recorded = _Reports()
    monkeypatch.setattr(
        module,
        "record_degradation",
        lambda *a, **kw: recorded.append_if_relevant(a, kw),
    )
    return recorded


async def _fail_inside(network, *, priority=1.0, absorb=None):
    async with network.rooted_flow(
        "a", "b", activity="do the thing", priority=priority,
        absorb_failures=absorb,
    ) as flow:
        raise ValueError("the thing did not happen")
    return flow  # noqa: F821 - unreachable unless the failure was absorbed


# --- absorption is still absorption ------------------------------------


async def test_a_priority_flow_still_absorbs(network):
    handle = None
    async with network.rooted_flow("a", "b", activity="x", priority=1.0) as flow:
        handle = flow
        raise ValueError("boom")
    assert handle.failed is True
    assert isinstance(handle.error, ValueError)


async def test_a_low_priority_flow_still_propagates(network):
    with pytest.raises(ValueError):
        async with network.rooted_flow("a", "b", activity="x", priority=0.1):
            raise ValueError("boom")


# --- absorption is now a decision, not a threshold ----------------------


async def test_a_caller_can_refuse_absorption_at_any_priority(network):
    with pytest.raises(ValueError):
        async with network.rooted_flow(
            "a", "b", activity="x", priority=5.0, absorb_failures=False
        ):
            raise ValueError("boom")


async def test_a_caller_can_request_absorption_at_any_priority(network):
    async with network.rooted_flow(
        "a", "b", activity="x", priority=0.1, absorb_failures=True
    ) as flow:
        raise ValueError("boom")
    assert flow.failed is True


# --- the unclaimed ones are reported ------------------------------------


async def test_an_unclaimed_absorption_is_reported_with_its_call_site(
    network, degradations
):
    async with network.rooted_flow("a", "b", activity="do the thing") as flow:
        raise ValueError("the thing did not happen")

    # One sweep is the grace period: acknowledgement can only happen after the
    # block exits, so a handle is never judged on the sweep that first sees it.
    network._report_unclaimed_absorptions()
    assert degradations == []

    network._report_unclaimed_absorptions()

    assert len(degradations) == 1
    action = degradations[0]["action"]
    assert "a->b" in action and "do the thing" in action
    assert "test_mycelium_absorbed_failure_visibility.py" in action
    assert flow.absorbed is True


@pytest.mark.parametrize("collect", [
    lambda f: f.failed,
    lambda f: f.error,
    lambda f: f.raise_for_status(),
])
async def test_collecting_the_failure_clears_it(network, degradations, collect):
    async with network.rooted_flow("a", "b", activity="x") as flow:
        raise ValueError("boom")

    try:
        collect(flow)
    except ValueError:
        pass

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    assert degradations == []


async def test_a_successful_flow_is_never_reported(network, degradations):
    async with network.rooted_flow("a", "b", activity="x"):
        pass

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    assert degradations == []


async def test_the_report_names_the_original_error(network, degradations):
    async with network.rooted_flow("a", "b", activity="x"):
        raise ValueError("the thing did not happen")

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    assert "the thing did not happen" in str(degradations[0]["args"][1])


async def test_reporting_does_not_fail_the_mycelium_subsystem(network):
    """The report is about a caller's lapse, not a mycelium outage; escalating
    it fail-closed would take the maintenance loop down with it."""
    recorded: list[dict] = []
    original = module.record_degradation

    def _capture(*a, **kw):
        if a and a[0] == "mycelium.rooted_flow":
            recorded.append(kw)
        return original(*a, **kw)

    module.record_degradation = _capture
    try:
        async with network.rooted_flow("a", "b", activity="x"):
            raise ValueError("boom")
        network._report_unclaimed_absorptions()
        network._report_unclaimed_absorptions()
    finally:
        module.record_degradation = original

    assert recorded and recorded[0]["enforce_failure_policy"] is False


# --- the counters are readable ------------------------------------------


async def test_the_integrity_read_model_counts_both_states(network, degradations):
    assert network.get_rooted_flow_integrity() == {
        "absorptions_awaiting_acknowledgement": 0,
        "absorptions_unclaimed": 0,
        "absorptions_untracked_overflow": 0,
    }

    async with network.rooted_flow("a", "b", activity="x"):
        raise ValueError("boom")

    assert network.get_rooted_flow_integrity()[
        "absorptions_awaiting_acknowledgement"
    ] == 1

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    integrity = network.get_rooted_flow_integrity()
    assert integrity["absorptions_unclaimed"] == 1
    assert integrity["absorptions_awaiting_acknowledgement"] == 0


async def test_the_tracking_list_is_bounded(network, degradations):
    for _ in range(module._MAX_TRACKED_ABSORPTIONS + 20):
        async with network.rooted_flow("a", "b", activity="x"):
            raise ValueError("boom")

    integrity = network.get_rooted_flow_integrity()
    assert (
        integrity["absorptions_awaiting_acknowledgement"]
        <= module._MAX_TRACKED_ABSORPTIONS
    )
    assert integrity["absorptions_untracked_overflow"] == 20


async def test_shutdown_drops_pending_handles(network):
    async with network.rooted_flow("a", "b", activity="x"):
        raise ValueError("boom")

    network.shutdown()

    assert network.get_rooted_flow_integrity()[
        "absorptions_awaiting_acknowledgement"
    ] == 0


# --- the live callers all collect ---------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "core/mind_tick.py",
        "core/cognition/meta_cognition.py",
        "core/orchestrator/mixins/incoming_logic.py",
        "core/orchestrator/mixins/learning_evolution.py",
    ],
)
async def test_every_rooted_flow_caller_collects_its_failure(path):
    """The sweep exists because this is not enforceable by the type system.
    It is still worth asserting that today's callers do it."""
    import pathlib

    source = pathlib.Path(path).read_text(encoding="utf-8")
    assert "rooted_flow(" in source
    assert "raise_for_status()" in source or '"failed"' in source
