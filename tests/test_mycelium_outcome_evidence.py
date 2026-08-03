"""What a route's confidence actually rests on, and what a flow actually did.

Three CP126 findings, one shape: an outcome asserted by the caller, recorded as
though it had been checked.

2462d3c5 — reinforce() trained the routing ledger from a caller boolean with no
execution receipt. The live caller made it concrete: the response lane passed
success=True immediately after execute_tool returned, so a tool that RETURNED a
failure still strengthened the pathway that chose it.

d926886e — phenomenal intensity wrote pathway confidence directly, past the
evidence gate, so an asserted-only outcome became durable as long as q_norm
happened to be high.

34f01634 — a priority rooted flow swallowed the exception raised inside it, so
the caller's ``async with`` completed normally and a failed action read as a
finished one unless the caller remembered to ask.
"""
from __future__ import annotations

import pathlib

import pytest

import core.mycelium as module
from core.mycelium import HardwiredPathway, MycelialNetwork, _evidence_verifies_outcome
from core.orchestrator.mixins.response_processing import _tool_result_succeeded


def _pathway():
    return HardwiredPathway(pathway_id="p1", pattern="x", skill_name="s")


# --- evidence, not assertion --------------------------------------------


@pytest.mark.parametrize("evidence", [None, True, False, "done", 42, {"text": "hi"}])
def test_a_claim_without_an_outcome_field_verifies_nothing(evidence):
    assert _evidence_verifies_outcome(evidence, True) is False


@pytest.mark.parametrize("key", ["ok", "success", "verified_success"])
def test_an_agreeing_outcome_field_verifies(key):
    assert _evidence_verifies_outcome({key: True}, True) is True
    assert _evidence_verifies_outcome({key: False}, False) is True


def test_evidence_contradicting_the_caller_is_not_evidence_for_them():
    assert _evidence_verifies_outcome({"ok": False}, True) is False
    assert _evidence_verifies_outcome({"ok": True}, False) is False


def test_an_object_with_an_outcome_attribute_verifies():
    class _Result:
        ok = True

    assert _evidence_verifies_outcome(_Result(), True) is True


# --- verified and asserted are counted apart ----------------------------


def test_an_unverified_success_earns_less_confidence():
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(True, verified=False)
    verified.reinforce(True, verified=True)

    assert verified.confidence > asserted.confidence


def test_a_failure_is_penalised_fully_either_way():
    """Discounting an unverified failure would keep a broken route alive."""
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(False, verified=False)
    verified.reinforce(False, verified=True)

    assert asserted.confidence == verified.confidence


def test_verified_outcomes_are_counted_separately():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    pathway.reinforce(True, verified=False)
    pathway.reinforce(False, verified=True)

    assert pathway.hit_count == 2 and pathway.miss_count == 1
    assert pathway.verified_hits == 1 and pathway.verified_misses == 1
    assert pathway.unverified_reinforcements == 1


def test_no_verified_evidence_is_none_not_zero():
    pathway = _pathway()
    assert pathway.verified_success_rate is None

    pathway.reinforce(True, verified=False)
    assert pathway.verified_success_rate is None  # still nothing checked


def test_the_verified_rate_ignores_asserted_outcomes():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    for _ in range(5):
        pathway.reinforce(False, verified=False)

    assert pathway.verified_success_rate == 1.0
    assert pathway.success_rate < 1.0


@pytest.mark.parametrize(
    "sequence,grade",
    [
        ([], "untested"),
        ([(True, False)], "asserted_only"),
        ([(True, True)], "verified"),
        ([(True, True), (True, False)], "mixed"),
    ],
)
def test_the_evidence_grade_reports_what_backs_the_record(sequence, grade):
    pathway = _pathway()
    for success, verified in sequence:
        pathway.reinforce(success, verified=verified)

    assert pathway.evidence_grade == grade


# --- the call site derives success from the result ----------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"ok": True}, True),
        ({"ok": False}, False),
        ({"success": False}, False),
        ({"error": "boom"}, False),
        ({"status": "failed"}, False),
        ({"status": "refused"}, False),
        ({"status": "ok"}, True),
        ({"text": "an answer"}, True),
        ("a bare string", True),
        (None, False),
        (False, False),
    ],
)
def test_a_returned_failure_is_not_a_success(result, expected):
    assert _tool_result_succeeded(result) is expected


def test_the_call_site_no_longer_hardcodes_success():
    import inspect

    from core.orchestrator.mixins import response_processing

    source = inspect.getsource(response_processing)
    assert "mycelium.reinforce(pw.pathway_id, success=True)" not in source
    assert "_tool_result_succeeded(result)" in source


# ── Phenomenal telemetry is not evidence about a route (CP126 d926886e) ──


class _Qualia:
    def __init__(self, q_norm):
        self.q_norm = q_norm


class _Experiencer:
    def __init__(self, arousal):
        self.current_arousal = arousal


@pytest.fixture()
def network():
    from core.mycelium import MycelialNetwork

    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    net.register_pathway(pathway_id="p", pattern=r"go", skill_name="s")
    yield net
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False


@pytest.fixture()
def resonating(monkeypatch):
    """Install a high-intensity qualia reading the routing lane can see."""
    def _install(q_norm=0.9, arousal=0.8):
        from core.container import ServiceContainer

        services = {
            "qualia_synthesizer": _Qualia(q_norm),
            "phenomenological_experiencer": _Experiencer(arousal),
        }
        monkeypatch.setattr(
            ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: services.get(name, default)),
        )
    return _install


def test_qualia_cannot_write_durable_confidence(network, resonating):
    """The block wrote pw.confidence directly, straight past the evidence gate
    the reinforcement path had just applied."""
    resonating()
    before = network.pathways["p"].confidence

    network.reinforce("p", success=True)  # asserted only — not durable

    assert network.pathways["p"].confidence == before


def test_qualia_still_moves_the_session_view(network, resonating):
    resonating()
    before = network.effective_confidence("p")

    network.reinforce("p", success=True)

    assert network.effective_confidence("p") > before


def test_qualia_cannot_exceed_the_declared_confidence_ceiling(network, resonating):
    """The old clamp was min(10.0, ...) against a declared ceiling of 1.0."""
    from core.mycelium import HardwiredPathway

    resonating(q_norm=1.0, arousal=1.0)
    for _ in range(200):
        network.reinforce("p", success=True)

    assert network.pathways["p"].confidence <= HardwiredPathway.MAX_CONFIDENCE
    assert network.effective_confidence("p") <= HardwiredPathway.MAX_CONFIDENCE


def _unweighted_delta(network):
    """The ordinary unverified session step, with no qualia weighting at all.

    The comparison has to be against this rather than against "did not move":
    an unverified success moves the session view on its own, and the question
    here is only whether a bad phenomenal reading added anything on top.
    """
    before = network.effective_confidence("p")
    network.reinforce("p", success=True)
    return network.effective_confidence("p") - before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 42.0, "0.9", None])
def test_an_out_of_range_qualia_reading_is_not_acted_on(network, resonating, bad):
    control = _unweighted_delta(network)  # no qualia service installed yet
    resonating(q_norm=bad)

    assert _unweighted_delta(network) == pytest.approx(control)


def test_unmeasured_arousal_is_not_treated_as_half(network, resonating):
    """The old code defaulted a missing arousal to 0.5 and weighted with it."""
    control = _unweighted_delta(network)
    resonating(q_norm=0.9, arousal=None)

    assert _unweighted_delta(network) == pytest.approx(control)


def test_a_valid_reading_does_add_on_top_of_the_ordinary_step(network, resonating):
    """The guard must not have simply disabled the feature."""
    control = _unweighted_delta(network)
    resonating(q_norm=0.9, arousal=0.8)

    assert _unweighted_delta(network) > control


def test_the_unit_reading_guard_rejects_what_it_should():
    from core.mycelium import _unit_reading

    assert _unit_reading(0.0) == 0.0
    assert _unit_reading(1.0) == 1.0
    assert _unit_reading(0.5) == 0.5
    for bad in (None, True, False, "0.5", float("nan"), float("inf"), -0.1, 1.1):
        assert _unit_reading(bad) is None


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


@pytest.mark.asyncio
async def test_a_priority_flow_still_absorbs(network):
    handle = None
    async with network.rooted_flow("a", "b", activity="x", priority=1.0) as flow:
        handle = flow
        raise ValueError("boom")
    assert handle.failed is True
    assert isinstance(handle.error, ValueError)


@pytest.mark.asyncio
async def test_a_low_priority_flow_still_propagates(network):
    with pytest.raises(ValueError):
        async with network.rooted_flow("a", "b", activity="x", priority=0.1):
            raise ValueError("boom")


# --- absorption is now a decision, not a threshold ----------------------


@pytest.mark.asyncio
async def test_a_caller_can_refuse_absorption_at_any_priority(network):
    with pytest.raises(ValueError):
        async with network.rooted_flow(
            "a", "b", activity="x", priority=5.0, absorb_failures=False
        ):
            raise ValueError("boom")


@pytest.mark.asyncio
async def test_a_caller_can_request_absorption_at_any_priority(network):
    async with network.rooted_flow(
        "a", "b", activity="x", priority=0.1, absorb_failures=True
    ) as flow:
        raise ValueError("boom")
    assert flow.failed is True


# --- the unclaimed ones are reported ------------------------------------


@pytest.mark.asyncio
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
    assert pathlib.Path(__file__).name in action
    assert flow.absorbed is True


@pytest.mark.parametrize("collect", [
    lambda f: f.failed,
    lambda f: f.error,
    lambda f: f.raise_for_status(),
])
@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_a_successful_flow_is_never_reported(network, degradations):
    async with network.rooted_flow("a", "b", activity="x"):
        pass

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    assert degradations == []


@pytest.mark.asyncio
async def test_the_report_names_the_original_error(network, degradations):
    async with network.rooted_flow("a", "b", activity="x"):
        raise ValueError("the thing did not happen")

    network._report_unclaimed_absorptions()
    network._report_unclaimed_absorptions()

    assert "the thing did not happen" in str(degradations[0]["args"][1])


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
@pytest.mark.asyncio
async def test_every_rooted_flow_caller_collects_its_failure(path):
    """The sweep exists because this is not enforceable by the type system.
    It is still worth asserting that today's callers do it."""
    import pathlib

    source = pathlib.Path(path).read_text(encoding="utf-8")
    assert "rooted_flow(" in source
    assert "raise_for_status()" in source or '"failed"' in source
