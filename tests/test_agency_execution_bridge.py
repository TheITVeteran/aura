import asyncio
from types import SimpleNamespace

import pytest

from core.agency.agency_core import AgencyCore
from core.container import ServiceContainer
from core.orchestrator.mixins.autonomy import AutonomyMixin


class _Recorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _AsyncRecorder(_Recorder):
    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _AgencyHarness(AutonomyMixin):
    def __init__(self, action, *, tool_result=None):
        self.message_queue = asyncio.Queue()
        self._agency_core = SimpleNamespace(
            pulse=_AsyncRecorder(action),
            claim_goal_for_execution=_Recorder(True),
            settle_goal_execution=_Recorder("completed"),
            state=SimpleNamespace(
                last_self_initiated_contact=0.0,
                last_observation_comment=0.0,
                unshared_observations=[],
            ),
        )
        self.execute_tool = _AsyncRecorder(tool_result)
        self.scheduled = []
        self.thoughts = []

    @staticmethod
    def _normalize_to_dict(value):
        return value

    def _emit_thought_stream(self, message):
        self.thoughts.append(message)

    def _fire_and_forget(self, awaitable, name=None):
        self.scheduled.append((name, awaitable))
        return SimpleNamespace(name=name)


class _RejectingSchedulerHarness(_AgencyHarness):
    def _fire_and_forget(self, awaitable, name=None):
        awaitable.close()
        self.scheduled.append((name, None))
        return None


@pytest.fixture
def approve_agency(monkeypatch):
    monkeypatch.setattr(
        "core.will.get_will",
        lambda: SimpleNamespace(
            decide=lambda **_kwargs: SimpleNamespace(
                is_approved=lambda: True,
                reason="approved_by_test",
            )
        ),
    )
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=_AsyncRecorder((True, "approved_by_test", None))
        ),
    )


@pytest.fixture(autouse=True)
def reset_services():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


@pytest.mark.asyncio
async def test_structured_embodiment_choice_executes_without_phrase_rerouting(approve_agency):
    harness = _AgencyHarness(
        {
            "id": "agency-physical-1",
            "type": "autonomous_action",
            "skill": "embodiment",
            "params": {"action": "inventory"},
            "message": "I want to inspect the physical interfaces available to me.",
            "source": "environmental_explorer",
        },
        tool_result={"ok": True, "devices": []},
    )

    await harness._pulse_agency_core()

    assert harness.message_queue.empty()
    assert [name for name, _ in harness.scheduled] == [
        "orchestrator.agency.skill.embodiment"
    ]
    await harness.scheduled[0][1]
    assert len(harness.execute_tool.calls) == 1
    args, kwargs = harness.execute_tool.calls[0]
    assert args == ("embodiment", {"action": "inventory"})
    assert kwargs["origin"] == "autonomy"
    assert kwargs["payload_context"]["requested_by"] == "aura"
    assert kwargs["payload_context"]["agency_action_id"] == "agency-physical-1"
    assert kwargs["payload_context"]["initiative_preflight"] == {}
    assert "intent_hint" not in kwargs["payload_context"]


@pytest.mark.asyncio
async def test_autonomous_research_executes_web_search_directly(approve_agency):
    harness = _AgencyHarness(
        {
            "type": "autonomous_research",
            "query": "distributed cognition in octopuses",
            "source": "curiosity",
        },
        tool_result={"ok": True, "results": [{"title": "Paper"}]},
    )

    await harness._pulse_agency_core()
    assert harness.message_queue.empty()
    await harness.scheduled[0][1]

    args, kwargs = harness.execute_tool.calls[0]
    assert args == ("web_search", {"query": "distributed cognition in octopuses"})
    assert kwargs["origin"] == "autonomy"


@pytest.mark.asyncio
async def test_initiative_preflight_receipt_is_bound_to_tool_execution(
    approve_agency,
    monkeypatch,
):
    decision = SimpleNamespace(
        executive_intent_id="intent-1",
        substrate_receipt_id="substrate-1",
        will_receipt_id="will-1",
        reason="approved",
        domain="tool_execution",
        source="agency_core",
    )
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=_AsyncRecorder((True, "approved", decision))
        ),
    )
    harness = _AgencyHarness(
        {
            "id": "agency-research-1",
            "type": "autonomous_research",
            "query": "causal abstraction",
            "source": "curiosity_drive",
        },
        tool_result={"ok": True},
    )

    await harness._pulse_agency_core()
    await harness.scheduled[0][1]

    _, kwargs = harness.execute_tool.calls[0]
    assert kwargs["payload_context"]["initiative_preflight"] == {
        "executive_intent_id": "intent-1",
        "substrate_receipt_id": "substrate-1",
        "will_receipt_id": "will-1",
        "reason": "approved",
        "domain": "tool_execution",
        "source": "agency_core",
        "preflight_complete": True,
    }


@pytest.mark.asyncio
async def test_denied_initiative_never_claims_or_executes_goal(approve_agency, monkeypatch):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=_AsyncRecorder((False, "denied_by_test", None))
        ),
    )
    goal = {"id": "goal-denied", "text": "Inspect embodied interfaces"}
    harness = _AgencyHarness(
        {"type": "pursue_goal", "goal": goal, "source": "goal_persistence"}
    )

    await harness._pulse_agency_core()

    assert harness._agency_core.claim_goal_for_execution.calls == []
    assert harness._agency_core.settle_goal_execution.calls == []
    assert harness.scheduled == []
    assert harness.execute_tool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("succeeded", "expected_status"),
    [(True, "completed"), (False, "pending")],
)
async def test_autonomous_goal_status_follows_verified_task_outcome(
    approve_agency,
    monkeypatch,
    succeeded,
    expected_status,
):
    goal = {"id": "goal-1", "text": "Inspect nearby embodied interfaces"}
    harness = _AgencyHarness(
        {"type": "pursue_goal", "goal": goal, "source": "goal_persistence"}
    )
    settlement = _Recorder(expected_status)
    harness._agency_core.settle_goal_execution = settlement
    task_result = SimpleNamespace(
        succeeded=succeeded,
        summary="done" if succeeded else "device unavailable",
        error="" if succeeded else "device unavailable",
        evidence=["receipt-1"] if succeeded else [],
        plan_id="plan-1",
        trace_id="trace-1",
    )
    execute_goal = _AsyncRecorder(task_result)
    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.get_task_engine",
        lambda _kernel=None: SimpleNamespace(execute_goal=execute_goal),
    )

    await harness._pulse_agency_core()
    assert harness.message_queue.empty()
    await harness.scheduled[0][1]

    assert len(execute_goal.calls) == 1
    args, kwargs = execute_goal.calls[0]
    assert args == ("Inspect nearby embodied interfaces",)
    assert kwargs["context"]["origin"] == "autonomous_task_engine"
    assert len(harness._agency_core.claim_goal_for_execution.calls) == 1
    _, claim_kwargs = harness._agency_core.claim_goal_for_execution.calls[0]
    execution_id = claim_kwargs["execution_id"]
    assert execution_id.startswith("agency-goal-")
    _, settlement_kwargs = settlement.calls[0]
    assert settlement_kwargs["execution_id"] == execution_id
    assert settlement_kwargs["succeeded"] is succeeded
    assert settlement_kwargs["execution_result"]["plan_id"] == "plan-1"


@pytest.mark.asyncio
async def test_scheduler_rejection_returns_claimed_goal_to_retry_state(approve_agency):
    goal = {"id": "goal-unscheduled", "text": "Inspect nearby embodied interfaces"}
    harness = _RejectingSchedulerHarness(
        {"type": "pursue_goal", "goal": goal, "source": "goal_persistence"}
    )
    settlement = _Recorder("pending")
    harness._agency_core.settle_goal_execution = settlement

    await harness._pulse_agency_core()

    assert len(harness._agency_core.claim_goal_for_execution.calls) == 1
    assert len(settlement.calls) == 1
    _, settlement_kwargs = settlement.calls[0]
    assert settlement_kwargs["succeeded"] is False
    assert settlement_kwargs["execution_result"] == {
        "error": "agency execution could not be scheduled"
    }


@pytest.mark.asyncio
async def test_cancelled_goal_execution_still_settles_claim(approve_agency, monkeypatch):
    goal = {"id": "goal-cancelled", "text": "Inspect nearby embodied interfaces"}
    harness = _AgencyHarness(
        {"type": "pursue_goal", "goal": goal, "source": "goal_persistence"}
    )
    settlement = _Recorder("pending")
    harness._agency_core.settle_goal_execution = settlement

    async def _cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.get_task_engine",
        lambda _kernel=None: SimpleNamespace(execute_goal=_cancelled),
    )

    await harness._pulse_agency_core()
    with pytest.raises(asyncio.CancelledError):
        await harness.scheduled[0][1]

    _, settlement_kwargs = settlement.calls[0]
    assert settlement_kwargs["succeeded"] is False
    assert "cancelled" in settlement_kwargs["execution_result"]["error"]


@pytest.mark.asyncio
async def test_invalid_structured_action_fails_before_any_dispatch(approve_agency):
    harness = _AgencyHarness(
        {
            "type": "autonomous_action",
            "skill": "embodiment",
            "params": "inventory",
            "source": "environmental_explorer",
        }
    )

    await harness._pulse_agency_core()

    assert harness.scheduled == []
    assert harness.execute_tool.calls == []
    assert harness.message_queue.empty()


@pytest.mark.parametrize(
    "receipt",
    [
        {"ok": True, "error": "transport failed"},
        {"ok": True, "success": False},
        {"status": "unknown"},
        object(),
    ],
)
def test_execution_success_requires_consistent_positive_evidence(receipt):
    from core.orchestrator.mixins.autonomy import _agency_execution_succeeded

    assert _agency_execution_succeeded(receipt) is False


def test_production_physical_pathway_chooses_embodiment_for_live_candidate(monkeypatch):
    agency = AgencyCore(orchestrator=None)
    agency.state.initiative_energy = 0.8
    candidate = SimpleNamespace(
        candidate_id="candidate-1",
        persistent_identity=True,
        privacy_sensitive=False,
    )
    broker = SimpleNamespace(candidates=lambda: (candidate,), requests=lambda: ())
    ServiceContainer.register_instance("reality_attachment_broker", broker, required=False)

    action = agency._pathway_physical_affordance(now=1_000.0, idle_seconds=500.0)

    assert action is not None
    assert action["type"] == "autonomous_action"
    assert action["skill"] == "embodiment"
    assert action["params"] == {
        "action": "request_connection",
        "candidate_id": "candidate-1",
        "access": "observe",
        "persistent": False,
    }


@pytest.mark.asyncio
async def test_live_candidate_flows_from_production_choice_to_governed_execution(
    approve_agency,
):
    agency = AgencyCore(orchestrator=None)
    agency.state.initiative_energy = 0.8
    candidate = SimpleNamespace(
        candidate_id="candidate-live",
        persistent_identity=True,
        privacy_sensitive=False,
    )
    ServiceContainer.register_instance(
        "reality_attachment_broker",
        SimpleNamespace(candidates=lambda: (candidate,), requests=lambda: ()),
        required=False,
    )
    action = agency._pathway_physical_affordance(now=2_000.0, idle_seconds=500.0)
    harness = _AgencyHarness(action, tool_result={"ok": True, "request_id": "request-1"})

    await harness._pulse_agency_core()
    await harness.scheduled[0][1]

    assert harness.message_queue.empty()
    args, kwargs = harness.execute_tool.calls[0]
    assert args == (
        "embodiment",
        {
            "action": "request_connection",
            "candidate_id": "candidate-live",
            "access": "observe",
            "persistent": False,
        },
    )
    assert kwargs["payload_context"]["requested_by"] == "aura"
    assert kwargs["payload_context"]["proposal_source"] == "physical_affordance"


def test_goal_execution_lease_retries_with_backoff_then_blocks(monkeypatch):
    agency = AgencyCore(orchestrator=None)
    monkeypatch.setattr(agency, "_constitutional_runtime_live", lambda: False)
    assert agency.add_goal({"id": "goal-1", "text": "Inspect Reality Reach", "priority": 0.8})
    goal = agency.state.pending_goals[0]

    for attempt in range(1, 6):
        execution_id = f"execution-{attempt}"
        now = 1_000.0 + attempt * 10_000.0
        assert agency.claim_goal_for_execution(goal, execution_id=execution_id, now=now)
        status = agency.settle_goal_execution(
            goal,
            execution_id=execution_id,
            succeeded=False,
            execution_result={"error": "device offline"},
            now=now + 1.0,
        )
        assert status == ("blocked" if attempt == 5 else "pending")
        if attempt < 5:
            assert goal["next_eligible_at"] > now
            assert agency._pathway_goal_persistence(
                now=now + 2.0,
                idle_seconds=500.0,
            ) is None

    assert goal["execution_failures"] == 5
    assert goal["status"] == "blocked"


def test_stale_goal_execution_claim_is_recovered(monkeypatch):
    agency = AgencyCore(orchestrator=None)
    monkeypatch.setattr(agency, "_constitutional_runtime_live", lambda: False)
    assert agency.add_goal({"id": "goal-1", "text": "Resume me"})
    goal = agency.state.pending_goals[0]
    assert agency.claim_goal_for_execution(
        goal,
        execution_id="execution-stale",
        now=100.0,
        lease_seconds=30.0,
    )

    assert agency.recover_stale_goal_claims(now=131.0) == 1
    assert goal["status"] == "pending"
    assert goal["active_execution_id"] == ""
    assert goal["execution_failures"] == 1
