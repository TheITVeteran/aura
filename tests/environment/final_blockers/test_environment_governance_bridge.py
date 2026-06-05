import pytest

from core.environment.governance_bridge import EnvironmentGovernanceBridge
from core.environment.command import ActionIntent
from core.environment.environment_kernel import EnvironmentKernel
from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter
from core.environment.action_gateway import GatewayDecision


class RecordingWillGateway:
    def __init__(self, events):
        self.events = events
        self.decisions = []

    async def decide(self, intent):
        self.events.append("will")
        self.decisions.append(intent)
        return type("WillDecision", (), {"status": "PROCEED", "receipt_id": "will-123"})()


class RecordingEnvironmentAuthority:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.calls = []

    async def authorize_environment_action(self, name, payload, **kwargs):
        self.events.append("authority")
        self.calls.append({"name": name, "payload": payload, "kwargs": kwargs})
        return type(
            "AuthorityDecision",
            (),
            {
                "approved": True,
                "will_receipt_id": "will-authority",
                "capability_token_id": "auth-456",
                "substrate_receipt_id": None,
                "executive_intent_id": "exec-456",
                "reason": "approved",
            },
        )()


class VetoingLocalGateway:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []
        self.failures = []

    def approve(self, intent, **kwargs):
        self.calls.append({"intent": intent, "kwargs": kwargs})
        return self.decision

    def record_failure(self, action_name, context_id):
        self.failures.append({"action_name": action_name, "context_id": context_id})


class CommandCompilerRecorder:
    def __init__(self):
        self.calls = []

    def compile(self, intent, **kwargs):
        self.calls.append({"intent": intent, "kwargs": kwargs})
        raise AssertionError("blocked action must not compile into a command")


@pytest.mark.asyncio
async def test_safe_observe_gets_environment_receipt_without_full_authority():
    bridge = EnvironmentGovernanceBridge(authority_gateway=RecordingEnvironmentAuthority())
    intent = ActionIntent(name="observe", requires_authority=False)
    decision = await bridge.decide_action(intent)
    
    assert decision.approved
    assert decision.will_receipt_id and decision.will_receipt_id != "not_required"
    assert decision.authority_receipt_id and decision.authority_receipt_id != "not_required"

@pytest.mark.asyncio
async def test_risky_action_calls_unified_will_before_gateway_approval():
    events = []
    will = RecordingWillGateway(events)
    authority = RecordingEnvironmentAuthority(events)
    bridge = EnvironmentGovernanceBridge(will_gateway=will, authority_gateway=authority)
    
    intent = ActionIntent(name="quaff", requires_authority=True)
    decision = await bridge.decide_action(intent)
    
    assert events == ["will", "authority"]
    assert will.decisions == [intent]
    assert decision.will_receipt_id == "will-123"

@pytest.mark.asyncio
async def test_authority_gateway_required_for_effectful_irreversible_action():
    authority = RecordingEnvironmentAuthority()
    bridge = EnvironmentGovernanceBridge(authority_gateway=authority)
    
    intent = ActionIntent(name="quaff", requires_authority=True)
    decision = await bridge.decide_action(intent)
    
    assert authority.calls
    assert authority.calls[0]["name"] == "quaff"
    assert decision.authority_receipt_id == "auth-456"
    
@pytest.mark.asyncio
async def test_local_gateway_can_veto_even_with_will_authority_receipts():
    adapter = ScriptedTerminalAdapter(["screen1"])
    kernel = EnvironmentKernel(adapter=adapter)
    kernel.command_compiler = CommandCompilerRecorder()
    
    intent = ActionIntent(name="dangerous_action", requires_authority=True)
    veto = GatewayDecision(
        action_intent=intent,
        approved=False,
        decision_id="veto_123",
        reason="critical_risk_blocks_risky_action",
    )
    kernel.gateway = VetoingLocalGateway(veto)
    kernel.governance_bridge = EnvironmentGovernanceBridge(
        will_gateway=RecordingWillGateway([]),
        authority_gateway=RecordingEnvironmentAuthority(),
    )
    
    await kernel.start(run_id="test_run")
    frame = await kernel.step(intent=intent)
    
    assert kernel.command_compiler.calls == []
    assert kernel.gateway.calls[0]["intent"] == intent
    assert kernel.gateway.failures == [{"action_name": "dangerous_action", "context_id": "unknown"}]
    assert frame.receipt.status == "blocked"
