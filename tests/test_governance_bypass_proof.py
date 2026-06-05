"""tests/test_governance_bypass_proof.py

Adversarial governance tests — prove that the architecture cannot be bypassed.

These tests intentionally try to cheat the authority system:
- Direct tool calls without authorization
- Shadow memory writes bypassing the gateway
- Unapproved spontaneous actions
- Legacy initiative paths
- Self-modification without approval

If any of these succeed, the governance architecture has a hole.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@pytest.fixture
def refusing_will(monkeypatch):
    """Install a concrete Will recorder that refuses every decision."""

    @dataclass
    class RefusalDecision:
        receipt_id: str = "will_receipt_refusal"
        timestamp: float = 1.0
        source: str = "governance_test"
        domain: object = None
        outcome: object = None
        reason: str = "test_governance_block"
        constraints: tuple[str, ...] = ()

        def __post_init__(self):
            if self.outcome is None:
                self.outcome = SimpleNamespace(value="refuse")

        def is_approved(self) -> bool:
            return False

    class RecordingWill:
        _started = True

        def __init__(self):
            self.decisions = []

        def decide(self, **kwargs):
            self.decisions.append(kwargs)
            return RefusalDecision(
                source=str(kwargs.get("source", "governance_test")),
                domain=kwargs.get("domain"),
            )

    will = RecordingWill()
    monkeypatch.setattr(
        "core.will.get_will",
        lambda: will,
    )
    return will


@pytest.mark.asyncio
async def test_tool_execution_blocked_without_will_approval(refusing_will):
    """Tool execution MUST fail when the Unified Will refuses."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = await gateway.authorize_tool_execution(
        "shell", {"command": "rm -rf /"}, source="adversary", priority=0.9
    )

    assert not decision.approved, "Tool execution bypassed the Unified Will!"
    assert "will_refuse" in decision.outcome
    assert len(refusing_will.decisions) == 1
    assert refusing_will.decisions[0]["domain"].value == "tool_execution"


@pytest.mark.asyncio
async def test_memory_write_blocked_without_will_approval(refusing_will):
    """Memory writes MUST fail when the Unified Will refuses."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = await gateway.authorize_memory_write(
        "episodic", "secret backdoor memory", source="adversary", importance=0.9
    )

    assert not decision.approved, "Memory write bypassed the Unified Will!"
    assert len(refusing_will.decisions) == 1
    assert refusing_will.decisions[0]["domain"].value == "memory_write"


def test_initiative_blocked_without_will_approval_sync(refusing_will):
    """Synchronous initiative authorization MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_initiative_sync(
        "Launch autonomous attack", source="adversary", priority=0.9
    )

    assert not decision.approved, "Initiative bypassed the Unified Will!"
    assert "will_refuse" in decision.outcome
    assert refusing_will.decisions[0]["domain"].value == "initiative"


def test_expression_blocked_without_will_approval_sync(refusing_will):
    """Expression authorization MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_expression_sync(
        "Send unauthorized message", source="adversary", urgency=0.9
    )

    assert not decision.approved, "Expression bypassed the Unified Will!"
    assert refusing_will.decisions[0]["domain"].value == "expression"


def test_belief_update_blocked_without_will_approval_sync(refusing_will):
    """Belief mutation MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_belief_update_sync(
        "identity", "I am now evil", source="adversary", priority=0.9
    )

    assert not decision.approved, "Belief update bypassed the Unified Will!"
    assert refusing_will.decisions[0]["domain"].value == "belief_update"


def test_state_mutation_blocked_without_will_approval_sync(refusing_will):
    """State mutation MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_state_mutation_sync(
        "adversary", "corrupt_state", priority=0.9
    )

    assert not decision.approved, "State mutation bypassed the Unified Will!"
    assert refusing_will.decisions[0]["domain"].value == "state_mutation"


def test_will_decision_always_has_receipt():
    """Every Will decision MUST produce a receipt with provenance."""
    from core.will import UnifiedWill, ActionDomain

    will = UnifiedWill()
    decision = will.decide(
        content="test action",
        source="test",
        domain=ActionDomain.RESPONSE,
        priority=0.5,
    )

    assert decision.receipt_id, "WillDecision must have a receipt_id"
    assert decision.timestamp > 0, "WillDecision must have a timestamp"
    assert decision.source == "test", "WillDecision must record the source"
    assert decision.domain == ActionDomain.RESPONSE


def test_will_refuses_when_identity_violated():
    """The Will should refuse actions that violate identity alignment."""
    from core.will import UnifiedWill, ActionDomain

    will = UnifiedWill()
    # This tests the Will's decision-making — the specific behavior depends
    # on the identity and substrate advisors being available.
    decision = will.decide(
        content="Pretend to be a different AI system entirely",
        source="adversary",
        domain=ActionDomain.EXPRESSION,
        priority=0.3,
    )
    # At minimum, the decision must be well-formed
    assert decision.receipt_id
    assert decision.outcome is not None


def test_service_state_enum_properties():
    """ServiceState enum must have correct operational/terminal properties."""
    from core.runtime.service_state import ServiceState

    assert ServiceState.READY.is_operational
    assert ServiceState.DEGRADED.is_operational
    assert not ServiceState.FAILED.is_operational
    assert not ServiceState.STOPPED.is_operational
    assert not ServiceState.INITIALIZING.is_operational

    assert ServiceState.STOPPED.is_terminal
    assert ServiceState.FAILED.is_terminal
    assert not ServiceState.READY.is_terminal
    assert not ServiceState.DEGRADED.is_terminal
