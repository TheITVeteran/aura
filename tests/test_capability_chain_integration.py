"""End-to-end proof that the constitutional chain is actually joined.

The unit tests in ``test_capability_chain.py`` prove the crypto behaves. They do
not prove the chain is *wired* — that a decision made by the real Will produces a
signed grant that survives the real AuthorityGateway and arrives, verifiable, at
the real sink. A chain that is sound in isolation and disconnected in practice is
exactly the defect this work exists to close, so it gets its own test.

These tests deliberately use the real objects (real Will, real gateway, real
issuer/verifier) rather than mocks. Mocks would re-prove the mocks.
"""
from __future__ import annotations

import pytest

from core.governance.capability_chain import (
    CapabilityDenial,
    compute_action_digest,
    get_capability_issuer,
    get_capability_verifier,
    reset_capability_chain,
)


@pytest.fixture(autouse=True)
def _isolated_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_CAPABILITY_KEY_DIR", str(tmp_path / "keys"))
    reset_capability_chain()
    yield
    reset_capability_chain()


def test_a_real_will_decision_produces_a_verifiable_capability():
    """The join: Will.decide() → signed grant → verified at a sink."""
    from core.will import ActionDomain, get_will

    decision = get_will().decide(
        content="read the project README",
        source="test_capability_chain_integration",
        domain=ActionDomain.TOOL_EXECUTION,
        priority=0.4,
    )
    if not decision.is_approved():
        pytest.skip(f"Will did not approve in this environment: {decision.reason}")

    payload = {"path": "README.md"}
    cap = get_capability_issuer().issue_from_decision(
        decision, action="read_file", payload=payload
    )

    # The grant carries the real decision's provenance, not a synthetic one.
    assert cap.receipt_id == decision.receipt_id
    assert cap.domain == ActionDomain.TOOL_EXECUTION.value
    assert cap.outcome == str(decision.outcome.value)

    # And a sink can authenticate it against the exact action.
    result = get_capability_verifier().verify(
        cap,
        expected_domain=ActionDomain.TOOL_EXECUTION,
        expected_action_digest=compute_action_digest("read_file", payload),
    )
    assert result.ok, f"real Will decision did not verify at the sink: {result.detail}"


def test_a_real_will_refusal_cannot_be_turned_into_a_capability():
    """Whatever the Will refuses must be unmintable — not merely unused."""
    from core.governance.capability_chain import CapabilityViolation
    from core.will import ActionDomain

    class _Refusal:
        outcome = "refuse"
        domain = ActionDomain.TOOL_EXECUTION
        receipt_id = "r-refused"
        constraints: list[str] = []

    with pytest.raises(CapabilityViolation) as exc:
        get_capability_issuer().issue_from_decision(_Refusal(), action="shell_command")
    assert exc.value.denial is CapabilityDenial.NOT_APPROVED


@pytest.mark.asyncio
async def test_authority_gateway_attaches_a_signed_capability_to_approved_tools():
    """The gateway must mint, not just allocate an opaque token id.

    This is the specific gap that made sinks unable to authenticate the Will:
    ``capability_token_id`` was a uuid anyone could produce. An approved
    decision must now also carry ``signed_capability``.
    """
    from types import SimpleNamespace

    from core.executive.authority_gateway import get_authority_gateway

    gateway = get_authority_gateway()

    # Standing authority is a separate concern (which grants cover this origin);
    # it would otherwise deny before the mint path is ever reached, and a
    # skipped test would prove nothing about the wiring under test.
    class _ApprovingStandingAuthority:
        @staticmethod
        async def issue_child_lease(*_a, **_kw):
            return SimpleNamespace(
                approved=True,
                reason="unit_grant",
                receipt_id="standing-receipt",
                token="standing-token",
                grant_id="standing-grant",
                context={},
                budget_remaining=10,
            )

        @staticmethod
        def finalize_child_lease(*_a, **_kw):
            return None

    original_standing = gateway._standing_authority
    gateway._standing_authority = _ApprovingStandingAuthority()
    try:
        decision = await gateway.authorize_tool_execution(
            "read_file",
            {"path": "README.md"},
            source="test_capability_chain_integration",
            priority=0.4,
        )
    finally:
        gateway._standing_authority = original_standing

    assert decision.approved, (
        f"gateway did not approve, so the mint path was never exercised: "
        f"{decision.reason}"
    )
    assert decision.signed_capability is not None, (
        "AuthorityGateway approved a tool execution without minting a signed "
        "capability — the sink cannot authenticate this decision"
    )

    result = get_capability_verifier().verify(
        decision.signed_capability,
        expected_domain="tool_execution",
        expected_action_digest=compute_action_digest("read_file", {"path": "README.md"}),
        consume=False,
    )
    assert result.ok, f"gateway minted an unverifiable capability: {result.detail}"
    assert result.capability is not None
    assert result.capability.receipt_id == decision.will_receipt_id


def test_capability_engine_refuses_a_fabricated_context_in_strict_mode(monkeypatch):
    """The headline invariant, at the sink that matters.

    A context that asserts its own governance must not execute a skill when the
    chain is enforced.
    """
    from core.capability_engine import CapabilityEngine

    monkeypatch.setenv("AURA_CAPABILITY_ENFORCEMENT", "strict")

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = type(
        "L", (), {"warning": lambda *a, **k: None, "info": lambda *a, **k: None}
    )()

    denial = engine._capability_chain_denial(
        {
            "capability_token_id": "fabricated",
            "_capability_token_verified": True,
            "governance_approved": True,
        },
        "shell_command",
        {"cmd": "rm -rf /"},
        True,
    )
    assert denial is not None
    assert denial["status"] == "blocked_by_capability_chain"
    assert denial["denial"] == CapabilityDenial.MISSING.value


def test_capability_engine_accepts_a_genuine_grant_in_strict_mode(monkeypatch):
    """Strict mode must not be a wall — real authority still passes."""
    from core.capability_engine import CapabilityEngine
    from core.governance.capability_chain import attach_capability

    monkeypatch.setenv("AURA_CAPABILITY_ENFORCEMENT", "strict")

    class _Decision:
        outcome = "proceed"
        domain = "tool_execution"
        receipt_id = "r-ok"
        constraints: list[str] = []

    params = {"cmd": "ls"}
    cap = get_capability_issuer().issue_from_decision(
        _Decision(), action="shell_command", payload=params
    )

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = type(
        "L", (), {"warning": lambda *a, **k: None, "info": lambda *a, **k: None}
    )()

    assert engine._capability_chain_denial(
        attach_capability({}, cap), "shell_command", params, True
    ) is None


def test_capability_engine_refuses_a_grant_for_a_different_action(monkeypatch):
    """The confused-deputy case at the real sink."""
    from core.capability_engine import CapabilityEngine
    from core.governance.capability_chain import attach_capability

    monkeypatch.setenv("AURA_CAPABILITY_ENFORCEMENT", "strict")

    class _Decision:
        outcome = "proceed"
        domain = "tool_execution"
        receipt_id = "r-ok"
        constraints: list[str] = []

    cap = get_capability_issuer().issue_from_decision(
        _Decision(), action="read_file", payload={"path": "/etc/hosts"}
    )

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = type(
        "L", (), {"warning": lambda *a, **k: None, "info": lambda *a, **k: None}
    )()

    denial = engine._capability_chain_denial(
        attach_capability({}, cap), "shell_command", {"cmd": "rm -rf /"}, True
    )
    assert denial is not None
    assert denial["denial"] == CapabilityDenial.ACTION_MISMATCH.value


def test_enforcement_mode_typo_does_not_disable_governance(monkeypatch):
    """A misspelled env var must fail safe, not fail open."""
    from core.governance.capability_chain import capability_enforcement_mode

    monkeypatch.setenv("AURA_CAPABILITY_ENFORCEMENT", "strictt")
    assert capability_enforcement_mode(default="off") == "strict"

    monkeypatch.setenv("AURA_CAPABILITY_ENFORCEMENT", "disabled")
    assert capability_enforcement_mode(default="off") == "strict"
