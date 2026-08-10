"""The governed scope a call already runs in must reach the Will.

LIVE DEFECT, 2026-08-10. Aura's screen perception was dead in production and
the only visible symptom was a warning card repeating on every ambient tick:

    WILL REFUSED: host_automation.screenshot_directory/file_write --
    denied_by_default: file_write requires validated scoped authority

``ActionExecutor.execute`` built its Will context from action_id,
request_digest, expectation objective and a rollback flag — no authority
provenance, and no parameter through which a caller could supply any. Under
strict default-deny that made every FILE_WRITE, NETWORK_CALL and
TOOL_EXECUTION routed through the executor refused *structurally*: passing the
gate was impossible, so ``take_screenshot`` failed before reaching
screencapture and retention cleanup could never delete a file.

These tests run against the REAL Will with strict default-deny active,
because that is the only configuration in which the defect exists — a bare
test process leaves strict mode off and admits the action either way, which
is precisely how this shipped unnoticed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def strict_will(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the live governance posture, not the permissive test one."""
    monkeypatch.setenv("AURA_STRICT_WILL", "1")
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    from core.governance.will import _strict_default_deny_enabled

    if not _strict_default_deny_enabled():
        pytest.skip("strict default-deny not resolvable in this runtime")


def _authority_context(domain):
    from core.runtime.action_executor import _ambient_authority_context

    return _ambient_authority_context(domain)


def test_ungoverned_call_carries_no_authority() -> None:
    """Ungoverned code must gain nothing — this is not a blanket bypass."""
    from core.governance.will import ActionDomain

    assert _authority_context(ActionDomain.FILE_WRITE) == {}


def test_governed_scope_authorizes_only_its_own_domain() -> None:
    """A state_mutation scope must not authorize a network call."""
    from core.governance.will import ActionDomain
    from core.governance_context import local_internal_governed_scope

    with local_internal_governed_scope("test.internal", domain="state_mutation"):
        context = _authority_context(ActionDomain.NETWORK_CALL)

    assert "scoped_authority" not in context
    assert "capability_token_id" not in context
    # The refusal must still say the scope was seen, so a mismatch never again
    # reads as "no governance at all".
    assert context["ambient_governed_scope"] == "state_mutation"
    assert context["ambient_scope_domain_mismatch"] == "network_call"


def test_matching_scope_supplies_scoped_authority() -> None:
    from core.governance.will import ActionDomain
    from core.governance_context import local_internal_governed_scope

    with local_internal_governed_scope(
        "host_automation.screenshot_directory",
        domain=ActionDomain.FILE_WRITE.value,
        constraints={"path": "/tmp/x", "op": "ensure_directory"},
    ):
        context = _authority_context(ActionDomain.FILE_WRITE)

    assert context["scoped_authority"] == "governed_scope:file_write"
    assert context["authority_origin"] == "host_automation.screenshot_directory"
    assert context["capability_token_id"]


def test_expired_scope_does_not_authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scope must not outlive its TTL.

    ``local_internal_governed_scope`` only ever lengthens a lease, so an
    expired token is installed directly — the point under test is that
    ``_ambient_authority_context`` trusts ``get_active_governance``'s validity
    check rather than reading the contextvar itself.
    """
    import time

    from core.governance import will as will_module
    from core.governance_context import GovernanceToken, _active_receipt

    stale = GovernanceToken(
        receipt_id="stale-receipt",
        domain=will_module.ActionDomain.FILE_WRITE.value,
        source="test.expiring",
        mono_timestamp=time.monotonic() - 10_000.0,
        ttl=1.0,
    )
    assert stale.expired

    reset = _active_receipt.set(stale)
    try:
        context = _authority_context(will_module.ActionDomain.FILE_WRITE)
    finally:
        _active_receipt.reset(reset)

    assert context == {}


@pytest.mark.usefixtures("strict_will")
def test_screenshot_directory_is_refused_without_a_scope(tmp_path: Path) -> None:
    """The exact live failure, reproduced: no scope ⇒ Will refuses."""
    from core.governance.will import ActionDomain
    from core.runtime.action_executor import ActionExecutor

    result = asyncio.run(
        ActionExecutor.execute(
            domain=ActionDomain.FILE_WRITE,
            action_name="host_automation.ensure_screenshot_directory",
            params={"path": str(tmp_path / "shots"), "op": "ensure_directory"},
            source="host_automation.screenshot_directory",
        )
    )

    assert result["ok"] is False
    assert "validated scoped authority" in str(result["error"])


@pytest.mark.usefixtures("strict_will")
def test_screenshot_directory_is_admitted_inside_its_scope(tmp_path: Path) -> None:
    """Screen perception's directory step must survive strict default-deny."""
    from core.governance.will import ActionDomain
    from core.governance_context import local_internal_governed_scope
    from core.runtime.action_executor import ActionExecutor

    target = tmp_path / "shots"

    async def _run() -> dict:
        with local_internal_governed_scope(
            "host_automation.screenshot_directory",
            domain=ActionDomain.FILE_WRITE.value,
            constraints={"path": str(target), "op": "ensure_directory"},
        ):
            return await ActionExecutor.execute(
                domain=ActionDomain.FILE_WRITE,
                action_name="host_automation.ensure_screenshot_directory",
                params={"path": str(target), "op": "ensure_directory"},
                source="host_automation.screenshot_directory",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True, result.get("error")


def test_take_screenshot_declares_its_own_scope() -> None:
    """The capture path must open the scope itself, not rely on its caller.

    screen_perception calls take_screenshot from an ambient loop that holds no
    scope of its own; if the declaration lived in the caller the live defect
    would simply return.
    """
    import inspect

    from core.capabilities import host_automation

    source = inspect.getsource(host_automation.HostAutomationProvider.take_screenshot)
    assert "local_internal_governed_scope" in source
    assert "ensure_screenshot_directory" in source


def test_screenshot_retention_declares_its_own_scope() -> None:
    """Retention deletion was refused the same way, so it never reclaimed."""
    import inspect

    from core.capabilities import host_automation

    source = inspect.getsource(host_automation.HostAutomationProvider)
    marker = "host_automation.screenshot_retention_delete"
    assert marker in source
    prefix = source.split(marker)[0]
    assert "local_internal_governed_scope" in prefix.rsplit("try:", 1)[-1]
