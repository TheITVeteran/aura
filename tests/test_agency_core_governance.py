from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_swarm_shard_tool_execution_creates_governed_subprocess_scope(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    from core.agency.agency_core import SovereignSwarm
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    class SubprocessBackedTool:
        async def route_and_execute(self, _name, _payload):
            result = get_subprocess_gateway().run(
                ["true"],
                capture_output=True,
                timeout=1.0,
                source="test.agency_core_governance.shard_tool",
            )
            return f"ok:{result.returncode}"

    owner = SimpleNamespace(tool_orchestrator=SubprocessBackedTool(), _last_tool_routing_error=None)
    swarm = SovereignSwarm(SimpleNamespace(), agency_core=owner)

    assert await swarm._execute_shard_tool("python_sandbox", "print('safe')") == "ok:0"
    assert owner._last_tool_routing_error is None


def test_agency_degradation_reporting_cannot_cascade(monkeypatch):
    from core.agency import agency_core

    def raising_reporter(*_args, **_kwargs):
        raise RuntimeError("failure policy fail-closed")

    monkeypatch.setattr(agency_core, "record_degradation", raising_reporter)

    agency_core._record_agency_degradation(
        RuntimeError("original shard failure"),
        action="shard cleanup failed",
    )


def test_ungoverned_subprocess_still_fails_closed(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    from core.governance_context import GovernanceViolation
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with pytest.raises(GovernanceViolation):
        get_subprocess_gateway().run(
            ["true"],
            capture_output=True,
            timeout=1.0,
            source="test.agency_core_governance.ungoverned",
        )
