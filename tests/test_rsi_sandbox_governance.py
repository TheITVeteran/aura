"""Regression: RSI candidate-evaluation sandbox must run governed.

Reproduces the production fail-closed path that crashed the in-process RSI
proof: ``subprocess_gateway.run`` raises ``GovernanceViolation`` when the
sandbox spawn happens outside a governed scope under hard governance. The fix
wraps the spawn in ``local_internal_governed_scope(domain="tool_execution")``.
"""

import os

import pytest


@pytest.fixture
def hard_governance(monkeypatch):
    # Force the runtime to enforce hard governance so the subprocess gateway
    # fail-closes on any ungoverned effect, exactly as in the live proof.
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    yield


def test_run_generated_solver_executes_under_hard_governance(hard_governance):
    from core.learning.autonomous_rsi import Task, _run_generated_solver

    source = (
        "import math\n"
        "\n"
        "def solve(task):\n"
        "    meta = task.metadata\n"
        "    if task.kind == 'gcd':\n"
        "        return math.gcd(int(meta['a']), int(meta['b']))\n"
        "    return None\n"
    )

    task = Task("gcd", "", 6, {"a": 54, "b": 24})
    ok, predicted, error = _run_generated_solver(task, source)

    assert ok, f"governed sandbox should execute; got error={error!r}"
    assert predicted == 6, f"expected gcd(54,24)=6, got {predicted!r}"


def test_ungoverned_effect_still_fails_closed(hard_governance):
    # Guard the teeth: an ungoverned effectful spawn must STILL fail closed
    # under hard governance — the fix governs the operation, it does not weaken
    # the gate.
    from core.governance_context import GovernanceViolation
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with pytest.raises(GovernanceViolation):
        get_subprocess_gateway().run(
            ["true"],
            capture_output=True,
            timeout=1.0,
            source="test_rsi_sandbox_governance.ungoverned",
        )
