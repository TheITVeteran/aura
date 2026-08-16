"""No skill effect runs outside the governed lane, whatever door it came through.

`BaseSkill.safe_execute` carried the active-runtime governance requirement and
its docstring called itself "the PUBLIC entry point that the skill router
should call". Nothing made it the only entry point. `DesktopPlanner`'s adapter
called `skill.execute(...)` directly, and so did `core/tools/computer_use.py`
and two capability modules — four live paths around the check, in the lane that
drives the mouse and keyboard.

`FluidExecutor` had the matching hole on the approval side: `gateway=None`
returned `(True, "")`, and `DesktopPlanner`, `GoalPursuitEngine` and
`ParallelExecutor` all construct it without a gateway.

The fix has to be structural, because an invariant maintained by every caller
remembering is not an invariant. `execute` governs itself, and a missing
gateway resolves the canonical one or refuses.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.governance_context import local_internal_governed_scope
from core.skills.action_gateway import ActionDecision, ActionGateway, get_action_gateway
from core.skills.base_skill import BaseSkill
from core.skills.fluid_executor import FluidExecutor, Step, _read_decision

pytestmark = pytest.mark.unit


class _Probe(BaseSkill):
    name = "governance_probe"
    description = "records whether its body ran"

    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, params, context):  # noqa: ANN001, ANN201
        self.ran += 1
        return {"ok": True, "ran": self.ran}


class _SyncProbe(BaseSkill):
    name = "sync_governance_probe"
    description = "a skill whose execute is not a coroutine function"

    def __init__(self) -> None:
        self.ran = 0

    def execute(self, params, context):  # noqa: ANN001, ANN201
        self.ran += 1
        return {"ok": True, "ran": self.ran}


async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# The raw execute() door
# ---------------------------------------------------------------------------


def test_execute_is_wrapped_on_every_subclass():
    assert getattr(_Probe.execute, "__aura_governed_execute__", False)
    assert getattr(_SyncProbe.execute, "__aura_governed_execute__", False)


def test_raw_execute_is_refused_under_strict_governance(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    probe = _Probe()
    result = asyncio.run(probe.execute({}, {}))
    assert result["ok"] is False
    assert "Ungoverned skill execution blocked" in result["error"]
    assert probe.ran == 0, "the skill body ran despite the refusal"


def test_raw_execute_on_a_sync_skill_is_refused_too(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    probe = _SyncProbe()
    result = probe.execute({}, {})
    assert result["ok"] is False
    assert probe.ran == 0


def test_raw_execute_proceeds_inside_a_governed_scope(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    probe = _Probe()

    async def main():
        with local_internal_governed_scope("test", domain="tool_execution"):
            return await probe.execute({}, {})

    result = asyncio.run(main())
    assert result["ok"] is True
    assert probe.ran == 1


def test_safe_execute_still_works_and_does_not_double_check(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    probe = _Probe()

    async def main():
        with local_internal_governed_scope("test", domain="tool_execution"):
            return await probe.safe_execute({}, {})

    result = asyncio.run(main())
    assert result["ok"] is True
    assert probe.ran == 1


def test_the_exemption_does_not_leak_to_a_nested_raw_call(monkeypatch):
    """A skill that calls another skill's raw execute must not inherit cover."""
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    inner = _Probe()

    class _Outer(BaseSkill):
        name = "outer"

        async def execute(self, params, context):  # noqa: ANN001, ANN201
            # safe_execute set the flag for THIS call; the nested call is a
            # separate raw entry and must be checked on its own.
            return await inner.execute({}, {})

    outer = _Outer()
    result = asyncio.run(outer.safe_execute({}, {}))
    assert inner.ran == 0
    assert result["ok"] is False


def test_wrapping_is_idempotent_across_reload():
    """Hot-reload re-runs __init_subclass__; a second layer would double-log."""
    first = _Probe.execute
    wrapped_again = BaseSkill.__init_subclass__.__func__  # type: ignore[attr-defined]
    assert callable(wrapped_again)
    from core.skills.base_skill import _governed_execute

    assert _governed_execute(first) is first


def test_the_unwrapped_body_stays_reachable():
    assert callable(getattr(_Probe.execute, "__aura_unwrapped_execute__", None))


# ---------------------------------------------------------------------------
# The FluidExecutor approval door
# ---------------------------------------------------------------------------


def test_a_missing_gateway_resolves_the_canonical_one():
    executor = FluidExecutor(verifier=None, sleep=lambda _s: asyncio.sleep(0))
    result = asyncio.run(executor.run_step(Step("open_app", _noop)))
    assert result.ok and not result.blocked


def test_no_gateway_anywhere_refuses_rather_than_allows(monkeypatch):
    executor = FluidExecutor(verifier=None, sleep=lambda _s: asyncio.sleep(0))
    monkeypatch.setattr(
        executor, "_resolve_gateway", lambda: asyncio.sleep(0, result=None)
    )
    result = asyncio.run(executor.run_step(Step("anything", _noop)))
    assert result.blocked, "a lane with no approver ran the step anyway"
    assert "no action gateway" in result.detail


def test_a_raising_gateway_refuses_rather_than_allows():
    class _Broken:
        def approve(self, name):  # noqa: ANN001, ANN201
            raise RuntimeError("gateway down")

    executor = FluidExecutor(
        verifier=None, gateway=_Broken(), sleep=lambda _s: asyncio.sleep(0)
    )
    result = asyncio.run(executor.run_step(Step("anything", _noop)))
    assert result.blocked
    assert "gateway down" in result.detail


def test_a_mapping_refusal_is_read_as_a_refusal():
    """core/security/conscience.py returns a dict; a non-empty dict is truthy.

    `bool(getattr(decision, "allowed", decision))` therefore read every dict
    refusal as approval — including `{"allowed": False}`.
    """
    assert _read_decision({"allowed": False, "reason": "vetoed"}) == (False, "vetoed")
    assert _read_decision({"allowed": True, "reason": ""}) == (True, "")
    assert _read_decision(SimpleNamespace(allowed=False, reason="no")) == (False, "no")
    assert _read_decision(ActionDecision(allowed=False, reason="no")) == (False, "no")


def test_a_dict_refusal_blocks_a_step():
    class _DictGate:
        def approve(self, name):  # noqa: ANN001, ANN201
            return {"allowed": False, "reason": "conscience veto"}

    executor = FluidExecutor(
        verifier=None, gateway=_DictGate(), sleep=lambda _s: asyncio.sleep(0)
    )
    result = asyncio.run(executor.run_step(Step("danger", _noop)))
    assert result.blocked and "conscience veto" in result.detail


# ---------------------------------------------------------------------------
# The gateway itself
# ---------------------------------------------------------------------------


def test_the_gateway_refuses_a_destructive_shell_command():
    decision = ActionGateway().approve("shell")
    assert decision.allowed  # no command supplied, nothing to refuse

    guarded = ActionGateway()

    class _Guard:
        @staticmethod
        def check_action(name, args):  # noqa: ANN001, ANN205
            return False

    import core.security.constitutional_guard as cg

    original = cg.ConstitutionalGuard
    cg.ConstitutionalGuard = lambda: _Guard()  # type: ignore[assignment]
    try:
        refused = guarded.approve("shell", {"command": "rm -rf /"})
    finally:
        cg.ConstitutionalGuard = original  # type: ignore[assignment]
    assert not refused.allowed
    assert "constitutional guard" in refused.reason


def test_unreadable_standing_directives_refuse(monkeypatch):
    """A prohibition file that will not parse is not an absence of rules."""
    import core.governance.standing_directives as sd

    class _Store:
        def check(self, *, tool_name, args, effect_scope=""):  # noqa: ANN001, ANN201
            return None, SimpleNamespace(unreadable=True, directives=[])

    monkeypatch.setattr(sd, "get_standing_directives", lambda: _Store())
    decision = ActionGateway().approve("open_app")
    assert not decision.allowed
    assert "unreadable" in decision.reason


def test_the_gateway_cannot_grant_only_decline():
    """An `allowed` result means no prohibition matched, not authorisation."""
    decision = get_action_gateway().approve("open_app")
    assert decision.allowed
    assert decision.reason == ""


# ---------------------------------------------------------------------------
# The source-level ratchet
#
# The runtime wrapper makes a raw call governed, not correct: it still skips
# input validation, the timeout, the circuit breaker, the retry policy and
# result normalisation. And a skill reached as a duck-typed object never goes
# through `__init_subclass__` at all. So the call sites are checked too.
# ---------------------------------------------------------------------------


def test_no_raw_skill_execute_call_sites_remain():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "check_raw_skill_execute.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_ratchet_can_fail(tmp_path):
    """Negative control: a fresh raw call site is rejected."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    gate = (root / "tools" / "check_raw_skill_execute.py").read_text(encoding="utf-8")

    (tmp_path / "tools").mkdir()
    (tmp_path / "core").mkdir()
    (tmp_path / "tools" / "check_raw_skill_execute.py").write_text(gate, encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "raw_skill_execute_baseline.json").write_text(
        '{"schema_version": 1, "call_sites": {}}\n', encoding="utf-8"
    )
    (tmp_path / "core" / "offender.py").write_text(
        "async def go(skill):\n    return await skill.execute({}, {})\n", encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, str(tmp_path / "tools" / "check_raw_skill_execute.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
    assert "core/offender.py" in result.stderr
