"""The registry enforces sequencing, not just existence.

Before this, ToolRegistry.execute_tool answered exactly one question — is a
tool by this name registered — and then ran it. Ordering obligations lived in
prompt text. These cover the wiring: that the solver is actually consulted on
the execution path, that a refusal carries its reason, and that a failed call
still moves the sequencing state.
"""
from __future__ import annotations

import pytest

from core.tools.tool_registry import ToolRegistry
from core.tools.tool_rules import (
    ChildToolRule,
    MaxCountPerStepToolRule,
    RequiredBeforeExitToolRule,
    ToolRuleSolver,
    UnsatisfiableRuleSet,
)

pytestmark = pytest.mark.unit


class _Manifest:
    """Minimal stand-in: a tool whose sandbox body returns a constant."""

    def __init__(self, name: str, value: str = "ok") -> None:
        self.code = (
            f"class {name}:\n"
            f"    def run(self, *a, **k):\n"
            f"        return {value!r}\n"
        )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    for name in ("read", "cite", "search"):
        reg.register_tool(name, _Manifest(name))
    return reg


# ── the wiring is live ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_without_rules_the_registry_behaves_as_before(registry):
    result = await registry.execute_tool("read")

    assert result["ok"] is True
    assert result["result"] == "ok"


@pytest.mark.asyncio
async def test_a_rule_actually_blocks_execution(registry):
    """The solver is consulted on the execution path, not merely stored."""
    registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["cite"])]))

    await registry.execute_tool("read")
    blocked = await registry.execute_tool("search")

    assert blocked["ok"] is False
    assert blocked["error"].startswith("tool_not_permitted_here")


@pytest.mark.asyncio
async def test_the_permitted_tool_still_runs(registry):
    registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["cite"])]))

    await registry.execute_tool("read")
    allowed = await registry.execute_tool("cite")

    assert allowed["ok"] is True


@pytest.mark.asyncio
async def test_a_refusal_carries_its_reason(registry):
    """A tool blocked for an unexplained reason is indistinguishable from one
    that is broken, and the model retries it either way."""
    registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["cite"])]))
    await registry.execute_tool("read")

    blocked = await registry.execute_tool("search")

    assert "cite" in blocked["reason"]


@pytest.mark.asyncio
async def test_an_unregistered_tool_is_still_not_found(registry):
    registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["cite"])]))

    result = await registry.execute_tool("nonexistent")

    assert result["error"].startswith("tool_not_found")


# ── step state ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_step_history_records_what_ran(registry):
    await registry.execute_tool("read")
    await registry.execute_tool("cite")

    assert [c.name for c in registry.step_history] == ["read", "cite"]


@pytest.mark.asyncio
async def test_beginning_a_step_resets_the_history(registry):
    await registry.execute_tool("read")

    registry.begin_step()

    assert registry.step_history == ()


@pytest.mark.asyncio
async def test_a_per_step_budget_resets_with_the_step(registry):
    registry.set_rules(ToolRuleSolver([MaxCountPerStepToolRule("search", max_count=1)]))

    assert (await registry.execute_tool("search"))["ok"] is True
    assert (await registry.execute_tool("search"))["ok"] is False

    registry.begin_step()
    assert (await registry.execute_tool("search"))["ok"] is True


@pytest.mark.asyncio
async def test_a_failed_call_still_consumes_its_budget(registry):
    """Counting only successes would let a tool retry past its own cap."""
    registry.register_tool("broken", _ExplodingManifest())
    registry.set_rules(ToolRuleSolver([MaxCountPerStepToolRule("broken", max_count=1)]))

    first = await registry.execute_tool("broken")
    second = await registry.execute_tool("broken")

    assert first["ok"] is False
    assert second["error"].startswith("tool_not_permitted_here")


class _ExplodingManifest:
    code = "class broken:\n    def run(self, *a, **k):\n        raise ValueError('no')\n"


# ── exit obligations ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_step_may_not_finish_with_a_required_tool_outstanding(registry):
    registry.set_rules(ToolRuleSolver([RequiredBeforeExitToolRule("cite")]))

    await registry.execute_tool("read")

    assert registry.may_finish() is False
    assert registry.outstanding() == {"cite"}


@pytest.mark.asyncio
async def test_the_obligation_clears_once_the_tool_runs(registry):
    registry.set_rules(ToolRuleSolver([RequiredBeforeExitToolRule("cite")]))

    await registry.execute_tool("read")
    await registry.execute_tool("cite")

    assert registry.may_finish() is True
    assert registry.outstanding() == frozenset()


def test_without_rules_nothing_is_outstanding(registry):
    assert registry.may_finish() is True
    assert registry.outstanding() == frozenset()


# ── validation at install time ─────────────────────────────────────────────


def test_a_rule_naming_an_unregistered_tool_is_refused_on_install(registry):
    """Otherwise it never fires and nothing ever says so."""
    with pytest.raises(UnsatisfiableRuleSet, match="do not exist"):
        registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["ctie"])]))


def test_rules_can_be_cleared(registry):
    registry.set_rules(ToolRuleSolver([ChildToolRule("read", ["cite"])]))

    registry.set_rules(None)

    assert registry.may_finish() is True
