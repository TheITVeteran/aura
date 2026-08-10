"""Tool sequencing: the constraint that used to live only in prompt text.

The gap these cover: ToolPermissionGuard could say a tool was allowed to touch
the network, and nothing anywhere could say a tool was allowed to run *here*.
"You must cite the file you read" was an instruction, not a constraint, and an
instruction is satisfied by a model that feels like satisfying it.
"""
from __future__ import annotations

import pytest

from core.tools.tool_rules import (
    ChildToolRule,
    ConditionalToolRule,
    ContinueToolRule,
    InitToolRule,
    MaxCountPerStepToolRule,
    ParentToolRule,
    RequiredBeforeExitToolRule,
    RequiresApprovalToolRule,
    TerminalToolRule,
    ToolCall,
    ToolRuleSolver,
    UnsatisfiableRuleSet,
)

pytestmark = pytest.mark.unit

TOOLS = frozenset({"read", "cite", "search", "verify", "answer", "deploy"})


def _solver(*rules, tools=TOOLS):
    return ToolRuleSolver(rules, available_tools=tools)


# ── opening moves ──────────────────────────────────────────────────────────


def test_init_rule_restricts_the_first_call_only():
    solver = _solver(InitToolRule("search"))

    assert solver.allowed().allowed == {"search"}
    # Once the step is underway the rule stops applying.
    assert solver.allowed([ToolCall("search")]).allowed == TOOLS


def test_init_args_are_prefilled_and_override_the_model():
    solver = _solver(InitToolRule("search", args={"scope": "repo"}))

    verdict = solver.allowed()

    assert verdict.prefilled_args == {"search": {"scope": "repo"}}


# ── routing ────────────────────────────────────────────────────────────────


def test_child_rule_forces_the_next_call():
    solver = _solver(ChildToolRule("read", ["cite"]))

    assert solver.allowed([ToolCall("read")]).allowed == {"cite"}
    # ...and only immediately after.
    assert solver.allowed([ToolCall("search")]).allowed == TOOLS


def test_parent_rule_makes_children_unreachable_by_any_other_path():
    solver = _solver(ParentToolRule("read", ["cite"]))

    assert "cite" not in solver.allowed([ToolCall("search")]).allowed
    assert solver.allowed([ToolCall("read")]).allowed == {"cite"}


def test_conditional_rule_routes_on_the_tool_output():
    solver = _solver(
        ConditionalToolRule("verify", {True: "answer", False: "search"}),
    )

    assert solver.allowed([ToolCall("verify", output=True)]).allowed == {"answer"}
    assert solver.allowed([ToolCall("verify", output=False)]).allowed == {"search"}


def test_conditional_rule_matches_stringified_output():
    """A verifier handing back "True" routes like one handing back True."""
    solver = _solver(ConditionalToolRule("verify", {True: "answer"}))

    assert solver.allowed([ToolCall("verify", output="True")]).allowed == {"answer"}


def test_unmapped_output_falls_through_to_the_default_child():
    solver = _solver(
        ConditionalToolRule("verify", {True: "answer"}, default_child="search"),
    )

    assert solver.allowed([ToolCall("verify", output="???")]).allowed == {"search"}


def test_strict_router_fails_closed_on_an_unmapped_output():
    """Guessing would hide the gap in the rule set. Nothing is legal instead."""
    solver = _solver(
        ConditionalToolRule("verify", {True: "answer"}, require_mapping=True),
    )

    assert solver.allowed([ToolCall("verify", output="???")]).allowed == frozenset()


def test_a_strict_router_with_a_catch_all_is_a_contradiction():
    with pytest.raises(ValueError, match="contradictory"):
        ConditionalToolRule(
            "verify", {True: "answer"}, default_child="search", require_mapping=True
        )


# ── budgets ────────────────────────────────────────────────────────────────


def test_max_count_removes_the_tool_once_spent():
    solver = _solver(MaxCountPerStepToolRule("search", max_count=2))

    assert "search" in solver.allowed([ToolCall("search")]).allowed
    two = [ToolCall("search"), ToolCall("search")]
    assert "search" not in solver.allowed(two).allowed


def test_exceeded_reports_overruns_for_ungated_callers():
    solver = _solver(MaxCountPerStepToolRule("search", max_count=1))

    assert solver.exceeded([ToolCall("search")]) == frozenset()
    assert solver.exceeded([ToolCall("search")] * 3) == {"search"}


# ── exit conditions ────────────────────────────────────────────────────────


def test_required_before_exit_blocks_the_turn_from_ending():
    solver = _solver(RequiredBeforeExitToolRule("cite"))

    assert not solver.may_exit([ToolCall("read")])
    assert solver.uncalled_required([ToolCall("read")]) == {"cite"}
    assert solver.may_exit([ToolCall("read"), ToolCall("cite")])


def test_required_before_exit_does_not_narrow_the_tool_set():
    """It is an exit condition, not a router. Narrowing would strand the model."""
    solver = _solver(RequiredBeforeExitToolRule("cite"))

    assert solver.allowed([ToolCall("read")]).allowed == TOOLS


def test_continue_rule_blocks_exit_even_when_nothing_is_owed():
    solver = _solver(ContinueToolRule("search"))

    assert not solver.may_exit([ToolCall("search")])
    assert solver.may_exit([ToolCall("read")])


def test_terminal_rule_is_reported_not_enforced_by_narrowing():
    solver = _solver(TerminalToolRule("answer"))

    assert solver.is_terminal("answer")
    assert not solver.is_terminal("read")


def test_approval_tools_stay_visible_to_the_model():
    """Hiding the capability would be a lie, and models route around lies."""
    solver = _solver(RequiresApprovalToolRule("deploy"))

    assert "deploy" in solver.allowed().allowed
    assert solver.requires_approval("deploy")


# ── provenance: the part that goes past the prior art ──────────────────────


def test_a_rejected_tool_carries_the_rule_that_killed_it():
    solver = _solver(ChildToolRule("read", ["cite"]))

    verdict = solver.allowed([ToolCall("read")])

    assert verdict.why_not("search") is not None
    assert "cite" in verdict.why_not("search")
    assert verdict.why_not("cite") is None


def test_the_verdict_is_falsy_when_nothing_survives():
    solver = _solver(ConditionalToolRule("verify", {True: "answer"}, require_mapping=True))

    assert not solver.allowed([ToolCall("verify", output="x")])


# ── construction-time refusal ──────────────────────────────────────────────


def test_a_rule_naming_a_nonexistent_tool_is_refused_at_construction():
    """Otherwise it fails silently forever: the rule simply never fires."""
    with pytest.raises(UnsatisfiableRuleSet, match="do not exist"):
        _solver(ChildToolRule("read", ["ctie"]))


def test_terminal_and_continue_on_the_same_tool_is_refused():
    with pytest.raises(UnsatisfiableRuleSet, match="both terminal and continue"):
        _solver(TerminalToolRule("answer"), ContinueToolRule("answer"))


def test_init_rules_naming_no_available_tool_are_refused():
    with pytest.raises(UnsatisfiableRuleSet):
        _solver(InitToolRule("search"), tools=frozenset({"read"}))


def test_a_valid_rule_set_constructs_without_complaint():
    solver = _solver(
        InitToolRule("search"),
        ChildToolRule("read", ["cite"]),
        RequiredBeforeExitToolRule("cite"),
        TerminalToolRule("answer"),
        MaxCountPerStepToolRule("search", max_count=3),
    )

    assert len(solver.rules) == 5


# ── composition ────────────────────────────────────────────────────────────


def test_rules_compose_by_intersection():
    """Two narrowings apply together, not last-one-wins."""
    solver = _solver(
        ChildToolRule("read", ["cite", "search"]),
        MaxCountPerStepToolRule("search", max_count=1),
    )

    verdict = solver.allowed([ToolCall("search"), ToolCall("read")])

    assert verdict.allowed == {"cite"}


def test_render_constraints_lists_every_rule():
    solver = _solver(InitToolRule("search"), TerminalToolRule("answer"))

    rendered = solver.render_constraints()

    assert "search" in rendered and "answer" in rendered


def test_no_rules_means_no_constraint_text_and_no_narrowing():
    solver = ToolRuleSolver([], available_tools=TOOLS)

    assert solver.render_constraints() == ""
    assert solver.allowed().allowed == TOOLS
