"""RSI loosening: deeper recursion + opt-in, reversible tool-creation proposals.

The critique asked to loosen RSI — depth 5 not 3, allow tool-creation proposals — while
keeping reversibility. These tests lock that exact contract: the new lever is opt-in,
env-gated, gap-driven, and produces a *reversible proposal* that is never executed.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.learning.recursive_self_improvement import RecursiveSelfImprovementLoop


def _loop(tmp_path):
    return RecursiveSelfImprovementLoop(
        ledger_path=tmp_path / "rsi.jsonl",
        require_will_authorization=False,  # exercise the mechanism, not Will here
        auto_recurse=False,
    )


def test_default_depth_is_five():
    loop = RecursiveSelfImprovementLoop(require_will_authorization=False)
    assert loop.max_depth == 5


def test_tool_creation_absent_without_optin(tmp_path):
    loop = _loop(tmp_path)
    plan = loop._make_plan(
        "close a gap",
        allow_weight_update=False,
        allow_code_modification=False,
        allow_tool_creation=False,
        force=True,
        depth=0,
    )
    assert "tool_creation" not in plan.actions


def test_tool_creation_requires_env_flag(tmp_path, monkeypatch):
    loop = _loop(tmp_path)
    monkeypatch.delenv("AURA_RSI_TOOL_CREATION", raising=False)
    plan = loop._make_plan(
        "close a gap",
        allow_weight_update=False,
        allow_code_modification=False,
        allow_tool_creation=True,   # opt-in, but env flag absent
        force=True,
        depth=0,
    )
    assert "tool_creation" not in plan.actions


def test_tool_creation_planned_when_optin_and_flag_and_gap(tmp_path, monkeypatch):
    loop = _loop(tmp_path)
    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "1")
    loop.record_signal("affordance", "capability_gap", severity=0.8, metric="capability")
    plan = loop._make_plan(
        "need a new tool",
        allow_weight_update=False,
        allow_code_modification=False,
        allow_tool_creation=True,
        force=False,
        depth=0,
    )
    assert "tool_creation" in plan.actions


def test_tool_creation_produces_reversible_unexecuted_proposal(tmp_path, monkeypatch):
    loop = _loop(tmp_path)
    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "1")

    plan = loop._make_plan(
        "need a new tool",
        allow_weight_update=False,
        allow_code_modification=False,
        allow_tool_creation=True,
        force=True,
        depth=0,
    )
    result = asyncio.run(loop._run_tool_creation(plan))
    assert result["ok"] is True
    assert result["reversible"] is True
    assert result["executed"] is False
    # the proposal was persisted as a reversible draft, not run
    path = result["path"]
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows and rows[-1]["status"] == "proposed"


def test_tool_creation_refuses_irreversible_self_modifier(tmp_path, monkeypatch):
    """If a self_modifier offers tool proposals but can't guarantee reversibility, refuse."""
    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "1")

    class _IrreversibleModifier:
        def propose_tool(self, proposal):
            return {"ok": True, "reversible": False}  # cannot be undone

    loop = RecursiveSelfImprovementLoop(
        ledger_path=tmp_path / "rsi.jsonl",
        self_modifier=_IrreversibleModifier(),
        require_will_authorization=False,
        auto_recurse=False,
    )
    plan = loop._make_plan(
        "need a new tool",
        allow_weight_update=False,
        allow_code_modification=False,
        allow_tool_creation=True,
        force=True,
        depth=0,
    )
    result = asyncio.run(loop._run_tool_creation(plan))
    assert result["ok"] is False
    assert "not_reversible" in result["reason"]
