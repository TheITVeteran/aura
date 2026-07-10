from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.planner import ExecutionPlan, Planner, ToolCall
from core.runtime.expectation_feedback import (
    expectation_feedback_fingerprint,
    format_expectation_repair_guidance,
    recent_expectation_repair_signals,
)
from core.runtime.receipts import (
    ToolExecutionReceipt,
    get_receipt_store,
    reset_receipt_store,
)


def _failed_expectation_receipt(
    *,
    receipt_id: str,
    objective: str,
    created_at: float | None = None,
) -> ToolExecutionReceipt:
    kwargs = {"created_at": created_at} if created_at is not None else {}
    return ToolExecutionReceipt(
        receipt_id=receipt_id,
        cause=objective,
        tool="web_search",
        status="success_unverified",
        verification_evidence={
            "expectation_verdict": {
                "passed": False,
                "status": "success_unverified",
                "missing_criteria": [],
                "missing_evidence": ["sources"],
                "next_step": "rerun_web_research_with_sources",
            }
        },
        metadata={
            "source": "capability_engine.action_expectation",
            "expectation_objective": objective,
            "expectation_next_step": "rerun_web_research_with_sources",
            "passed": False,
        },
        **kwargs,
    )


def test_expectation_feedback_selects_only_recent_goal_relevant_failures(tmp_path):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    try:
        store.emit(
            _failed_expectation_receipt(
                receipt_id="expect-relevant",
                objective="source-backed web research for Europa ocean chemistry",
                created_at=1_000.0,
            )
        )
        store.emit(
            _failed_expectation_receipt(
                receipt_id="expect-unrelated",
                objective="source-backed web research for quarterly payroll policy",
                created_at=1_000.0,
            )
        )
        store.emit(
            _failed_expectation_receipt(
                receipt_id="expect-stale",
                objective="source-backed web research for Europa ocean chemistry",
                created_at=100.0,
            )
        )

        signals = recent_expectation_repair_signals(
            "find Europa ocean chemistry",
            available_tools=["web_search"],
            receipt_store=store,
            now=1_100.0,
            max_age_s=200.0,
        )

        assert [signal.receipt_id for signal in signals] == ["expect-relevant"]
        assert signals[0].missing_evidence == ("sources",)
        guidance = format_expectation_repair_guidance(signals)
        assert "rerun_web_research_with_sources" in guidance
        assert "quarterly payroll" not in guidance
        assert expectation_feedback_fingerprint(signals) != "none"
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_planner_invalidates_shallow_cache_and_uses_expectation_receipt(
    tmp_path,
    monkeypatch,
):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")

    class Brain:
        def __init__(self):
            self.prompts = []

        async def think(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return SimpleNamespace(
                content={
                    "plan_steps": [
                        "Gather source-backed research",
                        "Verify citation evidence before completion",
                    ],
                    "tool_calls": [
                        {
                            "tool": "web_search",
                            "params": {
                                "query": "Europa ocean chemistry",
                                "deep": True,
                            },
                            "output_var": "research",
                        }
                    ],
                }
            )

    registry = SimpleNamespace(
        skills={
            "web_search": SimpleNamespace(
                description="Search the web with source evidence",
            )
        }
    )
    brain = Brain()
    planner = Planner(brain, registry=registry)
    monkeypatch.setattr(planner, "save_to_disk", lambda _plan: None)

    try:
        first = await planner.decompose("find Europa ocean chemistry")
        assert first.tool_calls[0].tool == "web_search"
        assert brain.prompts == []

        store.emit(
            _failed_expectation_receipt(
                receipt_id="expect-plan-repair",
                objective="source-backed web research for Europa ocean chemistry",
            )
        )

        second = await planner.decompose("find Europa ocean chemistry")

        assert len(brain.prompts) == 1
        assert "RECENT EXPECTATION FAILURES" in brain.prompts[0]
        assert "rerun_web_research_with_sources" in brain.prompts[0]
        assert second.metadata["expectation_feedback_receipt_ids"] == [
            "expect-plan-repair"
        ]
        assert planner.get_stats()["expectation_guided_plans"] == 1
        assert planner.get_stats()["cache_hits"] == 0
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_planner_critic_rejection_consumes_replan_budget(tmp_path, monkeypatch):
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")

    class Brain:
        def __init__(self):
            self.calls = 0

        async def think(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                content={
                    "plan_steps": ["Use a different evidence-gathering path"],
                    "tool_calls": [
                        {
                            "tool": "native_chat",
                            "params": {"message": "report bounded failure"},
                            "output_var": "report",
                        }
                    ],
                }
            )

    class RejectingCritic:
        async def critique_plan(self, plan, evidence):
            return SimpleNamespace(
                recommendation="backtrack",
                evidence="still lacks effect evidence",
            )

    brain = Brain()
    planner = Planner(brain)
    planner.critic = RejectingCritic()
    monkeypatch.setattr(planner, "save_to_disk", lambda _plan: None)
    original = ExecutionPlan(
        goal="repair a failed action",
        plan_steps=["Retry the failed action"],
        tool_calls=[
            ToolCall(tool="native_chat", params={"message": "retry"})
        ],
        replan_budget=2,
    )

    try:
        revised = await planner.revise_plan(original, "verification failed", 0)

        assert brain.calls == 2
        assert revised.metadata["source"] == "fallback"
        assert revised.replan_budget == 0
    finally:
        reset_receipt_store()
