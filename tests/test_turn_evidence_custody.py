"""Foreground evidence has exact turn ownership, including deliberate children."""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.failure_context import bind_failure_ledger, record_capability_failure
from core.conversation.surface_disposition import (
    begin_turn_tool_receipts,
    record_tool_receipt,
    turn_tool_receipts,
)
from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    join_turn_evidence_custody,
    record_turn_capability_availability,
    record_turn_grounding,
    turn_capability_availability,
    turn_grounding_evidence,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_deliberate_child_receipt_is_visible_to_parent() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
        begin_turn_tool_receipts()
        lease = custody.issue_child_lease("foreground tool")

        async def child() -> None:
            with join_turn_evidence_custody(lease):
                assert record_tool_receipt(
                    "desktop_task",
                    action="open_app",
                    object_ref="Notes",
                    ok=True,
                    effect_observed=True,
                )

        await asyncio.create_task(child())
        receipts = turn_tool_receipts()
        assert len(receipts) == 1
        assert (receipts[0]["session_id"], receipts[0]["turn_id"]) == ("s", "t")


@pytest.mark.asyncio
async def test_ambient_background_child_cannot_write_turn_evidence() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()

        async def background() -> bool:
            return record_tool_receipt("autonomous_scan", ok=True)

        assert await asyncio.create_task(background()) is False
        assert turn_tool_receipts() == ()


@pytest.mark.asyncio
async def test_failure_ledger_uses_the_same_explicit_child_custody() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
        with bind_failure_ledger() as ledger:
            async def ambient() -> object:
                return record_capability_failure(
                    "background",
                    intent="unrelated scan",
                    cause="failed",
                )

            assert await asyncio.create_task(ambient()) is None
            lease = custody.issue_child_lease("foreground failure")

            async def foreground() -> object:
                with join_turn_evidence_custody(lease):
                    return record_capability_failure(
                        "web_search",
                        intent="find current evidence",
                        cause="offline",
                    )

            assert await asyncio.create_task(foreground()) is not None
            assert [item.capability for item in ledger.records] == ["web_search"]


def test_a_lease_is_one_use_and_cannot_cross_turns() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t-a") as custody:
        lease = custody.issue_child_lease("one use")
        with join_turn_evidence_custody(lease):
            pass
        with pytest.raises(PermissionError):
            with join_turn_evidence_custody(lease):
                pass

    with bind_turn_evidence_custody(session_id="s", turn_id="t-b"):
        with pytest.raises(PermissionError):
            with join_turn_evidence_custody(lease):
                pass


def test_grounding_and_availability_are_exact_turn_owned() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert record_turn_grounding("Bryan said his favorite animal is the orca")
        assert record_turn_capability_availability(
            "web",
            available=False,
            reason="network disconnected",
            observed_at=123.0,
        )
        assert turn_grounding_evidence() == (
            "Bryan said his favorite animal is the orca",
        )
        assert turn_capability_availability()[0]["turn_id"] == "t"

    assert turn_grounding_evidence() == ()
    assert turn_capability_availability() == ()
