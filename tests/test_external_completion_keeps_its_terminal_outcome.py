"""Receipt closure follows the durable classification of a dispatched effect."""
from __future__ import annotations

import pytest

from core.brain.external_execute_coordinator import (
    ExternalExecuteCoordinator,
    _validate_post_action_receipt_outcome,
)
from core.runtime.action_executor import (
    _complete_external_execution_transaction,
    _terminal_effect_flags_invariant,
)
from tests.test_rlc_external_execution import _execute_trace, _offer, _record_handoff

pytestmark = pytest.mark.unit


def test_terminal_flags_canary_preserves_observations():
    assert _terminal_effect_flags_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,retry_safe,verified,state,manual", [
    (False, False, False, "UNKNOWN_EFFECT", True),
    (True, True, False, "UNKNOWN_EFFECT", True),
    (False, True, False, "FAILED", False),
    (True, False, True, "SUCCEEDED", False),
])
async def test_real_completion_seam_keeps_receipt_and_replay_consistent(
    tmp_path, transport, retry_safe, verified, state, manual,
):
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer()
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(offer, authorization_receipt_id=offer["will_receipt_id"], task_id="test-task")
    result = {"ok": verified, "status": "success_verified" if verified else "failed_recoverable",
              "error": "" if verified else "navigation_refused", "transport_succeeded": transport,
              "effect_verified": verified, "retry_safe": retry_safe, "manual_reconciliation_required": False,
              "steps": [{"completed": False}]}
    await _complete_external_execution_transaction(
        coordinator=coordinator, offer=offer, transaction=dispatch,
        dispatch_attempt_id=dispatch["dispatch_owner"]["attempt_id"], result=result, action_name=offer["action_name"],
    )
    terminal = result["external_execution_transaction"]
    assert terminal["state"] == state
    assert result["manual_reconciliation_required"] is manual
    assert result["retry_safe"] is (retry_safe and state == "FAILED")
    receipt = {**result, "error_status": result["error"]}
    _validate_post_action_receipt_outcome(receipt, transaction=terminal)
    replay = terminal["result"]["replay_payload"]
    assert replay["manual_reconciliation_required"] is manual
    assert replay["retry_safe"] is result["retry_safe"]
    assert coordinator.prepare(offer)["state"] == state
    assert result["steps"] == [{"completed": False}]
    if manual:
        with pytest.raises(ValueError, match="manual_reconciliation_required"):
            _validate_post_action_receipt_outcome({**receipt, "manual_reconciliation_required": False}, transaction=terminal)
