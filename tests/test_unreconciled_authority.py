"""A capability grant that never closed must not vanish into a log line.

``finalize_tool_execution`` builds a receipt naming exactly which of
intent / token / standing lease failed to close. Every call site discarded
it, so an unrevoked capability token — a live grant — was tracked by
nothing.
"""
from __future__ import annotations

import types

import pytest

from core.executive.authority_gateway import get_authority_gateway


@pytest.fixture
def gateway():
    gw = get_authority_gateway()
    with gw._unreconciled_lock:
        gw._unreconciled.clear()
        gw._unreconciled_total = 0
    original = gw._capabilities
    yield gw
    gw._capabilities = original
    with gw._unreconciled_lock:
        gw._unreconciled.clear()
        gw._unreconciled_total = 0


class _UnreachableCapabilities:
    def revoke_token(self, token_id):
        raise RuntimeError("token store unreachable")


def test_a_clean_finalization_queues_nothing(gateway):
    gateway.finalize_tool_execution(success=True)
    assert gateway.unreconciled_authority()["open"] == 0


def test_an_unrevoked_token_is_queued_even_when_the_receipt_is_ignored(gateway):
    """The defect: callers threw the receipt away and the leak disappeared."""
    gateway._capabilities = _UnreachableCapabilities()

    receipt = gateway.finalize_tool_execution(capability_token_id="cap-1", success=True)
    assert receipt["closed"] is False
    assert receipt["token_revoked"] is False

    queue = gateway.unreconciled_authority()
    assert queue["open"] == 1
    assert queue["entries"][0]["capability_token_id"] == "cap-1"


def test_the_queue_records_the_token_id_but_never_the_token(gateway):
    gateway._capabilities = _UnreachableCapabilities()
    gateway.finalize_tool_execution(
        capability_token_id="cap-2",
        standing_authority_token="SECRET-LEASE-MATERIAL",
        success=True,
    )
    entry = gateway.unreconciled_authority()["entries"][0]
    assert entry["capability_token_id"] == "cap-2"
    assert entry["standing_authority_token_present"] is True
    assert "SECRET-LEASE-MATERIAL" not in repr(entry)


def test_a_raised_finalization_is_recorded_by_the_caller_helper(gateway):
    """The path with no receipt to inspect at all."""
    from core.capability_engine import _record_unreconciled_authority

    _record_unreconciled_authority(
        types.SimpleNamespace(
            executive_intent_id="intent-7",
            capability_token_id="cap-7",
            standing_authority_token=None,
        ),
        reason="shell_finalize_raised:RuntimeError",
    )
    queue = gateway.unreconciled_authority()
    assert queue["open"] == 1
    entry = queue["entries"][0]
    assert entry["executive_intent_id"] == "intent-7"
    assert entry["reason"] == "shell_finalize_raised:RuntimeError"


def test_the_queue_is_bounded_and_counts_what_it_dropped(gateway):
    from core.executive.authority_gateway import UNRECONCILED_QUEUE_LIMIT

    for index in range(UNRECONCILED_QUEUE_LIMIT + 25):
        gateway.record_unreconciled_authority(
            capability_token_id=f"cap-{index}", reason="probe"
        )
    queue = gateway.unreconciled_authority()
    assert queue["open"] == UNRECONCILED_QUEUE_LIMIT
    assert queue["total_since_boot"] == UNRECONCILED_QUEUE_LIMIT + 25


def test_reading_the_queue_does_not_hand_out_mutable_state(gateway):
    gateway.record_unreconciled_authority(capability_token_id="cap-x", reason="probe")
    first = gateway.unreconciled_authority()
    first["entries"][0]["capability_token_id"] = "TAMPERED"
    assert gateway.unreconciled_authority()["entries"][0]["capability_token_id"] == "cap-x"
