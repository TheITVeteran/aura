"""Regression: a not-ready (warming/recovering) local cortex must be SKIPPED
for foreground routing, not submitted-to and blocked on the wall deadline.

Lived 2026-07-15 soak: one turn exceeded the 105s wall → force-abort
recycled the cortex worker → every later turn hit the warming cortex,
blocked 105s, re-recycled it → a busy Aura could never re-warm its 32B.
The old error-prefix allowlist missed 'foreground_warmup_timeout' /
'warmup_deferred', so the recycled cortex was never routed around.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm_health_router import (
    _is_transient_local_runtime_failure,
    _local_client_failure_reason,
)

pytestmark = pytest.mark.unit


def _client(state, ready, error=""):
    return SimpleNamespace(
        get_lane_status=lambda: {
            "state": state,
            "conversation_ready": ready,
            "last_failure_reason": error,
        }
    )


def test_warming_cortex_is_skipped():
    reason = _local_client_failure_reason(_client("warming", False, "foreground_warmup_timeout"))
    assert reason  # non-empty => the router routes around it


def test_recovering_cortex_is_skipped_even_without_named_error():
    reason = _local_client_failure_reason(_client("recovering", False, ""))
    assert reason == "lane_not_ready:recovering"


def test_ready_cortex_is_not_skipped():
    assert _local_client_failure_reason(_client("ready", True, "")) == ""


def test_failed_lane_is_skipped():
    assert _local_client_failure_reason(_client("failed", False, "boom")) == "boom"


def test_skip_reasons_are_transient_trips_not_permanent():
    # A skipped warming cortex must trip the circuit TEMPORARILY (retried
    # after recovery_timeout), never poison the endpoint permanently.
    assert _is_transient_local_runtime_failure("lane_not_ready:warming")
    assert _is_transient_local_runtime_failure("foreground_warmup_timeout")
    assert _is_transient_local_runtime_failure("warmup_deferred")
