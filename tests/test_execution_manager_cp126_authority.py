"""CP126 authority and transaction tests for the ExecutionManager.

The gate that matters: a caller must not be able to authorize its own
dangerous action, and a retry must not silently duplicate a side effect.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.execution import ExecutionManager, is_dangerous_action


class _Trace:
    def __init__(self):
        self.rows = []

    def log(self, event):
        self.rows.append(event)

    def kinds(self):
        return [row.get("type") for row in self.rows]


@pytest.fixture()
def trace() -> _Trace:
    return _Trace()


@pytest.fixture()
def manager(trace) -> ExecutionManager:
    return ExecutionManager(trace)


def _run(manager, **kwargs):
    kwargs.setdefault("action_fn", lambda: "done")
    return asyncio.run(manager.execute(**kwargs))


def _token(action: str) -> str:
    from core.agency.capability_token import get_token_store

    token = get_token_store().issue(
        origin="test", scope="unit", ttl_seconds=30.0,
        domain="tool_execution", requested_action=action,
        approver="cp126-test", parent_receipt="test-receipt",
    )
    return getattr(token, "token", None) or getattr(token, "token_str", "")


# --- c0dd3c01: the gate covers more than a caller-supplied list ----------


@pytest.mark.parametrize(
    "name",
    [
        "delete_all_records", "deleteAllRecords", "destroy_index", "drop_table",
        "purge_cache", "rm-rf-target", "shutdown_host", "deploy_release",
        "send_email", "transfer_funds", "grant_admin", "overwrite_config",
        "exec_shell",
    ],
)
def test_unlisted_dangerous_names_are_still_dangerous(name):
    """The old gate denied ONLY names present in dangerous_whitelist."""
    assert is_dangerous_action(name) is True


@pytest.mark.parametrize(
    "name", ["read_thing", "get_status", "summarize", "list_items", "observe"]
)
def test_ordinary_names_are_not_dangerous(name):
    assert is_dangerous_action(name) is False


def test_an_unlisted_dangerous_action_is_denied(manager):
    result = _run(manager, action_name="delete_all_records")

    assert result.ok is False
    assert "safe_mode" in result.error


def test_a_declared_whitelist_entry_is_still_honoured(trace):
    manager = ExecutionManager(trace, dangerous_whitelist={"totally_benign_name"})

    assert _run(manager, action_name="totally_benign_name").ok is False


def test_safe_mode_off_permits_the_action(trace):
    manager = ExecutionManager(trace, safe_mode=False)

    assert _run(manager, action_name="delete_all_records").ok is True


# --- the documented safety_check now exists ------------------------------


def test_the_safety_check_callback_is_consulted(trace):
    seen = []

    def check(name, context):
        seen.append((name, context))
        return False

    manager = ExecutionManager(trace, safety_check=check)
    result = _run(manager, action_name="read_thing", context="why")

    assert result.ok is False
    assert "safety_check" in result.error
    assert seen == [("read_thing", "why")]


def test_a_permitting_safety_check_allows_a_safe_action(trace):
    manager = ExecutionManager(trace, safety_check=lambda name, ctx: True)

    assert _run(manager, action_name="read_thing").ok is True


def test_a_raising_safety_check_denies(trace):
    def boom(name, context):
        raise RuntimeError("checker down")

    manager = ExecutionManager(trace, safety_check=boom)

    assert _run(manager, action_name="read_thing").ok is False


def test_the_safety_check_can_deny_a_non_dangerous_action(trace):
    manager = ExecutionManager(trace, safety_check=lambda name, ctx: name != "read_thing")

    assert _run(manager, action_name="read_thing").ok is False
    assert _run(manager, action_name="other_thing").ok is True


# --- 575d878c: allow_danger must be attested -----------------------------


def test_a_bare_allow_danger_boolean_does_not_authorize(manager):
    result = _run(manager, action_name="delete_all_records", allow_danger=True)

    assert result.ok is False
    assert "capability token" in result.error


def test_a_forged_token_does_not_authorize(manager):
    result = _run(
        manager,
        action_name="delete_all_records",
        allow_danger=True,
        metadata={"capability_token": "i-made-this-up"},
    )

    assert result.ok is False
    assert "rejected" in result.error


def test_a_valid_token_authorizes_the_action(manager):
    result = _run(
        manager,
        action_name="delete_all_records",
        allow_danger=True,
        metadata={"capability_token": _token("delete_all_records")},
    )

    assert result.ok is True
    assert "validated" in result.metadata["danger_authorization"]


def test_a_token_for_a_different_action_is_rejected(manager):
    result = _run(
        manager,
        action_name="delete_all_records",
        allow_danger=True,
        metadata={"capability_token": _token("some_other_action")},
    )

    assert result.ok is False


def test_the_capability_token_is_redacted_from_metadata(manager):
    """`token` is a sensitive marker, so it must not survive into a trace."""
    result = _run(
        manager,
        action_name="read_thing",
        metadata={"capability_token": "super-secret-value"},
    )

    assert "super-secret-value" not in str(result.metadata)


# --- 6e82df46 / 3c626bb4: no duplicate side effects ---------------------


def test_a_non_idempotent_timeout_is_not_retried(trace):
    """The sync callable may still be running; a retry would double it."""
    calls = []

    def slow():
        calls.append(1)
        import time as _time

        _time.sleep(0.5)
        return "done"

    manager = ExecutionManager(trace)
    result = asyncio.run(
        manager.execute(
            action_name="read_thing", action_fn=slow,
            timeout_seconds=0.05, retries=3,
        )
    )

    assert result.ok is False
    assert result.error == "timeout_outcome_uncertain"
    assert result.metadata["outcome"] == "uncertain_timeout"
    assert result.metadata["retry_suppressed"]
    assert len(calls) == 1
    assert "execution_retry_suppressed" in trace.kinds()


def test_an_idempotent_action_may_still_retry(trace):
    calls = []

    def slow():
        calls.append(1)
        import time as _time

        _time.sleep(0.3)
        return "done"

    manager = ExecutionManager(trace)
    asyncio.run(
        manager.execute(
            action_name="read_thing", action_fn=slow,
            timeout_seconds=0.05, retries=2, retry_delay=0.0,
            metadata={"idempotent": True},
        )
    )

    assert len(calls) == 2


def test_a_succeeded_effect_is_deduplicated(manager):
    calls = []

    def once():
        calls.append(1)
        return "done"

    first = _run(manager, action_name="read_thing", action_fn=once,
                 metadata={"idempotency_key": "k1"})
    second = _run(manager, action_name="read_thing", action_fn=once,
                  metadata={"idempotency_key": "k1"})

    assert first.ok and second.ok
    assert len(calls) == 1
    assert second.metadata["deduplicated"] is True


def test_different_keys_are_not_deduplicated(manager):
    calls = []

    def counted():
        calls.append(1)
        return "done"

    _run(manager, action_name="read_thing", action_fn=counted,
         metadata={"idempotency_key": "a"})
    _run(manager, action_name="read_thing", action_fn=counted,
         metadata={"idempotency_key": "b"})

    assert len(calls) == 2


def test_an_ordinary_error_still_retries(trace):
    calls = []

    def flaky():
        calls.append(1)
        raise ValueError("transient")

    manager = ExecutionManager(trace)
    result = asyncio.run(
        manager.execute(
            action_name="read_thing", action_fn=flaky, retries=3, retry_delay=0.0
        )
    )

    assert result.ok is False
    assert len(calls) == 3


# --- aae07b1a: a successful execution produces a receipt -----------------


def test_a_success_carries_a_state_mutation_receipt(manager):
    result = _run(
        manager,
        action_name="read_thing",
        context="because the operator asked",
        metadata={"principal": "operator", "target": "index-a"},
    )

    receipt = result.metadata["receipt"]
    assert receipt["action"] == "read_thing"
    assert receipt["principal"] == "operator"
    assert receipt["target"] == "index-a"
    assert receipt["context"].startswith("because the operator")
    assert len(receipt["result_sha256"]) == 64
    assert len(receipt["receipt_id"]) == 32
    assert receipt["at"] > 0


def test_the_receipt_binds_the_result(manager):
    one = _run(manager, action_name="read_thing", action_fn=lambda: "alpha")
    two = _run(manager, action_name="read_thing", action_fn=lambda: "beta")

    assert one.metadata["receipt"]["result_sha256"] != two.metadata["receipt"]["result_sha256"]


def test_the_receipt_records_the_authority_used(manager):
    plain = _run(manager, action_name="read_thing")
    authorized = _run(
        manager,
        action_name="delete_all_records",
        allow_danger=True,
        metadata={"capability_token": _token("delete_all_records")},
    )

    assert plain.metadata["receipt"]["authority"] == "not_required"
    assert "validated" in authorized.metadata["receipt"]["authority"]


def test_an_unattributed_principal_is_labelled(manager):
    result = _run(manager, action_name="read_thing")

    assert result.metadata["receipt"]["principal"] == "unattributed"


def test_the_receipt_id_reaches_the_trace(manager, trace):
    _run(manager, action_name="read_thing")

    execution_rows = [row for row in trace.rows if row.get("type") == "execution"]
    assert execution_rows and execution_rows[0]["receipt_id"]


def test_a_failed_execution_has_no_success_receipt(manager):
    result = _run(manager, action_name="read_thing", action_fn=lambda: {"ok": False, "error": "no"})

    assert result.ok is False
    assert "receipt" not in result.metadata
