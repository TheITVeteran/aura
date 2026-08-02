"""CP126 hardening contracts for core/actuation/world_actuator.py.

The world actuator coordinates external digital effects (files, cloud,
email/PR drafts). Physical device effects belong to Reality Reach. Tests cover:
terminal refusal of unknown or physical categories, parameter-aware
risk that reaches the executor, redacted+bounded+locked audit, operation
identity, executor-fault reconciliation, non-dict result guarding, and a
coordinator deadline. ActionExecutor is faked — no real effects run.
"""
from __future__ import annotations

import asyncio

import pytest

import core.actuation.world_actuator as wa
from core.actuation.world_actuator import (
    WorldActuator,
    _redact,
    get_world_actuator,
)


@pytest.fixture
def actuator(monkeypatch):
    captured = {}

    async def _fake_execute(*, domain, action_name, params, source, execution_timeout_s=None, **kw):
        captured["domain"] = domain
        captured["action_name"] = action_name
        captured["params"] = params
        captured["timeout"] = execution_timeout_s
        return {"ok": True}

    monkeypatch.setattr(wa.ActionExecutor, "execute", _fake_execute)
    act = WorldActuator()
    act._captured = captured  # type: ignore[attr-defined]
    return act


# ── d67c7d9e: unknown category refused terminally ──────────────────────────


@pytest.mark.asyncio
async def test_unknown_category_is_refused(actuator):
    res = await actuator.actuate("not_a_real_category", "do_thing", {})
    assert res["ok"] is False and res["error"] == "unknown_category"
    assert actuator._captured == {}  # executor never reached


@pytest.mark.asyncio
async def test_robotics_category_cannot_fall_through_generic_executor(actuator):
    res = await actuator.actuate("robotics_devices", "command_device", {})
    assert res["ok"] is False
    assert res["error"] == "physical_category_requires_reality_reach"
    assert actuator._captured == {}


# ── 5773a761 + 09dcc0bc: parameter-aware risk reaches the executor ─────────


@pytest.mark.asyncio
async def test_named_high_risk_action_flagged(actuator):
    await actuator.actuate("email_drafts", "send_message", {"body": "hi"})
    assert actuator._captured["params"]["_is_high_risk"] is True


@pytest.mark.asyncio
async def test_param_shape_escalates_risk(actuator):
    # An ordinary action name, but the params describe a payment recipient.
    await actuator.actuate("cloud_resources_owned", "run", {"recipient": "x", "amount": 100})
    assert actuator._captured["params"]["_is_high_risk"] is True


@pytest.mark.asyncio
async def test_destructive_value_escalates_risk(actuator):
    await actuator.actuate("databases_owned", "run", {"q": "DROP TABLE users"})
    assert actuator._captured["params"]["_is_high_risk"] is True


@pytest.mark.asyncio
async def test_benign_action_not_flagged(actuator):
    await actuator.actuate("local_files", "read", {"path": "notes.txt"})
    assert actuator._captured["params"]["_is_high_risk"] is False


# ── 5034a58c: audit redacts secrets and stores a digest ────────────────────


@pytest.mark.asyncio
async def test_audit_redacts_secrets_and_digests(actuator):
    await actuator.actuate("code_repos", "push", {"api_key": "sk-secret", "note": "ok"})
    record = actuator.last_actuations[-1]
    assert record["params_redacted"]["api_key"] == "***redacted***"
    assert record["params_redacted"]["note"] == "ok"
    assert len(record["params_digest"]) == 64
    # The raw secret value is not stored anywhere in the record.
    assert "sk-secret" not in str(record)


def test_redact_helper_truncates_and_masks():
    out = _redact({"password": "p", "big": "z" * 5000})
    assert out["password"] == "***redacted***"
    assert len(out["big"]) < 5000


# ── 8029366: audit history is bounded ──────────────────────────────────────


def test_audit_history_is_bounded(monkeypatch):
    monkeypatch.setattr(wa, "_MAX_AUDIT_RECORDS", 10)
    act = WorldActuator()
    for i in range(25):
        act._record({"i": i})
    assert len(act.last_actuations) == 10
    assert act.last_actuations[-1]["i"] == 24  # newest kept


# ── 3eb947ed: every request carries an operation id ────────────────────────


@pytest.mark.asyncio
async def test_operation_id_is_present(actuator):
    res = await actuator.actuate("local_files", "read", {})
    assert "operation_id" in res
    assert actuator.last_actuations[-1]["operation_id"] == res["operation_id"]


# ── 4a12f982: executor faults reconcile the pending record ─────────────────


@pytest.mark.asyncio
async def test_executor_exception_marks_error_not_pending(monkeypatch):
    async def _boom(**kw):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(wa.ActionExecutor, "execute", _boom)
    act = WorldActuator()
    res = await act.actuate("local_files", "write", {})
    assert res["ok"] is False and "executor_error" in res["error"]
    assert act.last_actuations[-1]["status"] == "error"  # never left pending


# ── 402e78f0: non-dict executor result is guarded ──────────────────────────


@pytest.mark.asyncio
async def test_non_dict_result_is_handled(monkeypatch):
    async def _weird(**kw):
        return "not a dict"

    monkeypatch.setattr(wa.ActionExecutor, "execute", _weird)
    act = WorldActuator()
    res = await act.actuate("local_files", "write", {})
    assert res["ok"] is False and res["error"] == "executor_returned_non_dict"
    assert act.last_actuations[-1]["status"] == "malformed_result"


# ── 9a68d683: a coordinator deadline bounds a hung executor ─────────────────


@pytest.mark.asyncio
async def test_coordinator_deadline(monkeypatch):
    async def _hang(**kw):
        await asyncio.sleep(30)

    monkeypatch.setattr(wa, "_COORDINATOR_GRACE_S", 0.1)
    monkeypatch.setattr(wa.ActionExecutor, "execute", _hang)
    act = WorldActuator()
    res = await act.actuate("browser", "open", {}, deadline_s=0.1)
    assert res["ok"] is False and res["outcome"] == "uncertain"
    assert act.last_actuations[-1]["status"] == "uncertain_timeout"


# ── 3faf6061: typed param contract ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_dict_params_refused(actuator):
    res = await actuator.actuate("local_files", "read", "not-a-dict")  # type: ignore[arg-type]
    assert res["ok"] is False and res["error"] == "params_must_be_a_mapping"


# ── a116fb38: singleton is stable ──────────────────────────────────────────


def test_singleton_is_stable():
    assert get_world_actuator() is get_world_actuator()
