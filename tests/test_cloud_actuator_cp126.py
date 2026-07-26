"""CP126 contract tests for the cloud actuator wrapper.

A thin wrapper is where risk classification gets lost. These pin that it
cannot misdescribe what it is about to do.
"""
from __future__ import annotations

import asyncio

import pytest

from core.actuation import cloud_actuator as module
from core.actuation.cloud_actuator import (
    KNOWN_INFRA_STATES,
    CloudActuator,
    classify_sql,
)


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else {"ok": True}

    async def actuate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.fixture()
def actuator(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(module, "get_world_actuator", lambda: recorder)
    return recorder


# --- fa7129ae: destructive SQL is not an ordinary query ------------------


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM users",
        "UPDATE users SET admin = 1",
        "DROP TABLE users",
        "ALTER TABLE users ADD COLUMN x INT",
        "TRUNCATE users",
        "GRANT ALL ON db TO attacker",
        "REVOKE SELECT ON db FROM analyst",
        "INSERT INTO audit VALUES (1)",
        "CREATE INDEX idx ON users (id)",
    ],
)
def test_mutating_sql_is_classified_as_mutating(query):
    assert classify_sql(query)["mutating"] is True


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM users",
        "select id, name from users where active = true",
        "SELECT count(*) FROM orders",
        "WITH t AS (SELECT 1) SELECT * FROM t",
    ],
)
def test_read_sql_is_not_classified_as_mutating(query):
    assert classify_sql(query)["mutating"] is False


def test_a_mutation_hidden_behind_a_select_is_caught():
    """A prefix check on the first verb would call this a read."""
    verdict = classify_sql("SELECT 1; DROP TABLE users")

    assert verdict["mutating"] is True
    assert "multiple statements" in verdict["reasons"]


def test_a_comment_smuggled_statement_is_caught():
    verdict = classify_sql("SELECT * FROM t -- ; DROP TABLE x")

    assert verdict["mutating"] is True
    assert "contains a comment" in verdict["reasons"]


def test_an_oversized_query_is_flagged():
    verdict = classify_sql("SELECT 1 " + "x" * 30_000)

    assert any("exceeds" in reason for reason in verdict["reasons"])


def test_a_read_routes_as_query_database(actuator):
    asyncio.run(CloudActuator.query_db("analytics", "SELECT 1"))

    call = actuator.calls[0]
    assert call["action_name"] == "query_database"
    assert call["high_risk_flag"] is None


def test_a_write_routes_as_a_distinct_high_risk_action(actuator):
    asyncio.run(
        CloudActuator.query_db("analytics", "DELETE FROM users", read_only=False)
    )

    call = actuator.calls[0]
    assert call["action_name"] == "mutate_database"
    assert call["high_risk_flag"] is True
    assert call["params"]["classification"]["mutating"] is True


def test_the_classification_reaches_the_domain(actuator):
    asyncio.run(CloudActuator.query_db("db", "GRANT ALL ON x TO y", read_only=False))

    classification = actuator.calls[0]["params"]["classification"]
    assert "grant" in classification["keywords"]


# --- 91be5450: the database identity and lane are constrained ------------


def test_a_write_through_the_read_lane_is_refused(actuator):
    result = asyncio.run(CloudActuator.query_db("db", "DELETE FROM users"))

    assert result["ok"] is False
    assert result["error"] == "mutating_query_requires_read_only_false"
    assert actuator.calls == []


@pytest.mark.parametrize(
    "db_name",
    ["", "   ", "db;DROP", "db name", "../../etc/passwd", "db\nname", "x" * 200],
)
def test_a_hostile_database_identifier_is_refused(actuator, db_name):
    result = asyncio.run(CloudActuator.query_db(db_name, "SELECT 1"))

    assert result["ok"] is False
    assert result["error"] == "invalid_db_identifier"
    assert actuator.calls == []


@pytest.mark.parametrize("db_name", ["analytics", "prod_db", "tenant-1", "a.b.c"])
def test_a_plain_identifier_is_accepted(actuator, db_name):
    assert asyncio.run(CloudActuator.query_db(db_name, "SELECT 1"))["ok"] is True


def test_an_empty_query_is_refused(actuator):
    result = asyncio.run(CloudActuator.query_db("db", "   "))

    assert result["ok"] is False
    assert result["error"] == "empty_query"


def test_the_read_only_lane_is_forwarded(actuator):
    asyncio.run(CloudActuator.query_db("db", "SELECT 1"))

    assert actuator.calls[0]["params"]["read_only"] is True


# --- 8aa639aa: infrastructure state is typed -----------------------------


@pytest.mark.parametrize("state", sorted(KNOWN_INFRA_STATES))
def test_known_states_are_accepted(actuator, state):
    result = asyncio.run(CloudActuator.modify_infra("api", state))

    assert result.get("ok") is not False


@pytest.mark.parametrize(
    "state", ["deleted", "obliterated", "whatever the model said", "", "RUNNING; DROP"]
)
def test_an_unknown_desired_state_is_refused(actuator, state):
    result = asyncio.run(CloudActuator.modify_infra("api", state))

    assert result["ok"] is False
    assert result["error"] == "unknown_desired_state"
    assert actuator.calls == []


@pytest.mark.parametrize("service", ["", "svc name", "svc;rm", "x" * 200])
def test_a_hostile_service_identifier_is_refused(actuator, service):
    result = asyncio.run(CloudActuator.modify_infra(service, "running"))

    assert result["ok"] is False
    assert result["error"] == "invalid_service_identifier"


def test_an_unknown_rollback_state_is_refused(actuator):
    result = asyncio.run(
        CloudActuator.modify_infra("api", "stopped", rollback_state="teleported")
    )

    assert result["ok"] is False
    assert result["error"] == "unknown_rollback_state"


# --- 8ceb4427: the change carries a plan and a compensating action -------


def test_the_change_submits_a_plan(actuator):
    asyncio.run(
        CloudActuator.modify_infra("api", "stopped", current_state="running")
    )

    params = actuator.calls[0]["params"]
    assert params["precondition_state"] == "running"
    assert params["rollback_state"] == "running"
    assert params["plan"] == {
        "from": "running", "to": "stopped", "compensating_action": "restore:running",
    }
    assert params["operation_id"]
    assert params["idempotency_key"]


def test_an_explicit_rollback_target_overrides_the_observed_state(actuator):
    asyncio.run(
        CloudActuator.modify_infra(
            "api", "stopped", current_state="running", rollback_state="drained"
        )
    )

    assert actuator.calls[0]["params"]["rollback_state"] == "drained"


def test_a_no_op_change_is_detected_before_acting(actuator):
    result = asyncio.run(
        CloudActuator.modify_infra("api", "running", current_state="running")
    )

    assert result["ok"] is True
    assert result["changed"] is False
    assert result["reason"] == "already_in_desired_state"
    assert actuator.calls == []


def test_an_unknown_precondition_is_declared_not_assumed(actuator):
    asyncio.run(CloudActuator.modify_infra("api", "stopped"))

    params = actuator.calls[0]["params"]
    assert params["precondition_state"] == "unknown"
    assert params["plan"]["compensating_action"] == "none_available"


def test_the_idempotency_key_distinguishes_transitions(actuator):
    asyncio.run(CloudActuator.modify_infra("api", "stopped", current_state="running"))
    asyncio.run(CloudActuator.modify_infra("api", "stopped", current_state="drained"))

    keys = [call["params"]["idempotency_key"] for call in actuator.calls]
    assert keys[0] != keys[1]


def test_infra_changes_stay_high_risk(actuator):
    asyncio.run(CloudActuator.modify_infra("api", "stopped", current_state="running"))

    assert actuator.calls[0]["high_risk_flag"] is True
