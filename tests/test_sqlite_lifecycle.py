from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from core.memory import db_config
from core.runtime.shutdown_coordinator import (
    clear_shutdown_request,
    request_shutdown,
)


@pytest.fixture(autouse=True)
def _clean_sqlite_lifecycle() -> Iterator[None]:
    clear_shutdown_request()
    db_config.close_all_connections()
    yield
    clear_shutdown_request()
    db_config.close_all_connections()


def test_cached_connections_are_globally_closed_and_recreated(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = db_config.configure_connection(str(path))
    assert db_config.configure_connection(str(path)) is first
    first.execute("CREATE TABLE durable (value TEXT)")
    first.execute("INSERT INTO durable VALUES ('saved')")

    report = db_config.close_all_connections()

    assert report == {"clean": True, "registered": 1, "closed": 1, "failures": []}
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        first.execute("SELECT 1")

    second = db_config.configure_connection(str(path))
    assert second is not first
    assert second.execute("SELECT value FROM durable").fetchone()[0] == "saved"


def test_shutdown_latch_allows_existing_connection_but_refuses_new_one(
    tmp_path: Path,
) -> None:
    existing_path = tmp_path / "existing.db"
    existing = db_config.configure_connection(str(existing_path))
    request_shutdown("unit-test")

    assert db_config.configure_connection(str(existing_path)) is existing
    with pytest.raises(RuntimeError, match="refused new SQLite connection during shutdown"):
        db_config.configure_connection(str(tmp_path / "new.db"))


def test_path_scoped_close_does_not_release_another_store(tmp_path: Path) -> None:
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    first = db_config.configure_connection(str(first_path))
    second = db_config.configure_connection(str(second_path))

    report = db_config.close_connections_for_path(first_path)

    assert report["clean"] is True
    assert report["registered"] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        first.execute("SELECT 1")
    assert second.execute("SELECT 1").fetchone() == (1,)


def test_db_config_registers_one_state_vault_shutdown_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.runtime import shutdown_coordinator

    coordinator = shutdown_coordinator.ShutdownCoordinator()
    monkeypatch.setattr(shutdown_coordinator, "get_shutdown_coordinator", lambda: coordinator)
    monkeypatch.setattr(db_config, "_registered_shutdown_coordinator", None)
    connection = db_config.configure_connection(str(tmp_path / "registered.db"))

    assert coordinator.handler_names("state_vault").count("db_config_connections") == 1
    assert db_config.configure_connection(str(tmp_path / "registered.db")) is connection
    assert coordinator.handler_names("state_vault").count("db_config_connections") == 1


def test_goal_engine_close_commits_and_releases_connection(tmp_path: Path) -> None:
    from core.goals.goal_engine import GoalEngine

    engine = GoalEngine(db_path=str(tmp_path / "goals.db"))
    connection = engine._conn
    assert connection is not None

    engine.close()
    engine.close()

    assert engine._conn is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_governance_vault_close_is_idempotent(tmp_path: Path) -> None:
    from core.security.governance_vault import GovernanceVault

    vault = GovernanceVault(tmp_path / "vault.db")
    connection = vault._conn
    assert connection is not None

    vault.close()
    vault.close()

    assert vault._conn is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_memory_services_expose_path_scoped_container_close(tmp_path: Path) -> None:
    from core.memory.episodic_memory import EpisodicMemory
    from core.memory.knowledge_graph import PersistentKnowledgeGraph

    episodic = EpisodicMemory(db_path=str(tmp_path / "episodic.db"))
    graph = PersistentKnowledgeGraph(db_path=str(tmp_path / "knowledge.db"))
    episodic_connection = episodic._get_conn()
    graph_connection = graph._get_conn()

    episodic.close()
    assert graph_connection.execute("SELECT 1").fetchone()[0] == 1
    graph.close()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        episodic_connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        graph_connection.execute("SELECT 1")


def test_mission_state_close_is_idempotent(tmp_path: Path) -> None:
    from core.planning.mission_state import MissionState

    state = MissionState(data_dir=str(tmp_path / "missions"))
    state._init_db()
    connection = state._conn
    assert connection is not None

    state.close()
    state.close()

    assert state._conn is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
