import pytest
################################################################################

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.container import ServiceContainer
from core.service_registration import register_all_services



# This file was fourteen lines of imports and nothing else. It named the dead
# letter queue — where work goes when it fails — and collected zero tests, so
# "the DLQ is tested" was true only of the filename.

import pytest

from core.tasks.dead_letter_queue import DeadLetterQueue


@pytest.fixture
def dlq(tmp_path):
    """A DLQ on its own file. Never the live one."""
    return DeadLetterQueue(db_path=str(tmp_path / "dlq.sqlite3"))


def test_a_failure_pushed_is_a_failure_retrievable(dlq):
    dlq.push("task_alpha", {"payload": 1}, "boom")
    failed = dlq.get_failed()
    assert any("task_alpha" in str(row) for row in failed)


def test_an_empty_queue_reports_nothing_failed(dlq):
    assert list(dlq.get_failed()) == []


def test_stats_are_reportable_on_an_empty_queue(dlq):
    """A stats call that raises when nothing has failed makes the health
    surface fail exactly when it is most needed."""
    assert dlq.stats() is not None


def test_resolving_removes_the_entry_from_failed(dlq):
    dlq.push("task_beta", {"payload": 2}, "boom")
    failed = list(dlq.get_failed())
    assert failed

    entry = failed[0]
    entry_id = entry.get("id") if isinstance(entry, dict) else entry[0]
    dlq.resolve(entry_id)

    remaining = [str(row) for row in dlq.get_failed()]
    assert not any("task_beta" in row for row in remaining)


def test_two_failures_are_recorded_separately(dlq):
    dlq.push("task_one", {}, "first")
    dlq.push("task_two", {}, "second")
    rows = [str(row) for row in dlq.get_failed()]
    assert any("task_one" in r for r in rows)
    assert any("task_two" in r for r in rows)


def test_the_queue_is_durable_across_instances(tmp_path):
    """A DLQ that forgets on restart loses exactly the failures worth keeping."""
    path = str(tmp_path / "durable.sqlite3")
    DeadLetterQueue(db_path=path).push("task_persist", {}, "boom")
    assert any("task_persist" in str(row) for row in DeadLetterQueue(db_path=path).get_failed())
