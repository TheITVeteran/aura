from __future__ import annotations

import asyncio
import json
import time


def _blank_journal(tmp_path):
    from core.introspection.insight_journal import InsightJournal

    journal = InsightJournal.__new__(InsightJournal)
    journal._insights = []
    journal._db_path = tmp_path / "insight_journal.json"
    return journal


def test_insight_journal_quarantines_corrupt_store(monkeypatch, tmp_path):
    from core import insight_journal as module

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    journal = _blank_journal(tmp_path)
    journal._db_path.write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    journal._load()

    assert journal._insights == []
    assert not journal._db_path.exists()
    assert list(tmp_path.glob("insight_journal.corrupt-*.json"))
    assert calls
    assert "quarantined corrupt store" in calls[-1][2]["action"]


def test_insight_journal_load_sanitizes_and_skips_invalid_entries(monkeypatch, tmp_path):
    from core import insight_journal as module

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    journal = _blank_journal(tmp_path)
    journal._db_path.write_text(
        json.dumps(
            [
                {
                    "id": "i1",
                    "title": "Recovered insight",
                    "content": "Evidence changes future planning.",
                    "domain": "agency",
                    "confidence": 2.0,
                    "timestamp": time.time() + 999,
                    "source": "test",
                    "tags": ["agency", 42],
                    "impact_score": -4,
                },
                {"title": "", "content": ""},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    journal._load()

    assert len(journal._insights) == 1
    insight = journal._insights[0]
    assert insight.confidence == 1.0
    assert insight.impact_score == 0.0
    assert insight.timestamp <= time.time()
    assert "42" in insight.tags
    assert any("skipped invalid insight" in call[2]["action"] for call in calls)


def test_insight_journal_broadcast_failure_does_not_block_belief_promotion(monkeypatch, tmp_path):
    from core import insight_journal as module

    calls = []
    promoted = []

    class FailingBus:
        def __init__(self):
            self.publish_calls = []

        async def publish(self, *_args, **_kwargs):
            self.publish_calls.append((_args, _kwargs))
            raise RuntimeError("event bus offline")

    class Beliefs:
        async def process_new_claim(self, **kwargs):
            promoted.append(kwargs)

    class Container:
        @staticmethod
        def get(name, default=None):
            if name == "belief_revision_engine":
                return Beliefs()
            return default

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    journal = _blank_journal(tmp_path)
    failing_bus = FailingBus()

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)
    monkeypatch.setitem(
        __import__("sys").modules,
        "core.event_bus",
        type("EventBusModule", (), {"get_event_bus": lambda: failing_bus}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "core.container",
        type("ContainerModule", (), {"ServiceContainer": Container}),
    )

    asyncio.run(
        journal.record_insight(
            title="Durable insight",
            content="Beliefs should still update when broadcast fails.",
            domain="agency",
            confidence=0.9,
            source="test",
        )
    )

    assert len(journal._insights) == 1
    assert len(failing_bus.publish_calls) == 1
    assert promoted
    assert calls
    assert "event-bus broadcast failed" in calls[0][2]["action"]
