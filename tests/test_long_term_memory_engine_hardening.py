import json

import pytest

from core import long_term_memory_engine as ltm_module


class ConstitutionScenario:
    async def approve_memory_write(self, **_kwargs):
        return True, "approved"


class ServiceScenario:
    _registration_locked = False

    @staticmethod
    def get(_name, default=None):
        return default

    @staticmethod
    def has(_name):
        return False


class EventBusScenario:
    def __init__(self):
        self.calls = 0

    async def publish(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("event bus unavailable")


@pytest.fixture(autouse=True)
def long_term_memory_environment(monkeypatch):
    monkeypatch.setattr(ltm_module, "ServiceContainer", ServiceScenario)


def test_corrupt_long_term_memory_store_is_quarantined(tmp_path, monkeypatch):
    records = []
    monkeypatch.setattr(
        ltm_module,
        "record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    engine = ltm_module.LongTermMemoryEngine()
    engine.db_path = tmp_path / "long_term_memories.json"
    engine.db_path.write_text("{not valid json", encoding="utf-8")

    loaded = engine._load_memories()

    assert loaded is False
    assert engine.memories == []
    assert not engine.db_path.exists()
    assert list(tmp_path.glob("long_term_memories.json.corrupt.*"))
    assert records[-1][0] == "long_term_memory_engine"
    assert "preserving corrupt store" in records[-1][2]["action"]


@pytest.mark.asyncio
async def test_store_uses_unique_ids_and_query_relevance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda: ConstitutionScenario(),
    )
    engine = ltm_module.LongTermMemoryEngine()
    engine.db_path = tmp_path / "long_term_memories.json"

    alpha = await engine.store("alpha calibration detail", importance=0.2, tags=["Alpha"])
    beta = await engine.store("beta deployment milestone", importance=0.95, tags=["Beta"])
    recalled = await engine.recall_relevant("alpha", limit=1)

    assert alpha is not None
    assert beta is not None
    assert alpha.id != beta.id
    assert recalled == [alpha]
    saved = json.loads(engine.db_path.read_text(encoding="utf-8"))
    assert {item["id"] for item in saved} == {alpha.id, beta.id}


@pytest.mark.asyncio
async def test_start_keeps_memory_online_when_optional_event_registration_fails(tmp_path, monkeypatch):
    records = []
    bus = EventBusScenario()
    monkeypatch.setattr(ltm_module, "get_event_bus", lambda: bus)
    monkeypatch.setattr(
        ltm_module,
        "record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    engine = ltm_module.LongTermMemoryEngine()
    engine.db_path = tmp_path / "long_term_memories.json"
    engine.consolidation_interval_s = 9999.0

    loaded = await engine.start()
    stopped = await engine.stop()

    assert loaded is True
    assert stopped is True
    assert bus.calls == 1
    assert records[-1][0] == "long_term_memory_engine"
    assert "skipped optional mycelium registration" in records[-1][2]["action"]
