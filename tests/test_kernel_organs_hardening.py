from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_organ_load_records_structured_fallback(monkeypatch):
    import core.kernel.organs as organs

    records = []
    organ = organs.OrganStub("voice", SimpleNamespace())

    async def failing_resolver():
        message = "voice engine unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(organ, "_resolve", failing_resolver)
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.load()

    assert organ.ready.is_set()
    assert organ.fallback_used is True
    assert organ.resolved_kind == "FallbackVoice"
    assert "voice engine unavailable" in organ.failure_reason
    assert records
    assert records[0][0][0] == "organs"
    assert records[0][1]["extra"]["organ"] == "voice"
    assert "bounded fallback" in records[0][1]["action"]
    assert organ.status()["fallback_used"] is True


@pytest.mark.asyncio
async def test_organ_load_marks_fallback_returned_by_resolver(monkeypatch):
    import core.kernel.organs as organs

    records = []
    organ = organs.OrganStub("neural", SimpleNamespace())

    async def fallback_resolver():
        return organs.FallbackNeural()

    monkeypatch.setattr(organ, "_resolve", fallback_resolver)
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.load()

    assert organ.fallback_used is True
    assert organ.resolved_kind == "FallbackNeural"
    assert "resolved to FallbackNeural" in organ.failure_reason
    assert records == []


@pytest.mark.asyncio
async def test_organ_shutdown_records_hook_failure(monkeypatch):
    import core.kernel.organs as organs

    records = []

    class ServiceWithBrokenStop:
        def stop(self):
            message = "cleanup failed"
            raise RuntimeError(message)

    organ = organs.OrganStub("voice", SimpleNamespace())
    organ.instance = ServiceWithBrokenStop()
    organ.resolved_kind = "ServiceWithBrokenStop"
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.shutdown()

    assert records
    assert "shutdown hook failed" in records[0][1]["action"]
    assert records[0][1]["extra"]["organ"] == "voice"
