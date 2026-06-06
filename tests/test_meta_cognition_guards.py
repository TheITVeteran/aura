from types import SimpleNamespace

import pytest

from core.orchestrator.meta_cognition_shard import MetaCognitionShard


class AsyncCallRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))


@pytest.mark.asyncio
async def test_meta_cognition_audit_loop_defers_when_background_policy_blocks(monkeypatch):
    orchestrator = SimpleNamespace(status=SimpleNamespace(healthy=True))
    shard = MetaCognitionShard(orchestrator)
    shard.is_running = True
    perform_audit = AsyncCallRecorder()
    shard.perform_audit = perform_audit

    monkeypatch.setattr(
        "core.orchestrator.meta_cognition_shard.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.24",
    )

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds):
        sleep_calls["count"] += 1
        shard.is_running = False

    monkeypatch.setattr("core.orchestrator.meta_cognition_shard.asyncio.sleep", _fake_sleep)

    await shard._audit_loop()

    assert perform_audit.calls == []
    assert sleep_calls["count"] == 1
