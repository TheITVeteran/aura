import os
from pathlib import Path
from types import SimpleNamespace

import pytest


class AsyncCallFixture:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value

    def assert_not_awaited(self):
        assert self.calls == []


@pytest.mark.asyncio
async def test_conversational_momentum_defers_during_proof_run(monkeypatch):
    from core.conversational_momentum_engine import (
        ConversationThread,
        ConversationalMomentumEngine,
    )

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        status=SimpleNamespace(is_processing=False),
        process_user_input=AsyncCallFixture(),
    )
    engine = ConversationalMomentumEngine(orchestrator)
    engine.running = True

    await engine._trigger_spontaneous_turn(
        ConversationThread(topic="sealed proof task", last_turn="sealed proof task", momentum=0.2)
    )

    orchestrator.process_user_input.assert_not_awaited()


def test_proof_boot_policy_disables_unsolicited_background_autonomy(monkeypatch):
    from aura_main import _activate_proof_runtime_policy

    managed_env = (
        "AURA_PROOF_RUN",
        "AURA_PROOF_MODEL_TIER",
        "AURA_ENABLE_PROACTIVE_SYSTEMS",
        "AURA_ENABLE_RESEARCH_CYCLE",
        "AURA_ENABLE_SENSORIMOTOR_GROUNDING",
        "AURA_ENABLE_PROACTIVE_VISION",
    )
    for name in managed_env:
        monkeypatch.delenv(name, raising=False)

    try:
        _activate_proof_runtime_policy("proof")

        assert os.environ["AURA_ENABLE_PROACTIVE_SYSTEMS"] == "0"
        assert os.environ["AURA_ENABLE_RESEARCH_CYCLE"] == "0"
        assert os.environ["AURA_ENABLE_SENSORIMOTOR_GROUNDING"] == "0"
        assert os.environ["AURA_ENABLE_PROACTIVE_VISION"] == "0"
    finally:
        for name in managed_env:
            os.environ.pop(name, None)


def test_proof_boot_defers_nonessential_background_loops():
    source = Path("core/orchestrator/main.py").read_text(encoding="utf-8")

    assert '_background_quiescent_runtime("pneuma_background")' in source
    assert '_background_quiescent_runtime("mhaf_background")' in source
    assert '_background_quiescent_runtime("terminal_watchdog")' in source
    assert '_proof_runtime_active("meta_evolution_cycle")' in source
