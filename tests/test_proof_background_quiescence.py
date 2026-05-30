import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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
        process_user_input=AsyncMock(),
    )
    engine = ConversationalMomentumEngine(orchestrator)
    engine.running = True

    await engine._trigger_spontaneous_turn(
        ConversationThread(topic="sealed proof task", last_turn="sealed proof task", momentum=0.2)
    )

    orchestrator.process_user_input.assert_not_awaited()


def test_proof_boot_policy_disables_unsolicited_background_autonomy(monkeypatch):
    from aura_main import _activate_proof_runtime_policy

    for name in (
        "AURA_ENABLE_PROACTIVE_SYSTEMS",
        "AURA_ENABLE_RESEARCH_CYCLE",
        "AURA_ENABLE_SENSORIMOTOR_GROUNDING",
    ):
        monkeypatch.delenv(name, raising=False)

    _activate_proof_runtime_policy("proof")

    assert os.environ["AURA_ENABLE_PROACTIVE_SYSTEMS"] == "0"
    assert os.environ["AURA_ENABLE_RESEARCH_CYCLE"] == "0"
    assert os.environ["AURA_ENABLE_SENSORIMOTOR_GROUNDING"] == "0"
