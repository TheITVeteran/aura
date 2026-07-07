from types import SimpleNamespace

import pytest

from core.brain.predictive_engine import Prediction, PredictiveEngine


class _NullRouter:
    async def think(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_predictive_evaluate_handles_empty_router_text_without_killing_mind_tick():
    engine = PredictiveEngine()
    engine.router = _NullRouter()
    state = SimpleNamespace()
    prediction = Prediction(
        content="the user will ask a follow-up",
        confidence=0.8,
        generated_at=0.0,
        context_hash="state",
    )

    error = await engine.evaluate(prediction, "the user asked a different follow-up", state)

    assert error.error_magnitude == 0.5
    assert error.surprise_signal == pytest.approx(0.4)
