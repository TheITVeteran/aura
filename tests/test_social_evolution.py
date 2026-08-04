import asyncio
import logging
import pytest
from core.state.aura_state import AuraState
from core.phases.inference_phase import InferencePhase
from core.phases.bonding_phase import BondingPhase
from core.phases.social_context_phase import SocialContextPhase
from core.phases.repair_phase import RepairPhase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Test.SocialEvolution")

class MockRouter:
    async def route(self, prompt, **kwargs):
        import json
        return json.dumps({
            "implicit_intent": "Connecting and sharing state",
            "affective_subtext": "Warmth",
            "momentum": 0.8,
            "conversation_hooks": ["your day", "feeling about projects"]
        })



# MockRouter above was defined and never used: this file collected zero tests
# while naming the social/bonding phase stack.


@pytest.mark.asyncio
async def test_mock_router_returns_parseable_social_reading():
    """The fixture's own contract — a malformed mock tests nothing."""
    import json

    payload = json.loads(await MockRouter().route("anything"))
    assert payload["implicit_intent"]
    assert payload["affective_subtext"]
    assert 0.0 <= float(payload["momentum"]) <= 1.0
    assert isinstance(payload["conversation_hooks"], list)


@pytest.mark.parametrize(
    "phase_cls", [InferencePhase, BondingPhase, SocialContextPhase, RepairPhase]
)
def test_every_social_phase_is_constructible(phase_cls):
    assert phase_cls() is not None


@pytest.mark.parametrize(
    "phase_cls", [InferencePhase, BondingPhase, SocialContextPhase, RepairPhase]
)
def test_every_social_phase_exposes_execute(phase_cls):
    """A phase without execute cannot run, however well it is registered."""
    assert callable(getattr(phase_cls, "execute", None))


def test_default_state_carries_the_affect_fields_the_phases_read():
    state = AuraState.default()
    for attribute in ("valence", "arousal"):
        assert isinstance(getattr(state.affect, attribute), float)
