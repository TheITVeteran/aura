import asyncio
from types import SimpleNamespace

from core.phases.bonding_phase import BondingPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


def test_bonding_phase_updates_user_turn_and_exports_modifier():
    async def scenario():
        get_degradation_tracker().reset()
        state = AuraState()
        state.cognition.current_origin = "user"
        state.cognition.modifiers["user_subtext"] = "warm trust and shared vulnerability"
        before = state.identity.bonding_level

        result = await BondingPhase().execute(state, objective=" ".join(["thoughtful"] * 60))

        assert result is state
        assert state.identity.bonding_level > before
        assert state.cognition.modifiers["bonding_phase"]["increment"] > 0
        assert state.identity.personality_growth["openness"] >= 0.0

    asyncio.run(scenario())


def test_bonding_phase_repairs_missing_growth_keys_for_deep_bonding():
    async def scenario():
        state = AuraState()
        state.cognition.current_origin = "voice"
        state.identity.bonding_level = 0.72
        state.identity.personality_growth = {}

        await BondingPhase().execute(state, objective="I trust this conversation")

        assert set(state.identity.personality_growth) >= {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
        assert state.identity.personality_growth["extraversion"] > 0
        assert state.identity.personality_growth["neuroticism"] < 0

    asyncio.run(scenario())


def test_bonding_phase_does_not_consume_person_specific_tom_state():
    class NoPersonModelReads:
        def get(self, *_args, **_kwargs):
            raise AssertionError("bonding phase must not read a person model")

    async def scenario():
        get_degradation_tracker().reset()
        state = AuraState()
        state.cognition.current_origin = "admin"
        before = state.identity.bonding_level

        await BondingPhase(NoPersonModelReads()).execute(
            state,
            objective="still a real user turn",
        )

        assert state.identity.bonding_level > before
        assert state.cognition.modifiers["bonding_phase"][
            "person_specific_relationship_claimed"
        ] is False
        assert get_degradation_tracker().count("bonding_phase") == 0

    asyncio.run(scenario())


def test_bonding_phase_ignores_background_origin():
    async def scenario():
        state = AuraState()
        state.cognition.current_origin = "system"
        before = state.identity.bonding_level

        await BondingPhase().execute(state, objective="background reflection")

        assert state.identity.bonding_level == before
        assert "bonding_phase" not in state.cognition.modifiers

    asyncio.run(scenario())


def test_bonding_phase_returns_state_on_malformed_surface():
    async def scenario():
        get_degradation_tracker().reset()
        state = SimpleNamespace(cognition=SimpleNamespace(current_origin="user", modifiers={}))

        result = await BondingPhase().execute(state, objective="hello")

        assert result is state
        assert get_degradation_tracker().count("bonding_phase") == 1

    asyncio.run(scenario())
