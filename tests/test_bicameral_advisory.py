from types import SimpleNamespace

import pytest

from core.brain.bicameral_advisory import (
    BicameralAdvisory,
    get_bicameral_advisory,
    render_bicameral_prompt_block,
)
from core.brain.cognitive_engine import CognitiveEngine
from core.brain.inference_gate import InferenceGate
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.types import ThinkingMode
from core.container import ServiceContainer
from core.state.aura_state import AuraState
from core.phases.response_generation import ResponseGenerationPhase


def teardown_function(_function=None):
    ServiceContainer.clear()


def test_bicameral_advisory_routes_desktop_effects_without_executing_them():
    advisor = BicameralAdvisory()
    frame = advisor.advise(
        "Open Notes, write a timestamped reflection, export it as a PDF, then verify the file exists.",
        state=AuraState.default(),
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert frame.salience >= 0.3
    assert frame.routing_bias["use_tool_gateway"] is True
    assert frame.routing_bias["seek_verification"] is True
    assert frame.routing_bias["compact_foreground"] is True
    assert frame.governance["advisory_only"] is True
    assert frame.governance["no_external_effects"] is True
    assert frame.governance["will_authority_required_for_effects"] is True
    assert "verification" in frame.attention_targets
    assert frame.causal_effects["verification_pressure"] >= 0.45


def test_bicameral_advisory_raises_metacognition_for_introspection_and_uncertainty():
    state = AuraState.default()
    state.affect.curiosity = 0.82
    advisor = BicameralAdvisory()

    frame = advisor.advise(
        "I am confused. Reflect on what Aura means by self-awareness, then imagine a novel way to test it.",
        state=state,
        origin="desktop",
    )

    assert frame.routing_bias["raise_metacognition"] is True
    assert frame.routing_bias["use_imagination"] is True
    assert frame.causal_effects["metacognition_depth"] >= 0.64
    assert frame.causal_effects["creative_pressure"] >= 0.6
    assert frame.causal_effects["self_model_update"] >= 0.45
    assert any(proposal.perspective == "critic" for proposal in frame.proposals)
    assert any(proposal.perspective == "explorer" for proposal in frame.proposals)


def test_bicameral_advisory_reads_experiential_emotions_as_causal_state():
    state = AuraState.default()
    state.affect.emotions["confused"] = 0.72
    state.affect.emotions["frustration"] = 0.46

    frame = BicameralAdvisory().advise(
        "Summarize the current plan.",
        state=state,
        origin="desktop",
    )

    assert frame.routing_bias["raise_metacognition"] is True
    assert frame.causal_effects["metacognition_depth"] >= 0.64
    assert any(proposal.perspective == "critic" for proposal in frame.proposals)


def test_bicameral_advisory_does_not_treat_generic_you_action_as_identity_reflection():
    frame = BicameralAdvisory().advise(
        "Can you open Notes and type the summary?",
        state=AuraState.default(),
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert frame.routing_bias["use_tool_gateway"] is True
    assert frame.causal_effects["self_model_update"] < 0.35


def test_bicameral_capability_reflection_drives_self_model_and_memory_grounding():
    engine = CognitiveEngine()
    state = AuraState.default()

    context = engine._apply_bicameral_advisory(
        state,
        "What tools can you use, and how do you know you can use them?",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    frame = state.response_modifiers["bicameral_advisory"]
    assert frame["causal_effects"]["self_model_update"] >= 0.35
    assert state.response_modifiers["self_model_update_pressure"] >= 0.35
    assert state.response_modifiers["requires_memory_grounding"] is True
    assert state.cognition.modifiers["self_model_update_pressure"] >= 0.35
    assert state.cognition.modifiers["requires_memory_grounding"] is True
    assert context["bicameral_advisory"]["frame_id"] == frame["frame_id"]


def test_cognitive_engine_records_bicameral_advisory_as_state_context_and_attention():
    engine = CognitiveEngine()
    state = AuraState.default()

    context = engine._apply_bicameral_advisory(
        state,
        "Could you use tools to verify this, then explain what you learned?",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    frame = state.response_modifiers["bicameral_advisory"]
    assert frame["governance"]["advisory_only"] is True
    assert state.response_modifiers["tool_governance_pressure"] is True
    assert state.response_modifiers["verification_pressure"] >= 0.45
    assert state.response_modifiers["metacognition_depth"] >= 0.35
    assert state.response_modifiers["bicameral_sampling_bias"] == frame["sampling_bias"]
    assert state.cognition.modifiers["bicameral_advisory"]["frame_id"] == frame["frame_id"]
    assert state.cognition.modifiers["bicameral_causal_effects"] == frame["causal_effects"]
    assert "advisory focus" in state.cognition.attention_focus
    assert context["bicameral_advisory"]["frame_id"] == frame["frame_id"]
    assert ServiceContainer.get("bicameral_advisory", default=None) is get_bicameral_advisory()


def test_context_assembler_injects_bicameral_advisory_prompt_block():
    engine = CognitiveEngine()
    state = AuraState.default()
    state.cognition.current_objective = "Reflect on uncertainty before using any external tool."
    engine._apply_bicameral_advisory(
        state,
        state.cognition.current_objective,
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    prompt = ContextAssembler.build_system_prompt(state)

    assert "BICAMERAL ADVISORY" in prompt
    assert "not a claim of voices" in prompt
    assert "governed tools" in prompt or "Verification is elevated" in prompt


@pytest.mark.asyncio
async def test_desktop_quick_path_consumes_bicameral_advisory():
    captured = {}

    class Router:
        async def think(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "I can explain the governed tool path and verify each external effect."

    ServiceContainer.register_instance("llm_router", Router(), required=False)
    frame = BicameralAdvisory().advise(
        "What tools can you use externally, and how would you verify them?",
        state=AuraState.default(),
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    ).to_dict()
    engine = CognitiveEngine()

    thought = await engine._direct_desktop_quick_reply(
        "What tools can you use externally, and how would you verify them?",
        ThinkingMode.FAST,
        "desktop",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "bicameral_advisory": frame,
            "max_tokens": 512,
        },
        timeout_s=20.0,
    )

    assert thought is not None
    assert thought.metadata["bicameral_advisory"]["frame_id"] == frame["frame_id"]
    assert thought.metadata["bicameral_advisory_feedback"]["outcome"] == "desktop_quick_reply"
    assert "Bicameral advisory" in captured["messages"][0]["content"]
    assert "phenomenal experience" in captured["messages"][0]["content"]
    assert captured["kwargs"]["protected_foreground_lane"] is True
    assert captured["kwargs"]["allow_cloud_fallback"] is False
    assert captured["kwargs"]["bicameral_sampling_bias"] == frame["sampling_bias"]


def test_bicameral_sampling_bias_reaches_inference_and_generation_with_bounds():
    state = AuraState.default()
    state.response_modifiers["bicameral_sampling_bias"] = {
        "temperature_delta": -0.08,
        "max_tokens_factor": 0.82,
    }

    temperature, tokens, applied = InferenceGate._apply_runtime_sampling_biases(
        base_temperature=0.70,
        max_tokens=1000,
        context={},
        state=state,
        allow_token_scaling=True,
    )
    gen_temperature, gen_tokens = ResponseGenerationPhase._apply_generation_sampling_bias(
        base_temperature=0.70,
        token_budget=1000,
        biases=[state.response_modifiers["bicameral_sampling_bias"]],
    )

    assert temperature == pytest.approx(0.62)
    assert tokens == 820
    assert applied["max_tokens_factor"] == pytest.approx(0.82)
    assert gen_temperature == pytest.approx(0.62)
    assert gen_tokens == 820


def test_bicameral_feedback_updates_perspective_reliability():
    advisor = BicameralAdvisory()
    frame = advisor.advise(
        "I am unsure; reflect and verify before acting.",
        state=SimpleNamespace(affect=SimpleNamespace(curiosity=0.3, arousal=0.2, valence=0.0)),
        origin="desktop",
    )
    before = advisor.get_status()["reliability"]

    result = advisor.learn_from_feedback(frame.to_dict(), reward=1.0, outcome="assistant_response")
    after = advisor.get_status()["reliability"]

    assert result["learned"] is True
    assert result["outcome"] == "assistant_response"
    assert any(after[key] > before[key] for key in result["reliability"])
    assert render_bicameral_prompt_block(frame)
