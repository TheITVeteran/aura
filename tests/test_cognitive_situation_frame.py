from __future__ import annotations

import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.cognitive_situation import (
    CognitiveSituationEngine,
    render_cognitive_situation_prompt_block,
)
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.types import ThinkingMode
from core.container import ServiceContainer
from core.phases.response_generation import ResponseGenerationPhase
from core.planning.task_decomposer import TaskDecomposer
from core.state.aura_state import AuraState, CognitiveMode


def test_cognitive_situation_frame_models_semantic_analogy_and_embodiment():
    ServiceContainer.clear()
    state = AuraState.default()
    state.affect.curiosity = 0.82
    state.affect.emotions["confused"] = 0.31

    frame = CognitiveSituationEngine().frame(
        "Can you see the screen, compare this workflow to a navigation system, "
        "and explain what the ambiguous user intent means before you open Notes?",
        state=state,
        origin="desktop",
        context={"desktop_cognitive_engine_required": True},
    )

    assert frame.semantic_flexibility > 0.45
    assert frame.analogical_leap_pressure > 0.35
    assert frame.sensorimotor_grounding > 0.45
    assert frame.routing_bias["use_tool_gateway"] is True
    assert frame.routing_bias["seek_verification"] is True
    assert frame.routing_bias["bind_sensorimotor_evidence"] is True
    assert frame.governance["external_effects_require_authority_gateway"] is True
    assert frame.semantic_interpretations
    assert frame.analogy_bridges
    assert frame.embodied_affordances
    assert "sensorimotor-evidence" in frame.attention_targets
    assert "tool-effect-verification" in frame.attention_targets


def test_cognitive_engine_applies_situation_frame_to_live_state_and_context():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    state.affect.curiosity = 0.76
    state.affect.emotions["confused"] = 0.42

    context = engine._apply_cognitive_situation_frame(
        state,
        "Interpret this visible desktop task, then open Chrome and verify the result.",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    frame = state.response_modifiers["cognitive_situation_frame"]
    assert context["cognitive_situation_frame"]["frame_id"] == frame["frame_id"]
    assert state.response_modifiers["semantic_flexibility_pressure"] == frame["semantic_flexibility"]
    assert state.response_modifiers["sensorimotor_grounding_pressure"] == frame["sensorimotor_grounding"]
    assert state.response_modifiers["tool_governance_pressure"] is True
    assert state.response_modifiers["verification_pressure"] >= frame["verification_pressure"]
    assert state.response_modifiers["metacognition_depth"] >= frame["metacognition_pressure"]
    assert state.cognition.modifiers["cognitive_situation_frame"]["frame_id"] == frame["frame_id"]
    assert state.cognition.modifiers["bind_sensorimotor_evidence"] is True
    assert "situation focus" in state.cognition.attention_focus
    assert state.cognition.current_mode is CognitiveMode.DELIBERATE


def test_context_assembler_injects_cognitive_situation_prompt_block():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    objective = "Use an analogy to explain what this visible screen workflow means."
    state.cognition.current_objective = objective
    engine._apply_cognitive_situation_frame(
        state,
        objective,
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    prompt = ContextAssembler.build_system_prompt(state)

    assert "COGNITIVE SITUATION FRAME" in prompt
    assert "Semantic" in prompt or "semantic" in prompt
    assert "sensorimotor" in prompt.lower()


def test_response_generation_consumes_cognitive_situation_sampling_bias():
    temperature, tokens = ResponseGenerationPhase._apply_generation_sampling_bias(
        base_temperature=0.7,
        token_budget=1000,
        biases=[
            {"temperature_delta": 0.02, "max_tokens_factor": 1.05},
            {"temperature_delta": -0.05, "max_tokens_factor": 0.90},
        ],
    )

    assert temperature == pytest.approx(0.67)
    assert tokens == 945


def test_response_generation_injects_cognitive_situation_runtime_block():
    frame = CognitiveSituationEngine().frame(
        "What would this visible app workflow look like as an analogy?",
        state=AuraState.default(),
        origin="desktop",
        context={"desktop_cognitive_engine_required": True},
    ).to_dict()
    messages = [{"role": "system", "content": "Base"}]

    ResponseGenerationPhase._inject_live_runtime_grounding(
        messages,
        {"cognitive_situation_frame": frame},
    )

    assert "COGNITIVE SITUATION FRAME" in messages[0]["content"]
    assert "Ground screen/tool claims" in messages[0]["content"]


@pytest.mark.asyncio
async def test_desktop_quick_path_consumes_cognitive_situation_frame():
    ServiceContainer.clear()
    captured = {}

    class Router:
        async def think(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "I would first ground the visible state, then act through the governed tool lane."

        def get_last_generation_metadata(self):
            return {}

    ServiceContainer.register_instance("llm_router", Router(), required=False)
    frame = CognitiveSituationEngine().frame(
        "Can you see the screen and compare this task to a cockpit checklist?",
        state=AuraState.default(),
        origin="desktop",
        context={"desktop_cognitive_engine_required": True},
    ).to_dict()
    engine = CognitiveEngine()

    thought = await engine._direct_desktop_quick_reply(
        "Can you see the screen and compare this task to a cockpit checklist?",
        ThinkingMode.FAST,
        "desktop",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_situation_frame": frame,
            "max_tokens": 512,
        },
        timeout_s=20.0,
    )

    assert thought is not None
    assert thought.metadata["cognitive_situation_frame"]["frame_id"] == frame["frame_id"]
    assert "COGNITIVE SITUATION FRAME" in captured["messages"][0]["content"]
    assert captured["kwargs"]["cognitive_situation_sampling_bias"] == frame["sampling_bias"]
    assert captured["kwargs"]["protected_foreground_lane"] is True


def test_render_cognitive_situation_prompt_block_handles_malformed_frame():
    assert render_cognitive_situation_prompt_block({"salience": object()}) == ""


def test_fused_perception_changes_attention_routing_response_and_planning() -> None:
    class Pump:
        def get_status(self):
            return {
                "running": True,
                "frames_produced": 42,
                "substrate_injections": 21,
                "errors": 0,
                "pump_hz": 10.0,
                "fusion": {
                    "frame_id": "fusion-test-42",
                    "confidence": 0.22,
                    "uncertainty": 0.84,
                    "observations": {"device": {"source": "unit"}},
                    "missing": {
                        "vision": "stale",
                        "audio": "permission_denied",
                    },
                    "unresolved_contradictions": 1,
                    "directives": {
                        "attention_targets": [
                            "perception-gap:vision:stale",
                            "sensor-conflict:scene.person_present",
                        ],
                        "memory_candidates": [],
                        "planning_constraints": [
                            "verify-before-action:scene.person_present",
                            "prefer-reversible-information-gathering",
                        ],
                        "repair_requirements": [
                            "refresh-sensor:vision",
                            "request-consent:audio",
                        ],
                    },
                },
            }

    ServiceContainer.register_instance("perceptual_pump", Pump(), required=False)
    state = AuraState.default()
    engine = CognitiveSituationEngine()
    frame = engine.frame(
        "Open the visible app and click the person shown on screen.",
        state=state,
        origin="desktop",
        context={"desktop_cognitive_engine_required": True},
    )

    assert frame.perception_summary["multimodal_fusion"]["frame_id"] == "fusion-test-42"
    assert frame.ambiguity > 0.35
    assert frame.verification_pressure >= 0.58
    assert frame.routing_bias["perception_abstention_required"] is True
    assert frame.routing_bias["perception_repair_required"] is True
    assert "sensor-conflict:scene.person_present" in frame.attention_targets
    assert frame.causal_effects["multimodal_confidence"] == 0.22
    assert frame.causal_effects["unresolved_sensor_conflicts"] == 1

    rendered = render_cognitive_situation_prompt_block(frame.to_dict())
    planning = TaskDecomposer._render_cognitive_situation_for_planning(
        {"cognitive_situation_frame": frame.to_dict()}
    )
    assert "abstain from unsupported scene claims" in rendered
    assert "verify-before-action:scene.person_present" in rendered
    assert "verify-before-action:scene.person_present" in planning
    assert "request-consent:audio" in planning

    cognitive_engine = CognitiveEngine()
    cognitive_engine._apply_cognitive_situation_frame(
        state,
        "Open the visible app and click the person shown on screen.",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )
    assert state.response_modifiers["perception_abstention_required"] is True
    assert "verify-before-action:scene.person_present" in state.response_modifiers[
        "perception_planning_constraints"
    ]
    assert state.cognition.modifiers["perception_repair_required"] is True


@pytest.fixture(autouse=True)
def _restore_service_container():
    yield
    ServiceContainer.clear()
