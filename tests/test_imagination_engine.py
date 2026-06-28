import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.imagination import ImaginationEngine, get_imagination_engine
from core.brain.inference_gate import InferenceGate
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.types import ThinkingMode
from core.container import ServiceContainer
from core.phases.response_generation import ResponseGenerationPhase
from core.state.aura_state import AuraState


def test_imagination_engine_reports_ready_before_first_frame():
    engine = ImaginationEngine()

    status = engine.get_status()

    assert status["running"] is True
    assert status["status"] == "idle"
    assert status["frames_built"] == 0
    assert status["latest"] is None


def test_imagination_engine_models_visual_counterfactual_and_connections():
    state = AuraState.default()
    state.affect.curiosity = 0.86
    state.affect.emotions["curiosity"] = 0.78
    state.affect.emotions["confused"] = 0.34
    state.cognition.working_memory.append(
        {
            "role": "user",
            "content": "Earlier we talked about memory as architecture.",
        }
    )
    engine = ImaginationEngine()

    frame = engine.imagine(
        "What would a city made of memory look like, and what novel connection does it suggest?",
        state=state,
        origin="desktop",
    )
    replay_frame = engine.imagine(
        "What would a city made of memory look like, and what novel connection does it suggest?",
        state=state,
        origin="desktop",
    )

    assert frame.salience > 0.5
    assert replay_frame.frame_id == frame.frame_id
    assert frame.memory_pressure > 0.45
    assert frame.verification_pressure > 0.15
    assert "visual" in frame.modalities
    assert "counterfactual" in frame.modalities
    assert "conceptual" in frame.modalities
    assert "memory" in frame.attention_targets
    assert frame.visual_model
    assert frame.mental_canvas["modality"] == "visual"
    assert frame.mental_canvas["image_prompt"]
    assert frame.mental_canvas["externalization_path"].startswith("If the user asks")
    assert frame.associative_links
    assert frame.novel_thoughts
    assert frame.simulation_steps
    assert frame.action_affordances
    assert frame.ablation_predictions["no_imagination"]
    assert frame.causal_effects["memory_priority"] == pytest.approx(frame.memory_pressure)
    assert "memory_retrieval_bias" in frame.causal_effects["expected_downstream"]
    assert frame.conceptual_bridge
    assert frame.counterfactuals
    assert frame.governance["advisory_only"] is True
    assert frame.governance["no_external_effects"] is True
    assert "not external perception" in frame.verification_boundary
    assert frame.working_memory["admission"] in {"admit", "compress_foreground", "thin_frame"}
    assert frame.attractor_state["selected"]
    assert frame.attractor_state["recurrent_depth"] >= 1
    assert frame.eligibility_trace
    assert frame.causal_effects["working_memory_admission"] == frame.working_memory["admission"]


def test_imagination_engine_load_gate_compresses_background_under_pressure():
    engine = ImaginationEngine()

    frame = engine.imagine(
        "Invent a new mental model for desktop tool use and imagine what it looks like.",
        state=AuraState.default(),
        origin="background",
        is_background=True,
        context={
            "memory_pressure": {
                "level": "high",
                "pressure_pct": 86.0,
                "reason": "test high memory pressure",
            }
        },
    )

    assert frame.working_memory["admission"] == "defer_background"
    assert frame.working_memory["admitted"] is False
    assert frame.routing_bias["compress_imagination"] is True
    assert frame.sampling_bias["max_tokens_factor"] <= 0.70
    assert frame.causal_effects["load_shed_requested"] is True
    assert "runtime_load_shed" in frame.causal_effects["expected_downstream"]
    assert frame.governance["no_external_effects"] is True


def test_imagination_feedback_updates_future_attractor_bias():
    engine = ImaginationEngine()
    frame = engine.imagine(
        "Create a novel analogy for memory, curiosity, and tool governance.",
        state=AuraState.default(),
        origin="desktop",
    )
    selected = frame.attractor_state["selected"]

    feedback = engine.learn_from_feedback(
        frame,
        reward=1.0,
        outcome="assistant_response",
    )
    snapshot = engine.snapshot()

    assert feedback is not None
    assert feedback["selected_attractor"] == selected
    assert feedback["updated_bias"] > 0.0
    assert snapshot["attractor_bias"][selected] == pytest.approx(feedback["updated_bias"])
    assert snapshot["recent_outcomes"][-1]["outcome"] == "assistant_response"


def test_cognitive_engine_records_imagination_workspace_as_state_and_context():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    state.affect.curiosity = 0.8
    state.cognition.working_memory.append(
        {
            "role": "user",
            "content": "Earlier we were discussing how desktop actions should stay governed.",
        }
    )

    context = engine._apply_imagination_workspace(
        state,
        "Imagine what a governed desktop action system should look like.",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    frame = state.response_modifiers["imagination_workspace"]
    assert frame["governance"]["advisory_only"] is True
    assert state.response_modifiers["creative_pressure"] > 0.0
    assert state.response_modifiers["novelty_pressure"] > 0.0
    assert "imagination_sampling_bias" in state.response_modifiers
    assert state.response_modifiers["imagination_memory_pressure"] == frame["memory_pressure"]
    assert state.response_modifiers["imagination_verification_pressure"] == frame["verification_pressure"]
    assert state.response_modifiers["imagination_working_memory"] == frame["working_memory"]
    assert state.response_modifiers["imagination_attractor_state"] == frame["attractor_state"]
    assert state.response_modifiers["verification_pressure"] == frame["verification_pressure"]
    assert state.response_modifiers["tool_governance_pressure"] is True
    assert state.cognition.modifiers["imagination_workspace"]["frame_id"] == frame["frame_id"]
    assert state.cognition.modifiers["imagination_attention_targets"] == frame["attention_targets"]
    assert state.cognition.modifiers["imagination_causal_effects"]["memory_priority"] == frame["memory_pressure"]
    assert state.cognition.modifiers["imagination_working_memory"] == frame["working_memory"]
    assert state.cognition.modifiers["imagination_attractor_state"] == frame["attractor_state"]
    assert state.cognition.modifiers["requires_memory_grounding"] is True
    assert "imagined focus" in state.cognition.attention_focus
    assert context["imagination_workspace"]["frame_id"] == frame["frame_id"]
    assert ServiceContainer.get("imagination_engine", default=None) is get_imagination_engine()


def test_context_assembler_injects_imagination_workspace_prompt_block():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    state.cognition.current_objective = (
        "What would this architecture look like as a mental model?"
    )
    state.affect.curiosity = 0.74
    engine._apply_imagination_workspace(
        state,
        state.cognition.current_objective,
        "desktop",
        {},
        is_background=False,
    )

    prompt = ContextAssembler.build_system_prompt(state)

    assert "IMAGINATION WORKSPACE" in prompt
    assert "Private hypothetical model" in prompt or "private generative scratchpad" in prompt
    assert "do not claim" in prompt.lower()


@pytest.mark.asyncio
async def test_desktop_quick_path_consumes_imagination_workspace():
    ServiceContainer.clear()
    captured = {}

    class Router:
        async def think(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "I can model that privately first, then verify anything external through tools."

    ServiceContainer.register_instance("llm_router", Router(), required=False)
    frame = ImaginationEngine().imagine(
        "What would this look like as a visible workflow?",
        state=AuraState.default(),
        origin="desktop",
    ).to_dict()
    engine = CognitiveEngine()

    thought = await engine._direct_desktop_quick_reply(
        "What would this look like as a visible workflow?",
        ThinkingMode.FAST,
        "desktop",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "imagination_workspace": frame,
            "max_tokens": 512,
        },
        timeout_s=20.0,
    )

    assert thought is not None
    assert thought.metadata["imagination_workspace"]["frame_id"] == frame["frame_id"]
    assert thought.metadata["imagination_workspace_feedback"]["outcome"] == "desktop_quick_reply"
    assert "Imagination workspace" in captured["messages"][0]["content"]
    assert "Mental canvas" in captured["messages"][0]["content"]
    assert "Novel thought candidates" in captured["messages"][0]["content"]
    assert captured["kwargs"]["protected_foreground_lane"] is True
    assert captured["kwargs"]["allow_cloud_fallback"] is False
    assert captured["kwargs"]["imagination_sampling_bias"] == frame["sampling_bias"]
    assert 512 < captured["kwargs"]["max_tokens"] <= 768


def test_inference_gate_applies_bounded_runtime_imagination_sampling_bias():
    state = AuraState.default()
    state.response_modifiers["imagination_sampling_bias"] = {
        "temperature_delta": 0.11,
        "max_tokens_factor": 1.1,
    }
    state.response_modifiers["sampling_bias"] = {
        "temperature_delta": -0.02,
        "max_tokens_factor": 0.8,
    }

    temperature, tokens, applied = InferenceGate._apply_runtime_sampling_biases(
        base_temperature=0.70,
        max_tokens=1000,
        context={},
        state=state,
        allow_token_scaling=True,
    )

    assert temperature == pytest.approx(0.79)
    assert tokens == 880
    assert applied["temperature_delta"] == pytest.approx(0.09)
    assert applied["max_tokens_factor"] == pytest.approx(0.88)


def test_inference_gate_rejects_unbounded_runtime_sampling_bias_values():
    temperature, tokens, applied = InferenceGate._apply_runtime_sampling_biases(
        base_temperature=0.70,
        max_tokens=1000,
        context={
            "imagination_sampling_bias": {
                "temperature_delta": 9.0,
                "max_tokens_factor": 99.0,
            }
        },
        state=None,
        allow_token_scaling=True,
    )

    assert temperature == pytest.approx(0.88)
    assert tokens == 1000
    assert applied["temperature_delta"] == pytest.approx(0.18)
    assert applied["max_tokens_factor"] == pytest.approx(1.0)


def test_imagination_prompt_block_exports_canvas_without_claiming_perception():
    frame = ImaginationEngine().imagine(
        "Invent a new phrase and picture for curiosity connected to memory.",
        state=AuraState.default(),
        origin="desktop",
    )

    block = frame.prompt_block()

    assert "Mental canvas" in block
    assert "Novel thought candidates" in block
    assert "Association map" in block
    assert "Causal effects" in block
    assert "not as evidence" in block
    assert frame.mental_canvas["externalization_path"].endswith("otherwise keep it private.")


def test_response_generation_sampling_combines_imagination_and_load_biases():
    temperature, tokens = ResponseGenerationPhase._apply_generation_sampling_bias(
        base_temperature=0.70,
        token_budget=4096,
        biases=[
            {"temperature_delta": -0.05, "max_tokens_factor": 0.5},
            {"temperature_delta": 0.11, "max_tokens_factor": 1.1},
        ],
    )

    assert temperature == pytest.approx(0.76)
    assert tokens == 2252
