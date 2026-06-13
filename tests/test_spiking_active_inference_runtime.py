import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.latent_bridge import compute_inference_params
from core.brain.types import ThinkingMode
from core.cognitive.spiking_active_inference import (
    MultiCompartmentSpikeResponseModel,
    SpikingActiveInferenceAdvisor,
    get_spiking_active_inference_advisor,
)
from core.container import ServiceContainer
from core.state.aura_state import AuraState


def teardown_function(_function=None):
    ServiceContainer.clear()


def test_spiking_active_inference_flags_governed_tools_without_executing_them():
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Open Chrome, search for three articles, create a document, and export it as a PDF.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert advice.routing_bias["use_tool_gateway"] is True
    assert advice.features["tool_pressure"] >= 0.58
    assert advice.governance["advisory_only"] is True
    assert advice.governance["executes_tools"] is False
    assert advice.governance["authority_gateway_required_for_effects"] is True


def test_spiking_active_inference_repair_pressure_changes_general_tendency():
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Aura crashed with a memory spike, Cortex unavailable, timeout, and broken tool loop.",
        origin="desktop",
    )

    assert advice.routing_bias["repair_first"] is True
    assert advice.features["error_pressure"] >= 0.70
    assert advice.sampling_bias["temperature_delta"] < 0.0
    assert advice.sampling_bias["repetition_penalty_delta"] > 0.0


def test_multi_compartment_srm_stays_bounded_under_repeated_pressure():
    model = MultiCompartmentSpikeResponseModel()

    summary = {}
    for _ in range(200):
        summary = model.tick([1.0, 0.2, 0.8, 0.9, 0.7, 0.9, 0.2, 0.4], modulation=1.8)

    assert 0.0 <= summary["spike_rate"] <= 1.0
    assert 0.0 <= summary["plateau_rate"] <= 1.0
    assert 0.0 <= summary["weight_mean"] <= 2.0
    assert 0.0 <= summary["threshold_mean"] <= 2.0


def test_spiking_advisor_reregisters_after_container_reset():
    ServiceContainer.clear()
    advisor = get_spiking_active_inference_advisor()
    assert ServiceContainer.get("spiking_active_inference", default=None) is advisor

    ServiceContainer.clear()
    same_advisor = get_spiking_active_inference_advisor()

    assert same_advisor is advisor
    assert ServiceContainer.get("spiking_active_inference", default=None) is advisor


def test_cognitive_engine_records_spiking_active_inference_on_state():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()

    context = engine._apply_spiking_active_inference(
        state,
        "Can you open my notes app and create a timestamped journal entry?",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    advice = state.response_modifiers["spiking_active_inference"]
    assert advice["governance"]["advisory_only"] is True
    assert state.response_modifiers["tool_governance_pressure"] is True
    assert "spiking_active_inference" in context
    assert context["spiking_active_inference"]["advice_id"] == advice["advice_id"]


def test_cognitive_engine_closes_feedback_loop_for_neurodynamic_advice():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    context = engine._apply_spiking_active_inference(
        state,
        "I am confused; reason carefully before acting.",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    feedback = engine._learn_spiking_active_inference_outcome(
        context,
        outcome="assistant_response",
        reward=1.0,
    )

    assert feedback is not None
    assert feedback["outcome"] == "assistant_response"
    assert feedback["action"] == context["spiking_active_inference"]["action"]
    assert "prediction_error" in feedback


def test_runtime_capabilities_expose_neurodynamic_status():
    ServiceContainer.clear()
    advisor = SpikingActiveInferenceAdvisor()
    advisor.advise(
        "Open a tool, verify the result, and remember the lesson.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )
    ServiceContainer.register_instance("spiking_active_inference", advisor, required=False)

    from interface.routes.system import _collect_runtime_capabilities

    payload = _collect_runtime_capabilities(
        {"conversation_ready": True, "state": "ready", "desired_model": "cortex"}
    )

    status = payload["neurodynamic_advisor"]
    assert status["status"] == "active"
    assert status["advisory_only"] is True
    assert status["authority_gateway_required_for_effects"] is True
    assert status["features"]["tool_pressure"] > 0.0
    assert status["features"]["memory_pressure"] > 0.0


@pytest.mark.asyncio
async def test_desktop_quick_path_consumes_neurodynamic_advisory():
    ServiceContainer.clear()
    captured = {}

    class Router:
        async def think(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "I can help with that through the governed desktop tool path."

    ServiceContainer.register_instance("llm_router", Router(), required=False)
    engine = CognitiveEngine()
    advice = {
        "action": "use_governed_tools",
        "uncertainty": 0.42,
        "routing_bias": {
            "use_tool_gateway": True,
            "ask_clarification": False,
            "reduce_load": False,
            "repair_first": False,
        },
        "sampling_bias": {"max_tokens_factor": 0.50},
    }

    thought = await engine._direct_desktop_quick_reply(
        "Open a document and write a short summary.",
        ThinkingMode.FAST,
        "desktop",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "spiking_active_inference": advice,
            "max_tokens": 512,
        },
        timeout_s=20.0,
    )

    assert thought is not None
    assert thought.metadata["spiking_active_inference"] == advice
    assert "Neurodynamic advisory" in captured["messages"][0]["content"]
    assert captured["kwargs"]["protected_foreground_lane"] is True
    assert captured["kwargs"]["allow_cloud_fallback"] is False
    assert captured["kwargs"]["max_tokens"] == 256


def test_latent_bridge_sampling_consumes_spiking_active_inference():
    ServiceContainer.clear()

    class Advisor:
        def snapshot(self):
            return {
                "uncertainty": 0.80,
                "features": {"tool_pressure": 0.70, "error_pressure": 0.65},
                "routing_bias": {"reduce_load": True, "seek_information": True},
            }

    base = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)
    ServiceContainer.register_instance("spiking_active_inference", Advisor(), required=False)
    steered = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)

    assert steered.temperature < base.temperature
    assert steered.top_p < base.top_p
    assert steered.max_tokens < base.max_tokens
    assert steered.presence_penalty > base.presence_penalty
    assert any("active_uncert" in item for item in steered.rationale)
