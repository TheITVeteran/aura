import asyncio
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from core.brain.generation_provenance import generation_metadata_of
from core.phases.response_generation import ResponseGenerationPhase
from core.state.aura_state import AuraState, CognitiveMode


class _Container:
    def __init__(self, services):
        self.services = services

    def get(self, name, default=None):
        return self.services.get(name, default)


class _Router:
    def __init__(self):
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "Thermal-safe response: I am keeping the architectural audit grounded, "
            "reducing the local load, and preserving the conversation thread."
        )

    def get_last_generation_metadata(self):
        return {
            "surface_control_receipt": {
                "enabled": True,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_validation_prompt_present": True,
                "surface_alpha_applied": 0.30,
                "surface_alpha_applied_ok": True,
                "recurrent_runtime_loops_applied": 2,
                "recurrent_runtime_loops_applied_ok": True,
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 1,
                "surface_quality_gate_reasons": [],
                "applied": True,
            }
        }


class _MemoryAckRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return "I’ll remember that the blue lantern is under the desk for later in this conversation."


class _SearchCapability:
    def __init__(self):
        self.calls = []

    def resolve_skill_name(self, name):
        return str(name)

    async def execute(self, skill_name, params, context=None):
        self.calls.append((skill_name, dict(params), dict(context or {})))
        return {
            "ok": True,
            "query": params.get("query"),
            "answer": "NASA describes Europa as an icy moon of Jupiter with a subsurface ocean.",
            "results": [
                {
                    "title": "Europa: Jupiter's Ocean World",
                    "url": "https://science.nasa.gov/jupiter/moons/europa/",
                    "snippet": "Europa is one of Jupiter's moons and is a target in the search for habitable worlds.",
                }
            ],
            "source": "https://science.nasa.gov/jupiter/moons/europa/",
        }


class _EvidenceRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "I searched it live. Source title: Europa: Jupiter's Ocean World. "
            "NASA describes Europa as an icy moon of Jupiter with evidence for a subsurface ocean."
        )


class _FalseSearchInabilityRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "I can't execute web searches directly. But I know NASA has a Europa page."
        )


class _TimeoutSearchRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        raise TimeoutError()


class _BlankSearchRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return ""


class _ConcurrentAmplifierRouter(_Router):
    def __init__(self):
        super().__init__()
        self.completion_order = []
        self._metadata = ContextVar(
            "test_amplifier_generation_metadata",
            default=None,
        )

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        candidate_id = int(prompt.rsplit("-", 1)[-1])
        await asyncio.sleep(0.03 if candidate_id == 0 else 0.005)
        self._metadata.set(
            {
                "candidate_id": candidate_id,
                "surface_control_receipt": {"applied": True},
            }
        )
        self.completion_order.append(candidate_id)
        return f"candidate answer {candidate_id}"

    def get_last_generation_metadata(self):
        return dict(self._metadata.get() or {})


class _LatentService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def deep_reason(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _TimedOutLatentService(_LatentService):
    async def deep_reason(self, **kwargs):
        self.calls.append(kwargs)
        raise TimeoutError("resident latent deadline")


def _live_latent_receipt():
    return {
        "episode_id": "episode-live-32b",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 32,
        "worker_boot_id": "b" * 32,
        "worker_pid": 4200,
        "worker_model_path": "/models/Aura-32B",
        "worker_model_parameter_count": 32_000_000_000,
        "worker_model_stored_parameter_element_count": 5_000_000_000,
        "worker_model_parameter_count_basis": "architecture_config_logical",
        "worker_source_sha256": "c" * 64,
        "worker_affective_steering_active": True,
        "worker_affective_steering_alpha": 0.30,
        "episode_affective_steering_applied": True,
        "episode_affective_steering_alpha": 0.30,
        "request_payload_sha256": "d" * 64,
        "input_tokens_sha256": "e" * 64,
        "input_token_count": 128,
        "steps_taken": 5,
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 37,
        "decode_termination": "eos",
        "decode_temperature": 0.61,
        "decode_top_p": 0.87,
        "runtime_identity": {
            "identity_bound": True,
            "launch_mode": "signed_app",
            "installed_app_required": True,
            "installed_app_verified": True,
            "source_verified": True,
            "source_commit": "f" * 40,
            "workspace_state_sha256": "1" * 64,
            "shell_assets_sha256": "2" * 64,
        },
    }


def _latent_context(objective):
    return {
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "visible_user_message": objective,
        "compact_desktop_chat_contract": False,
        "prompt_shape": {
            "question_parts": 2,
            "prefers_extended_answer": True,
            "requires_single_reply_coverage": True,
        },
        "max_tokens": 512,
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.87,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
    }


@pytest.mark.asyncio
async def test_depth_worthy_desktop_turn_uses_one_latent_generation(monkeypatch):
    objective = (
        "Compare both failure modes, explain the causal tradeoff, and give one "
        "coherent implementation decision with its verification plan."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService(
        {
            "ok": True,
            "text": (
                "The first failure mode loses identity evidence, while the second "
                "duplicates generation. I would bind one episode receipt at the phase "
                "boundary and verify both single invocation and exact runtime identity."
            ),
            "receipt": _live_latent_receipt(),
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "full live context"},
            {"role": "user", "content": objective},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert latent.calls[0]["messages"][-1]["content"] == objective
    assert latent.calls[0]["config_overrides"] == {
        "decode_max_tokens": 512,
        "decode_temperature": 0.61,
        "decode_top_p": 0.87,
    }
    assert latent.calls[0]["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert result.response_modifiers["latent_cortex_succeeded"] is True
    assert result.response_modifiers["latent_cortex_fallback_used"] is False
    assert result.response_modifiers["latent_cortex_identity_bound"] is True
    assert result.response_modifiers["live_mind_controls_worker_applied"] is True


@pytest.mark.asyncio
async def test_selected_latent_refusal_falls_back_to_exactly_one_generation(monkeypatch):
    objective = "Analyze both branches and recommend the safer architecture."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService({"ok": False, "reason": "worker_not_ready"})
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert len(router.calls) == 1
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["latent_cortex_fallback_used"] is True
    assert result.response_modifiers["latent_cortex_failure_reason"] == "worker_not_ready"


@pytest.mark.asyncio
async def test_latent_timeout_preserves_attempt_and_suppresses_second_model_owner(
    monkeypatch,
):
    objective = "Compare both architectures, then choose and verify the safer one."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _TimedOutLatentService({})
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["latent_cortex_selected"] is True
    assert result.response_modifiers["latent_cortex_attempted"] is True
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert (
        result.response_modifiers["generation_failure_class"]
        == "response_generation_deadline_exhausted"
    )


@pytest.mark.asyncio
async def test_returned_latent_timeout_preserves_receipt_and_suppresses_fallback(
    monkeypatch,
):
    objective = "Compare both architectures, then choose and verify the safer one."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    receipt = {
        "episode_id": "live-timeout",
        "input_token_count": 4096,
        "params_unchanged": True,
        "last_stage": "prefill",
        "stage_timings_s": {"prefill": 119.2},
    }
    progress = {
        "stage": "prefill",
        "elapsed_s": 119.2,
        "input_tokens": 4096,
    }
    latent = _LatentService(
        {
            "ok": False,
            "reason": "latent_timeout:cooperative_cancelled",
            "receipt": receipt,
            "progress": progress,
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert (
        result.response_modifiers["generation_failure_class"]
        == "latent_timeout:cooperative_cancelled"
    )
    assert result.response_modifiers["latent_cortex_receipt"] == receipt
    assert result.response_modifiers["latent_cortex_progress"] == progress
    assert (
        result.response_modifiers["response_path"]
        == "cognitive_engine_latent_owner_exhausted"
    )


@pytest.mark.asyncio
async def test_response_generation_downshifts_on_thermal_pressure(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Perform a deep architectural audit"
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["model_tier"] = "secondary"
    state.response_modifiers["deep_handoff"] = True
    state.soma.hardware["temperature"] = 96.0
    state.soma.hardware["cpu_usage"] = 63.0

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(state)

    assert router.calls
    call = router.calls[0]
    assert call["prefer_tier"] == "tertiary"
    assert call["deep_handoff"] is False
    assert call["max_tokens"] < 6144
    assert new_state.response_modifiers["thermal_guard"] is True
    # Downstream voice shaping may add punctuation/styling, so verify content
    # presence rather than exact equality.
    assert "Thermal-safe response" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_executes_required_search_before_answering(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for current NASA Europa page and tell me source title only."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    def _messages_from_state(state_arg, _objective):
        skill_blocks = [
            str(item.get("content") or "")
            for item in state_arg.cognition.working_memory
            if isinstance(item, dict)
            and (item.get("metadata") or {}).get("type") == "skill_result"
        ]
        return [{"role": "system", "content": "\n".join(["context", *skill_blocks])}]

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        _messages_from_state,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    skill_name, params, context = capability.calls[0]
    assert skill_name == "web_search"
    assert params["query"] == "current NASA Europa page"
    assert params["retain"] is True
    assert context["effect_scope"] == "read_only_external_io"
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert new_state.response_modifiers["last_skill_run"] == "web_search"
    assert "[SKILL RESULT: web_search]" in router.calls[0]["messages"][0]["content"]
    assert "Europa: Jupiter's Ocean World" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_preserves_source_definition_search_tail(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Please search the web for one current NASA page about Europa. "
        "Tell me the source title and what NASA says Europa is."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda state_arg, _objective: [
            {
                "role": "system",
                "content": "\n".join(
                    str(item.get("content") or "")
                    for item in state_arg.cognition.working_memory
                    if isinstance(item, dict)
                ),
            }
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    _, params, _ = capability.calls[0]
    assert "one current NASA page about Europa" in params["query"]
    assert "what NASA says Europa is" in params["query"]
    assert "Tell me the source title" not in params["query"]
    assert new_state.response_modifiers["last_skill_ok"] is True


def test_required_search_cleaner_uses_original_request_inside_repair_prompt():
    repair_prompt = (
        "The prior draft for this same user turn did not satisfy the user-facing response contract.\n"
        "Observed problems: truncated_tail.\n\n"
        "Rewrite from scratch for the original user request below.\n\n"
        "Original user request:\n"
        "Please search the web for one current NASA page about Europa. "
        "Tell me the source title and what NASA says Europa is.\n\n"
        "Rejected draft for avoidance only:\n"
        "I've searched and found a relevant page from NASA. The source title is"
    )

    cleaned = ResponseGenerationPhase._clean_required_search_query(repair_prompt)

    assert "one current NASA page about Europa" in cleaned
    assert "what NASA says Europa is" in cleaned
    assert "Rejected draft" not in cleaned
    assert "I've searched" not in cleaned


@pytest.mark.asyncio
async def test_response_generation_required_search_uses_service_container_fallback(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for current NASA Europa page and tell me source title only."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))

    def _messages_from_state(state_arg, _objective):
        skill_blocks = [
            str(item.get("content") or "")
            for item in state_arg.cognition.working_memory
            if isinstance(item, dict)
            and (item.get("metadata") or {}).get("type") == "skill_result"
        ]
        return [{"role": "system", "content": "\n".join(["context", *skill_blocks])}]

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        _messages_from_state,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ServiceContainer.get",
        lambda name, default=None: capability if name == "capability_engine" else default,
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert "[SKILL RESULT: web_search]" in router.calls[0]["messages"][0]["content"]
    assert "Europa: Jupiter's Ocean World" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_repairs_false_search_inability_after_evidence(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for one current NASA page about Europa, then answer with the source title and one sentence."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _FalseSearchInabilityRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda state_arg, _objective: [
            {
                "role": "system",
                "content": "\n".join(
                    str(item.get("content") or "")
                    for item in state_arg.cognition.working_memory
                    if isinstance(item, dict)
                ),
            }
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert (
        new_state.response_modifiers["required_tool_false_inability_repaired"]["skill"]
        == "web_search"
    )
    reply = new_state.cognition.last_response
    assert "Europa: Jupiter's Ocean World" in reply
    assert "science.nasa.gov/jupiter/moons/europa" in reply
    assert "can't execute web searches" not in reply.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router_cls", "receipt_key"),
    [
        (_TimeoutSearchRouter, "required_tool_timeout_repaired"),
        (_BlankSearchRouter, "required_tool_empty_repaired"),
    ],
)
async def test_response_generation_answers_from_search_evidence_when_cortex_fails(
    monkeypatch,
    router_cls,
    receipt_key,
):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for one current NASA page about Europa, then answer with the source title and one sentence."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = router_cls()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert new_state.response_modifiers[receipt_key]["skill"] == "web_search"
    reply = new_state.cognition.last_response
    assert "Europa: Jupiter's Ocean World" in reply
    assert "science.nasa.gov/jupiter/moons/europa" in reply
    mutation_stages = {
        item["stage"]
        for item in new_state.response_modifiers["live_mind_surface_control_receipt"][
            "text_mutations"
        ]
    }
    expected_stage = (
        "response_generation.required_tool_timeout_recovery"
        if router_cls is _TimeoutSearchRouter
        else "response_generation.required_tool_blank_recovery"
    )
    assert expected_stage in mutation_stages
    assert new_state.response_modifiers["live_mind_surface_control_receipt"][
        "deterministic_repair_applied"
    ] is True


@pytest.mark.asyncio
async def test_response_generation_honors_live_caller_token_cap_after_biases(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "How would that uncertainty change your next decision?"
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["sampling_bias"] = {"max_tokens_factor": 1.25}
    state.response_modifiers["imagination_sampling_bias"] = {"max_tokens_factor": 1.25}
    state.response_modifiers["bicameral_sampling_bias"] = {"max_tokens_factor": 1.25}

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 1,
        },
    )

    assert router.calls
    assert router.calls[0]["max_tokens"] == 1


@pytest.mark.asyncio
async def test_response_generation_forwards_compiled_visible_output_contract(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = 'Reply exactly: "yes"'
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(
        state,
        context={
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 4096,
        },
    )

    assert router.calls
    call = router.calls[0]
    assert call["max_tokens"] == len(b"yes") + 16
    assert call["requested_output_contract"]["kind"] == "exact_reply"
    assert call["semantic_output_token_cap"] == 8
    assert call["hard_output_token_ceiling"] == len(b"yes") + 16
    assert result.cognition.last_response == "yes"


@pytest.mark.asyncio
async def test_response_amplifier_adopts_selected_candidate_metadata_not_last_completion(
    monkeypatch,
):
    router = _ConcurrentAmplifierRouter()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    state = AuraState()

    async def fake_amplify_turn(_objective, generate, **_kwargs):
        candidates = await asyncio.gather(
            generate("candidate-0", 0.2),
            generate("candidate-1", 0.8),
        )
        winner = candidates[0]
        return SimpleNamespace(
            answer=str(winner),
            confidence=0.95,
            verified=True,
            generation_metadata=generation_metadata_of(winner),
            receipt=SimpleNamespace(to_dict=lambda: {"winner": 0}),
        )

    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.is_amplifiable",
        lambda _objective: "planning",
    )
    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.amplify_turn",
        fake_amplify_turn,
    )

    answer = await phase._maybe_amplify_response(
        objective="Plan a causally valid migration",
        draft="initial draft",
        router=router,
        state=state,
        request_timeout=30.0,
        origin="user",
        tier="primary",
        runtime_context={"visible_user_message": "Plan the migration"},
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )

    assert answer == "candidate answer 0"
    assert router.completion_order == [1, 0]
    assert generation_metadata_of(answer)["candidate_id"] == 0


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_identity_refresh_when_runtime_is_not_idle(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\nSummarize recent continuity."
    state.cognition.current_origin = "system"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_noise_objective(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Task exception: database is locked while background cognitive state retries."
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_treats_prefixed_user_origin_as_foreground(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Hello Aura."
    state.cognition.current_origin = "routing_user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    background_gate_calls = []

    def _unexpected_background_gate(*_args, **_kwargs):
        background_gate_calls.append((_args, _kwargs))
        raise AssertionError("foreground origins should not consult background gating")

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _unexpected_background_gate,
    )

    result = await phase.execute(state)

    # The router must have been called as a foreground request.
    # We don't assert the exact response text because downstream voice shaping
    # (SubstrateVoiceEngine) may legitimately restyle it — but the routing
    # decision (foreground vs background) is what this test validates.
    assert background_gate_calls == []
    assert router.calls, "Router should have been called for a user-facing origin"
    assert router.calls[0]["is_background"] is False
    assert result.cognition.last_response, "A response should have been generated"


@pytest.mark.asyncio
async def test_response_generation_full_phase_injects_live_desktop_grounding(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "What tools can you use externally?"
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "What tools can you use externally?"},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "live_mind_context_required": True,
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "lane": {"state": "ready", "conversation_ready": True},
                "voice": {"mood": "steady"},
                "substrate": {"coherence": 0.91},
                "governance": {"legacy_fallback_allowed": False},
            },
            "mind_context_contract": "Use the live mind context as causal grounding.",
            "live_speech_grounding_frame": {
                "attention_focus": "Bryan's live desktop capability question",
                "dominant_action": "answer",
                "mood": "steady",
            },
            "grounded_capability_inventory_context": (
                "Aura can use governed desktop, browser, file, document, and terminal lanes "
                "only with authorization and effect receipts."
            ),
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What tools can you use externally?",
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.61,
                "top_p": 0.87,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert router.calls
    system_prompt = router.calls[0]["messages"][0]["content"]
    assert "LIVE MIND CONTEXT" in system_prompt
    assert "must_answer_from_full_mind_path" in system_prompt
    assert "LIVE SPEECH GROUNDING" in system_prompt
    assert "GOVERNED CAPABILITY INVENTORY EVIDENCE" in system_prompt
    call = router.calls[0]
    assert call["clean_user_surface_contract"] is True
    assert call["user_surface_validation_prompt"] == "What tools can you use externally?"
    assert call["live_mind_controls_bound"] is True
    assert call["live_mind_snapshot_ready"] is True
    assert call["live_mind_required_subsystems_ok"] is True
    assert call["clean_user_surface_recurrent_loops"] == 2
    assert call["clean_user_surface_steering_alpha"] == 0.30
    assert call["temperature"] == 0.61
    assert call["top_p"] == 0.87
    assert result.response_modifiers["live_mind_controls_worker_applied"] is True
    assert result.response_modifiers["live_mind_surface_control_receipt"]["applied"] is True


@pytest.mark.asyncio
async def test_response_generation_quality_gate_uses_visible_desktop_prompt(monkeypatch):
    state = AuraState()
    visible = (
        "Remember this note for later in this conversation: "
        "the blue lantern is under the desk."
    )
    contract_wrapped = (
        f"{visible}\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n"
        "- Runtime path contract: governed tool and model lane status must remain available.\n"
        "[END LIVE DESKTOP FULL-MIND CONTRACT]"
    )
    state.cognition.current_objective = contract_wrapped
    state.cognition.current_origin = "desktop_quick_user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _MemoryAckRouter()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    prompts_seen = []

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": contract_wrapped},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    from core.conversation.response_reliability import assess_user_facing_reply as real_assess

    def _record_assess(prompt, reply):
        prompts_seen.append(str(prompt))
        return real_assess(prompt, reply)

    monkeypatch.setattr(
        "core.phases.response_generation.assess_user_facing_reply",
        _record_assess,
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "clean_user_surface_contract": True,
            "visible_user_message": visible,
            "user_surface_validation_prompt": visible,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert router.calls
    assert router.calls[0]["visible_user_message"] == visible
    assert router.calls[0]["user_surface_validation_prompt"] == visible
    assert prompts_seen
    assert all(prompt == visible for prompt in prompts_seen)
    assert "blue lantern" in result.cognition.last_response
    assert "Runtime path contract" not in result.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_dialogue_retry_preserves_live_mind_contract(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Explain how confusion changes your reasoning."
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    replies = iter(
        (
            "Confusion increases uncertainty, so I slow down and check competing interpretations before answering.",
            "When confusion rises, I compare alternatives explicitly and preserve uncertainty until the evidence separates them.",
        )
    )

    async def _distinct_retry_think(**kwargs):
        router.calls.append(kwargs)
        return next(replies)

    router.think = _distinct_retry_think
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "Explain how confusion changes your reasoning."},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    async def _force_retry(response, contract, *, retry_generate, state):
        retried = await retry_generate("Repair the answer without leaving the live mind path.")
        return retried, SimpleNamespace(to_dict=lambda: {"valid": True}), True

    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _force_retry,
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "visible_user_message": "Explain how confusion changes your reasoning.",
            "clean_user_surface_contract": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.49,
                "top_p": 0.81,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.34,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert len(router.calls) == 2
    retry_call = router.calls[1]
    assert retry_call["desktop_cognitive_engine_required"] is True
    assert retry_call["live_runtime_payload_required"] is True
    assert retry_call["clean_user_surface_contract"] is True
    assert retry_call["live_mind_controls_bound"] is True
    assert retry_call["clean_user_surface_recurrent_loops"] == 2
    assert retry_call["clean_user_surface_steering_alpha"] == 0.34
    assert retry_call["temperature"] == 0.49
    assert retry_call["top_p"] == 0.81
    assert router.calls[0]["desktop_cognitive_engine_required"] is True
    assert router.calls[0]["allow_cloud_fallback"] is False
    mutations = result.response_modifiers["live_mind_surface_control_receipt"][
        "text_mutations"
    ]
    assert any(
        item["stage"] == "response_generation.dialogue_contract_retry"
        for item in mutations
    )


@pytest.mark.asyncio
async def test_response_generation_ledgers_executive_guard_visible_replacement(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Summarize the architectural audit."
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "Summarize the architectural audit."},
        ],
    )

    def _align(text):
        return str(text).replace("Thermal-safe", "Identity-aligned"), True, [
            "identity_alignment"
        ]

    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=_align),
    )

    result = await phase.execute(
        state,
        context={
            "visible_user_message": "Summarize the architectural audit.",
            "clean_user_surface_contract": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.55,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        },
    )

    mutations = result.response_modifiers["live_mind_surface_control_receipt"][
        "text_mutations"
    ]
    assert any(
        item["stage"] == "response_generation.executive_guard"
        and item["deterministic"] is True
        for item in mutations
    )


def test_response_generation_timeout_fits_owning_cognitive_deadline(monkeypatch):
    monkeypatch.setattr("core.phases.response_generation.time.monotonic", lambda: 100.0)

    timeout = ResponseGenerationPhase._bounded_request_timeout(
        {"cognitive_cycle_deadline_monotonic": 130.0},
        180.0,
        reserve_s=8.0,
    )

    assert timeout == 22.0


def test_response_generation_timeout_keeps_requested_budget_without_deadline():
    assert (
        ResponseGenerationPhase._bounded_request_timeout({}, 180.0, reserve_s=8.0)
        == 180.0
    )
