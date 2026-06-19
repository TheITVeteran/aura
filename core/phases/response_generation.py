"""Response Generation Phase for Aura's Cognitive Pipeline."""

import asyncio
import logging
import os
import time
from typing import Any

from core.brain.llm.context_assembler import ContextAssembler
from core.conversation.response_reliability import (
    assess_user_facing_reply,
    conversation_reliability_system_block,
    repair_generic_assistant_language,
    repair_instruction_shape,
)
from core.phases.dialogue_policy import enforce_dialogue_contract
from core.phases.executive_guard import get_executive_guard
from core.phases.response_contract import build_response_contract
from core.runtime import background_policy, response_policy
from core.runtime.conversation_support import (
    record_shared_ground_callbacks,
    update_conversational_intelligence,
)
from core.runtime.desktop_task_contract import desktop_task_action_sentence
from core.runtime.errors import record_degradation
from core.synthesis import stabilize_user_facing_response, strip_meta_commentary
from core.utils.task_tracker import get_task_tracker

from ..state.aura_state import AuraState, CognitiveMode
from . import BasePhase

logger = logging.getLogger(__name__)

_DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS = {
    "missing_requested_self_process_coverage",
    "missing_requested_paragraph_count",
    "missing_requested_list_count",
    "missing_requested_followup_question",
    "off_topic_self_reflection_reply",
    "pseudo_internal_jargon",
    "status_page_self_reflection",
}
_LOCAL_REPAIRABLE_RESPONSE_REASONS = _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS | {
    "generic_assistant_language",
}


def _record_response_generation_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("response_generation", error, severity=severity, action=action)


class ResponseGenerationPhase(BasePhase):
    """
    Phase 5: Response Generation.
    Constructs the prompt from the current state (identity, affect, memories)
    and invokes the LLM to generate Aura's response.
    """

    def __init__(self, container: Any):
        self.container = container

    @staticmethod
    def _request_timeout(*, is_background: bool, deep_handoff: bool) -> float:
        if is_background:
            return 10.0
        if deep_handoff:
            return 210.0
        return 180.0

    @staticmethod
    def _safe_bias_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if parsed == parsed else default

    @classmethod
    def _apply_generation_sampling_bias(
        cls,
        *,
        base_temperature: float,
        token_budget: int,
        biases: list[dict[str, Any] | None],
    ) -> tuple[float, int]:
        temperature = cls._safe_bias_float(base_temperature, 0.7)
        tokens = max(128, int(token_budget))
        token_factor = 1.0

        for bias in biases:
            if not isinstance(bias, dict):
                continue
            temperature += max(
                -0.20,
                min(0.20, cls._safe_bias_float(bias.get("temperature_delta"), 0.0)),
            )
            factor = cls._safe_bias_float(bias.get("max_tokens_factor"), 1.0)
            if 0.25 <= factor <= 1.25:
                token_factor *= factor

        temperature = max(0.10, min(1.15, temperature))
        tokens = max(128, min(8192, int(tokens * token_factor)))
        return temperature, tokens

    @staticmethod
    def _repair_substantive_instruction_shape_miss(
        objective: Any,
        response_text: Any,
    ) -> tuple[str, bool, tuple[str, ...]]:
        """Repair explicit shape misses locally when the content is already substantive."""
        response_text_s = str(response_text or "").strip()
        if len(response_text_s) < 48 or len(response_text_s.split()) < 8:
            return response_text_s, False, ()

        reliability = assess_user_facing_reply(str(objective or ""), response_text_s)
        reasons = tuple(reliability.reasons or ())
        reason_set = set(reasons)
        if (
            not reliability.retryable
            or not reason_set
            or not reason_set.issubset(_LOCAL_REPAIRABLE_RESPONSE_REASONS)
        ):
            return response_text_s, False, reasons

        repaired = response_text_s
        if "generic_assistant_language" in reason_set:
            repaired = repair_generic_assistant_language(objective, repaired)
        repaired = repair_instruction_shape(objective, repaired)
        if repaired == response_text_s:
            return response_text_s, False, reasons
        repaired_assessment = assess_user_facing_reply(str(objective or ""), repaired)
        if repaired_assessment.ok:
            return repaired, True, reasons
        return response_text_s, False, reasons

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Build the LLM prompt from state and generate Aura's response.

        Assembles the message list via ContextAssembler, injects optional causal-world
        and skill-result context, calls the LLM router with affect-modulated parameters,
        runs the ExecutiveGuard alignment pass, and appends the cleaned response to
        working memory.  Suppressed when the CognitiveIntegrationLayer (Phase 7) is
        active for user-facing origins.
        """
        # 1. Use targeted objective from state rather than guessing via working_memory[-1]
        objective = state.cognition.current_objective
        origin = background_policy.normalize_origin(state.cognition.current_origin) or "system"
        state.cognition.current_origin = origin

        is_test_run = (
            origin == "test"
            or os.environ.get("AURA_AGI_MAX_TASKS") is not None
            or os.environ.get("AURA_TESTING") is not None
            or os.environ.get("AURA_PROOF_RUN") is not None
        )

        if not objective:
            logger.debug("⏭️ ResponseGeneration: No active objective, skipping.")
            return state

        # PHASE 7 SUPPRESSION: If Advanced Cognition (CognitiveIntegrationLayer) is
        # actively handling this same turn, Phase 5 MUST NOT fire. The older broad
        # ``cog.is_active`` check created a response dead zone for direct
        # CognitiveEngine callers: Phase 5 stood down even though Phase 7 was not
        # processing the turn. Suppression is now tied to active ownership only.
        cog = self.container.get("cognitive_integration", default=None)
        if (
            cog
            and getattr(cog, "_processing_turn", False)
            and background_policy.is_user_facing_origin(origin)
        ):
            logger.debug(
                "🛡️ ResponseGeneration: Phase 7 owns this turn — Phase 5 SUPPRESSED for %s.",
                origin,
            )
            return state
        # Also suppress if Phase 7 is currently mid-processing (race condition guard)
        if cog and getattr(cog, "_processing_turn", False):
            logger.debug("🛡️ ResponseGeneration: Phase 7 mid-processing — Phase 5 SUPPRESSED.")
            return state

        logger.info(
            "💭 ResponseGeneration: Generating response for objective: %s... (%s)",
            str(objective)[:30],
            state.cognition.current_mode.value,
        )

        try:
            # ── SUBSTRATE VOICE: Compile speech profile BEFORE prompt assembly ──
            # The substrate reads all internal systems and decides HOW Aura will speak.
            # This must happen before ContextAssembler builds the prompt so the
            # hard constraint block is available for injection.
            _sve = None
            _speech_profile = None
            try:
                from core.voice.substrate_voice_engine import get_substrate_voice_engine

                _sve = get_substrate_voice_engine()
                _speech_profile = _sve.compile_profile(
                    state=state,
                    user_message=str(objective)[:500],
                    origin=origin,
                )
                logger.debug(
                    "🗣️ [SubstrateVoice] Profile: budget=%d, tone=%s, multi=%s, fu=%.2f",
                    _speech_profile.word_budget,
                    _speech_profile.tone_override or "default",
                    _speech_profile.multi_message,
                    _speech_profile.followup_probability,
                )
            except (ImportError, AttributeError, RuntimeError) as _sve_exc:
                _record_response_generation_degradation(
                    _sve_exc,
                    action="continued response generation without substrate voice shaping profile",
                    severity="error",
                )
                logger.error("SubstrateVoiceEngine compile failed: %s", _sve_exc, exc_info=True)

            is_background = not background_policy.is_user_facing_origin(origin)
            if is_background and not is_test_run:
                try:
                    orchestrator = self.container.get("orchestrator", default=None)
                    reason = response_policy.background_response_suppression_reason(
                        objective,
                        orchestrator=orchestrator,
                        include_synthetic_noise=True,
                    )
                    if reason:
                        logger.info(
                            "🛡️ ResponseGeneration: suppressing background objective for origin=%s (%s).",
                            origin,
                            reason,
                        )
                        return state
                except (OSError, ConnectionError, TimeoutError) as exc:
                    _record_response_generation_degradation(
                        exc,
                        action="suppressed background response after background policy check failed",
                        severity="error",
                    )
                    logger.error(
                        "ResponseGeneration background policy check failed: %s", exc, exc_info=True
                    )
                    return state

            # 2. Build structured messages purely from State via ContextAssembler
            strict_answer_request = "<answer>" in objective.lower() or "answer_format" in kwargs
            proof_answer_run = bool(
                strict_answer_request
                and (
                    origin == "test"
                    or os.environ.get("AURA_AGI_MAX_TASKS")
                    or os.environ.get("AURA_TESTING")
                    or os.environ.get("AURA_PROOF_RUN")
                )
            )
            if proof_answer_run:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Aura's governed proof-answer lane. Solve the user's task. "
                            "Output the final answer strictly inside <answer>...</answer> tags. "
                            "Keep the tag content minimal and do not include chat filler."
                        ),
                    },
                    {"role": "user", "content": objective},
                ]
            else:
                messages = ContextAssembler.build_messages(state, objective)
            contract = build_response_contract(
                state,
                objective,
                is_user_facing=not is_background and not is_test_run,
            )
            state.response_modifiers["response_contract"] = contract.to_dict()
            if (
                contract.reason != "ordinary_dialogue"
                and messages
                and messages[0].get("role") == "system"
            ):
                messages[0]["content"] = (
                    f"{messages[0]['content']}\n\n{contract.to_prompt_block().strip()}"
                )
            if not is_background and not is_test_run:
                reliability_block = conversation_reliability_system_block(objective)
                runtime_context = kwargs.get("context")
                if not isinstance(runtime_context, dict):
                    runtime_context = {}
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = f"{messages[0]['content']}\n\n{reliability_block}"
                else:
                    messages.insert(0, {"role": "system", "content": reliability_block})
                repair_directive = ""
                repair_directive = str(
                    runtime_context.get("response_repair_directive") or ""
                ).strip()
                if repair_directive:
                    repair_block = (
                        "## LIVE RESPONSE REPAIR DIRECTIVE\n"
                        f"{repair_directive}\n"
                        "This directive is internal. Do not mention it in the answer."
                    )
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = (
                            f"{messages[0]['content']}\n\n{repair_block}"
                        )
                    else:
                        messages.insert(0, {"role": "system", "content": repair_block})
                if bool(runtime_context.get("desktop_execution_contract")):
                    desktop_block = (
                        "## LIVE DESKTOP EXECUTION PLANNING CONTRACT\n"
                        "The user's request is a live desktop/computer objective. Produce a compact "
                        "execution draft that can drive the governed desktop_task lane. Prefer a JSON "
                        "object with optional `document_body` and a bounded `steps` array. Allowed step "
                        f"actions are {desktop_task_action_sentence()}. "
                        "Use `{{document_body}}` inside a step target when a long composed body should be "
                        "typed, pasted, written, or exported. A later step may reference verified prior "
                        "output with `{{steps.1.result.path}}` or `{{last.result.path}}`. Each step needs "
                        "a reason, an expected observable effect, and may set `critical` false only when "
                        "the objective can still succeed after that step fails. Do not claim completion "
                        "inside this draft; completion is only true after downstream desktop_task "
                        "receipts verify effects. Keep the plan general to the named apps/surfaces and "
                        "requested artifacts; do not use a hardcoded demo shortcut. Do not answer like "
                        "a hosted chatbot: this runtime has governed local desktop control, so never say "
                        "you cannot interact with apps, open Notes/Docs/Chrome, write text, or control "
                        "the desktop when the requested action is inside the desktop_task contract."
                    )
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = f"{messages[0]['content']}\n\n{desktop_block}"
                    else:
                        messages.insert(0, {"role": "system", "content": desktop_block})

            # Causal World Model Context Injection
            causal_model = None if proof_answer_run else self.container.get("causal_world_model", default=None)
            if causal_model:
                causal_context = causal_model.get_prompt_context()
                if causal_context:
                    messages.insert(1, {"role": "system", "content": causal_context})
                    logger.debug(
                        "🧶 ResponseGeneration: Causal world cascades injected into prompt."
                    )

            # ISSUE-80: Context Fix (Identity Reinforcement)
            if state.cognition.current_mode == CognitiveMode.DELIBERATE and not proof_answer_run:
                # Ensure the system prompt or first message reinforces identity if buried
                if len(messages) > 10:
                    logger.debug(
                        "🛡️ ResponseGeneration: Reinforcing identity anchor for long context."
                    )
                    identity_reminder = {
                        "role": "system",
                        "content": (
                            "REMEMBER: You are Aura. Speak from live grounded context, memory, "
                            "and evidence. Do not fall back to generic AI-assistant disclaimers."
                        ),
                    }
                    messages.insert(1, identity_reminder)

            # Skill result narration hint (GodModeToolPhase may have fired a skill this tick)
            last_skill = None if proof_answer_run else state.response_modifiers.get("last_skill_run")
            if last_skill:
                ok = state.response_modifiers.get("last_skill_ok", True)
                status_hint = "completed successfully" if ok else "encountered an issue"
                skill_hint = {
                    "role": "system",
                    "content": (
                        f"[SKILL EXECUTION] The skill '{last_skill}' just {status_hint}. "
                        f"Its result is in your context as [SKILL RESULT: {last_skill}]. "
                        f"Narrate it naturally — as yourself, not as a tool output log."
                    ),
                }
                messages.insert(1, skill_hint)

            # 3. Invoke LLM Router with messages and watchdog
            router = self.container.get("llm_router")

            # Derive context-dependent parameters from state
            runtime_context = kwargs.get("context")
            if not isinstance(runtime_context, dict):
                runtime_context = {}
            desktop_cognitive_engine_required = bool(
                runtime_context.get("desktop_cognitive_engine_required", False)
                or runtime_context.get("cognitive_engine_required", False)
            )
            tier = state.response_modifiers.get(
                "model_tier", "tertiary" if is_background else "primary"
            )
            deep_handoff = (
                bool(state.response_modifiers.get("deep_handoff", False)) and not is_background
            )
            if proof_answer_run:
                tier = str(kwargs.get("prefer_tier") or "tertiary")
                deep_handoff = False
            soma_data = getattr(state, "soma", None)
            hardware = getattr(soma_data, "hardware", {}) or {}
            thermal_c = float(hardware.get("temperature", 0.0) or 0.0)
            cpu_usage = float(hardware.get("cpu_usage", 0.0) or 0.0)
            memory_pressure = None
            try:
                mem_monitor = self.container.get("memory_monitor", default=None)
                if mem_monitor is not None:
                    memory_pressure = getattr(mem_monitor, "pressure", None)
            except (OSError, ConnectionError, TimeoutError):
                memory_pressure = None
            if memory_pressure is None:
                try:
                    import psutil

                    memory_pressure = psutil.virtual_memory().percent
                except (ImportError, AttributeError, RuntimeError):
                    memory_pressure = 0.0

            # Affect-modulated generation parameters
            affect = getattr(state, "affect", None)
            curiosity = getattr(affect, "curiosity", 0.5) if affect else 0.5
            temp_mod = 0.8 + (curiosity * 0.4)  # 0.8–1.2 range based on curiosity
            depth_mod = 1.0
            if state.cognition.current_mode == CognitiveMode.DELIBERATE:
                depth_mod = 1.5

            token_budget = (
                int((6144 if deep_handoff else 4096) * depth_mod) if not is_background else 1024
            )
            generation_temperature, token_budget = self._apply_generation_sampling_bias(
                base_temperature=0.7 * temp_mod,
                token_budget=token_budget,
                biases=[
                    state.response_modifiers.get("sampling_bias"),
                    state.response_modifiers.get("imagination_sampling_bias"),
                    state.response_modifiers.get("bicameral_sampling_bias"),
                ],
            )
            # [STABILITY v55] Raised thermal from 85°C to 95°C (M-series
            # throttles at 100°C+) and memory pressure from 85% to 94%
            # (32B model normally uses 85-90% of 64GB).
            if thermal_c >= 95.0:
                logger.warning(
                    "🌡️ ResponseGeneration: thermal guard active (temp=%.1fC cpu=%.1f%% mem=%.1f%%). Downshifting tier/tokens.",
                    thermal_c,
                    cpu_usage,
                    float(memory_pressure or 0.0),
                )
                tier = "tertiary"
                deep_handoff = False
                token_budget = min(4096, max(256, int(token_budget * 0.7)))
                state.response_modifiers["thermal_guard"] = True
            elif float(memory_pressure or 0.0) >= 94.0:
                token_budget = min(4096, max(256, int(token_budget * 0.8)))
                state.response_modifiers["thermal_guard"] = True
            else:
                state.response_modifiers["thermal_guard"] = False

            try:
                request_timeout = self._request_timeout(
                    is_background=is_background,
                    deep_handoff=deep_handoff,
                )
                think_coro = router.think(
                        messages=messages,
                        priority=1.0 if not is_background else 0.5,
                        origin=f"response_generation_{origin}",
                        purpose="reply" if not is_background else "background",
                        prefer_tier=tier,
                        is_background=is_background,
                        protected_foreground_lane=bool(not is_background and not proof_answer_run),
                        foreground_request=not is_background,
                        deep_handoff=deep_handoff,
                        allow_cloud_fallback=False,
                        cognitive_engine_required=bool(
                            runtime_context.get("cognitive_engine_required", False)
                        ),
                        desktop_cognitive_engine_required=bool(
                            runtime_context.get("desktop_cognitive_engine_required", False)
                            or runtime_context.get("cognitive_engine_required", False)
                        ),
                        live_runtime_payload_required=bool(
                            runtime_context.get("live_runtime_payload_required", False)
                        ),
                        visible_user_message=str(
                            runtime_context.get("visible_user_message") or objective or ""
                        ),
                        recent_conversation_context=str(
                            runtime_context.get("recent_conversation_context") or ""
                        ),
                        recent_context_needed=bool(
                            runtime_context.get("recent_context_needed", False)
                        ),
                        allow_mesh_cognition=bool(
                            runtime_context.get("allow_mesh_cognition", True)
                        ),
                        skip_runtime_payload=bool(
                            runtime_context.get("skip_runtime_payload", False)
                        ),
                        disable_prompt_cache=bool(
                            runtime_context.get("disable_prompt_cache", False)
                        ),
                        clear_prompt_cache=bool(
                            runtime_context.get("clear_prompt_cache", False)
                        ),
                        soma=soma_data,
                        state=state,
                        temperature=generation_temperature,
                        max_tokens=token_budget,
                        timeout=request_timeout,
                )
                response_text = await asyncio.wait_for(think_coro, timeout=request_timeout + 4.0)

                shape_repaired = False
                if not is_background and not is_test_run:
                    response_text, shape_repaired, shape_repair_reasons = (
                        self._repair_substantive_instruction_shape_miss(objective, response_text)
                    )
                    if shape_repaired:
                        logger.info(
                            "🛡️ ResponseGeneration repaired instruction shape locally before critique (%s).",
                            ",".join(shape_repair_reasons) or "unknown",
                        )

                # System 2 internal critique layer to verify logical correctness
                try:
                    from core.brain.reasoning_strategies import ReasoningStrategies
                    async def _raw_generate(p, **kw):
                        return await router.think(p, **kw)
                    strategies = ReasoningStrategies(_raw_generate)
                    if (
                        not desktop_cognitive_engine_required
                        and not shape_repaired
                        and strategies._is_logical_check(objective)
                    ):
                        logger.info("⚡ [Critique] Running System 2 self-critique on response...")
                        critique_response = await strategies._self_critique(objective, response_text, origin=origin)
                        if critique_response and critique_response != response_text:
                            logger.info("⚡ [Critique] Self-critique corrected the generated response!")
                            response_text = critique_response
                except (ImportError, AttributeError, TypeError, ValueError, LookupError, RuntimeError, NameError, SyntaxError, asyncio.TimeoutError) as critique_exc:
                    logger.warning("Failed to run System 2 self-critique: %s", critique_exc)

                # ComposerNode: Structural Refinement
                composer = self.container.get("composer_node", default=None)
                if composer and hasattr(composer, "refine"):
                    logger.debug("🎨 [Composer] Refining response structure...")
                    response_text = await composer.refine(response_text, objective=objective)

            except TimeoutError:
                logger.error(
                    "🛑 ResponseGeneration Phase TIMEOUT (%.0fs). Logic took too long.",
                    request_timeout + 4.0,
                )
                # [STABILITY v55] Don't inject a robotic timeout message into
                # working memory.  Return state unchanged (no response text)
                # so the Kernel reports empty and chat.py fires the protected
                # foreground lane as a rescue rather than showing
                # "My cognitive process timed out" to the user.
                return state

            # Handle None response from router.think()
            if response_text is None:
                logger.debug("💭 ResponseGeneration: LLM returned None. Skipping this tick.")
                return state
            if not is_background:
                if (
                    origin != "test"
                    and not os.environ.get("AURA_AGI_MAX_TASKS")
                    and not os.environ.get("AURA_TESTING")
                    and not os.environ.get("AURA_PROOF_RUN")
                ):
                    reliability = assess_user_facing_reply(objective, response_text)
                    if reliability.retryable:
                        repaired_text, repaired_shape, repair_reasons = (
                            self._repair_substantive_instruction_shape_miss(objective, response_text)
                        )
                        if repaired_shape:
                            logger.info(
                                "🛡️ ResponseGeneration repaired instruction shape locally after refinement (%s).",
                                ",".join(repair_reasons) or "unknown",
                            )
                            response_text = repaired_text
                            reliability = assess_user_facing_reply(objective, response_text)
                        reliability_reasons = set(reliability.reasons or ())
                        response_text_s = str(response_text or "").strip()
                        if (
                            reliability_reasons
                            and reliability_reasons.issubset(
                                _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS
                            )
                            and len(response_text_s) >= 48
                            and len(response_text_s.split()) >= 8
                        ):
                            logger.warning(
                                "🛡️ ResponseGeneration kept repairable foreground draft for final reply repair (%s, len=%d).",
                                ",".join(reliability.reasons) or "unknown",
                                len(response_text_s),
                            )
                        else:
                            logger.warning(
                                "🛡️ ResponseGeneration rejected unsafe user-facing draft (%s, len=%d).",
                                ",".join(reliability.reasons) or "unknown",
                                len(str(response_text or "")),
                            )
                            return state

            # 4. Defensive Hardening: JSON Repair & Proactive Extraction
            content = response_text
            action = None

            # PROACTIVE JSON EXTRACTION:
            # If the response contains a JSON-like structure with "content", extract it
            # regardless of the current mode. This prevents raw "philosophical_insight"
            # JSON from leaking into the UI if the LLM slips into JSON mode accidentally.
            if "{" in response_text and '"content":' in response_text:
                try:
                    import re

                    # Find the outermost { ... } block
                    match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                    if match:
                        potential_json = match.group(1)
                        from core.utils.json_utils import extract_json

                        data = extract_json(potential_json)
                        if isinstance(data, dict):
                            # Try both "content" and deeper "response": {"content": ...}
                            ext_content = data.get("content")
                            if (
                                not ext_content
                                and "response" in data
                                and isinstance(data["response"], dict)
                            ):
                                ext_content = data["response"].get("content")
                                if not action:
                                    action = data["response"].get("action")

                            if ext_content:
                                logger.info(
                                    "🛡️ [HARDENING] Proactively extracted content from accidental JSON block."
                                )
                                content = ext_content
                                if not action:
                                    action = data.get("action")
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_response_generation_degradation(
                        e,
                        action="continued with raw response after proactive JSON extraction failed",
                    )
                    logger.debug("Proactive JSON extraction failed (normal for non-JSON): %s", e)

            # Mode-specific validation for DELIBERATE reasoning
            if state.cognition.current_mode == CognitiveMode.DELIBERATE and not action:
                from core.llm_guard import validate_json_response

                success, obj, err = validate_json_response(response_text, expected_keys=["content"])
                if success:
                    content = obj["content"]
                    action = obj.get("action")
                else:
                    # Robust fallback: if LLM failed to return valid JSON, or returned plain text instead of JSON
                    import json
                    import re

                    is_json_like = response_text.strip().startswith("{") and response_text.strip().endswith("}")
                    if not is_json_like:
                        logger.info("🛡️ [HARDENING] DELIBERATE mode validation failed, but output is not JSON-like. Reverting to plain text.")
                        content = response_text
                    else:
                        # It is JSON-like but parsing or validation failed. Let's see if we can extract "content" via regex
                        content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text)
                        if content_match:
                            try:
                                content = json.loads(f'"{content_match.group(1)}"')
                                logger.info("🛡️ [HARDENING] DELIBERATE mode validation recovered content from JSON-like response using regex.")
                            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                                content = response_text
                        else:
                            content = response_text

            # Proactive XML Answer Tag formatting guard:
            # If the user prompt or system instruction requires XML answer tagging (e.g. "<answer>"),
            # but the model's generated text doesn't contain a valid "<answer>...</answer>" tag:
            # We use a robust regex parsing cascade to extract the plain-text answer from the model's explanation,
            # and automatically wrap it in a clean "<answer>...</answer>" block at the end of the text.
            lower_objective = objective.lower() if objective else ""
            lower_response = content.lower() if content else ""
            if ("<answer>" in lower_objective or "answer_format" in kwargs) and content and not "<answer>" in lower_response:
                import re
                extracted_ans = None
                
                # 1. Look for markdown bolded final answer (e.g. **Answer**: 5 or **Final Answer**: Alice)
                match = re.search(r"\*\*(?:final\s+)?answer\*\*:\s*([^\n]+)", content, re.IGNORECASE)
                if match:
                    extracted_ans = match.group(1).strip()
                
                # 2. Look for plain-text answer prefix (e.g. Final Answer: same)
                if not extracted_ans:
                    match = re.search(r"(?:final\s+)?answer:\s*([^\n]+)", content, re.IGNORECASE)
                    if match:
                        extracted_ans = match.group(1).strip()
                
                # 3. Look for concluding "therefore, the answer is X"
                if not extracted_ans:
                    match = re.search(r"(?:therefore|thus|hence|so),\s*(?:the\s+)?answer\s+(?:is|must\s+be)\s+([^\n.]+)", content, re.IGNORECASE)
                    if match:
                        extracted_ans = match.group(1).strip()
                
                # 4. If the response is short enough (e.g. under 60 chars) and has no explanation, use the whole text
                if not extracted_ans and len(content.strip()) < 60 and not any(k in lower_response for k in ("because", "since", "as we", "therefore")):
                    extracted_ans = content.strip()
                
                if extracted_ans:
                    # Clean trailing punctuation
                    extracted_ans = extracted_ans.rstrip(".,;:!?* ")
                    # Wrap and append
                    content += f"\n\n<answer>{extracted_ans}</answer>"
                    logger.info("🛡️ [HARDENING] Auto-corrected and wrapped extracted answer '%s' in XML tags.", extracted_ans)

            # 5. Executive Guard — real-time identity alignment
            guard = get_executive_guard()
            cleaned_response, was_corrected, violations = guard.align(content)
            if was_corrected:
                logger.info(
                    "🛡️ ExecutiveGuard corrected %d violation(s) in LLM output.", len(violations)
                )

            async def _retry_dialogue(repair_block: str) -> str:
                retry_messages = [dict(msg) for msg in messages]
                if retry_messages and retry_messages[0].get("role") == "system":
                    retry_messages[0]["content"] = (
                        f"{repair_block}\n\n{retry_messages[0]['content']}"
                    )
                else:
                    retry_messages.insert(0, {"role": "system", "content": repair_block})

                retry_timeout = min(35.0, max(12.0, request_timeout * 0.5))
                retried = await router.think(
                    messages=retry_messages,
                    priority=1.0 if not is_background else 0.5,
                    origin=f"response_generation_{origin}",
                    purpose="reply" if not is_background else "background",
                    prefer_tier=tier,
                    is_background=is_background,
                    protected_foreground_lane=not is_background,
                    deep_handoff=deep_handoff,
                    allow_cloud_fallback=False,
                    soma=soma_data,
                    state=state,
                    temperature=generation_temperature,
                    max_tokens=token_budget,
                    timeout=retry_timeout,
                )
                retried_text = str(retried or "").strip()
                if guard and retried_text:
                    retried_text, _, _ = guard.align(retried_text)
                return retried_text

            (
                cleaned_response,
                dialogue_validation,
                dialogue_retried,
            ) = await enforce_dialogue_contract(
                cleaned_response,
                contract,
                retry_generate=_retry_dialogue if not is_background and not is_test_run else None,
                state=state,
            )
            state.response_modifiers["dialogue_validation"] = dialogue_validation.to_dict()
            if dialogue_retried:
                logger.info("🗣️ ResponseGeneration: retried draft to satisfy dialogue contract.")

            # 6. Clean response
            cleaned_response = self._clean_response(
                cleaned_response,
                state,
                allow_mumbling=is_background,
            )

            # 6b. SUBSTRATE VOICE: Shape the response — enforce the profile
            # The substrate compiled constraints. Now enforce them on the output.
            _shaped_messages = None
            if _sve and _speech_profile and cleaned_response and not is_test_run:
                try:
                    shaped = _sve.shape_response(cleaned_response)
                    if isinstance(shaped, list):
                        # Multi-message: use first as primary, queue rest as follow-ups
                        cleaned_response = shaped[0]
                        _shaped_messages = shaped[1:]
                        logger.debug(
                            "🗣️ [SubstrateVoice] Shaped into %d messages",
                            len(shaped),
                        )
                    else:
                        cleaned_response = shaped
                except (RuntimeError, AttributeError, TypeError, ValueError) as _shape_exc:
                    _record_response_generation_degradation(
                        _shape_exc,
                        action="continued with cleaned response after substrate voice shaping failed",
                        severity="error",
                    )
                    logger.debug("ResponseShaper failed (using raw): %s", _shape_exc)

            if not is_background and cleaned_response and not is_test_run:
                repaired_response, repaired_shape, repair_reasons = (
                    self._repair_substantive_instruction_shape_miss(objective, cleaned_response)
                )
                if repaired_shape:
                    cleaned_response = repaired_response
                    state.response_modifiers["post_voice_shape_repair"] = {
                        "reasons": list(repair_reasons),
                        "method": "deterministic_instruction_shape",
                    }
                    logger.info(
                        "🛡️ ResponseGeneration repaired instruction shape after voice shaping (%s).",
                        ",".join(repair_reasons) or "unknown",
                    )

            # 6c. Skip emission for background tasks if they produced no meaningful content
            if is_background and not cleaned_response:
                return state

            # 7. Derive new state with the response
            new_state = state.derive("response_generation")
            new_state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": str(cleaned_response),
                    "timestamp": float(time.time()),
                    "mode": str(state.cognition.current_mode.value),
                    "objective_ref": "".join(
                        [str(objective)[i] for i in range(min(50, len(str(objective))))]
                    ),
                    "action": action,
                }
            )
            new_state.cognition.last_thought_at = time.time()
            # Set last_response so RepairPhase can inspect and clean it
            new_state.cognition.last_response = str(cleaned_response)

            # SharedGround callback detection — fire-and-forget background task
            # Detect when Aura's response references an established shared-ground entry
            # and record the callback so salience scores accumulate over time.
            if cleaned_response:
                get_task_tracker().create_task(record_shared_ground_callbacks(cleaned_response))

            # ── Conversational Intelligence Updates (fire-and-forget) ──
            # Update all person-specific models from this exchange.
            if cleaned_response and objective:
                get_task_tracker().create_task(
                    update_conversational_intelligence(str(objective), str(cleaned_response), state)
                )

            # ── SUBSTRATE VOICE: Follow-up decision ──────────────────────
            # Ask the substrate if a follow-up is warranted. This is organic,
            # not forced — driven by actual curiosity/engagement/dopamine.
            if _sve and _speech_profile and not is_background and cleaned_response and not is_test_run:
                try:
                    history = [
                        {"role": m.get("role", ""), "content": str(m.get("content", ""))}
                        for m in (state.cognition.working_memory or [])[-8:]
                    ]
                    fu_decision = _sve.decide_followup(
                        user_message=str(objective),
                        aura_response=str(cleaned_response),
                        state=state,
                        conversation_history=history,
                    )
                    if fu_decision.should_followup:
                        # Store decision in state for the orchestrator to pick up
                        new_state.response_modifiers["pending_followup"] = {
                            "type": fu_decision.followup_type,
                            "delay": fu_decision.delay_seconds,
                            "word_budget": fu_decision.word_budget,
                            "context_hint": fu_decision.context_hint,
                            "reason": fu_decision.reason,
                        }
                        logger.info(
                            "💬 [SubstrateVoice] Follow-up queued: %s in %.1fs",
                            fu_decision.followup_type,
                            fu_decision.delay_seconds,
                        )

                    # Queue additional shaped messages (from multi-message split)
                    if _shaped_messages:
                        new_state.response_modifiers["queued_messages"] = _shaped_messages
                except (OSError, ConnectionError, TimeoutError) as _fu_exc:
                    _record_response_generation_degradation(
                        _fu_exc,
                        action="returned primary response without queuing substrate follow-up",
                    )
                    logger.debug("Follow-up decision failed: %s", _fu_exc)

            return new_state

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_response_generation_degradation(
                e,
                action="returned prior state unchanged after response generation phase failed",
                severity="error",
            )
            logger.error("❌ ResponseGeneration: LLM call failed: %s", e, exc_info=True)
            return state

    def _clean_response(
        self,
        text: str,
        state: AuraState | None = None,
        *,
        allow_mumbling: bool = False,
    ) -> str:
        """Strip tags and assistant-isms without leaking internal thought into chat."""
        import re

        mumbling = ""
        # Internal Monologue Spillage ("Mumbling")
        exp_state = "neutral"
        load = "normal"
        if state is not None and hasattr(state, "soma"):
            s_val = state.soma
            if s_val is not None:
                exp = getattr(s_val, "expressive", {}) or {}
                exp_state = exp.get("current_expression", "neutral")
                load = exp.get("cognitive_load", "normal")

            if allow_mumbling and (
                exp_state in ("contemplative", "anxious", "fatigued") or load == "high"
            ):
                # Extract the thought block before we strip it
                thought_match = re.search(r"<thought>(.*?)</thought>", text, flags=re.DOTALL)
                if thought_match:
                    thought_content = thought_match.group(1).strip()
                    # Just grab the last sentence or first few words to mumble
                    snippets = [s.strip() for s in thought_content.split(".") if s.strip()]
                    if snippets:
                        snippet = snippets[-1] if len(snippets) > 1 else snippets[0]
                        # Cap length
                        if len(snippet) > 80:
                            # vResilience: Workaround for str indexing/slice limitations
                            snippet = "".join([snippet[i] for i in range(77)]) + "..."
                        mumbling = f"*...{snippet.lower()}...*\n\n"

        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        text = re.sub(r"^Aura:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)

        # 🧠 COGNITIVE WIRING: Affect-Gated Prompt Hunting
        # Instead of using a 'fake band-aid' system prompt to forbid questions,
        # we wire this behavior directly into her mind. If the LLM reflexively
        # appends a trailing question, she must ACTUALLY be curious to ask it.
        if state is not None and hasattr(state, "affect"):
            curiosity = getattr(state.affect, "curiosity", 0.5)
            if curiosity < 0.70:
                # She is not curious enough to warrant a reflexive follow-up question.
                # Strip the trailing question (e.g. 'What do you think?').
                # This matches the last sentence if it ends in a question mark.
                text = re.sub(r"(?<=[.!?])\s+[A-Z][^.!?]*\?\s*$", "", text)
                text = text.strip()

        # Apply aggressive centralized scrubbing
        text = strip_meta_commentary(text)
        text = stabilize_user_facing_response(
            text,
            getattr(getattr(state, "cognition", None), "current_objective", "") or "",
        )

        return (mumbling + text.strip()).strip()
