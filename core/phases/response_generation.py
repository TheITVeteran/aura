"""Response Generation Phase for Aura's Cognitive Pipeline."""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from core.brain.live_mind_contract import normalize_live_mind_surface_control_receipt
from core.brain.llm.context_assembler import ContextAssembler
from core.container import ServiceContainer
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
from core.runtime.connectivity import render_connectivity_prompt_block
from core.runtime.conversation_support import (
    schedule_conversation_support_updates,
)
from core.runtime.desktop_task_contract import desktop_task_action_sentence
from core.runtime.errors import record_degradation
from core.synthesis import stabilize_user_facing_response, strip_meta_commentary

from ..state.aura_state import AuraState, CognitiveMode
from . import BasePhase

logger = logging.getLogger(__name__)

# Explicit tool compositions (e.g. composing an outbound message to ANOTHER AI
# in a web-interlocutor conversation) are non-user-facing, so they correctly
# skip the user-facing reply gates — but they are NOT autonomous background
# chatter either. They were explicitly requested and MUST produce real cortex
# output, so they are exempt from the idle/background suppression that protects
# against autonomous thought leaking. This is what lets her actually THINK a
# reply to ChatGPT instead of falling back to a canned default.
_EXPLICIT_TOOL_COMPOSITION_ORIGINS = frozenset({"web_interlocutor"})

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
_TOOL_FALSE_INABILITY_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:can't|cannot|can\s+not|am\s+unable\s+to|don't\s+have\s+access\s+to|"
    r"do\s+not\s+have\s+access\s+to|lack(?:\s+the)?\s+ability\s+to|can't\s+directly)|"
    r"i'm\s+unable\s+to|i\s+am\s+unable\s+to"
    r")\b[^.\n]{0,180}\b(?:"
    r"browse|search|web|internet|look\s+up|fetch|access|open|visit"
    r")\b",
    re.IGNORECASE,
)
_SEARCH_SKILL_NAMES = {
    "free_search",
    "grounded_search",
    "search_web",
    "web_search",
}

_SOURCE_DEFINITION_TAIL_RE = re.compile(
    r"\b(?:what|who|where|when|how)\s+"
    r"(?:the\s+)?(?:source|page|site|article|[A-Z][A-Za-z0-9 .&'_-]{1,80})\s+"
    r"(?:says?|calls?|defines?|describes?|explains?)\s+"
    r"[^.?!;]{1,180}",
    re.IGNORECASE,
)


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
        self._last_reasoning_receipt: dict[str, Any] | None = None

    @staticmethod
    def _compact_prompt_payload(value: Any, *, limit: int = 3000) -> str:
        """Render runtime grounding payloads without letting them dominate the prompt."""

        try:
            rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        except (TypeError, ValueError, OverflowError):
            rendered = str(value)
        rendered = rendered.strip()
        if len(rendered) <= limit:
            return rendered
        return f"{rendered[: max(0, limit - 16)]}...[truncated]"

    @classmethod
    def _append_system_block(
        cls,
        messages: list[dict[str, Any]],
        title: str,
        body: str,
    ) -> None:
        body = str(body or "").strip()
        if not body:
            return
        block = f"## {title}\n{body}"
        if messages and str(messages[0].get("role", "") or "").strip().lower() == "system":
            messages[0]["content"] = f"{str(messages[0].get('content', '')).rstrip()}\n\n{block}"
        else:
            messages.insert(0, {"role": "system", "content": block})

    @staticmethod
    def _sanitize_grounding_payload(payload: Any) -> dict[str, Any]:
        """Keep tool evidence useful without flooding prompts or memory."""

        if not isinstance(payload, dict):
            return {"ok": False, "result": str(payload or "")[:1200]}

        compact: dict[str, Any] = {}
        scalar_keys = (
            "ok",
            "query",
            "answer",
            "summary",
            "message",
            "source",
            "url",
            "title",
            "provenance",
            "offline_fallback",
            "web_error",
            "confidence",
            "count",
            "mode",
        )
        for key in scalar_keys:
            if key not in payload:
                continue
            value = payload.get(key)
            if isinstance(value, str):
                compact[key] = value[:4000]
            elif isinstance(value, (bool, int, float)) or value is None:
                compact[key] = value
            else:
                compact[key] = str(value)[:1000]

        def _compact_items(items: Any, *, limit: int = 5) -> list[dict[str, Any]]:
            compact_items: list[dict[str, Any]] = []
            if not isinstance(items, list):
                return compact_items
            for item in items[:limit]:
                if not isinstance(item, dict):
                    continue
                compact_items.append(
                    {
                        "title": str(item.get("title") or "")[:300],
                        "url": str(item.get("url") or item.get("source") or "")[:500],
                        "snippet": str(
                            item.get("snippet")
                            or item.get("content")
                            or item.get("text")
                            or ""
                        )[:1200],
                    }
                )
            return [item for item in compact_items if any(item.values())]

        results = _compact_items(payload.get("results"), limit=5)
        if results:
            compact["results"] = results
        citations = _compact_items(payload.get("citations"), limit=5)
        if citations:
            compact["citations"] = citations
        chunks = _compact_items(payload.get("chunks"), limit=3)
        if chunks:
            compact["chunks"] = chunks

        content = str(payload.get("content") or payload.get("result") or "").strip()
        if content and "content" not in compact:
            compact["content"] = content[:6000]
        return compact

    @classmethod
    def _render_skill_result_block(
        cls,
        *,
        skill_name: str,
        payload: dict[str, Any],
    ) -> str:
        status = "✅" if payload.get("ok") else "⚠️"
        parts: list[str] = []
        for key in ("query", "answer", "summary", "message", "title", "source", "url", "content"):
            value = str(payload.get(key) or "").strip()
            if value:
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {value[:1600]}")
        results = payload.get("results")
        if isinstance(results, list) and results:
            rendered_results = []
            for idx, item in enumerate(results[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                rendered_results.append(
                    f"{idx}. {title or 'Untitled'}"
                    + (f" — {url}" if url else "")
                    + (f" — {snippet[:500]}" if snippet else "")
                )
            if rendered_results:
                parts.append("Results:\n" + "\n".join(rendered_results))
        if not parts:
            parts.append(json.dumps(payload, ensure_ascii=True, default=str)[:2400])
        return f"[SKILL RESULT: {skill_name}] {status} " + "\n".join(parts)

    @staticmethod
    def _first_sentence(text: str, *, fallback: str = "") -> str:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return fallback
        match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
        return (match.group(1) if match else cleaned).strip()

    @classmethod
    def _successful_required_search_payload(
        cls,
        state: AuraState,
        contract: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        if not getattr(contract, "requires_search", False):
            return None
        if state.response_modifiers.get("last_skill_ok") is not True:
            return None
        skill_name = str(state.response_modifiers.get("last_skill_run") or "").strip()
        if skill_name not in _SEARCH_SKILL_NAMES:
            return None
        payload = state.response_modifiers.get("last_skill_result_payload")
        if not isinstance(payload, dict):
            return None
        return skill_name, payload

    @classmethod
    def _render_required_search_answer_from_payload(
        cls,
        *,
        payload: dict[str, Any],
    ) -> str:
        results = payload.get("results")
        first_result = next(
            (item for item in results if isinstance(item, dict)),
            {},
        ) if isinstance(results, list) else {}
        title = str(
            first_result.get("title")
            or payload.get("title")
            or payload.get("source_title")
            or ""
        ).strip()
        url = str(
            first_result.get("url")
            or payload.get("url")
            or payload.get("source")
            or ""
        ).strip()
        evidence_text = str(
            payload.get("answer")
            or payload.get("summary")
            or first_result.get("snippet")
            or payload.get("message")
            or ""
        ).strip()
        if not evidence_text:
            evidence_text = "The search returned evidence, but the result did not include a usable snippet."
        sentence = cls._first_sentence(evidence_text, fallback=evidence_text)

        if title and url:
            return f"I found {title}. {sentence} Source: {url}"
        if title:
            return f"I found {title}. {sentence}"
        if url:
            return f"I found a relevant source. {sentence} Source: {url}"
        return sentence

    @classmethod
    def _repair_false_required_tool_inability(
        cls,
        *,
        state: AuraState,
        contract: Any,
        response_text: str,
    ) -> str:
        hit = cls._successful_required_search_payload(state, contract)
        if not hit:
            return response_text
        skill_name, payload = hit
        if not _TOOL_FALSE_INABILITY_RE.search(str(response_text or "")):
            return response_text
        repaired = cls._render_required_search_answer_from_payload(payload=payload)
        state.response_modifiers["required_tool_false_inability_repaired"] = {
            "skill": skill_name,
            "method": "deterministic_grounded_evidence",
        }
        logger.warning(
            "🛡️ ResponseGeneration replaced false %s inability after successful required evidence.",
            skill_name,
        )
        return repaired

    async def _execute_required_search_evidence(
        self,
        *,
        state: AuraState,
        objective: str,
        contract: Any,
        origin: str,
        runtime_context: dict[str, Any],
    ) -> bool:
        """Run mandatory read-only search before the model narrates a search turn."""

        if not getattr(contract, "requires_search", False):
            return False
        if getattr(contract, "tool_evidence_available", False):
            return False

        query = self._clean_required_search_query(
            str(getattr(contract, "search_query", "") or objective or "").strip()
        )
        if not query:
            return False

        cap = self.container.get("capability_engine", default=None)
        if cap is None:
            try:
                cap = ServiceContainer.get("capability_engine", default=None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                cap = None
        if cap is None or not hasattr(cap, "execute"):
            logger.warning(
                "🔎 ResponseGeneration: required search evidence skipped because capability_engine is unavailable."
            )
            return False

        skill_name = "web_search"
        matched = state.response_modifiers.get("matched_skills") or []
        if isinstance(matched, str):
            matched = [matched]
        for candidate in matched:
            resolved = candidate
            if hasattr(cap, "resolve_skill_name"):
                try:
                    resolved = cap.resolve_skill_name(str(candidate))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    resolved = str(candidate)
            if str(resolved) in {"web_search", "search_web", "free_search", "grounded_search"}:
                skill_name = str(resolved)
                break

        context = {
            "origin": origin,
            "source": origin,
            "route": "response_generation.required_search_evidence",
            "objective": objective,
            "message": objective,
            "user_requested_action": True,
            "risk_level": "low",
            "effect_scope": "read_only_external_io",
            "skill_name": skill_name,
            "tool_name": skill_name,
            "foreground_request": True,
            "desktop_cognitive_engine_required": bool(
                runtime_context.get("desktop_cognitive_engine_required")
                or runtime_context.get("cognitive_engine_required")
            ),
        }

        try:
            result = await asyncio.wait_for(
                cap.execute(
                    skill_name,
                    {
                        "query": query,
                        "num_results": 5,
                        "deep": False,
                        "retain": True,
                    },
                    context,
                ),
                timeout=35.0,
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_response_generation_degradation(
                exc,
                action="continued search turn after mandatory search evidence execution failed",
                severity="warning",
            )
            return False

        payload = self._sanitize_grounding_payload(result)
        ok = bool(payload.get("ok"))
        payload.setdefault("query", query)
        payload.setdefault("ok", ok)

        state.response_modifiers["last_skill_run"] = skill_name
        state.response_modifiers["last_skill_ok"] = ok
        state.response_modifiers["last_skill_turn_marker"] = state.response_modifiers.get(
            "evidence_turn_marker"
        )
        state.response_modifiers["last_skill_result_payload"] = payload
        state.response_modifiers["required_search_evidence_executed"] = {
            "skill": skill_name,
            "ok": ok,
            "query": query[:240],
        }
        state.cognition.working_memory.append(
            {
                "role": "system",
                "content": self._render_skill_result_block(
                    skill_name=skill_name,
                    payload=payload,
                ),
                "metadata": {
                    "type": "skill_result",
                    "skill": skill_name,
                    "ok": ok,
                    "query": query[:240],
                    "turn_marker": state.response_modifiers.get("evidence_turn_marker"),
                },
                "timestamp": time.time(),
            }
        )
        try:
            state.cognition.trim_working_memory()
        except AttributeError:
            pass
        logger.info(
            "🔎 ResponseGeneration: executed required search evidence via %s (ok=%s query=%s).",
            skill_name,
            ok,
            query[:120],
        )
        return True

    @staticmethod
    def _clean_required_search_query(query: str) -> str:
        """Remove response-format instructions without losing source semantics.

        Live regression: "Tell me the source title and what NASA says Europa is"
        was reduced to "one current NASA page about Europa". That made the
        search/cache layer satisfy the wrong task with a Clipper article instead
        of a definition-bearing NASA source. The cleaner may drop pure formatting
        instructions ("source title only"), but source-definition clauses stay in
        the retrieval query because they change what evidence is relevant.
        """

        raw = str(query or "").strip()
        if not raw:
            return ""
        repair_match = re.search(
            r"(?is)\boriginal\s+user\s+request\s*:\s*(.*?)"
            r"(?:\n\s*\n\s*rejected\s+draft\s+for\s+avoidance\s+only\s*:|$)",
            raw,
        )
        if repair_match:
            raw = repair_match.group(1).strip()
            if not raw:
                return ""
        source_definition_tails = [
            " ".join(match.group(0).strip(" .?!,:;").split())
            for match in _SOURCE_DEFINITION_TAIL_RE.finditer(raw)
        ]
        cleaned = re.sub(
            r"^\s*(?:please\s+)?(?:search|look\s+up|find)\s+"
            r"(?:(?:the\s+)?(?:web|internet)\s+)?(?:for\s+)?",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:and\s+tell me|then\s+tell me|and\s+answer|then\s+answer|"
            r"and\s+give me|then\s+give me)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .?!,:;")
        cleaned = re.sub(
            r"(?:^|[.?!]\s*)(?:tell\s+me|include|give\s+me)\s+"
            r"(?:the\s+)?source\s+title(?:\s+only)?(?:\s+and)?\s*",
            ". ",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .?!,:;")
        for tail in source_definition_tails:
            if tail and tail.lower() not in cleaned.lower():
                cleaned = f"{cleaned}. {tail}" if cleaned else tail
        return re.sub(r"\s+", " ", cleaned).strip(" .?!,:;")

    @classmethod
    def _inject_live_runtime_grounding(
        cls,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
    ) -> None:
        """Make live desktop mind/body/tool context visible to the full phase path.

        The compact desktop router path already receives these fields directly.
        The full response-generation phase must receive them too, otherwise a
        required desktop turn can be technically routed through CognitiveEngine
        while the model only sees a generic prompt.
        """

        live_mind = runtime_context.get("live_mind_context")
        if isinstance(live_mind, dict) and live_mind:
            compact_mind = {
                "required_for_live_desktop": live_mind.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind.get("must_answer_from_full_mind_path"),
                "required_subsystems_ok": live_mind.get("required_subsystems_ok"),
                "required_subsystems": live_mind.get("required_subsystems"),
                "lane": live_mind.get("lane"),
                "voice": live_mind.get("voice"),
                "substrate": live_mind.get("substrate"),
                "timescale_reconciliation": live_mind.get("timescale_reconciliation"),
                "governance": live_mind.get("governance"),
            }
            contract = str(runtime_context.get("mind_context_contract") or "").strip()
            timescale_block = ""
            timescale = live_mind.get("timescale_reconciliation")
            if isinstance(timescale, dict) and timescale:
                try:
                    from core.runtime.timescale_bridge import render_timescale_prompt_block

                    timescale_block = render_timescale_prompt_block(timescale)
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_response_generation_degradation(
                        exc,
                        action="continued response generation without timescale reconciliation block",
                    )
            cls._append_system_block(
                messages,
                "LIVE MIND CONTEXT",
                (
                    f"{cls._compact_prompt_payload(compact_mind, limit=3200)}\n"
                    "This is causal grounding for the reply, not text to recite. "
                    "Use memory, current state, substrate, governance, and the live lane as one context. "
                    "Do not answer as a generic assistant persona."
                    + (f"\n{timescale_block}" if timescale_block else "")
                    + (f"\n{contract}" if contract else "")
                ),
            )

        speech_frame = runtime_context.get("live_speech_grounding_frame")
        if isinstance(speech_frame, dict) and speech_frame:
            compact_frame = {
                key: speech_frame.get(key)
                for key in (
                    "attention_focus",
                    "dominant_action",
                    "dominant_emotions",
                    "interests",
                    "mood",
                    "tone",
                    "requires_explicit_live_grounding",
                )
                if speech_frame.get(key) not in (None, "", [], {})
            }
            if compact_frame:
                cls._append_system_block(
                    messages,
                    "LIVE SPEECH GROUNDING",
                    (
                        f"{cls._compact_prompt_payload(compact_frame, limit=1200)}\n"
                        "This frame is grounding, not prose to repeat. Convert it into ordinary speech only when it helps."
                    ),
                )

        cognitive_situation = runtime_context.get("cognitive_situation_frame")
        if isinstance(cognitive_situation, dict) and cognitive_situation:
            try:
                from core.brain.cognitive_situation import (
                    render_cognitive_situation_prompt_block,
                )

                situation_block = render_cognitive_situation_prompt_block(
                    cognitive_situation,
                    compact=True,
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_response_generation_degradation(
                    exc,
                    action="continued response generation without cognitive situation block",
                )
                situation_block = ""
            if situation_block:
                cls._append_system_block(
                    messages,
                    "COGNITIVE SITUATION FRAME",
                    situation_block,
                )

        evidence_blocks = (
            (
                "CONTEXT CHALLENGE EVIDENCE",
                runtime_context.get("contextual_relevance_evidence"),
                "Use this to avoid inventing prior context; answer from the actual recent thread.",
            ),
            (
                "CONVERSATION RECALL EVIDENCE",
                runtime_context.get("conversation_recall_evidence"),
                "Use this as source-of-truth memory for the current recall question.",
            ),
            (
                "DEEP MEMORY RECALL",
                runtime_context.get("deep_memory_context"),
                "Silent background recall from long-term memory; draw on it only where "
                "genuinely relevant, never recite it.",
            ),
            (
                "GOVERNED CAPABILITY INVENTORY EVIDENCE",
                runtime_context.get("grounded_capability_inventory_context"),
                "Use this for capability questions; do not claim execution without receipts.",
            ),
            (
                "EVIDENCE-BOUND SELF-CLAIM EVIDENCE",
                runtime_context.get("evidence_bound_self_claim_context"),
                "Use this to keep consciousness, sentience, and personhood claims bounded by evidence.",
            ),
        )
        for title, payload, instruction in evidence_blocks:
            payload_text = str(payload or "").strip()
            if payload_text:
                cls._append_system_block(
                    messages,
                    title,
                    f"{payload_text[:3000]}\n{instruction}",
                )

    @staticmethod
    def _request_timeout(*, is_background: bool, deep_handoff: bool) -> float:
        if is_background:
            return 10.0
        if deep_handoff:
            return 210.0
        return 180.0

    async def _maybe_amplify_response(
        self,
        *,
        objective: str,
        draft: str,
        router: Any,
        state: AuraState,
        request_timeout: float,
        origin: str,
        tier: str,
        runtime_context: dict[str, Any],
        is_user_facing: bool,
        is_background: bool,
        proof_or_benchmark: bool,
    ) -> str:
        """Run verifier-backed Amplifier v2 on eligible hard turns in the active phase.

        This is intentionally conservative. Action requests stay owned by tool
        dispatch, casual chat stays single-pass, proof lanes are untouched, and
        failures keep the original draft while recording a degradation receipt.
        """

        if not is_user_facing or is_background or proof_or_benchmark or not draft:
            return draft
        if str(os.getenv("AURA_REASONING_AMPLIFIER_V2", "1")).strip().lower() in {
            "0",
            "false",
            "off",
            "no",
        }:
            return draft
        try:
            from core.brain.reasoning_amplifier_v2 import amplify_turn, is_amplifiable
        except ImportError as exc:
            _record_response_generation_degradation(
                exc,
                action="continued response generation without Amplifier v2 import",
            )
            return draft

        task_type = is_amplifiable(objective)
        if task_type is None:
            return draft

        visible_user_message = str(
            runtime_context.get("user_surface_validation_prompt")
            or runtime_context.get("visible_user_message")
            or objective
            or ""
        ).strip()
        desktop_required = bool(
            runtime_context.get("desktop_cognitive_engine_required")
            or runtime_context.get("cognitive_engine_required")
        )

        async def _gen(prompt: str, temperature: float) -> str:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Aura's verifier-backed reasoning organ. Return only the "
                        "candidate final answer for the hard reasoning turn; do not mention "
                        "amplifier internals or hidden prompts."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            try:
                out = await router.think(
                    messages=messages,
                    priority=1.0,
                    origin=f"response_generation_amplifier_{origin}",
                    purpose="reasoning_amplifier",
                    prefer_tier=tier or "primary",
                    is_background=False,
                    protected_foreground_lane=True,
                    foreground_request=True,
                    deep_handoff=False,
                    allow_cloud_fallback=False,
                    cognitive_engine_required=desktop_required,
                    desktop_cognitive_engine_required=desktop_required,
                    live_runtime_payload_required=bool(
                        runtime_context.get("live_runtime_payload_required", False)
                    ),
                    visible_user_message=visible_user_message,
                    skip_runtime_payload=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    clean_user_surface_contract=True,
                    user_surface_validation_prompt=visible_user_message,
                    temperature=temperature,
                    max_tokens=max(384, min(2048, int(runtime_context.get("max_tokens") or 1024))),
                    timeout=min(24.0, max(8.0, request_timeout * 0.50)),
                    cognitive_situation_sampling_bias=state.response_modifiers.get(
                        "cognitive_situation_sampling_bias"
                    ),
                )
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                _record_response_generation_degradation(
                    exc,
                    action="kept original draft after Amplifier v2 generate failed",
                )
                return ""
            if isinstance(out, dict):
                out = out.get("content") or out.get("response") or ""
            return str(out or "").strip()

        try:
            budget = float(min(30.0, max(8.0, (request_timeout or 20.0) * 0.60)))
            result = await amplify_turn(
                objective,
                _gen,
                task_type=task_type,
                time_budget_s=budget,
                extra_context={
                    "live_response_phase": True,
                    "cognitive_situation_frame": state.response_modifiers.get(
                        "cognitive_situation_frame"
                    ),
                },
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_response_generation_degradation(
                exc,
                action="kept original draft after Amplifier v2 failed",
            )
            return draft

        receipt = result.receipt.to_dict()
        self._last_reasoning_receipt = receipt
        state.response_modifiers["reasoning_receipt"] = receipt
        state.response_modifiers["reasoning_amplifier_v2_active_phase"] = {
            "task_type": task_type,
            "verified": bool(result.verified),
            "confidence": float(result.confidence),
            "adopted": bool(result.verified and result.answer),
        }
        logger.info(
            "🧠 [AmplifyV2-active-phase] task=%s verified=%s conf=%.2f -> %s",
            task_type,
            result.verified,
            result.confidence,
            "adopted" if (result.verified and result.answer) else "kept draft",
        )
        if result.verified and result.answer and len(result.answer.strip()) >= 3:
            return result.answer.strip()
        return draft

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
            explicit_tool_composition = origin in _EXPLICIT_TOOL_COMPOSITION_ORIGINS
            if is_background and not is_test_run and not explicit_tool_composition:
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
            if not is_background and not is_test_run:
                search_executed = await self._execute_required_search_evidence(
                    state=state,
                    objective=objective,
                    contract=contract,
                    origin=origin,
                    runtime_context=kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {},
                )
                if search_executed:
                    messages = ContextAssembler.build_messages(state, objective)
                    contract = build_response_contract(
                        state,
                        objective,
                        is_user_facing=True,
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
            connectivity_block = render_connectivity_prompt_block(
                getattr(state, "response_modifiers", {}).get("connectivity")
                or getattr(getattr(state, "world", None), "facts", {}).get("connectivity")
            )
            if connectivity_block and messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{connectivity_block}"
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
                self._inject_live_runtime_grounding(messages, runtime_context)

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
            live_mind_generation_controls = runtime_context.get(
                "live_mind_generation_controls"
            )
            if not isinstance(live_mind_generation_controls, dict):
                live_mind_generation_controls = {}
            live_mind_controls_bound = bool(
                runtime_context.get("live_mind_controls_bound", False)
            )
            clean_user_surface_contract = bool(
                runtime_context.get("clean_user_surface_contract", False)
                or desktop_cognitive_engine_required
            )
            user_surface_validation_prompt = str(
                runtime_context.get("user_surface_validation_prompt")
                or runtime_context.get("visible_user_message")
                or objective
                or ""
            ).strip()
            runtime_fact_status_contract = bool(
                runtime_context.get("runtime_fact_status_contract", False)
                or runtime_context.get("grounded_runtime_status_contract", False)
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
                    from core.runtime import resource_psutil as psutil

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
                    state.response_modifiers.get("cognitive_situation_sampling_bias"),
                ],
            )
            # The caller owns the user-facing latency envelope.  Sampling and
            # cognitive biases may spend less of that budget, but they must not
            # multiply past it.  This keeps the full phase stack available on
            # live desktop turns without turning an ordinary follow-up into a
            # multi-minute local generation.
            requested_token_cap = runtime_context.get("max_tokens")
            if requested_token_cap is not None:
                try:
                    token_budget = min(token_budget, max(64, int(requested_token_cap)))
                except (TypeError, ValueError, OverflowError):
                    logger.warning(
                        "ResponseGeneration ignored invalid caller token cap %r.",
                        requested_token_cap,
                    )
            if live_mind_controls_bound:
                generation_temperature = max(
                    0.10,
                    min(
                        1.15,
                        self._safe_bias_float(
                            live_mind_generation_controls.get("temperature"),
                            generation_temperature,
                        ),
                    ),
                )
            generation_top_p = max(
                0.05,
                min(
                    1.0,
                    self._safe_bias_float(
                        live_mind_generation_controls.get("top_p"),
                        0.90,
                    ),
                ),
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
                        visible_user_message=user_surface_validation_prompt,
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
                        memory_state_contract=bool(
                            runtime_context.get("memory_state_contract", False)
                        ),
                        runtime_fact_status_contract=runtime_fact_status_contract,
                        grounded_runtime_status_contract=runtime_fact_status_contract,
                        clean_user_surface_contract=clean_user_surface_contract,
                        user_surface_validation_prompt=user_surface_validation_prompt,
                        clean_user_surface_recurrent_loops=int(
                            live_mind_generation_controls.get(
                                "clean_user_surface_recurrent_loops", 1
                            )
                        ),
                        clean_user_surface_steering_alpha=self._safe_bias_float(
                            live_mind_generation_controls.get(
                                "clean_user_surface_steering_alpha"
                            ),
                            0.25,
                        ),
                        live_mind_controls_bound=live_mind_controls_bound,
                        live_mind_generation_controls=dict(
                            live_mind_generation_controls
                        ),
                        live_mind_snapshot_ready=bool(
                            runtime_context.get("live_mind_snapshot_ready", False)
                        ),
                        live_mind_required_subsystems_ok=bool(
                            runtime_context.get(
                                "live_mind_required_subsystems_ok", False
                            )
                        ),
                        cognitive_situation_sampling_bias=state.response_modifiers.get(
                            "cognitive_situation_sampling_bias"
                        ),
                        soma=soma_data,
                        state=state,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        max_tokens=token_budget,
                        timeout=request_timeout,
                )
                response_text = await asyncio.wait_for(think_coro, timeout=request_timeout + 4.0)

                shape_repaired = False
                if not is_background and not is_test_run:
                    response_text, shape_repaired, shape_repair_reasons = (
                        self._repair_substantive_instruction_shape_miss(
                            user_surface_validation_prompt, response_text
                        )
                    )
                    if shape_repaired:
                        logger.info(
                            "🛡️ ResponseGeneration repaired instruction shape locally before critique (%s).",
                            ",".join(shape_repair_reasons) or "unknown",
                        )

                response_text = await self._maybe_amplify_response(
                    objective=objective,
                    draft=response_text,
                    router=router,
                    state=state,
                    request_timeout=request_timeout,
                    origin=origin,
                    tier=tier,
                    runtime_context=runtime_context,
                    is_user_facing=not is_background and not is_test_run,
                    is_background=is_background,
                    proof_or_benchmark=proof_answer_run,
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
                except (ImportError, AttributeError, TypeError, ValueError, LookupError, RuntimeError, NameError, SyntaxError, TimeoutError) as critique_exc:
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
                required_tool_hit = self._successful_required_search_payload(state, contract)
                if required_tool_hit and not is_background:
                    skill_name, payload = required_tool_hit
                    response_text = self._render_required_search_answer_from_payload(
                        payload=payload,
                    )
                    state.response_modifiers["required_tool_timeout_repaired"] = {
                        "skill": skill_name,
                        "method": "deterministic_grounded_evidence",
                    }
                    logger.warning(
                        "🛡️ ResponseGeneration answered from successful %s evidence after Cortex timeout.",
                        skill_name,
                    )
                else:
                    # [STABILITY v55] Don't inject a robotic timeout message into
                    # working memory.  Return state unchanged (no response text)
                    # so the Kernel reports empty and chat.py fires the protected
                    # foreground lane as a rescue rather than showing
                    # "My cognitive process timed out" to the user.
                    return state

            # Handle None response from router.think()
            if response_text is None:
                required_tool_hit = self._successful_required_search_payload(state, contract)
                if required_tool_hit and not is_background:
                    skill_name, payload = required_tool_hit
                    response_text = self._render_required_search_answer_from_payload(
                        payload=payload,
                    )
                    state.response_modifiers["required_tool_empty_repaired"] = {
                        "skill": skill_name,
                        "method": "deterministic_grounded_evidence",
                    }
                    logger.warning(
                        "🛡️ ResponseGeneration answered from successful %s evidence after empty Cortex result.",
                        skill_name,
                    )
                else:
                    logger.debug("💭 ResponseGeneration: LLM returned None. Skipping this tick.")
                    return state
            if not str(response_text or "").strip():
                required_tool_hit = self._successful_required_search_payload(state, contract)
                if required_tool_hit and not is_background:
                    skill_name, payload = required_tool_hit
                    response_text = self._render_required_search_answer_from_payload(
                        payload=payload,
                    )
                    state.response_modifiers["required_tool_empty_repaired"] = {
                        "skill": skill_name,
                        "method": "deterministic_grounded_evidence",
                    }
                    logger.warning(
                        "🛡️ ResponseGeneration answered from successful %s evidence after blank Cortex result.",
                        skill_name,
                    )
                else:
                    logger.debug("💭 ResponseGeneration: LLM returned blank text. Skipping this tick.")
                    return state
            if not is_background:
                if (
                    origin != "test"
                    and not os.environ.get("AURA_AGI_MAX_TASKS")
                    and not os.environ.get("AURA_TESTING")
                    and not os.environ.get("AURA_PROOF_RUN")
                ):
                    reliability = assess_user_facing_reply(
                        user_surface_validation_prompt, response_text
                    )
                    if reliability.retryable:
                        repaired_text, repaired_shape, repair_reasons = (
                            self._repair_substantive_instruction_shape_miss(
                                user_surface_validation_prompt, response_text
                            )
                        )
                        if repaired_shape:
                            logger.info(
                                "🛡️ ResponseGeneration repaired instruction shape locally after refinement (%s).",
                                ",".join(repair_reasons) or "unknown",
                            )
                            response_text = repaired_text
                            reliability = assess_user_facing_reply(
                                user_surface_validation_prompt, response_text
                            )
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
                from core.llm.llm_guard import validate_json_response

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
            if ("<answer>" in lower_objective or "answer_format" in kwargs) and content and "<answer>" not in lower_response:
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
                    cognitive_engine_required=bool(
                        runtime_context.get("cognitive_engine_required", False)
                    ),
                    desktop_cognitive_engine_required=desktop_cognitive_engine_required,
                    live_runtime_payload_required=bool(
                        runtime_context.get("live_runtime_payload_required", False)
                    ),
                    visible_user_message=user_surface_validation_prompt,
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
                    memory_state_contract=bool(
                        runtime_context.get("memory_state_contract", False)
                    ),
                    runtime_fact_status_contract=runtime_fact_status_contract,
                    grounded_runtime_status_contract=runtime_fact_status_contract,
                    clean_user_surface_contract=clean_user_surface_contract,
                    user_surface_validation_prompt=user_surface_validation_prompt,
                    clean_user_surface_recurrent_loops=int(
                        live_mind_generation_controls.get(
                            "clean_user_surface_recurrent_loops", 1
                        )
                    ),
                    clean_user_surface_steering_alpha=self._safe_bias_float(
                        live_mind_generation_controls.get(
                            "clean_user_surface_steering_alpha"
                        ),
                        0.25,
                    ),
                    live_mind_controls_bound=live_mind_controls_bound,
                    live_mind_generation_controls=dict(
                        live_mind_generation_controls
                    ),
                    live_mind_snapshot_ready=bool(
                        runtime_context.get("live_mind_snapshot_ready", False)
                    ),
                    live_mind_required_subsystems_ok=bool(
                        runtime_context.get(
                            "live_mind_required_subsystems_ok", False
                        )
                    ),
                    cognitive_situation_sampling_bias=state.response_modifiers.get(
                        "cognitive_situation_sampling_bias"
                    ),
                    soma=soma_data,
                    state=state,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
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

            cleaned_response = self._repair_false_required_tool_inability(
                state=state,
                contract=contract,
                response_text=cleaned_response,
            )

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
                    self._repair_substantive_instruction_shape_miss(
                        user_surface_validation_prompt, cleaned_response
                    )
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

            cleaned_response = self._repair_false_required_tool_inability(
                state=state,
                contract=contract,
                response_text=cleaned_response,
            )

            # 6c. Skip emission for background tasks if they produced no meaningful content
            if is_background and not cleaned_response:
                return state

            surface_control_receipt: dict[str, Any] = {}
            if hasattr(router, "get_last_generation_metadata"):
                try:
                    generation_metadata = router.get_last_generation_metadata()
                    if isinstance(generation_metadata, dict):
                        candidate = generation_metadata.get("surface_control_receipt")
                        if isinstance(candidate, dict):
                            surface_control_receipt = dict(candidate)
                except (AttributeError, RuntimeError, TypeError) as exc:
                    logger.debug(
                        "ResponseGeneration could not read surface-control receipt: %s",
                        exc,
                    )
            surface_control_receipt = normalize_live_mind_surface_control_receipt(
                surface_control_receipt,
                controls_bound=live_mind_controls_bound,
                generation_controls=live_mind_generation_controls,
                source="response_generation_live_mind_controls",
            )
            state.response_modifiers["live_mind_surface_control_receipt"] = dict(
                surface_control_receipt
            )
            state.response_modifiers["live_mind_controls_worker_applied"] = bool(
                surface_control_receipt.get("live_mind_controls_bound")
                and surface_control_receipt.get("applied")
            )

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

            # One bounded owner observes conversational model updates and
            # shared-ground callbacks under the same stable partner identity.
            schedule_conversation_support_updates(
                str(objective or ""),
                str(cleaned_response or ""),
                state,
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

        # A background generation that deferred/failed can hand us None, and every
        # re.* below then raises "expected string or bytes-like object, got
        # 'NoneType'" — which tripped the mind_tick circuit and held the runtime
        # DEGRADED (observed live 2026-07-04). A cleaner must never crash on None.
        if not isinstance(text, str):
            text = "" if text is None else str(text)

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
