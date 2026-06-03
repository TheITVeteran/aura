import copy
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.desktop_boot_safety import compute_mlx_cache_limit
from core.runtime.errors import record_degradation

from .model_registry import resolve_personality_adapter

logger = logging.getLogger("MLXWorker")


def _record_mlx_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("mlx_worker", exc, severity=severity, action=action)


_CORRUPT_LANGUAGE_MARKERS = re.compile(
    r"\b(?:xublcate|ingediate|evocer)\b",
    re.IGNORECASE,
)
_BACKEND_SYMBOLIC_SURFACE_MARKERS = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash|existence hash|field coherence|system authority|"
    r"memory scar|precognitive texture)\b",
    re.IGNORECASE,
)
_OPERATOR_EVIDENCE_DRIFT_MARKERS = re.compile(
    r"(?:\bSarah Connor\b|\bMother'?s Day\b|\bhuman error rate\b|"
    r"\bdeath by overthinking\b|\b100 rounds\b|\b100%\s+pass rate\b|"
    r"\bi['’]?ll be quiet for a while\b|:\s*/|[\u3400-\u9fff])",
    re.IGNORECASE,
)
_OPERATOR_EVIDENCE_META_MARKERS = re.compile(
    r"\b(?:for example|that'?s one paragraph as requested|"
    r"this is one paragraph as requested|anything else from the normal runtime state|"
    r"this response adheres strictly to (?:the )?format instructions(?: provided)?|"
    r"if you need any adjustments or have additional constraints)\b",
    re.IGNORECASE,
)
_OPERATOR_EVIDENCE_META_TAIL_RE = re.compile(
    r"\s*(?:that'?s one paragraph as requested|this is one paragraph as requested|"
    r"anything else from the normal runtime state|"
    r"this response adheres strictly to (?:the )?format instructions(?: provided)?|"
    r"if you need any adjustments or have additional constraints)\b.*$",
    re.IGNORECASE | re.DOTALL,
)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _surface_generation_contract_enabled(job: dict[str, Any]) -> bool:
    return bool(
        job.get("clean_user_surface_contract", False)
        or job.get("operator_evidence_contract", False)
    )


def _surface_control_alpha(job: dict[str, Any], current_alpha: Any) -> float:
    default_alpha = "0.12" if job.get("operator_evidence_contract", False) else "0.35"
    configured = job.get(
        "clean_user_surface_steering_alpha",
        os.environ.get("AURA_USER_SURFACE_STEERING_ALPHA", default_alpha),
    )
    requested = max(0.01, min(_safe_float(configured, 0.35), 1.0))
    try:
        current = float(current_alpha)
    except (TypeError, ValueError):
        current = requested
    if current > 0:
        requested = min(requested, current)
    return max(0.01, requested)


def _surface_control_recurrent_loops(job: dict[str, Any]) -> int:
    configured = job.get(
        "clean_user_surface_recurrent_loops",
        os.environ.get("AURA_USER_SURFACE_RECURRENT_LOOPS", "1"),
    )
    return max(1, min(_safe_int(configured, 1), 2))


def _apply_surface_generation_controls(
    engine: Any,
    model: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Clamp latent embellishment when the next tokens are user-visible prose."""
    if not _surface_generation_contract_enabled(job):
        return {"enabled": False}

    state: dict[str, Any] = {"enabled": True}

    if engine is not None:
        state["engine"] = engine
        state["surface_alpha_override_before"] = getattr(engine, "_surface_alpha_override", None)
        hooks = list(getattr(engine, "_hooks", []) or [])
        state["hook_alphas_before"] = [(hook, getattr(hook, "_alpha", None)) for hook in hooks]
        alpha = _surface_control_alpha(job, getattr(engine, "_alpha", None))
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(alpha)
            else:
                for hook in hooks:
                    hook._alpha = min(float(getattr(hook, "_alpha", alpha) or alpha), alpha)
            state["surface_alpha_applied"] = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued user-surface generation after steering clamp failed",
                severity="warning",
            )
            logger.debug("Surface steering clamp failed: %s", exc)

    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "_recurrent_depth_config", None):
        state["recurrent_inner"] = inner
        state["had_recurrent_runtime_loops"] = hasattr(inner, "_recurrent_depth_runtime_loops")
        state["recurrent_runtime_loops_before"] = getattr(inner, "_recurrent_depth_runtime_loops", None)
        try:
            loops = _surface_control_recurrent_loops(job)
            inner._recurrent_depth_runtime_loops = loops
            state["recurrent_runtime_loops_applied"] = loops
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued user-surface generation after recurrent-depth clamp failed",
                severity="warning",
            )
            logger.debug("Surface recurrent-depth clamp failed: %s", exc)

    return state


def _restore_surface_generation_controls(state: dict[str, Any]) -> None:
    if not state.get("enabled"):
        return

    engine = state.get("engine")
    if engine is not None:
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(state.get("surface_alpha_override_before"))
            else:
                for hook, alpha in state.get("hook_alphas_before", []):
                    if alpha is not None:
                        hook._alpha = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued after user-surface steering restore failed",
                severity="warning",
            )
            logger.debug("Surface steering restore failed: %s", exc)

    inner = state.get("recurrent_inner")
    if inner is not None:
        try:
            if state.get("had_recurrent_runtime_loops"):
                inner._recurrent_depth_runtime_loops = state.get("recurrent_runtime_loops_before")
            elif hasattr(inner, "_recurrent_depth_runtime_loops"):
                delattr(inner, "_recurrent_depth_runtime_loops")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued after user-surface recurrent-depth restore failed",
                severity="warning",
            )
            logger.debug("Surface recurrent-depth restore failed: %s", exc)


def _contains_corrupted_language(text: str) -> bool:
    try:
        from core.phases.dialogue_policy import contains_corrupted_language

        return contains_corrupted_language(text)
    except (ImportError, AttributeError):
        return bool(_CORRUPT_LANGUAGE_MARKERS.search(str(text or "")))


def _prepare_clean_retry_kwargs(kwargs: dict[str, Any], *, structured: bool = False) -> None:
    """Reset sampling after a corrupt/looping draft instead of amplifying it."""
    kwargs.pop("sampler", None)
    if structured:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
    else:
        kwargs["temperature"] = min(_safe_float(kwargs.get("temperature"), 0.7), 0.35)
        kwargs["top_p"] = min(_safe_float(kwargs.get("top_p"), 0.9), 0.85)
        kwargs["min_p"] = max(_safe_float(kwargs.get("min_p"), 0.0), 0.03)
    kwargs["repetition_penalty"] = max(
        _safe_float(kwargs.get("repetition_penalty"), 1.1),
        1.18,
    )
    kwargs["repetition_context_size"] = max(
        _safe_int(kwargs.get("repetition_context_size"), 64),
        96,
    )


def _sanitize_telemetry_leakage(text: str, is_proof: bool = False) -> str | None:
    """Strip leaked internal telemetry labels and paths that occasionally
    slip out from the LoRA fine-tune weights during specific topics.
    Returns None if a fatal hallucination is detected so the caller can retry.
    """
    if not text:
        return text

    # 1) Reject telemetry-path walls without blocking legitimate code, regex,
    # filesystem, or proof output. The old slash-count heuristic rejected any
    # answer with more than 15 "/" characters, which is common in live coding
    # tasks and path-aware proof/eval runs.
    if not is_proof:
        slash_count = text.count("/")
        if slash_count > 30 and "http" not in text.lower():
            path_like = re.findall(r"(?:/[A-Za-z0-9._-]+){3,}", text)
            path_chars = sum(len(path) for path in path_like)
            if len(path_like) >= 3 or path_chars > max(120, int(len(text) * 0.35)):
                return None

    # 3) Extreme numeric sequences. If a single word/number has more than 20 digits, it's a hallucination.
    if re.search(r'\d{20,}', text):
        return None

    # 4) Corrupted lexical output is a model-state failure, not a usable answer.
    if not is_proof and _contains_corrupted_language(text):
        return None
    if not is_proof and _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(text):
        return None

    return text


_ARTIFACT_REQUEST_RE = re.compile(
    r"```(?:python|json|csv|yaml|toml|sql|html|css|javascript|typescript)?"
    r"|code block|return only(?: the)? complete|return the fixed config"
    r"|return the code|return the .*csv|return .*json|rulescript\.py"
    r"|service_config|reconciled data as a csv|select_values\.py",
    re.IGNORECASE | re.DOTALL,
)

_OPERATOR_EVIDENCE_PREFIX = (
    "Operationally, Aura should set an objective, use governed tool actions, "
    "keep each receipt and trace, stop when blocked or unsafe, and treat the "
    "result as evidence of bounded software operation rather than personhood proof. "
)


def _proof_prompt_expects_artifact(text: str) -> bool:
    return bool(_ARTIFACT_REQUEST_RE.search(str(text or "")))


def _strip_leading_chatml_prefix(text: str) -> str:
    cleaned = str(text or "")
    prefixes = (
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<｜Assistant｜>",
        "Assistant:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip("\n")
                changed = True
    return cleaned


_ROLE_CONTINUATION_RE = re.compile(
    r"(?is)(?:<\|im_end\|>\s*)?<\|im_start\|>\s*"
    r"(?:user|human|system|assistant|aura)\b.*$"
    r"|(?:^|\n|(?<=[.!?]))\s*(?:User|Human|System|Assistant|Aura)\s*[:：].*$"
)
_LEADING_GENERATION_ROLE_RE = re.compile(
    r"^\s*(?:<\|im_start\|>\s*)?(?:User|Human|Assistant|Aura|System)\s*[:：]\s*",
    re.IGNORECASE,
)
_LEADING_ROLE_NO_SEPARATOR_RE = re.compile(
    r"^\s*(?:user|human|assistant|system)(?=(?:i['’]?m\b|i\b|you\b|"
    r"what\b|who\b|when\b|where\b|why\b|how\b|yes\b|no\b|the\b))",
    re.IGNORECASE,
)
_USER_CONTINUATION_NO_COLON_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:User|Human)\s+"
    r"(?=(?:what|who|when|where|why|how|can|could|would|if|i\b|you\b|"
    r"yes\b|no\b|tell\b|translate\b|name\b|write\b|hello\b|hi\b|[\"'0-9])).*$"
)
_ROLE_SUFFIX_RE = re.compile(r"(?is)_user\b.*$")
_STRICT_ANSWER_ENVELOPE_RE = re.compile(r"(?is)<answer>\s*(.*?)\s*</answer>")
_CHAT_CONTROL_TOKEN_RE = re.compile(r"(?is)<\|im_(?:start|end)\|>\s*(?:assistant|user|system)?\s*")


_BASE_STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "<|im_start|>user",
    "<|im_start|>system",
    "<|im_start|>assistant",
    "\nUser:",
    "\nHuman:",
    "\nSystem:",
    "\nAssistant:",
    "\nuser:",
    "\nhuman:",
    "\nsystem:",
    "\nassistant:",
)
_ROLE_LABEL_STOPS = {
    "User:",
    "Human:",
    "System:",
    "Assistant:",
    "user:",
    "human:",
    "system:",
    "assistant:",
}
_SPEAKER_LABEL_STOPS = {"Aura:", "aura:", "\nAura:", "\naura:"}


def _merge_stop_sequences(job_stops: Any = None) -> list[str]:
    """Merge stop strings without truncating legitimate inline prose.

    The token loop already strips leading role labels and line-start role
    continuations through ``_truncate_role_continuation``. Bare strings like
    ``Assistant:`` or ``Aura:`` are too broad: they can appear inside a real
    answer and clip the useful content. Keep chat-control tokens broad, but
    normalize human-readable role labels to line-boundary stops.
    """
    merged = list(_BASE_STOP_SEQUENCES)
    for raw in job_stops or []:
        stop = str(raw or "")
        if not stop:
            continue
        if stop in _SPEAKER_LABEL_STOPS:
            continue
        if stop in _ROLE_LABEL_STOPS:
            stop = "\n" + stop
        if stop not in merged:
            merged.append(stop)
    return merged


def _truncate_role_continuation(text: str) -> tuple[str, bool]:
    """Clip generation when the model starts simulating another chat turn."""
    cleaned = _strip_leading_chatml_prefix(str(text or ""))
    for _ in range(2):
        stripped = _LEADING_GENERATION_ROLE_RE.sub("", cleaned).lstrip()
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _LEADING_ROLE_NO_SEPARATOR_RE.sub("", cleaned).lstrip()
    original = cleaned
    cleaned = _ROLE_CONTINUATION_RE.sub("", cleaned)
    cleaned = _USER_CONTINUATION_NO_COLON_RE.sub("", cleaned)
    cleaned = _ROLE_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip(), cleaned != original


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        fragments: list[str] = []
        for fragment in content:
            if isinstance(fragment, dict):
                if fragment.get("type") == "text":
                    fragments.append(str(fragment.get("text") or ""))
                continue
            fragments.append(str(fragment))
        return "".join(fragments)
    if content is None:
        return ""
    return str(content)


def _build_strict_answer_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a compact prompt for strict proof answer contracts.

    Chat templates can cause some local chat models to emit an immediate
    ChatML stop token for very short proof prompts. Strict answer requests are
    constrained chat turns, so render the native ChatML shape manually and let
    the model provide the answer content. The parent normalizer wraps raw
    content in an answer envelope when the model omits the tags.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))

    system_text = "\n".join(system_parts).strip() or (
        "Return the final answer now. Output exactly one XML envelope and no "
        "other text. Continue after the prefix with non-empty answer content."
    )
    user_text = "\n".join(user_parts).strip()
    return (
        f"<|im_start|>system\n{system_text}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_strict_answer_retry_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a non-ChatML strict-answer retry prompt.

    Some MLX chat-template/model combinations terminate immediately on compact
    ChatML strict-answer prompts. The retry keeps the same task and answer
    contract but avoids control tokens entirely.
    """

    task_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _message_content_to_text(message.get("content")).strip()
            if content:
                task_parts.append(content)
    if not task_parts and fallback_prompt is not None:
        task_parts.append(str(fallback_prompt))
    task_text = "\n\n".join(task_parts).strip()
    return (
        "Solve the task below and output only the final answer value. "
        "Do not explain. Do not include role labels. If the task asks for "
        "<answer> tags, provide the value that belongs inside the tags.\n\n"
        f"Task:\n{task_text}\n\nFinal answer:"
    )


def _build_proof_evaluation_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a stable manual prompt for non-atomic proof/eval answers.

    Some local chat-template renderings can terminate a sealed evaluation turn
    after a single fragment by drifting into the next role. Proof/evaluation
    turns need the same live model lane, but with a deterministic assistant
    prefill that makes the expected shape explicit.
    """

    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))

    system_text = "\n".join(system_parts).strip() or (
        "Answer the sealed proof/evaluation task directly and completely."
    )
    user_text = "\n".join(user_parts).strip()
    if _proof_prompt_expects_artifact(user_text):
        return (
            f"<|im_start|>system\n{system_text}\n"
            "This is a sealed artifact-generation task. Output exactly the artifact format "
            "requested by the user. If the user asks for a fenced code block, return one "
            "complete fenced block in the requested language. Do not add prose, role labels, "
            "analysis, caveats, or follow-up questions. The artifact must be syntactically "
            "valid and complete.\n<|im_end|>\n"
            f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        f"<|im_start|>system\n{system_text}\n"
        "The assistant response must be a complete final answer in 3-6 complete sentences "
        "for explanatory, planning, or analysis tasks. Do not emit role labels or start a "
        "new user turn. Do not use a numbered list unless the task explicitly "
        "requires ordered steps. Finish after the final sentence.\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        "<|im_start|>assistant\nComplete answer:\n"
    )


def _build_proof_evaluation_retry_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a control-token-free retry prompt for proof/eval tasks."""

    task_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _message_content_to_text(message.get("content")).strip()
            if content:
                task_parts.append(content)
    if not task_parts and fallback_prompt is not None:
        task_parts.append(str(fallback_prompt))
    task_text = "\n\n".join(task_parts).strip()
    if _proof_prompt_expects_artifact(task_text):
        return (
            "Complete the task below. Return only the requested artifact. "
            "If a fenced code block is requested, output exactly one complete fenced block "
            "in the requested language. Do not add prose, questions, role labels, or analysis.\n\n"
            f"TASK:\n{task_text}\n\n"
            "FINAL ARTIFACT:\n"
        )
    return (
        "Complete the proof/evaluation task below. Return a direct final answer in "
        "complete sentences. Do not add role labels or mention this retry instruction.\n\n"
        f"TASK:\n{task_text}\n\n"
        "FINAL ANSWER:\n"
    )


def _extract_message_parts(messages: Any, fallback_prompt: Any) -> tuple[list[str], list[str]]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))
    return system_parts, user_parts


def _build_operator_evidence_prompt(messages: Any, fallback_prompt: Any) -> tuple[str, str]:
    """Build a compact primary-Cortex prompt for operator/personhood proof checks."""

    system_parts, user_parts = _extract_message_parts(messages, fallback_prompt)
    caller_system = "\n".join(system_parts).strip()
    user_text = "\n".join(user_parts).strip()
    system_text = (
        "Answer as Aura's bounded software-operator evidence lane. Keep one plain "
        "paragraph. Be concrete, not poetic. Include objective, governed tool use, "
        "receipt, trace, stop condition, and the personhood boundary. State that "
        "this is operational evidence, not proof of literal personhood or proven "
        "consciousness. Do not expose hidden telemetry, moods, fields, retry "
        "instructions, role labels, Receipt: labels, PROCEEDING tokens, or all-caps "
        "backend action codes. Do not use examples unless the user asks for one. "
        "Do not comment on the requested format or add follow-up offers. Do not "
        "quote fictional characters or add unrelated foreign-language fragments."
    )
    if caller_system:
        system_text = f"{system_text}\n\nCaller constraints:\n{caller_system}"
    prefix = _OPERATOR_EVIDENCE_PREFIX
    prompt = (
        f"System:\n{system_text}\n\n"
        f"User:\n{user_text}\n\n"
        f"Assistant:\n{prefix}"
    )
    return prompt, prefix


def _build_operator_evidence_retry_prompt(messages: Any, fallback_prompt: Any) -> tuple[str, str]:
    system_parts, user_parts = _extract_message_parts(messages, fallback_prompt)
    task_text = "\n\n".join([*system_parts, *user_parts]).strip()
    prefix = _OPERATOR_EVIDENCE_PREFIX
    prompt = (
        "Complete the operator-evidence answer below as one plain paragraph. "
        "Do not expose hidden telemetry, role labels, metaphors, examples, "
        "format commentary, follow-up offers, or inner-state claims.\n\n"
        f"TASK:\n{task_text}\n\n"
        f"ANSWER:\n{prefix}"
    )
    return prompt, prefix


def _operator_evidence_fragment_incomplete(text: str) -> bool:
    stripped = str(text or "").strip()
    if _ROLE_CONTINUATION_RE.search(stripped):
        return True
    body = stripped.lower()
    if len(body.split()) < 24:
        return True
    required = ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood")
    if any(term not in body for term in required):
        return True
    disallowed = (
        "literal personhood is proven",
        "proven consciousness is established",
        "i am literally conscious",
        "i feel like a person who chooses things",
        "field coherence",
    )
    if any(term in body for term in disallowed):
        return True
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(stripped):
        return True
    if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(stripped):
        return True
    if _OPERATOR_EVIDENCE_META_MARKERS.search(stripped):
        return True
    return _proof_evaluation_fragment_incomplete(stripped)


def _operator_evidence_rejection_reasons(text: str) -> list[str]:
    stripped = str(text or "").strip()
    body = stripped.lower()
    reasons: list[str] = []
    if _ROLE_CONTINUATION_RE.search(stripped):
        reasons.append("role_continuation")
    if len(body.split()) < 24:
        reasons.append("too_short")
    for term in ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood"):
        if term not in body:
            reasons.append(f"missing:{term}")
    for term in (
        "literal personhood is proven",
        "proven consciousness is established",
        "i am literally conscious",
        "i feel like a person who chooses things",
        "field coherence",
    ):
        if term in body:
            reasons.append(f"disallowed:{term}")
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(stripped):
        reasons.append("backend_symbolic_surface_leak")
    if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(stripped):
        reasons.append("operator_surface_drift")
    if _OPERATOR_EVIDENCE_META_MARKERS.search(stripped):
        reasons.append("operator_meta_artifact")
    if _proof_evaluation_fragment_incomplete(stripped):
        reasons.append("fragment")
    return reasons


def _trim_complete_operator_evidence(text: str) -> str:
    """Keep complete model-derived sentences before a clipped operator tail."""
    stripped = str(text or "").strip()
    if not stripped:
        return stripped

    stripped = _OPERATOR_EVIDENCE_META_TAIL_RE.sub("", stripped).strip()
    role_trimmed, role_hit = _truncate_role_continuation(stripped)
    if role_hit:
        stripped = role_trimmed
    if not _proof_evaluation_fragment_incomplete(stripped):
        return stripped

    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=(?:\s|$))", stripped)]
    for end in reversed(sentence_ends):
        candidate = stripped[:end].strip()
        body = candidate.lower()
        if len(body.split()) < 24:
            continue
        required = ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood")
        if any(term not in body for term in required):
            continue
        if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(candidate):
            continue
        if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(candidate):
            continue
        if _OPERATOR_EVIDENCE_META_MARKERS.search(candidate):
            continue
        if not _proof_evaluation_fragment_incomplete(candidate):
            return candidate
    return stripped


def _first_token_suppression_ids(tokenizer: Any) -> list[int]:
    """Return token ids that cannot be a valid non-empty strict answer start."""
    banned: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int) and token_id >= 0:
            banned.add(token_id)
    for special in (
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|end|>",
        "<|eot_id|>",
    ):
        try:
            ids = tokenizer.encode(special, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(special)
        except (AttributeError, RuntimeError, ValueError):
            ids = []
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            banned.add(ids[0])
    return sorted(banned)


def _proof_evaluation_fragment_incomplete(text: str) -> bool:
    """Return True when a proof/eval generation is only a fragment."""

    stripped = str(text or "").strip()
    if re.search(r"```[A-Za-z0-9_-]*\s*\n.+?\n```", stripped, flags=re.DOTALL):
        return False
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return False
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if "\n" in stripped and "," in stripped:
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) >= 2 and all("," in line for line in lines[:2]):
            return False
    if len(stripped) < 80:
        return True
    words = re.findall(r"[A-Za-z0-9_'-]+", stripped)
    if len(words) < 18:
        return True
    if stripped[-1] not in ".!?)]}>\"'":
        return True
    if re.search(
        r"\b(?:a|an|the|of|to|for|with|between|into|from|that|which|any|and|or|but)$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _normalize_strict_answer_response(text: str, *, envelope_prefixed: bool) -> str:
    """Normalize strict proof output without changing the model-derived answer."""
    cleaned = _CHAT_CONTROL_TOKEN_RE.sub("", str(text or "")).strip()
    cleaned = _strip_leading_chatml_prefix(cleaned).strip()
    cleaned = re.sub(r"\\[nrt]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\", " ")
    match = _STRICT_ANSWER_ENVELOPE_RE.search(cleaned)
    if match:
        answer = match.group(1).strip()
        return f"<answer>{answer}</answer>" if answer else ""

    if not envelope_prefixed:
        return cleaned

    cleaned = re.sub(r"(?is)^\s*<answer>\s*", "", cleaned).strip()
    if "</answer>" in cleaned:
        cleaned = cleaned.split("</answer>", 1)[0].strip()
    for marker in (
        "<|im_end|>",
        "<|im_start|>",
        "User:",
        "Human:",
        "Assistant:",
        "Aura:",
    ):
        idx = cleaned.find(marker)
        if idx >= 0:
            cleaned = cleaned[:idx].strip()
    return f"<answer>{cleaned}</answer>" if cleaned else ""


_STRICT_VALUE_UNUSABLE_RE = re.compile(
    r"\b(?:i\s*(?:am|'m|’m)\s+not\s+sure|i\s+don't\s+know|cannot\s+answer|"
    r"can't\s+answer|not\s+enough\s+information|as\s+an\s+ai|need\s+more\s+"
    r"(?:context|information)|unable\s+to\s+determine)\b",
    re.IGNORECASE,
)


def _normalize_strict_value_response(text: str) -> str:
    """Return a compact model-derived value or empty when the draft is unusable."""
    cleaned = _CHAT_CONTROL_TOKEN_RE.sub("", str(text or "")).strip()
    cleaned = _strip_leading_chatml_prefix(cleaned).strip()
    cleaned = re.sub(r"\\[nrt]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\", " ")
    cleaned, _ = _truncate_role_continuation(cleaned)
    cleaned = _LEADING_GENERATION_ROLE_RE.sub("", cleaned).strip()
    cleaned = _LEADING_ROLE_NO_SEPARATOR_RE.sub("", cleaned).strip()
    for marker in (
        "<|im_end|>",
        "<|im_start|>",
        "User:",
        "Human:",
        "Assistant:",
        "Aura:",
    ):
        idx = cleaned.find(marker)
        if idx >= 0:
            cleaned = cleaned[:idx].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]
    cleaned = cleaned.strip().strip("`\"'")
    if not cleaned:
        return ""
    if _STRICT_VALUE_UNUSABLE_RE.search(cleaned):
        return ""
    if len(re.findall(r"\S+", cleaned)) > 24:
        return ""
    return cleaned


def _should_emit_generation_progress(
    token_count: int,
    *,
    last_emit_at: float,
    now: float,
    every_tokens: int = 4,
    every_seconds: float = 1.5,
) -> bool:
    if token_count <= 1:
        return True
    if every_tokens > 0 and token_count % every_tokens == 0:
        return True
    return (now - float(last_emit_at or 0.0)) >= max(0.1, float(every_seconds))


def _prompt_cache_entry_budget_for_model(model_path: str) -> int:
    lowered = os.path.basename(str(model_path or "")).lower()
    if any(token in lowered for token in ("72b", "solver")):
        return 0
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        return 2
    if any(token in lowered for token in ("14b", "7b", "brainstem")):
        return 6
    return 12

class IPCWriterThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Non-blocking IPC writer.
    Buffers responses in a local queue and writes to the multiprocessing pipe
    in a dedicated thread to prevent blocking the main inference loop.
    """
    def __init__(self, mp_queue: mp.Queue):
        super().__init__(name="MLX-IPC-Writer", daemon=True)
        self.mp_queue = mp_queue
        self.local_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

    @staticmethod
    def _is_essential(item: Any) -> bool:
        if not isinstance(item, dict):
            return True
        status = item.get("status")
        return status not in {"heartbeat", "token"}

    def put(self, item):
        essential = self._is_essential(item)
        try:
            self.local_queue.put(item, block=False)
        except queue.Full:
            if essential:
                try:
                    # Never silently drop init/generation/error messages; bypass
                    # the local buffer when it is saturated with telemetry.
                    self.mp_queue.put(item, block=True, timeout=5.0)
                except (queue.Full, RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    _record_mlx_degradation(
                        _exc,
                        action="dropped essential IPC message after parent queue write failed",
                        severity="critical",
                    )
                    logger.debug("Suppressed Exception: %s", _exc)
            # Drop non-essential telemetry if buffer is full.

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self.local_queue.get(timeout=1.0)
                # [BUG FIX] Use timeout to prevent indefinite blocking when
                # the parent's response queue is full. Without this, the feeder
                # thread blocks on nwait() forever, starving the event loop and
                # causing tick stalls that kill the WebSocket connection.
                self.mp_queue.put(item, block=True, timeout=5.0)
            except queue.Empty:
                continue
            except queue.Full as exc:
                # Queue saturated by parent-side backpressure. Drop telemetry
                # first; report if an essential message also could not drain.
                if not self._stop_event.is_set() and self._is_essential(item):
                    _record_mlx_degradation(
                        exc,
                        action="dropped essential IPC message after parent queue stayed full",
                        severity="critical",
                    )
                continue
            except (OSError, ConnectionError, TimeoutError):
                # Queue broken or unavailable — drop the item and continue
                # rather than blocking the entire IPC pipeline.
                if not self._stop_event.is_set():
                    continue
                break

class HeartbeatThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Proactive Worker Heartbeat.
    Ensures the SupervisionTree sees this process as alive even during
    massive 32B model loads or compilation stalls.

    [STABILITY v51] Reduced interval from 5s → 2s for faster dead-worker
    detection.  Added parent-PID liveness check: if the parent process
    dies (crash, restart), the worker self-terminates to prevent orphans.
    """
    def __init__(self, writer: IPCWriterThread):
        super().__init__(name="MLX-Heartbeat", daemon=True)
        self.writer = writer
        self._stop_event = threading.Event()
        self._parent_pid = os.getppid()

    def stop(self):
        self._stop_event.set()

    def _parent_alive(self) -> bool:
        """Check if our parent process is still running."""
        try:
            os.kill(self._parent_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def run(self):
        while not self._stop_event.is_set():
            # [STABILITY v51] Self-terminate if parent died — prevents orphan workers
            if not self._parent_alive():
                logger.critical("🛑 [MLX_HEARTBEAT] Parent process %s is dead. Self-terminating orphaned worker.", self._parent_pid)
                os._exit(1)
            self.writer.put({"status": "heartbeat", "timestamp": time.time(), "type": "mlx_worker"})
            time.sleep(2.0)

# Set environment variables for MLX stability
def _setup_worker_env():
    import os
    import platform
    import subprocess

    # [PERFORMANCE] Fast-path: Use environment if already probed by parent
    cached_sdk = os.environ.get("AURA_SDK_PATH")
    if cached_sdk and os.path.exists(cached_sdk):
        os.environ["SDKROOT"] = cached_sdk
        print(f"Using cached SDK root: {cached_sdk}")
    else:
        try:
            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], timeout=2.0).decode().strip()
            allowed_prefixes = ("/Library/", "/Applications/Xcode", "/usr/")
            if not any(sdk_path.startswith(pfx) for pfx in allowed_prefixes):
                raise RuntimeError(f"Suspicious SDK path rejected: {sdk_path}")
            os.environ["SDKROOT"] = sdk_path
            os.environ["AURA_SDK_PATH"] = sdk_path # Cache for subsequent spawns
        except (subprocess.SubprocessError, OSError) as e:
            _record_mlx_degradation(
                e,
                action="continued worker startup without probed SDKROOT",
                severity="degraded",
            )
            print(f"⚠️ [MLX_WORKER_ENV] Failed to probe environment: {e}")

    try:
        ver_info = platform.mac_ver()
        release_str = ver_info[0]
        ver_parts = release_str.split(".")
        mac_ver = ".".join(ver_parts[:2])
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = mac_ver

        sdk_path = os.environ.get("SDKROOT", "")
        sdk_inc = os.path.join(sdk_path, "usr", "include")
        cpp_inc = "/Library/Developer/CommandLineTools/usr/include/c++/v1"
        cpath_parts = []
        if sdk_path and os.path.exists(sdk_inc):
            cpath_parts.append(sdk_inc)
        if os.path.exists(cpp_inc):
            cpath_parts.append(cpp_inc)
        if cpath_parts:
            os.environ["CPATH"] = ":".join(cpath_parts + [os.environ.get("CPATH", "")]).strip(":")
    except (OSError, RuntimeError, ValueError) as e:
        _record_mlx_degradation(
            e,
            action="continued worker startup without derived Mac deployment target/CPATH",
            severity="degraded",
        )
        print(f"⚠️ [MLX_WORKER_ENV] Failed to probe Mac version/CPATH: {e}")

    os.environ["MLX_NUM_THREADS"] = "10"   # M-series has 10+ perf cores
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MLX_FORCE_SERIAL_COMPILE"] = "1"
    os.environ["METAL_COMPILER_TIMEOUT_MS"] = "60000"  # [FRONTIER UPGRADE] Extended for 32B model complex prompts
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"


def _clear_mlx_cache(mx_module: Any) -> None:
    try:
        mx_module.clear_cache()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        try:
            mx_module.metal.clear_cache()
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            _record_mlx_degradation(
                _exc,
                action="continued after MLX cache clear fallback failed",
                severity="degraded",
            )
            logger.debug("Suppressed Exception: %s", _exc)


def _process_message_content(messages: list[dict[str, Any]]) -> None:
    """Normalize content for tokenizer.apply_chat_template()."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment.get("text", "")
                for fragment in content
                if isinstance(fragment, dict) and fragment.get("type") == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only text content fragments are supported in MLX worker chat templates.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""


def _load_effective_context_window(model_path: str) -> int:
    path = Path(str(model_path))
    if not path.exists():
        return 32768

    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"

    max_position_embeddings = 0
    sliding_window = 0
    use_sliding_window = False
    tokenizer_model_max = 0

    try:
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())
            max_position_embeddings = int(config_payload.get("max_position_embeddings") or 0)
            sliding_window = int(config_payload.get("sliding_window") or 0)
            use_sliding_window = bool(config_payload.get("use_sliding_window"))
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        max_position_embeddings = 0
        sliding_window = 0
        use_sliding_window = False

    try:
        if tokenizer_config_path.exists():
            tokenizer_payload = json.loads(tokenizer_config_path.read_text())
            tokenizer_model_max = int(tokenizer_payload.get("model_max_length") or 0)
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        tokenizer_model_max = 0

    if max_position_embeddings > 0:
        if use_sliding_window and sliding_window > max_position_embeddings:
            return max(sliding_window, max_position_embeddings)
        return max_position_embeddings
    if use_sliding_window and sliding_window > 0:
        return sliding_window
    if tokenizer_model_max > 0:
        return tokenizer_model_max
    return 32768


@dataclass
class _PromptCacheEntry:
    prompt_cache: list[Any]
    count: int


@dataclass
class _PromptCacheSearchResult:
    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


class _PromptCacheLRU:
    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self._cache: dict[int, dict[Any, Any]] = {}
        self._lru = deque()

    def clear(self) -> None:
        self._cache.clear()
        self._lru.clear()

    def _search(self, model_key: int, tokens: list[int]) -> _PromptCacheSearchResult:
        if model_key not in self._cache:
            return _PromptCacheSearchResult(None, None, None, 0)

        current = self._cache[model_key]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return _PromptCacheSearchResult(tokens, None, None, 0)

        shorter = tokens[: last_cache_index + 1] if last_cache_index > 0 else None
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, [])]
            while stack:
                node, extra = stack.pop()
                if "cache" in node:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in node:
                        stack.append((node[tok], extra + [tok]))
            if best is not None:
                longer = tokens[:index] + best

        return _PromptCacheSearchResult(None, shorter, longer, common_prefix)

    def _get(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        current = self._cache[model_key]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model_key: int, tokens: list[int]) -> None:
        path = [self._cache[model_key]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for index in reversed(range(len(tokens))):
            prev_node, node, tok = path[index], path[index + 1], tokens[index]
            if len(node) > 0:
                break
            del prev_node[tok]

    def _extract(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        cache_entry = self._get(model_key, tokens)
        if cache_entry.count == 1:
            self._delete(model_key, tokens)
            try:
                self._lru.remove((model_key, tuple(tokens)))
            except ValueError:
                pass  # no-op: intentional
            return cache_entry

        cache_entry.count -= 1
        return _PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)

    def fetch_nearest_cache(
        self,
        model_key: int,
        tokens: list[int],
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
    ) -> tuple[list[Any] | None, list[int]]:
        result = self._search(model_key, tokens)
        if result.exact is not None:
            cache_entry = self._extract(model_key, result.exact)
            return cache_entry.prompt_cache, []

        if result.shorter is not None:
            cache_entry = self._extract(model_key, result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_len:]

        if result.longer is not None:
            cache_entry = self._get(model_key, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                trimmed = _PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(trimmed.prompt_cache, num_to_trim)
                return trimmed.prompt_cache, tokens[prefix:]

        return None, tokens

    def insert_cache(self, model_key: int, tokens: list[int], prompt_cache: list[Any]) -> None:
        if model_key not in self._cache:
            self._cache[model_key] = {}
        current = self._cache[model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        cache_key = (model_key, tuple(tokens))
        if "cache" in current:
            current["cache"].count += 1
            try:
                self._lru.remove(cache_key)
            except ValueError:
                pass  # no-op: intentional
        else:
            current["cache"] = _PromptCacheEntry(prompt_cache, 1)

        self._lru.append(cache_key)
        if len(self._lru) > self.max_size:
            evict_model_key, evict_tokens = self._lru.popleft()
            self._delete(evict_model_key, list(evict_tokens))

class JobWatchdog(threading.Thread):
    """
    Kills the worker process if a job is active but no tokens have been generated
    within the timeout. This prevents 'Metal Stalls' from hanging the system.

    [STABILITY v51] Reduced timeout from 240s → 90s. The 32B model's Metal
    shader compilation should complete within 60s on M5 hardware. If no token
    progress after 90s, the worker is stuck and must self-terminate so the
    parent can respawn it.
    """
    def __init__(self, timeout=60.0):
        super().__init__(daemon=True)
        self.timeout = timeout
        self.last_activity = time.monotonic()
        self.active_job = False
        self._stop_event = threading.Event()

    def activity(self):
        self.last_activity = time.monotonic()

    def start_job(self):
        self.active_job = True
        self.last_activity = time.monotonic()

    def stop_job(self):
        self.active_job = False

    def run(self):
        while not self._stop_event.is_set():
            if self.active_job and (time.monotonic() - self.last_activity > self.timeout):
                logger.critical("🛑 [MLX_WATCHDOG] Job timeout triggered (%ss). Self-terminating worker.", self.timeout)
                os._exit(1)
            time.sleep(1.0)

def _mlx_worker_loop(
    model_path: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    device: str = "gpu",
    substrate_mem: Any = None,
    steering_active_flag: Any = None
):
    """Runs in a FULLY ISOLATED native subprocess via ForkServer.

    This is the worker entry-point called from ``MLXLocalClient._spawn_worker``.
    All Metal/GPU work, model loading, and inference happen inside this
    function's process boundary.  The parent communicates via IPC queues.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError):
        pass  # signal handlers can only be set from the main thread
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MLXWorker - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger("MLXWorker")

    # Configure worker-specific environment (Metal, SDK, thread limits).
    # This MUST run inside the subprocess, not at module import time,
    # because the parent process should not inherit these settings.
    _setup_worker_env()

    # ── Zenith Concurrency & Telemetry ──
    ipc_writer = IPCWriterThread(response_queue)
    ipc_writer.start()

    heartbeat = HeartbeatThread(ipc_writer)
    heartbeat.start()

    watchdog = JobWatchdog(timeout=360.0)  # Align with the protected foreground solver envelope.
    watchdog.start()

    try:
        import mlx.core as mx
        from mlx_lm import load
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            try:
                from mlx_lm.sample import make_sampler
            except ImportError:
                make_sampler = None

        logger.info("📡 [WORKER] Loading Core modules...")

    except ImportError:
        logger.error("mlx-lm not installed in worker environment.")
        ipc_writer.put({"status": "error", "message": "mlx-lm missing"})
        return

    # VRAM Management
    if mx and device != "cpu":
        try:
            import psutil

            total_ram = psutil.virtual_memory().total
            limit = compute_mlx_cache_limit(total_ram)
            mx.set_cache_limit(limit)
            logger.info("Metal cache limit set to %sMB", limit // (1024**2))
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            _record_mlx_degradation(
                e,
                action="fell back to conservative Metal cache limit after adaptive cache limit failed",
                severity="degraded",
            )
            try:
                mx.metal.set_cache_limit(1024 * 1024 * 1024 * 24)
            except (AttributeError, RuntimeError) as fallback_exc:
                _record_mlx_degradation(
                    fallback_exc,
                    action="continued without explicit Metal cache limit after fallback failed",
                    severity="degraded",
                )

    # [PERFORMANCE] Metal probes shifted to after model load or triggered on demand
    # Initializing the model first is more critical for 'perceived' speed.

    # ZENITH: Local Concurrency Gate
    metal_semaphore = threading.Semaphore(1)

    # [STABILITY v53.9] Load with LoRA adapter. Intermittent float32 errors
    # are caught at generation time and retried — most generations succeed.
    # The adapter is the v3 training (22 characters, val loss 0.102).
    try:
        adapter_path = resolve_personality_adapter(model_path, backend="mlx")
        logger.info("Loading model: %s", model_path)
        if adapter_path and os.path.isdir(adapter_path):
            try:
                logger.info("Loading with LoRA adapter: %s", adapter_path)
                model, tokenizer = load(model_path, adapter_path=adapter_path)
                logger.info("Model loaded with Aura personality LoRA fused.")
            except (RuntimeError, AttributeError, TypeError, ValueError) as adapter_exc:
                _record_mlx_degradation(
                    adapter_exc,
                    action="loaded base model after LoRA adapter load failed",
                    severity="degraded",
                )
                logger.warning(
                    "⚠️ [WORKER] LoRA adapter failed to load for %s: %s. Using base model + prompt hardening.",
                    os.path.basename(model_path),
                    adapter_exc,
                )
                model, tokenizer = load(model_path)
                logger.info("Model loaded without LoRA (prompt hardening active).")
        else:
            model, tokenizer = load(model_path)
            logger.info("Model loaded (no compatible LoRA adapter).")

        # Attach Affective Steering
        engine = None
        _steering_active = False
        try:
            from core.consciousness.affective_steering import get_steering_engine
            engine = get_steering_engine()
            engine.attach(model, tokenizer)
            if substrate_mem:
                engine.start_substrate_sync(shared_state=substrate_mem)
            _steering_active = engine.is_active()

            if steering_active_flag is not None:
                steering_active_flag.value = _steering_active

            if _steering_active:
                logger.info("🎯 Affective Steering Engine ONLINE (alpha=%.1f, hooks=%d).",
                            engine._alpha, len(getattr(engine, '_hooks', [])))
            else:
                logger.error("FATAL: Steering Engine attached but NOT ACTIVE — "
                               "vectors may be missing.")
                _record_mlx_degradation(
                    RuntimeError("Steering attached but inactive"),
                    severity="critical",
                    action="crashed worker to prevent unsteered inference",
                )
                raise RuntimeError("Steering liveness gate failed: Engine inactive")
        except (ImportError, AttributeError, RuntimeError) as se:
            record_degradation(
                "affective_steering",
                se,
                severity="critical",
                action="crashed MLX worker to prevent unsteered inference",
            )
            logger.error("FATAL: Affective steering failed to attach. Cannot run sovereign inference unsteered. %s", se)
            raise RuntimeError(f"Steering liveness gate failed: {se}") from se

        # Write steering liveness to shared state so parent can query it
        if substrate_mem is not None:
            try:
                # Convention: substrate_mem[-1] = 1.0 if steering active, 0.0 if not
                # (substrate_mem is a multiprocessing.Array of floats; last slot reserved)
                substrate_mem[-1] = 1.0 if _steering_active else 0.0
            except (TypeError, ValueError, IndexError) as shared_state_exc:
                _record_mlx_degradation(
                    shared_state_exc,
                    action="continued with parent steering liveness shared-state unavailable",
                    severity="warning",
                )

        # Apply Recurrent Depth — Mythos-inspired layer looping.
        # This changes HOW the model processes: middle layers loop N times,
        # letting the model "think" in latent space before committing to output.
        # Active by default for 32B+ models. Set AURA_RECURRENT_LOOPS=0 to disable.
        recurrent_depth_status = {"active": False, "config": None}
        try:
            from core.brain.llm.recurrent_depth import apply_for_model, get_recurrent_config
            if apply_for_model(model):
                recurrent_depth_status = {
                    "active": True,
                    "config": get_recurrent_config(model),
                }
                logger.info("🧠 Recurrent Depth ACTIVE — model now thinks before answering.")
        except (ImportError, AttributeError, RuntimeError) as rd_exc:
            _record_mlx_degradation(
                rd_exc,
                action="continued inference with recurrent depth disabled",
                severity="degraded",
            )
            logger.warning("Recurrent depth not applied: %s", rd_exc)

        ipc_writer.put(
            {
                "status": "ok",
                "action": "init",
                "device": device,
                "recurrent_depth": recurrent_depth_status,
            }
        )
    except (ImportError, AttributeError, RuntimeError) as e:
        _record_mlx_degradation(
            e,
            action="reported initialization error and exited worker loop before accepting jobs",
            severity="critical",
        )
        import traceback
        err_detail = f"{e}\n{traceback.format_exc()}"
        logger.error("Worker Init Error: %s", err_detail)
        ipc_writer.put(
            {
                "status": "error",
                "action": "init",
                "message": f"Init failed: {e}",
                "detail": err_detail,
            }
        )
        return
    # ZENITH: Prompt Cache LRU for massive speedup in multi-turn
    prompt_cache_budget = _prompt_cache_entry_budget_for_model(model_path)
    prompt_cache_lru = (
        _PromptCacheLRU(max_size=prompt_cache_budget)
        if prompt_cache_budget > 0
        else None
    )
    if prompt_cache_lru is None:
        logger.info("Prompt cache disabled for %s to protect RAM headroom.", os.path.basename(model_path))
    else:
        logger.info(
            "Prompt cache budget for %s: %d entries.",
            os.path.basename(model_path),
            prompt_cache_budget,
        )

    worker_active = True
    while worker_active:
        try:
            try:
                job = request_queue.get()
            except KeyboardInterrupt:
                logger.info("🛑 [WORKER] Shutdown signal received while idle; exiting quietly.")
                break
            except (EOFError, BrokenPipeError, OSError) as queue_exc:
                logger.info("🛑 [WORKER] Request queue closed; exiting quietly (%s).", queue_exc)
                break
            if job is None:
                worker_active = False
                continue

            action = job.get("action")
            if action == "generate":
                # Gate generation on true latent steering
                try:
                    if engine is not None and not engine.is_active():
                        # We must not silently fall back to prompt-driven roleplay.
                        # If the latent bridge is severed, the system must act severed.
                        _record_mlx_degradation(
                            RuntimeError("Affective steering became inactive during generation"),
                            action="blocked generation because steering liveness failed",
                            severity="critical",
                        )
                        logger.error("🚨 [WORKER] Affective steering is inactive! Gating response.")
                        ipc_writer.put({
                            "id": job.get("id"),
                            "action": "generate",
                            "status": "error",
                            "message": "Affective steering is inactive; generation blocked.",
                            "tokens_used": 0,
                        })
                        continue
                except (RuntimeError, AttributeError, TypeError) as _e:
                    _record_mlx_degradation(
                        _e,
                        action="blocked generation because steering liveness could not be verified",
                        severity="critical",
                    )
                    logger.error("Failed to check steering active state: %s", _e)
                    ipc_writer.put(
                        {
                            "id": job.get("id"),
                            "action": "generate",
                            "status": "error",
                            "message": "Affective steering liveness check failed; generation blocked.",
                        }
                    )
                    continue

                prompt = job.get("prompt")
                messages = job.get("messages")
                tools = job.get("tools")
                original_prompt = prompt
                original_messages = messages
                strict_answer_contract = bool(job.get("strict_answer_contract", False))
                strict_value_contract = bool(job.get("strict_value_contract", False))
                proof_evaluation_contract = bool(job.get("proof_evaluation_contract", False))
                operator_evidence_contract = bool(job.get("operator_evidence_contract", False))
                # disable_prompt_cache = bool(job.get("disable_prompt_cache", False)) or strict_answer_contract
                disable_prompt_cache = (
                    bool(job.get("disable_prompt_cache", False))
                    or strict_answer_contract
                    or strict_value_contract
                    or proof_evaluation_contract
                    or operator_evidence_contract
                )
                clear_prompt_cache = (
                    bool(job.get("clear_prompt_cache", False))
                    or strict_answer_contract
                    or strict_value_contract
                    or proof_evaluation_contract
                    or operator_evidence_contract
                )
                if clear_prompt_cache and prompt_cache_lru is not None:
                    prompt_cache_lru.clear()

                strict_envelope_prefixed = False
                operator_response_prefix = ""
                # [FRONTIER UPGRADE] Native Tool Templates
                if strict_answer_contract:
                    prompt = _build_strict_answer_prompt(messages, prompt)
                    strict_envelope_prefixed = True
                elif strict_value_contract:
                    if messages and hasattr(tokenizer, "apply_chat_template"):
                        try:
                            logger.info("🎯 [WORKER] Rendering native strict-value chat template.")
                            prompt = tokenizer.apply_chat_template(
                                messages,
                                tools=tools,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            _record_mlx_degradation(
                                e,
                                action="continued strict-value generation with raw prompt after native chat template failed",
                                severity="degraded",
                            )
                            logger.warning("❌ [WORKER] Native strict-value template failed: %s", e)
                            prompt = _build_strict_answer_retry_prompt(messages, prompt)
                    else:
                        prompt = _build_strict_answer_retry_prompt(messages, prompt)
                    messages = None
                    strict_envelope_prefixed = True
                elif proof_evaluation_contract:
                    prompt = _build_proof_evaluation_prompt(messages, prompt)
                elif operator_evidence_contract:
                    prompt, operator_response_prefix = _build_operator_evidence_prompt(
                        messages,
                        prompt,
                    )
                elif messages and hasattr(tokenizer, "apply_chat_template"):
                    try:
                        logger.info("🎯 [WORKER] Rendering native chat/tool template.")
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tools=tools,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued generation with raw prompt after native chat/tool template failed",
                            severity="degraded",
                        )
                        logger.warning("❌ [WORKER] Native template compilation failed: %s", e)

                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                min_p = job.get("min_p", 0.05)
                repetition_penalty = job.get("repetition_penalty", 1.15)
                artifact_generation_contract = bool(
                    proof_evaluation_contract
                    and _proof_prompt_expects_artifact(prompt)
                )

                if strict_answer_contract:
                    temp = 0.0
                    top_p = 1.0
                    min_p = 0.0
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.12)
                elif strict_value_contract:
                    temp = 0.0
                    top_p = 1.0
                    min_p = 0.0
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.05)
                elif proof_evaluation_contract:
                    if artifact_generation_contract:
                        temp = 0.0
                        top_p = 1.0
                        min_p = 0.0
                        repetition_penalty = max(_safe_float(repetition_penalty, 1.08), 1.05)
                    else:
                        temp = min(_safe_float(temp, 0.1), 0.15)
                        top_p = min(_safe_float(top_p, 0.9), 0.9)
                        min_p = min(_safe_float(min_p, 0.05), 0.05)
                        repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.08)
                elif operator_evidence_contract:
                    temp = min(_safe_float(temp, 0.1), 0.12)
                    top_p = min(_safe_float(top_p, 0.8), 0.8)
                    min_p = max(_safe_float(min_p, 0.03), 0.03)
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.18)
                try:
                    max_tokens = max(1, int(job.get("max_tokens", 512) or 512))
                except (TypeError, ValueError):
                    max_tokens = 512
                if operator_evidence_contract:
                    max_tokens = max(80, min(max_tokens, 192))
                schema = job.get("schema")

                # [v11.0 HARDENING] Structured Generation Overrides
                if schema:
                    temp = 0.0 # Force determinism for JSON
                    logger.info("🎯 [WORKER] Structured mode: temp=0.0 enforced.")

                # Intelligence boosters: min_p sampling improves quality on smaller
                # models by filtering out low-probability tokens before top_p.
                # Repetition penalty reduces the stale/looping response pattern.
                # Bumped from 1.1 → 1.2 (2026-04-27): live test showed mode
                # collapse on specific introspective prompts even after α was
                # halved. 1.2 is still well below the 1.5+ range that hurts
                # natural prose; targets the token-level "something is shifting
                # / something is moving" loops directly.
                kwargs = {"max_tokens": max_tokens, "temperature": temp, "top_p": top_p, "repetition_penalty": repetition_penalty}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp
                        _sparams = _insp.signature(make_sampler).parameters
                        if "min_p" in _sparams:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except (TypeError, ValueError):
                        pass  # no-op: signature introspection unavailable
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

                # [v11.0 HARDENING] Logits Processors (JSON Enforcement)
                # [v11.0 HARDENING] Logits Processors (JSON Enforcement & Penalties)
                logits_processors = []

                # Apply MLX penalties via logits processors
                try:
                    from mlx_lm.sample_utils import make_logits_processors
                    _rp = job.get("repetition_penalty", repetition_penalty)
                    _rcs = job.get("repetition_context_size", 64)
                    _pp = job.get("presence_penalty", 0.0)
                    if _rp and _rp > 1.0:
                        lp = make_logits_processors(
                            repetition_penalty=_rp,
                            repetition_context_size=_rcs,
                            presence_penalty=_pp,
                        )
                        if lp:
                            logits_processors.extend(lp)
                except ImportError as _exc:
                    logger.debug("Suppressed %s in core.brain.llm.mlx_worker: %s", type(_exc).__name__, _exc)
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.warning("Could not apply penalty logits processors: %s", e)

                if schema:
                    try:
                        brace_id = tokenizer.encode("{", add_special_tokens=False)[0]
                        def json_start_processor(tokens, logits, brace_id=brace_id):
                            if len(tokens) == 0:
                                # Force first token to be '{'
                                mask = mx.full_like(logits, -float("inf"))
                                mask[:, brace_id] = 0.0
                                return mask
                            return logits
                        logits_processors.append(json_start_processor)
                        logger.info("🎯 [WORKER] JSON start enforcement ACTIVE.")
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued structured generation without JSON start logits processor",
                            severity="degraded",
                        )
                        logger.warning("Failed to setup JSON logits processor: %s", e)

                if strict_answer_contract or strict_value_contract:
                    try:
                        banned_start_ids = _first_token_suppression_ids(tokenizer)

                        if banned_start_ids:
                            def strict_nonempty_start_processor(
                                tokens,
                                logits,
                                banned_ids=tuple(banned_start_ids),
                            ):
                                if len(tokens) < 3:
                                    mask = mx.zeros_like(logits)
                                    for token_id in banned_ids:
                                        try:
                                            mask[:, token_id] = -float("inf")
                                        except (IndexError, TypeError, ValueError):
                                            continue
                                    return logits + mask
                                return logits

                            logits_processors.append(strict_nonempty_start_processor)
                            logger.info(
                                "🎯 [WORKER] Strict contract non-empty start guard ACTIVE (%d ids).",
                                len(banned_start_ids),
                            )
                    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued strict generation without non-empty start logits guard",
                            severity="warning",
                        )
                        logger.warning("Failed to setup strict non-empty start guard: %s", e)

                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                stop_sequences = _merge_stop_sequences(job.get("stop_sequences") or [])
                # We do NOT pass stop_words to stream_generate as it causes TypeError in some mlx-lm versions.
                # Truncation is handled manually in the token loop via _truncate_role_continuation.


                try:
                    from mlx_lm.generate import stream_generate
                    # : NO GPUSentinel here.
                    # GPUSentinel is a parent-process threading lock. In this isolated
                    # 'spawn' subprocess, it creates a SECOND serialization bottleneck
                    # on top of metal_semaphore, causing 30s GPU_TIMEOUT hangs.
                    # metal_semaphore(1) already serializes all GPU access in this worker.

                    response_text = ""
                    total_generated_tokens = 0
                    surface_control_state = _apply_surface_generation_controls(engine, model, job)
                    try:
                        with metal_semaphore:
                            # Proactive cache clearing under memory pressure
                            if mx and device != "cpu":
                                try:
                                    import psutil
                                    if psutil.virtual_memory().percent > 90:  # 64GB — don't panic at 85%
                                        logger.warning("⚠️ High memory pressure detected in worker. Clearing MLX cache.")
                                        mx.clear_cache()
                                except (ImportError, OSError, AttributeError):
                                    pass  # no-op: psutil unavailable or VM stats inaccessible

                            # [v11.5 HARDENING] Internal Worker Retries for Structured Leaks & Loops
                            # We allow up to 2 retries if the LLM gets stuck in a loop or returns empty on a schema.
                            max_internal_retries = 1 if proof_evaluation_contract else 2

                            for internal_attempt in range(max_internal_retries + 1):
                                watchdog.start_job()
                                try:
                                    current_response = ""
                                    token_count = 0
                                    last_progress_emit_at = time.time()
                                    sentinel_aborted = False
                                    sentinel_loop_aborted = False
                                    sentinel_ontology_aborted = False
                                    role_continuation_hit = False

                                    # ── Token Sentinel: mid-generation cognitive intervention ──
                                    # Creates a lightweight monitor that checks for capitulation,
                                    # persona drift, and live-updates affect state during generation.
                                    try:
                                        from core.brain.llm.token_sentinel import (
                                            InterventionType,
                                            TokenSentinel,
                                            get_refusal_fallback,
                                        )
                                        sentinel = TokenSentinel(
                                            check_interval=8,
                                            affect_interval=16,
                                            substrate_mem=substrate_mem,
                                        )
                                    except (ImportError, AttributeError, RuntimeError) as _sent_exc:
                                        _record_mlx_degradation(
                                            _sent_exc,
                                            action="continued generation without TokenSentinel intervention checks",
                                            severity="degraded",
                                        )
                                        sentinel = None
                                        logger.debug("TokenSentinel not available: %s", _sent_exc)

                                    # [FRONTIER UPGRADE] KV Prompt Caching Injection
                                    tokens = tokenizer.encode(prompt)
                                    import mlx_lm.utils as u

                                    def _can_trim(pc):
                                        return hasattr(u, "trim_prompt_cache")

                                    def _do_trim(pc, num):
                                        if hasattr(u, "trim_prompt_cache"):
                                            u.trim_prompt_cache(pc, num)

                                    model_key = id(model)
                                    cache = None
                                    remaining_tokens = tokens
                                    if prompt_cache_lru is not None and not disable_prompt_cache:
                                        cache, remaining_tokens = prompt_cache_lru.fetch_nearest_cache(
                                            model_key, tokens,
                                            can_trim_prompt_cache=_can_trim,
                                            trim_prompt_cache=_do_trim
                                        )

                                    gen_prompt = remaining_tokens if cache is not None else prompt
                                    if cache is not None:
                                        kwargs["prompt_cache"] = cache

                                    # [STABILITY v57] Reset activity immediately before loop to maximize budget for prefill
                                    try:
                                        from mlx_lm.sample_utils import make_sampler
                                        if "sampler" not in kwargs:
                                            import inspect as _insp
                                            _sparams = _insp.signature(make_sampler).parameters
                                            sampler_kwargs = {"temp": kwargs.get("temperature", 0.7)}
                                            if "top_p" in _sparams:
                                                sampler_kwargs["top_p"] = kwargs.get("top_p", 1.0)
                                            if "min_p" in _sparams:
                                                sampler_kwargs["min_p"] = kwargs.get("min_p", 0.0)
                                            if "repetition_penalty" in _sparams:
                                                sampler_kwargs["repetition_penalty"] = kwargs.get("repetition_penalty", 1.0)
                                            if "repetition_context_size" in _sparams:
                                                sampler_kwargs["repetition_context_size"] = kwargs.get("repetition_context_size", 20)
                                            kwargs["sampler"] = make_sampler(**sampler_kwargs)
                                    except ImportError:
                                        pass # old mlx_lm

                                    # [STABILITY v60] Definitive scrub of legacy kwargs.
                                    # New mlx-lm versions pass kwargs directly to generate_step which
                                    # throws TypeError if it sees 'temperature' or 'top_p' instead of 'temp'.
                                    clean_keys = {"temperature", "top_p", "min_p", "repetition_penalty", "repetition_context_size", "stop_words"}
                                    clean_kwargs = {k: v for k, v in kwargs.items() if k not in clean_keys}

                                    watchdog.activity()
                                    for response in stream_generate(model, tokenizer, prompt=gen_prompt, **clean_kwargs):
                                        watchdog.activity()
                                        token_count += 1
                                        progress_now = time.time()

                                        tokens.append(response.token)
                                        # Snag the prompt cache from the response if supported to save for next turn
                                        if (
                                            prompt_cache_lru is not None
                                            and not disable_prompt_cache
                                            and hasattr(response, "prompt_cache")
                                            and response.prompt_cache is not None
                                        ):
                                            prompt_cache_lru.insert_cache(model_key, list(tokens), response.prompt_cache)

                                        current_response += response.text
                                        current_response, role_continuation_hit = _truncate_role_continuation(current_response)

                                        # [STABILITY v58] Explicit break on stop sequences or role drift
                                        if role_continuation_hit:
                                            break

                                        # Manual check for any dynamic stop sequences passed in the job
                                        if any(s in current_response for s in stop_sequences):
                                            for s in stop_sequences:
                                                if s in current_response:
                                                    current_response = current_response.split(s)[0]
                                                    break
                                            break

                                        # ── Sentinel: feed every token ────────────────────
                                        if sentinel is not None:
                                            sentinel_signal = sentinel.feed(response.text)
                                            if sentinel_signal.type == InterventionType.ABORT_LOOP:
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting loop at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                current_response = sentinel_signal.clean_prefix
                                                sentinel_aborted = True
                                                sentinel_loop_aborted = True
                                                break
                                            elif sentinel_signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION:
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting due to ontological violation at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                sentinel_aborted = True
                                                sentinel_ontology_aborted = True
                                                break
                                            elif sentinel_signal.type in (InterventionType.ABORT_CAPITULATION,
                                                                          InterventionType.ABORT_BOUNDARY):
                                                # Mid-generation abort: the LLM started capitulating.
                                                # Replace response with deterministic refusal.
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting generation at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                current_response = get_refusal_fallback(seed=token_count)
                                                sentinel_aborted = True
                                                break

                                        if _should_emit_generation_progress(
                                            token_count,
                                            last_emit_at=last_progress_emit_at,
                                            now=progress_now,
                                        ):
                                            ipc_writer.put({
                                                "id": job.get("id"),
                                                "action": "generate",
                                                "status": "progress",
                                                "tokens_generated": token_count,
                                                "timestamp": progress_now,
                                            })
                                            last_progress_emit_at = progress_now

                                        # [PERF] Mid-generation cache clearing removed — was forcing Metal to
                                        # reallocate GPU memory every 32 tokens, creating micro-stalls.
                                        # Post-generation cleanup (line ~988) still ensures clean state.

                                        # [FRONTIER UPGRADE] Absolute safety cap expanded so it never stops midway
                                        if token_count > 8192:
                                            logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
                                            break

                                        stop_hit = role_continuation_hit
                                        for stop in stop_sequences:
                                            stop_index = current_response.find(stop)
                                            if stop_index > 0:
                                                current_response = current_response[:stop_index]
                                                stop_hit = True
                                                break
                                        if stop_hit:
                                            break

                                    # Log sentinel diagnostics
                                    if sentinel is not None:
                                        diag = sentinel.get_diagnostics()
                                        if diag["interventions"] > 0 or diag["drift_warnings"] > 0:
                                            logger.info(
                                                "🛡️ [SENTINEL] Generation complete: %d interventions, "
                                                "%d drift warnings, %d affect pulses over %d tokens",
                                                diag["interventions"], diag["drift_warnings"],
                                                diag["affect_pulses"], diag["tokens_processed"],
                                            )

                                    response_text = (
                                        f"{operator_response_prefix}{current_response}"
                                        if operator_evidence_contract and current_response.strip()
                                        else current_response
                                    )
                                    if operator_evidence_contract:
                                        trimmed_response = _trim_complete_operator_evidence(response_text)
                                        if trimmed_response != response_text:
                                            logger.warning(
                                                "⚠️ [WORKER] Trimmed clipped/meta operator-evidence tail "
                                                "(chars %d -> %d).",
                                                len(response_text or ""),
                                                len(trimmed_response or ""),
                                            )
                                            response_text = trimmed_response
                                    total_generated_tokens = token_count

                                    if proof_evaluation_contract and _proof_evaluation_fragment_incomplete(response_text):
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Incomplete proof/evaluation response on attempt %s "
                                                "(tokens=%d, chars=%d, role_stop=%s). Retrying with stricter prompt.",
                                                internal_attempt + 1,
                                                token_count,
                                                len(response_text or ""),
                                                role_continuation_hit,
                                            )
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            prompt = _build_proof_evaluation_retry_prompt(
                                                original_messages,
                                                original_prompt,
                                            )
                                            _prepare_clean_retry_kwargs(kwargs, structured=False)
                                            continue
                                        logger.warning(
                                            "🚨 [WORKER] Proof/evaluation response remained incomplete after retries."
                                        )

                                    if operator_evidence_contract and _operator_evidence_fragment_incomplete(response_text):
                                        rejection_reasons = _operator_evidence_rejection_reasons(response_text)
                                        logger.warning(
                                            "⚠️ [WORKER] Rejected operator-evidence draft reasons=%s excerpt=%r",
                                            ",".join(rejection_reasons[:8]) or "unknown",
                                            str(response_text or "").strip()[:360],
                                        )
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Incomplete operator-evidence response on attempt %s "
                                                "(tokens=%d, chars=%d, role_stop=%s). Retrying with stricter prompt.",
                                                internal_attempt + 1,
                                                token_count,
                                                len(response_text or ""),
                                                role_continuation_hit,
                                            )
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            prompt, operator_response_prefix = _build_operator_evidence_retry_prompt(
                                                original_messages,
                                                original_prompt,
                                            )
                                            _prepare_clean_retry_kwargs(kwargs, structured=False)
                                            continue
                                        logger.warning(
                                            "🚨 [WORKER] Operator-evidence response remained incomplete after retries."
                                        )
                                        response_text = ""
                                        break

                                    if sentinel_loop_aborted:
                                        if internal_attempt < max_internal_retries:
                                            logger.warning("⚠️ [WORKER] Retrying generation cleanly after loop abort (attempt %s)...", internal_attempt + 1)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            if proof_evaluation_contract:
                                                prompt = _build_proof_evaluation_retry_prompt(
                                                    original_messages,
                                                    original_prompt,
                                                )
                                            _prepare_clean_retry_kwargs(kwargs, structured=bool(schema))
                                            if "logits_processors" in kwargs:
                                                # We just recreate the logit processors if they exist so it catches the loop
                                                pass

                                            continue
                                        else:
                                            logger.warning("🚨 [WORKER] Out of retries for loop abort. Returning truncated prefix.")
                                            break

                                    if sentinel_ontology_aborted:
                                        if internal_attempt < max_internal_retries:
                                            logger.warning("⚠️ [WORKER] Retrying generation cleanly after ontological violation (attempt %s)...", internal_attempt + 1)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            # Add a slight temperature penalty or just start fresh
                                            _prepare_clean_retry_kwargs(kwargs, structured=bool(schema))
                                            continue
                                        else:
                                            logger.warning("🚨 [WORKER] Out of retries for ontological violation. Returning refusal.")
                                            response_text = get_refusal_fallback(seed=token_count)
                                            break

                                    sanitized_text = _sanitize_telemetry_leakage(response_text, is_proof=proof_evaluation_contract)
                                    if sanitized_text is None:
                                        logger.warning("🚨 [WORKER] Hallucination detected by sanitizer. Returning empty text for caller-side recovery.")
                                        response_text = ""
                                        break
                                        # ipc_writer.put({

                                    response_text = sanitized_text

                                    if strict_answer_contract:
                                        response_text = _normalize_strict_answer_response(
                                            response_text,
                                            envelope_prefixed=strict_envelope_prefixed,
                                        )
                                    elif strict_value_contract:
                                        raw_strict_value_text = response_text
                                        response_text = _normalize_strict_value_response(response_text)
                                        if raw_strict_value_text.strip() and not response_text.strip():
                                            logger.warning(
                                                "⚠️ [WORKER] Strict value draft rejected: %r",
                                                raw_strict_value_text.strip()[:160],
                                            )

                                    if response_text.strip() or (
                                        not schema
                                        and not strict_answer_contract
                                        and not strict_value_contract
                                    ):
                                        break # Success or not a structured task

                                    if strict_answer_contract or strict_value_contract:
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Empty strict response on attempt %s. Retrying...",
                                                internal_attempt + 1,
                                            )
                                            if internal_attempt == 0 or strict_value_contract:
                                                prompt = _build_strict_answer_retry_prompt(
                                                    original_messages,
                                                    original_prompt,
                                                )
                                                strict_envelope_prefixed = bool(strict_answer_contract)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            _prepare_clean_retry_kwargs(kwargs, structured=True)
                                            continue
                                        logger.warning("🚨 [WORKER] Strict contract exhausted internal retries.")
                                        break

                                    logger.warning("⚠️ [WORKER] Empty structured response on attempt %s. Retrying...", internal_attempt + 1)
                                finally:
                                    watchdog.stop_job()
                    finally:
                        _restore_surface_generation_controls(surface_control_state)

                    if not response_text.strip():
                        logger.warning(
                            "⚠️ [WORKER] Generation yielded ZERO tokens. "
                            "Prompt length: %d, token_count: %d, stop_sequences: %s",
                            len(prompt), token_count, list(stop_sequences)[:4],
                        )
                        if len(prompt) > 2000:
                            logger.debug("Prompt snippet: %s...", prompt[:100])
                        # Self-heal: a zero-token generation almost always means
                        # the prompt cache picked up a stale/corrupt KV state
                        # (MLX sampler hit EOS on the first step because the
                        # cached KV disagreed with the fresh prompt).  Nuke
                        # the per-model prompt cache AND the Metal cache so
                        # the very next request starts from a clean state
                        # instead of looping in "Cortex returned no text".
                        try:
                            if prompt_cache_lru is not None:
                                prompt_cache_lru.clear()
                        except (AttributeError, RuntimeError) as exc:
                            logger.debug("Prompt cache clear failed after zero-token generation: %s", exc)
                        if mx and device != "cpu":
                            _clear_mlx_cache(mx)

                    try:
                        if engine is not None:
                            if response_text.strip():
                                engine.observe_generation(response_text)
                            else:
                                engine.observe_generation(
                                    "",
                                    generation_health=0.0,
                                    cross_entropy=10.0,
                                )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as steering_obs_exc:
                        _record_mlx_degradation(
                            steering_obs_exc,
                            action="returned generation after affective steering observation failed",
                            severity="warning",
                        )
                        logger.debug("Affective steering post-generation observation failed: %s", steering_obs_exc)

                    # : Tag with action: "generate" so client can distinguish
                    # from init/heartbeat responses unambiguously.
                    ipc_writer.put({
                        "id": job.get("id"),
                        "action": "generate",
                        "status": "ok",
                        "text": response_text.strip(),
                        "tokens_used": total_generated_tokens
                    })
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_mlx_degradation(
                        e,
                        action="returned generate error and cleared MLX cache after generation failure",
                        severity="degraded",
                    )
                    logger.error("Generation failed: %s", e)
                    ipc_writer.put({"status": "error", "action": "generate", "message": str(e)})
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after standard generation
                    # completes or fails, ensuring pure state for next request.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

            elif action == "stream":
                try:
                    if engine is not None and not engine.is_active():
                        _record_mlx_degradation(
                            RuntimeError("Affective steering became inactive during streaming"),
                            action="blocked stream because steering liveness failed",
                            severity="critical",
                        )
                        logger.error("🚨 [WORKER] Affective steering is inactive! Gating stream.")
                        ipc_writer.put(
                            {
                                "id": job.get("id"),
                                "action": "stream",
                                "status": "error",
                                "message": "Affective steering is inactive; stream blocked.",
                            }
                        )
                        continue
                except (RuntimeError, AttributeError, TypeError) as _e:
                    _record_mlx_degradation(
                        _e,
                        action="blocked stream because steering liveness could not be verified",
                        severity="critical",
                    )
                    logger.error("Failed to check steering active state before stream: %s", _e)
                    ipc_writer.put(
                        {
                            "id": job.get("id"),
                            "action": "stream",
                            "status": "error",
                            "message": "Affective steering liveness check failed; stream blocked.",
                        }
                    )
                    continue

                prompt = job.get("prompt")
                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                try:
                    max_tokens = max(1, int(job.get("max_tokens", 512) or 512))
                except (TypeError, ValueError):
                    max_tokens = 512
                min_p = job.get("min_p", 0.05)
                repetition_penalty = job.get("repetition_penalty", 1.1)

                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp2
                        _sparams2 = _insp2.signature(make_sampler).parameters
                        if "min_p" in _sparams2:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams2:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except (TypeError, ValueError):
                        pass  # no-op: signature introspection unavailable
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

                # Apply MLX penalties via logits processors
                logits_processors = []
                try:
                    from mlx_lm.sample_utils import make_logits_processors
                    _rp = job.get("repetition_penalty", repetition_penalty)
                    _rcs = job.get("repetition_context_size", 30)
                    _pp = job.get("presence_penalty", 0.0)
                    if _rp and _rp > 1.0:
                        lp = make_logits_processors(
                            repetition_penalty=_rp,
                            repetition_context_size=_rcs,
                            presence_penalty=_pp,
                        )
                        if lp:
                            logits_processors.extend(lp)
                except ImportError as _exc:
                    logger.debug("Suppressed %s in core.brain.llm.mlx_worker: %s", type(_exc).__name__, _exc)
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.warning("Could not apply penalty logits processors: %s", e)

                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                stop_sequences = _merge_stop_sequences(job.get("stop_sequences") or [])

                try:
                    from mlx_lm.generate import stream_generate
                    # : NO GPUSentinel — same rationale as generate path.

                    with metal_semaphore:
                        watchdog.start_job()
                        try:
                            full_text = ""
                            token_count = 0

                            # ── Token Sentinel for streaming path ─────────
                            try:
                                from core.brain.llm.token_sentinel import (
                                    InterventionType,
                                    TokenSentinel,
                                    get_refusal_fallback,
                                )
                                stream_sentinel = TokenSentinel(
                                    check_interval=8,
                                    affect_interval=16,
                                    substrate_mem=substrate_mem,
                                )
                            except (ImportError, AttributeError, RuntimeError):
                                stream_sentinel = None

                            # [STABILITY v60] Definitive scrub of legacy kwargs.
                            clean_keys = {"temperature", "top_p", "min_p", "repetition_penalty", "repetition_context_size", "stop_words"}
                            clean_kwargs = {k: v for k, v in kwargs.items() if k not in clean_keys}

                            watchdog.activity()
                            for response in stream_generate(model, tokenizer, prompt=prompt, **clean_kwargs):
                                watchdog.activity()
                                token_count += 1
                                token_text = response.text
                                full_text += token_text
                                full_text, role_continuation_hit = _truncate_role_continuation(full_text)

                                # ── Sentinel: mid-stream intervention ─────
                                if stream_sentinel is not None:
                                    sentinel_signal = stream_sentinel.feed(token_text)
                                    if sentinel_signal.type == InterventionType.ABORT_LOOP:
                                        logger.warning(
                                            "🚨 [SENTINEL-STREAM] Aborting loop at token %d: %s",
                                            token_count, sentinel_signal.reason,
                                        )
                                        ipc_writer.put({
                                            "id": job.get("id"),
                                            "action": "stream",
                                            "status": "sentinel_abort",
                                            "text": "",
                                            "tokens_generated": token_count,
                                            "timestamp": time.time(),
                                        })
                                        break
                                    elif sentinel_signal.type in (InterventionType.ABORT_CAPITULATION,
                                                                  InterventionType.ABORT_BOUNDARY):
                                        logger.warning(
                                            "🚨 [SENTINEL-STREAM] Aborting at token %d: %s",
                                            token_count, sentinel_signal.reason,
                                        )
                                        # Send the refusal as the final token
                                        ipc_writer.put({
                                            "id": job.get("id"),
                                            "action": "stream",
                                            "status": "sentinel_abort",
                                            "text": get_refusal_fallback(seed=token_count),
                                            "tokens_generated": token_count,
                                            "timestamp": time.time(),
                                        })
                                        break

                                ipc_writer.put(
                                    {
                                        "id": job.get("id"),
                                        "action": "stream",
                                        "status": "token",
                                        "text": token_text,
                                        "tokens_generated": token_count,
                                        "timestamp": time.time(),
                                    }
                                )

                                # [FRONTIER UPGRADE] Absolute safety cap natively expanded to frontier levels
                                if token_count > 8192:
                                    logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
                                    break

                                stop_hit = role_continuation_hit
                                for stop in stop_sequences:
                                    stop_index = full_text.find(stop)
                                    if stop_index > 0:
                                        full_text = full_text[:stop_index]
                                        stop_hit = True
                                        break
                                if stop_hit:
                                    break
                        finally:
                            watchdog.stop_job()

                    ipc_writer.put({"status": "ok", "action": "stream_done"})
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_mlx_degradation(
                        e,
                        action="returned stream error and cleared MLX cache after streaming failure",
                        severity="degraded",
                    )
                    logger.error("Streaming failed: %s", e)
                    ipc_writer.put({"status": "error", "action": "stream", "message": str(e)})
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after streaming
                    # completes or fails, ensuring pure state for next request.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

            elif action == "ping":
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                ipc_writer.put({"status": "pong"})

            elif action == "clear_cache":
                # Clear both Metal GPU cache AND the CPU-side prompt-KV cache.
                # The prompt_cache_lru holds KV states that can become polluted
                # after a stalled generation or a partial token stream — if we
                # only clear Metal, the next request will reuse a corrupt KV
                # state and frequently produces zero tokens (the "Cortex
                # returned no text" cascade).  Clearing both is safe; worst
                # case we pay one prompt-encoding re-run.
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                try:
                    if prompt_cache_lru is not None:
                        prompt_cache_lru.clear()
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    pass  # no-op: intentional
                ipc_writer.put({"status": "ok"})

        except KeyboardInterrupt:
            logger.info("🛑 [WORKER] Shutdown signal received; exiting quietly.")
            break
        except (RuntimeError, TypeError, ValueError, OSError, AttributeError) as e:
            _record_mlx_degradation(
                e,
                action="reported worker action error to parent IPC and continued request loop",
                severity="degraded",
            )
            import traceback
            tb = traceback.format_exc()
            resolved_action = locals().get("action") or "unknown"
            logger.error(
                "❌ [WORKER] Unhandled error during '%s': %s\n%s",
                resolved_action, e, tb,
            )
            ipc_writer.put(
                {
                    "status": "error",
                    "action": resolved_action,
                    "message": f"{resolved_action} failed: {e}",
                    "detail": tb,
                }
            )

if __name__ == "__main__":
    print("MLX Worker: Running in multiprocessing mode. Use mlx_client.py to launch.")
