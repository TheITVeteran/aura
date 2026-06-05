import asyncio
import json
import logging
import os
import time
from typing import Any

from core.config import config
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.LocalLLM")

_DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_LOCAL_LLM_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_llm_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def _normalize_base_url(value: Any) -> str | None:
    candidate = str(value or "").strip().rstrip("/")
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate


def _resolve_ollama_base_url() -> str:
    """Resolve the raw Ollama fallback endpoint from current and legacy contracts."""
    for candidate in (
        getattr(config.llm, "base_url", None),
        os.getenv("AURA_OLLAMA_URL"),
        os.getenv("OLLAMA_HOST"),
    ):
        resolved = _normalize_base_url(candidate)
        if resolved:
            return resolved
    return _DEFAULT_OLLAMA_BASE_URL


def _resolve_default_model() -> str:
    return str(
        getattr(config.llm, "model", None)
        or getattr(config.llm, "fast_model", None)
        or getattr(config.llm, "chat_model", None)
        or "llama3.1"
    )


# Configurable timeouts (Increased for Llama 3.1 8b)
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 300.0  # 5 minutes to allow slow model loading
_DEFAULT_REQUEST_TIMEOUT = _READ_TIMEOUT

# ── Model Tier Defaults (v27: Upgraded for M5/64GB) ─────────────────────────
# Old values (v26): num_ctx=4096, num_predict=512 — cripplingly low
_MODEL_TIER_DEFAULTS = {
    "coding": {"num_ctx": 16384, "num_predict": 4096, "temperature": 0.4},
    "chat": {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.7},
    "summary": {"num_ctx": 8192, "num_predict": 1024, "temperature": 0.3},
    "default": {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.7},
}

# ── Thought Circulation (from gemini-skills) ────────────────────────────────
_CODING_KEYWORDS = {
    "code",
    "function",
    "class",
    "implement",
    "debug",
    "fix",
    "refactor",
    "script",
    "api",
    "endpoint",
    "database",
    "sql",
    "query",
    "test",
    "error",
    "bug",
    "traceback",
    "exception",
    "compile",
    "build",
    "deploy",
    "docker",
    "git",
    "commit",
    "merge",
    "python",
    "javascript",
    "typescript",
    "rust",
    "java",
    "create a",
    "write a",
    "modify",
    "html",
    "css",
    "react",
    "fastapi",
    "flask",
    "django",
}

_THOUGHT_CIRCULATION_DIRECTIVE = """Before responding, reason step-by-step in <think> tags about:
1. What the user is asking for and any implicit requirements
2. What files, APIs, or systems are involved
3. Edge cases, error handling, and potential pitfalls
4. Your exact implementation approach
Then provide your implementation outside the think tags."""


def detect_task_tier(prompt: str, system_prompt: str = "") -> str:
    """Detect the task tier from the prompt content."""
    combined = (prompt + " " + (system_prompt or "")).lower()
    for keyword in _CODING_KEYWORDS:
        if keyword in combined:
            return "coding"
    # Check for summary/compression tasks
    if any(kw in combined for kw in ["summarize", "compress", "distill", "snapshot"]):
        return "summary"
    return "chat"


class LocalBrain:
    """Sovereign Intelligence Bridge connecting to local Ollama instance.
    v6.1: Retry logic, proper client lifecycle, configurable timeouts.
    """

    def __init__(self, model_name: str = None):
        self.base_url = _resolve_ollama_base_url()
        self.model = model_name or _resolve_default_model()
        self.timeout = getattr(config.llm, "timeout", config.llm_request_timeout_s)
        self._warmed = False
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    # --- Lifecycle ---
    async def close(self):
        return None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, Any], str]:
        url = f"{self.base_url}{path}"
        response = await asyncio.to_thread(
            get_network_gateway().request,
            method,
            url,
            headers={"Content-Type": "application/json"} if payload is not None else None,
            data=json.dumps(payload) if payload is not None else None,
            timeout=timeout or _DEFAULT_REQUEST_TIMEOUT,
            source=f"llm_provider:local_ollama:{self.model}",
            read_only=True,
        )
        body = response.get("content") or b""
        text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
        data: dict[str, Any] = {}
        if text.strip():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {}
        return int(response.get("status_code") or 0), data, text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # --- Health Checks ---
    def check_health(self) -> bool:
        """Sovereign health check through the canonical network gateway."""
        try:
            response = get_network_gateway().request(
                "GET",
                f"{self.base_url}/api/tags",
                timeout=3,
                source=f"llm_provider:local_ollama_health:{self.model}",
                read_only=True,
            )
            return int(response.get("status_code") or 0) == 200
        except (OSError, RuntimeError, TypeError, ValueError, ConnectionError, TimeoutError):
            return False

    async def check_health_async(self) -> bool:
        """Sovereign health check (Async)."""
        return await asyncio.to_thread(self.check_health)

    async def warmup(self) -> bool:
        """v6.0: Pre-warm the model by sending a minimal generate request.
        This loads the model into GPU memory for faster first-response.
        """
        if self._warmed:
            return True
        try:
            logger.info("🔥 Warming up model: %s", self.model)
            status_code, _data, text = await self._request_json(
                "POST",
                "/api/generate",
                payload={
                    "model": self.model,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=120,
            )
            if status_code != 200:
                raise RuntimeError(f"warmup_http_{status_code}:{text[:160]}")
            self._warmed = True
            logger.info("🔥 Model %s warmed up successfully", self.model)
            return True
        except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
            _record_llm_degradation(
                "local_llm",
                e,
                action="left local model cold after warmup failed",
                severity="warning",
                extra={"model": self.model},
            )
            logger.warning("Model warmup failed: %s", e)
            return False

    # --- Circuit Breaker ---
    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows the call."""
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                logger.warning("Circuit breaker OPEN — skipping LLM call")
                return False
            # Half-open: allow one attempt
            self._circuit_open = False
        return True

    def _record_success(self):
        """Record a successful call."""
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    def _record_failure(self):
        """Record a failed call. Opens circuit after 5 consecutive failures."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 30.0  # 30s recovery
            logger.error(
                "Circuit breaker OPENED after %s failures. Cooldown: 30s",
                self._consecutive_failures,
            )

    # --- Generation ---
    # --- Helper: DeepSeek Think Extraction ---
    THOUGHT_STREAM_PREFIX = "__THOUGHT__:"
    THOUGHT_STREAM_SUFFIX = ":__ENDTHOUGHT__"

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from static text (legacy compat)."""
        return self._extract_think_segments(text)[0]

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        """Extract thinking content and cleaned response from <think> tagged text.

        Returns:
            (cleaned_response, thought_content)
        """
        import re

        thoughts: list[str] = []
        for m in re.finditer(r"<think>(.*?)</think>", text, flags=re.DOTALL):
            thought_text = m.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        # Remove balanced tags
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove stray closing tags if any
        cleaned = cleaned.replace("</think>", "")
        # Remove stray opening tags if at the end (incomplete thought)
        cleaned = cleaned.replace("<think>", "")
        return cleaned.strip(), "\n\n".join(thoughts)

    # --- Generation ---
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, str]:
        """Send a generation request to the local Ollama instance (Async).
        v15: Task-tier-aware defaults, thought circulation, 3 retries w/ exp backoff.
        """
        if not self._check_circuit():
            return {
                "response": "Error: Local brain temporarily unavailable (circuit breaker open).",
                "thought": "",
            }

        try:
            from core.brain.unified_inference import UnifiedInferenceEngine

            engine = UnifiedInferenceEngine()
            res = await engine.generate_unified(
                prompt=prompt,
                system_prompt=system_prompt,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
            self._record_success()
            return res
        except _LOCAL_LLM_RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="fell back to raw Ollama generation after unified inference failed",
                severity="warning",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            logger.warning(
                "UnifiedInferenceEngine failed, falling back to raw Ollama call: %s", exc
            )

        # v27: Task-tier-aware defaults (replaces hard 4K/512 ceiling)
        task_tier = detect_task_tier(prompt, system_prompt or "")
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # v27: Thought circulation injection for coding tasks
        effective_system = system_prompt or ""
        if task_tier == "coding" and _THOUGHT_CIRCULATION_DIRECTIVE not in effective_system:
            effective_system = (
                f"{effective_system}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}"
                if effective_system
                else _THOUGHT_CIRCULATION_DIRECTIVE
            )

        url = "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": final_options,
        }
        if effective_system:
            payload["system"] = effective_system

        last_error = None
        max_retries = 4  # v27: Increased from 2 to 4 (3 retries)
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Ollama Request: %s | Tier: %s | Prompt len: %d | Attempt: %d/%d",
                    self.model,
                    task_tier,
                    len(prompt),
                    attempt + 1,
                    max_retries,
                )
                status_code, data, text = await self._request_json(
                    "POST",
                    url,
                    payload=payload,
                    timeout=float(self.timeout or _DEFAULT_REQUEST_TIMEOUT),
                )
                if status_code != 200:
                    raise RuntimeError(f"ollama_generate_http_{status_code}:{text[:160]}")
                self._record_success()

                # Extract thinking segments
                raw_response = data.get("response", "").strip()
                cleaned, thought = self._extract_think_segments(raw_response)
                return {"response": cleaned, "thought": thought}

            except (OSError, TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = min(2**attempt, 8)  # 1s, 2s, 4s, 8s
                    logger.warning(
                        "Transient LLM error (retry %d in %ds): %s", attempt + 1, backoff, e
                    )
                    await asyncio.sleep(backoff)
                    continue
            except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
                _record_llm_degradation(
                    "local_llm",
                    e,
                    action="stopped raw Ollama generation retries and returned unreachable response",
                    extra={"model": self.model, "attempt": attempt + 1},
                )
                last_error = e
                break

        self._record_failure()
        logger.error("Local LLM Error: %s", last_error)
        return {
            "response": f"Error: Local brain (Ollama) is unreachable. {str(last_error)}",
            "thought": "",
        }

    async def generate_text_stream_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Stream tokens from the local Ollama instance.
        v13B: Filters <think> blocks in real-time.
        """
        if not self._check_circuit():
            yield "Error: Local brain temporarily unavailable."
            return

        # v27: Task-tier-aware defaults for text streaming
        task_tier = detect_task_tier(prompt, system_prompt or "")
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # Homeostatic modulation injection for streaming
        try:
            from core.brain.homeostatic_modulator import HomeostaticModulator

            _mod = HomeostaticModulator().compute_modulation()
            final_options.setdefault("temperature", _mod.temperature)
            final_options["top_p"] = _mod.top_p
            final_options["repeat_penalty"] = _mod.repetition_penalty
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            _record_llm_degradation(
                "local_llm_stream_modulation",
                exc,
                action="streamed with tier defaults after homeostatic modulation lookup failed",
                severity="warning",
                extra={"model": self.model},
            )
            logger.debug("Homeostatic modulation unavailable for stream: %s", exc)

        # v27: Thought circulation for coding
        effective_system = system_prompt or ""
        if task_tier == "coding" and _THOUGHT_CIRCULATION_DIRECTIVE not in effective_system:
            effective_system = (
                f"{effective_system}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}"
                if effective_system
                else _THOUGHT_CIRCULATION_DIRECTIVE
            )

        url = "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": final_options,
        }
        if effective_system:
            payload["system"] = effective_system

        try:
            if cancel_event and cancel_event.is_set():
                logger.info("Stream cancelled by caller")
                return
            status_code, data, text = await self._request_json(
                "POST",
                url,
                payload={**payload, "stream": False},
                timeout=float(self.timeout or _DEFAULT_REQUEST_TIMEOUT),
            )
            if status_code != 200:
                raise RuntimeError(f"ollama_stream_http_{status_code}:{text[:160]}")
            cleaned, thought = self._extract_think_segments(str(data.get("response", "") or ""))
            if cleaned:
                yield cleaned
            if thought:
                yield f"{self.THOUGHT_STREAM_PREFIX}{thought}{self.THOUGHT_STREAM_SUFFIX}"
            self._record_success()
        except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
            _record_llm_degradation(
                "local_llm",
                e,
                action="yielded stream interruption marker after Ollama stream failed",
                extra={"model": self.model},
            )
            self._record_failure()
            logger.error("Streaming Error: %s", e)
            yield f" [Error: Sovereign stream interrupted: {str(e)}]"

    # --- Chat ---
    async def chat(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, str]:
        """Async chat interface for the orchestrator.
        v15: Task-tier-aware defaults, thought circulation, 3 retries w/ exp backoff.
        """
        if not self._check_circuit():
            return {
                "response": "Error: Local brain temporarily unavailable (circuit breaker open).",
                "thought": "",
            }

        try:
            from core.brain.unified_inference import UnifiedInferenceEngine

            engine = UnifiedInferenceEngine()
            res = await engine.generate_unified(
                prompt="",
                messages=messages,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
            self._record_success()
            return res
        except _LOCAL_LLM_RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="fell back to raw Ollama chat after unified inference failed",
                severity="warning",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            logger.warning(
                "UnifiedInferenceEngine failed, falling back to raw Ollama chat: %s", exc
            )

        # v27: Detect task tier from message content
        all_content = " ".join(m.get("content", "") for m in messages[-3:])  # Last 3 messages
        task_tier = detect_task_tier(all_content)
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # v27: Thought circulation — inject into system message for coding tasks
        effective_messages = list(messages)
        if task_tier == "coding" and effective_messages:
            if effective_messages[0].get("role") == "system":
                sys_content = effective_messages[0]["content"]
                if _THOUGHT_CIRCULATION_DIRECTIVE not in sys_content:
                    effective_messages[0] = {
                        **effective_messages[0],
                        "content": f"{sys_content}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}",
                    }
            else:
                effective_messages.insert(
                    0, {"role": "system", "content": _THOUGHT_CIRCULATION_DIRECTIVE}
                )

        url = "/api/chat"
        payload = {
            "model": self.model,
            "messages": effective_messages,
            "stream": False,
            "keep_alive": "30m",
            "options": final_options,
        }

        last_error = None
        max_retries = 4
        for attempt in range(max_retries):
            try:
                status_code, data, text = await self._request_json(
                    "POST",
                    url,
                    payload=payload,
                    timeout=float(self.timeout or _DEFAULT_REQUEST_TIMEOUT),
                )
                if status_code != 200:
                    raise RuntimeError(f"ollama_chat_http_{status_code}:{text[:160]}")
                self._record_success()

                # Extract thinking segments
                raw_response = data.get("message", {}).get("content", "").strip()
                cleaned, thought = self._extract_think_segments(raw_response)
                return {"response": cleaned, "thought": thought}

            except (OSError, TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = min(2**attempt, 8)
                    logger.warning(
                        "Transient chat error (retry %d in %ds): %s", attempt + 1, backoff, e
                    )
                    await asyncio.sleep(backoff)
                    continue
            except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
                _record_llm_degradation(
                    "local_llm",
                    e,
                    action="stopped raw Ollama chat retries and returned interruption response",
                    extra={"model": self.model, "attempt": attempt + 1},
                )
                last_error = e
                break

        self._record_failure()
        logger.error("Local Chat Error: %s", last_error)
        return {
            "response": "Internal Error: Sovereign cognitive stream interrupted.",
            "thought": "",
        }

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Stream chat tokens from the local Ollama instance.
        v13B: Filters <think> blocks in real-time.
        """
        if not self._check_circuit():
            yield "Error: Local brain temporarily unavailable."
            return

        # v27: Task-tier-aware defaults for streaming chat
        all_content = " ".join(m.get("content", "") for m in messages[-3:])
        task_tier = detect_task_tier(all_content)
        tier_defaults = _MODEL_TIER_DEFAULTS.get(task_tier, _MODEL_TIER_DEFAULTS["default"])

        final_options = {
            "temperature": tier_defaults["temperature"],
            "num_predict": tier_defaults["num_predict"],
            "num_ctx": tier_defaults["num_ctx"],
        }
        if options:
            final_options.update(options)

        # Homeostatic modulation injection for streaming chat
        try:
            from core.brain.homeostatic_modulator import HomeostaticModulator

            _mod = HomeostaticModulator().compute_modulation()
            final_options.setdefault("temperature", _mod.temperature)
            final_options["top_p"] = _mod.top_p
            final_options["repeat_penalty"] = _mod.repetition_penalty
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            _record_llm_degradation(
                "local_llm_chat_stream_modulation",
                exc,
                action="streamed chat with tier defaults after homeostatic modulation lookup failed",
                severity="warning",
                extra={"model": self.model},
            )
            logger.debug("Homeostatic modulation unavailable for chat stream: %s", exc)

        # v27: Thought circulation for coding
        effective_messages = list(messages)
        if task_tier == "coding" and effective_messages:
            if effective_messages[0].get("role") == "system":
                sys_content = effective_messages[0]["content"]
                if _THOUGHT_CIRCULATION_DIRECTIVE not in sys_content:
                    effective_messages[0] = {
                        **effective_messages[0],
                        "content": f"{sys_content}\n\n{_THOUGHT_CIRCULATION_DIRECTIVE}",
                    }

        url = "/api/chat"
        payload = {
            "model": self.model,
            "messages": effective_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": final_options,
        }

        try:
            if cancel_event and cancel_event.is_set():
                logger.info("Chat stream cancelled by caller")
                return
            status_code, data, text = await self._request_json(
                "POST",
                url,
                payload={**payload, "stream": False},
                timeout=float(self.timeout or _DEFAULT_REQUEST_TIMEOUT),
            )
            if status_code != 200:
                raise RuntimeError(f"ollama_chat_stream_http_{status_code}:{text[:160]}")
            raw_response = str(data.get("message", {}).get("content", "") or "")
            cleaned, thought = self._extract_think_segments(raw_response)
            if cleaned:
                yield cleaned
            if thought:
                yield f"{self.THOUGHT_STREAM_PREFIX}{thought}{self.THOUGHT_STREAM_SUFFIX}"
            self._record_success()
        except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
            _record_llm_degradation(
                "local_llm",
                e,
                action="yielded chat stream interruption marker after Ollama stream failed",
                extra={"model": self.model},
            )
            self._record_failure()
            logger.error("Chat Streaming Error: %s (%s)", e, type(e).__name__)
            yield f" [Error: Sovereign chat stream interrupted: {type(e).__name__}: {str(e) or repr(e)}]"
