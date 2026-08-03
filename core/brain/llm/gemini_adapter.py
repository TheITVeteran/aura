"""core/brain/llm/gemini_adapter.py
Frontier LLM Adapter for Google Gemini API.

Provides PRIMARY-tier reasoning via Gemini 1.5 Pro/Flash while respecting
free tier rate limits to prevent charges and 429 errors.

Free tier limits (as of March 2026):
    - Gemini Pro:        50 RPD,  2 RPM
    - Gemini Flash:      1500 RPD, 15 RPM

Strategy: Use Flash for streaming chat (250 RPD budget), Pro for deep
reasoning only when explicitly requested (100 RPD budget). Automatic
fallback to local models when daily quota is exhausted.
"""
import asyncio
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx

from core.resilience.factory import circuit_breaker
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.utils.exceptions import capture_and_log
from core.runtime.flags import FlagKind as _FlagKind, declare as _declare_flag

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_GEMINI_CHAT_MODEL = _declare_flag(
    "AURA_GEMINI_CHAT_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-flash",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_GEMINI_DEEP_MODEL = _declare_flag(
    "AURA_GEMINI_DEEP_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-flash",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_GEMINI_THINKING_MODEL = _declare_flag(
    "AURA_GEMINI_THINKING_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-pro",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


logger = logging.getLogger("Brain.Gemini")


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return max(low, min(high, result))


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, result))


GEMINI_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    httpx.HTTPError,
)


def _record_gemini_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "gemini_adapter",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class GeminiProviderUnavailableError(RuntimeError):
    """Non-crash provider/configuration failure; local lanes should continue."""


GeminiProviderUnavailable = GeminiProviderUnavailableError


class DailyRateLimiter:
    """Tracks per-model daily usage and enforces free-tier limits.
    
    Resets at midnight Pacific Time (Google's reset schedule).
    Saves state to disk so restarts don't lose the count.
    """
    
    DEFAULT_LIMITS = {
        "gemini-pro": int(os.environ.get("AURA_GEMINI_RPD_PRO", 2000)),
        "gemini-2.5-flash": int(os.environ.get("AURA_GEMINI_RPD_DEEP", 2000)),
        "gemini-flash-latest": int(os.environ.get("AURA_GEMINI_RPD_FLASH", 10000)),
        "gemini-2.0-flash": int(os.environ.get("AURA_GEMINI_RPD_FLASH", 10000)),
        "gemini-2.5-pro": int(os.environ.get("AURA_GEMINI_RPD_THINKING", 2000)),
        "gemini-3.5-flash": int(os.environ.get("AURA_GEMINI_RPD_FLASH", 10000)),
        "gemini-3.5-pro": int(os.environ.get("AURA_GEMINI_RPD_THINKING", 2000)),
    }
    
    # Per-minute limits (High-performance baseline for paid tiers)
    RPM_LIMITS = {
        "gemini-pro": int(os.environ.get("AURA_GEMINI_RPM_PRO", 50)),
        "gemini-2.5-flash": int(os.environ.get("AURA_GEMINI_RPM_DEEP", 50)),
        "gemini-flash-latest": int(os.environ.get("AURA_GEMINI_RPM_FLASH", 500)),
        "gemini-2.0-flash": int(os.environ.get("AURA_GEMINI_RPM_FLASH", 500)),
        "gemini-2.5-pro": int(os.environ.get("AURA_GEMINI_RPM_THINKING", 50)),
        "gemini-3.5-flash": int(os.environ.get("AURA_GEMINI_RPM_FLASH", 500)),
        "gemini-3.5-pro": int(os.environ.get("AURA_GEMINI_RPM_THINKING", 50)),
    }
    
    def __init__(self, state_path: str | None = None):
        self._counts: dict[str, int] = defaultdict(int)
        self._reset_date: str = self._today()
        self._state_path = state_path
        self._minute_timestamps: dict[str, deque] = defaultdict(deque)
        self._backoff_until: dict[str, float] = {}  # model -> timestamp
        self._cluster_backoff_until: float = 0.0     # All Gemini models
        self._boot_time: float = time.monotonic()
        self.COLD_START_GRACE_S: float = 90.0  # 90s boot grace window
        # Guards the check-and-increment sequence: can_call read the counters
        # and record_call incremented them separately, so N concurrent
        # requests could all pass the check and then all consume, blowing the
        # quota (and, on a paid tier, the spend cap).
        self._lock = threading.RLock()
        self._load_state()

    def try_reserve(self, model: str, is_background: bool = False,
                    priority: float = 0.5) -> bool:
        """Atomically admit AND reserve one call's quota.

        Reserving up front (rather than can_call now, record_call after the
        response) makes admission race-free and also counts the attempt
        against quota immediately — the provider counts failed, 4xx/5xx,
        cancelled, and partial requests too, so recording only completed
        successes let those slip the local counter and overrun the real
        quota.
        """
        with self._lock:
            if not self.can_call(model, is_background=is_background, priority=priority):
                return False
            self.record_call(model)
            return True
    
    def _today(self) -> str:
        """Current date in Pacific time (Google's billing day boundary)."""
        import datetime as dt
        # Approximate Pacific time as UTC-8
        pt = datetime.now(dt.UTC).astimezone(
            dt.timezone(dt.timedelta(hours=-8))
        )
        return pt.strftime("%Y-%m-%d")
    
    def _load_state(self):
        """Load daily counts from disk if available."""
        if self._state_path:
            try:
                import json
                from pathlib import Path
                p = Path(self._state_path)
                if p.exists():
                    data = json.loads(p.read_text())
                    if data.get("date") == self._today():
                        self._counts = defaultdict(int, data.get("counts", {}))
                        self._reset_date = data["date"]
                        logger.info("📊 Loaded Gemini usage: %s", dict(self._counts))
                    else:
                        logger.info("📊 New day — resetting Gemini usage counters")
            except GEMINI_RECOVERABLE_ERRORS as e:
                _record_gemini_degradation(
                    e,
                    action="started Gemini rate limiter with empty in-memory counters after state load failed",
                    extra={"state_path": self._state_path},
                )
                logger.debug("Failed to load rate limiter state: %s", e)
    
    def _save_state(self):
        """Persist daily counts to disk."""
        if self._state_path:
            try:
                from pathlib import Path
                state_path = Path(self._state_path)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(state_path, json.dumps({
                    "date": self._reset_date,
                    "counts": dict(self._counts),
                }))
            except GEMINI_RECOVERABLE_ERRORS as e:
                _record_gemini_degradation(
                    e,
                    action="kept Gemini rate limiter counters in memory after durable state save failed",
                    extra={"state_path": self._state_path},
                )
                capture_and_log(e, {'module': __name__})
    
    def _maybe_reset(self):
        """Reset counters if it's a new day."""
        today = self._today()
        if today != self._reset_date:
            logger.info("📊 Daily reset: Gemini quotas refreshed (%s → %s)", 
                       self._reset_date, today)
            self._counts.clear()
            self._reset_date = today
            self._save_state()
    
    def _check_rpm(self, model: str) -> bool:
        """Check if we're within per-minute rate limit."""
        now = time.monotonic()
        rpm_limit = self.RPM_LIMITS.get(model, 3)
        
        # Clean old timestamps with O(1) amortized popleft
        dq = self._minute_timestamps[model]
        while dq and now - dq[0] >= 60:
            dq.popleft()
        
        return len(dq) < rpm_limit
    
    def _record_rpm(self, model: str):
        """Record a call for RPM tracking."""
        self._minute_timestamps[model].append(time.monotonic())
    
    def mark_429(self, model: str, retry_after: float = 60.0):
        """Mark a 429 backoff for this model."""
        self._backoff_until[model] = time.monotonic() + retry_after
        logger.warning("🚫 Gemini %s: 429 backoff for %.0fs", model, retry_after)
    
    def is_backed_off(self, model: str) -> bool:
        """Check if this model is in a 429 backoff period."""
        now = time.monotonic()
        
        # Check cluster-level backoff first
        if now < self._cluster_backoff_until:
            return True
            
        until = self._backoff_until.get(model, 0)
        if now < until:
            return True
        return False

    def mark_cluster_429(self, retry_after: float = 5.0):
        """Mark a brief cooldown for ALL Gemini models to prevent project quota thumping."""
        self._cluster_backoff_until = time.monotonic() + retry_after
        logger.info("🛡️ Gemini Cluster: Entering %.0fs project-level backoff", retry_after)
    
    def is_cold_start(self) -> bool:
        """True during the first 90s after boot — protect against startup RPM storms."""
        return (time.monotonic() - self._boot_time) < self.COLD_START_GRACE_S

    def can_call(self, model: str, is_background: bool = False, priority: float = 0.5) -> bool:
        """Check if we have remaining quota for this model.
        Background calls are prioritized lower to save credits for chat.
        """
        self._maybe_reset()
        
        # Block non-urgent background calls during boot grace window
        if is_background and self.is_cold_start():
            logger.debug("⏳ Cold-start grace: deferring background Gemini call for %s", model)
            return False
            
        # Check 429 backoff first
        if self.is_backed_off(model):
            return False
        
        # Check RPM
        if not self._check_rpm(model):
            logger.debug("⏳ Gemini %s: RPM limit reached, waiting...", model)
            return False
        
        # Check daily limit
        limit = self.DEFAULT_LIMITS.get(model, 80)
        
        # [Pipeline Hardening] Conservative background limit: Stop using Gemini for background tasks
        # once we hit the preservation threshold (default 30%), preserving 70% for the User.
        preservation_threshold = float(os.environ.get("AURA_GEMINI_BACKGROUND_THRESHOLD", 0.3))
        if is_background and self._counts[model] > (limit * preservation_threshold):
            logger.debug("📉 Preserving Gemini %s quota: background call diverted (threshold: %.1f).", model, preservation_threshold)
            return False

        return self._counts[model] < limit

    def reset_manual(self):
        """Force a reset of all daily counters and backoffs."""
        self._counts.clear()
        self._backoff_until.clear()
        self._reset_date = self._today()
        self._save_state()
        logger.info("📊 Gemini rate limits manually RESET.")
    
    def record_call(self, model: str):
        """Record one API call (attempt) against quota."""
        with self._lock:
            self._maybe_reset()
            self._counts[model] += 1
            self._record_rpm(model)
            remaining = self.DEFAULT_LIMITS.get(model, 80) - self._counts[model]
            if remaining <= 10:
                logger.warning("⚠️ Gemini %s: %d calls remaining today", model, remaining)
            self._save_state()
    
    def get_usage(self) -> dict:
        """Return current usage stats."""
        self._maybe_reset()
        return {
            model: {
                "used": self._counts.get(model, 0),
                "limit": limit,
                "remaining": limit - self._counts.get(model, 0),
            }
            for model, limit in self.DEFAULT_LIMITS.items()
        }


class GeminiAdapter:
    """Adapter for Google Gemini API — slots into IntelligentLLMRouter as PRIMARY tier.
    
    Provides both streaming (generate_text_stream_async) and non-streaming (call)
    interfaces so the router's race_think_stream and think_stream both work.
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    # Gemini 3.5 family: Flash for speed and deep, Pro for thinking/advanced reasoning
    CHAT_MODEL = _FLAG_GEMINI_CHAT_MODEL.value()
    DEEP_MODEL = _FLAG_GEMINI_DEEP_MODEL.value()
    THINKING_MODEL = _FLAG_GEMINI_THINKING_MODEL.value()
    
    def __init__(self, api_key: str, model: str = None, 
                 rate_limiter: DailyRateLimiter | None = None,
                 timeout: float = 120.0):
        self.api_key = api_key
        self.model = model or self.CHAT_MODEL
        self.timeout = timeout
        self.rate_limiter = rate_limiter or DailyRateLimiter()
        self._disabled_until: float = 0.0
        self._disabled_reason: str = ""
        # Last call's telemetry, so the string-only compatibility methods
        # (generate/generate_text_async) do not ERASE a provider failure into
        # an indistinguishable empty string — the router reads this to tell a
        # failed call apart from a genuinely empty answer.
        self._last_generation_metadata: dict[str, Any] = {}
        logger.info("✨ GeminiAdapter initialized: model=%s", self.model)

    def get_last_generation_metadata(self) -> dict[str, Any]:
        """Telemetry from the most recent call, including {'error': ...} on
        failure. Consulted by the router when a string result is empty."""
        return dict(self._last_generation_metadata)

    def _reserve_quota(self, is_background: bool) -> bool:
        """Atomically admit and count one call. Prefers the limiter's
        race-free ``try_reserve``; falls back to the older
        can_call/record_call pair for limiters that predate it."""
        limiter = self.rate_limiter
        reserve = getattr(limiter, "try_reserve", None)
        if callable(reserve):
            return bool(reserve(self.model, is_background=is_background))
        if not limiter.can_call(self.model, is_background=is_background):
            return False
        limiter.record_call(self.model)
        return True

    def is_available(self) -> bool:
        return time.monotonic() >= self._disabled_until

    def availability_reason(self) -> str:
        if self.is_available():
            return ""
        return self._disabled_reason or "gemini_provider_disabled"

    def _mark_provider_unavailable(self, reason: str, cooldown_s: float = 3600.0) -> None:
        self._disabled_reason = str(reason or "gemini_provider_unavailable")
        self._disabled_until = time.monotonic() + max(60.0, float(cooldown_s or 3600.0))
    
    async def close(self):
        return None

    def _get_client(self) -> Any | None:
        """Optional test/diagnostic client seam; production uses NetworkGateway."""
        return None

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """API key travels in the x-goog-api-key HEADER, never the URL.

        A key in the query string leaks into gateway, proxy, exception,
        tracing, and access logs — a far wider exposure surface than a
        header, which logging sinks routinely redact.
        """
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        if extra:
            headers.update(extra)
        return headers

    async def _post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
        response = await asyncio.to_thread(
            get_network_gateway().request,
            "POST",
            url,
            headers=self._auth_headers(),
            data=json.dumps(payload),
            timeout=self.timeout,
            source=f"llm_provider:gemini:{self.model}",
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
    
    async def _handle_error(self, response: httpx.Response):
        """Standardized error handling for Gemini API."""
        error_body = await response.aread()
        await self._handle_error_payload(response.status_code, error_body)

    async def _handle_error_payload(self, status_code: int, error_body: bytes | str):
        """Standardized error handling for Gemini API gateway responses."""
        if isinstance(error_body, str):
            error_body = error_body.encode("utf-8", errors="replace")
        text = error_body.decode('utf-8', errors='replace')
        lowered = text.lower()
        
        if status_code == 429:
            retry_after = self._parse_retry_after(error_body)
            # Permanent Quota exhaustion detection
            if "quota" in text.lower():
                logger.error("🚫 Gemini %s: DAILY QUOTA EXHAUSTED", self.model)
                self.rate_limiter._counts[self.model] = self.rate_limiter.DEFAULT_LIMITS.get(self.model, 80)
                self.rate_limiter._save_state()
            else:
                self.rate_limiter.mark_429(self.model, retry_after)
                # Phase 39: Protect project quota by pulsing a cluster-wide backoff
                self.rate_limiter.mark_cluster_429(min(retry_after, 5.0))
            
            msg = f"🚫 Gemini {self.model}: 429 rate limited, backoff {retry_after:.0f}s"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)
        elif status_code in {401, 403} or any(
            marker in lowered
            for marker in ("permission_denied", "api key", "leaked", "api_key_invalid")
        ):
            reason = f"Gemini {self.model}: provider_auth_failed_http_{status_code}"
            self._mark_provider_unavailable(reason, cooldown_s=24 * 60 * 60)
            logger.warning("%s", reason)
            raise GeminiProviderUnavailable(reason)
        else:
            msg = f"Gemini API error {status_code}: {text[:500]}"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)

    def _parse_retry_after(self, error_body: bytes) -> float:
        """Extract retry-after duration from a 429 error response."""
        try:
            text = error_body.decode('utf-8', errors='replace')
            match = re.search(r'retry in (\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="used default Gemini retry-after backoff after parse failed",
                severity="debug",
            )
            capture_and_log(e, {'module': __name__})
        return 60.0  # Default 60s backoff
    
    @circuit_breaker(service_name="gemini-api")
    async def generate_text_stream_async(
        self, prompt: str, 
        system_prompt: str | None = None,
        cancel_event=None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream-compatible Gemini path routed through the canonical network gateway."""
        is_background = kwargs.get("is_background", False)
        if not self.is_available():
            raise GeminiProviderUnavailable(self.availability_reason())
        # Atomic reserve: admits AND counts the attempt in one step.
        if not self._reserve_quota(is_background):
            msg = f"🚫 Gemini {self.model} local rate limited"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)

        if cancel_event and cancel_event.is_set():
            return

        try:
            injected_client = self._get_client()
            if injected_client is not None and hasattr(injected_client, "stream"):
                # Key in the auth header, not the URL (see _auth_headers).
                url = f"{self.BASE_URL}/models/{self.model}:streamGenerateContent?alt=sse"
                stream_payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
                if system_prompt:
                    stream_payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                async with injected_client.stream(
                    "POST",
                    url,
                    json=stream_payload,
                    headers=self._auth_headers(),
                    timeout=self.timeout,
                ) as response:
                    if getattr(response, "status_code", 200) != 200:
                        await self._handle_error(response)
                    if hasattr(response, "aiter_lines"):
                        async for line in response.aiter_lines():
                            if cancel_event and cancel_event.is_set():
                                return
                            if line:
                                yield line
                    # Already counted at reserve time; do not double-count.
                    return
            ok, text, metadata = await self.call(prompt, system_prompt=system_prompt, **kwargs)
            if not ok:
                raise GeminiProviderUnavailable(str(metadata.get("error") or "gemini_stream_call_failed"))
            if cancel_event and cancel_event.is_set():
                return
            yield text
        except GeminiProviderUnavailable as e:
            logger.warning("Gemini stream unavailable: %s", e)
            raise
        except httpx.TimeoutException as e:
            self._mark_provider_unavailable("gemini_stream_timeout", cooldown_s=300.0)
            _record_gemini_degradation(
                e,
                action="raised provider-unavailable signal so router can fail over after Gemini stream timeout",
                extra={"model": self.model},
            )
            logger.error("Gemini stream timeout: %s", e)
            raise GeminiProviderUnavailable(str(e)) from e
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="raised provider-unavailable signal so router can fail over after Gemini stream error",
                extra={"model": self.model},
            )
            logger.error("Gemini stream error: %s", e)
            raise GeminiProviderUnavailable(str(e)) from e

    @circuit_breaker(service_name="gemini-api")
    async def call(
        self, prompt: str, **kwargs
    ) -> tuple[bool, str, dict[str, Any]]:
        """Non-streaming call — compatible with LLMRouter's _think_internal fallback."""
        is_background = kwargs.get("is_background", False)
        if not self.is_available():
            return False, "", {"error": self.availability_reason()}
        # Atomic reserve: admits AND counts the attempt in one step (the
        # provider counts failed/cancelled requests too).
        if not self._reserve_quota(is_background):
            logger.warning("🚫 Gemini %s rate limited", self.model)
            return False, "", {"error": "Rate limit reached"}

        contents = []
        system_instruction = None
        
        sys_prompt = kwargs.get("system_prompt") or kwargs.get("system", "")
        if sys_prompt:
            system_instruction = {"parts": [{"text": sys_prompt}]}
        
        parts = kwargs.get("parts")
        if not parts:
            # Handle standard 'messages' format if provided
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                # Map to Gemini-style contents
                for msg in messages:
                    role = str(msg.get("role", "user") or "user").strip().lower()
                    content = msg.get("content", "")
                    if role == "system":
                        # Multiple system messages accumulate rather than the
                        # last silently overwriting earlier governance text.
                        if system_instruction is None:
                            system_instruction = {"parts": [{"text": content}]}
                        else:
                            system_instruction["parts"].append({"text": content})
                    elif role in {"user", "model", "assistant"}:
                        contents.append({
                            "role": "user" if role == "user" else "model",
                            "parts": [{"text": content}]
                        })
                    else:
                        # tool/function/developer/malformed/attacker-chosen
                        # roles are NOT silently promoted to model authority —
                        # they carry no authority and are dropped with a note.
                        _record_gemini_degradation(
                            ValueError(f"unrecognized_message_role:{role[:32]}"),
                            action="dropped a message with an unrecognized role from the Gemini request",
                            severity="warning",
                            extra={"model": self.model},
                        )
                
                # If we have contents, we skip the prompt-based construction below
                if contents:
                    parts = True # sentinel to skip next block
            
        if not parts:
            # Guard: prompt can be empty when user input is in system_prompt
            if prompt and prompt.strip():
                parts = [{"text": prompt}]
            elif sys_prompt:
                # Move system_prompt to user content so Gemini has valid data
                parts = [{"text": sys_prompt}]
                system_instruction = None  # Don't double-send
            else:
                return False, "", {"error": "No content to send to Gemini"}
        
        if contents:
            # Already built via messages
            pass  # no-op: intentional
        else:
            # Final guard: filter out any parts with empty/None text unless they have inlineData
            parts = [p for p in parts] if isinstance(parts, list) else []
            parts = [p for p in parts if p.get("text") or p.get("inlineData")]
            if not parts:
                return False, "", {"error": "All parts were empty after filtering"}
                
            contents.append({
                "role": "user",
                "parts": parts
            })
        
        # Validate generation params to finite, in-range values: an invalid
        # temperature or max_tokens forwarded verbatim can be rejected by the
        # provider (a wasted, quota-counted call) or produce undefined
        # sampling behavior.
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": _clamp_float(kwargs.get("temperature"), 0.8, 0.0, 2.0),
                "maxOutputTokens": _clamp_int(kwargs.get("max_tokens"), 2048, 1, 32768),
                "topP": 0.95,
            },
        }
        
        # [v11.0 HARDENING] Native JSON Mode & Schema Enforcement
        schema = kwargs.get("schema")
        if schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(schema, dict):
                payload["generationConfig"]["responseSchema"] = schema
            logger.info("🎯 Gemini: JSON Mode ACTIVE with schema enforcement.")

        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        # Key is in the auth header (see _auth_headers), not the URL.
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        
        metadata = {
            "model": self.model,
            "endpoint": "gemini_frontier",
            "latency_ms": 0,
        }
        
        t0 = time.monotonic()
        
        try:
            injected_client = self._get_client()
            if injected_client is not None and hasattr(injected_client, "post"):
                response = await injected_client.post(url, json=payload, timeout=self.timeout)
                metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
                if response.status_code != 200:
                    try:
                        await self._handle_error(response)
                    except GeminiProviderUnavailable as e:
                        return False, "", {"error": str(e)}
                    except GEMINI_RECOVERABLE_ERRORS as e:
                        _record_gemini_degradation(
                            e,
                            action="returned failed Gemini call result after provider error handler failed",
                            extra={"model": self.model, "status_code": response.status_code},
                        )
                        return False, "", {"error": str(e)}
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    return False, "", {"error": str(e)}
            else:
                status_code, data, body_text = await self._post_json(url, payload)
                metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)

                if status_code != 200:
                    try:
                        await self._handle_error_payload(status_code, body_text)
                    except GeminiProviderUnavailable as e:
                        return False, "", {"error": str(e)}
                    except GEMINI_RECOVERABLE_ERRORS as e:
                        _record_gemini_degradation(
                            e,
                            action="returned failed Gemini call result after provider error handler failed",
                            extra={"model": self.model, "status_code": status_code},
                        )
                        return False, "", {"error": str(e)}

            # Already counted at reserve time; do not double-count.
            candidates = data.get("candidates", [])
            if not candidates:
                return False, "", {"error": "No candidates in response"}
            
            # Surface the candidate's terminal status instead of blindly
            # concatenating parts: a SAFETY/RECITATION block or a MAX_TOKENS
            # truncation is a FAILURE, not a clean answer, and a prompt-level
            # block (promptFeedback.blockReason) means no valid content at all.
            prompt_block = str(
                data.get("promptFeedback", {}).get("blockReason", "") or ""
            )
            if prompt_block:
                return False, "", {"error": f"gemini_prompt_blocked:{prompt_block}"}
            candidate = candidates[0]
            finish_reason = str(candidate.get("finishReason", "") or "")
            metadata["finish_reason"] = finish_reason
            if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
                return False, "", {"error": f"gemini_finish_{finish_reason.lower()}"}

            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

            # PROVIDER RECEIPT (CP126 inference-gate 8ff3084b /
            # llm_health_router 3bc237f4). Provenance was previously asserted
            # from the ROUTER's own endpoint record, so a misregistered,
            # proxied or deceptive client was described identically to a real
            # one. These fields come from the PROVIDER's response body, not
            # from our configuration: responseId is generated server-side and
            # modelVersion is the model that actually answered.
            response_id = str(data.get("responseId") or "").strip()
            model_version = str(data.get("modelVersion") or "").strip()
            if response_id or model_version:
                metadata["provider_receipt"] = {
                    "provider": "gemini",
                    "response_id": response_id,
                    "model_version": model_version,
                    "requested_model": str(kwargs.get("model") or self.model or ""),
                    "source": "provider_response_body",
                }
                # The model that ANSWERED must be the one we asked for.
                requested = str(kwargs.get("model") or self.model or "")
                if model_version and requested and not model_version.startswith(requested.split("-latest")[0]):
                    metadata["provider_receipt"]["model_version_mismatch"] = True

            # Extract token usage
            usage = data.get("usageMetadata", {})
            metadata["tokens_used"] = usage.get("totalTokenCount", 0)
            metadata["prompt_tokens"] = usage.get("promptTokenCount", 0)
            metadata["completion_tokens"] = usage.get("candidatesTokenCount", 0)
            metadata["truncated"] = finish_reason == "MAX_TOKENS"

            if not text:
                return False, "", {"error": "Empty response from Gemini"}

            # If a JSON schema was requested, VERIFY the output actually parses
            # as JSON rather than trusting the provider to have enforced it —
            # provider omission or an unsupported schema silently returns prose
            # that downstream JSON consumers then choke on.
            requested_schema = kwargs.get("schema")
            if requested_schema:
                try:
                    json.loads(text)
                    metadata["schema_verified"] = True
                except (json.JSONDecodeError, TypeError, ValueError):
                    metadata["schema_verified"] = False
                    return False, "", {"error": "gemini_schema_not_satisfied"}
            
            # [Phase 18] Record Metabolic Cost
            try:
                from core.ops.metabolic_monitor import get_cost_tracker
                get_cost_tracker().record_operation(
                    op_type="gemini_call",
                    tokens=metadata.get("tokens_used", 0),
                    duration_s=(time.monotonic() - t0),
                    model_tier="PRIMARY" if self.model == self.DEEP_MODEL else "SECONDARY"
                )
            except GEMINI_RECOVERABLE_ERRORS as _e:
                _record_gemini_degradation(
                    _e,
                    action="returned Gemini call result after metabolic cost recording failed",
                    severity="debug",
                    extra={"model": self.model, "tokens_used": metadata.get("tokens_used", 0)},
                )
                logger.debug('Ignored Exception in gemini_adapter.py: %s', _e)
            
            return True, text, metadata
            
        except TimeoutError:
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": f"Timeout after {self.timeout}s"}
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="returned failed Gemini call result after recoverable client error",
                extra={"model": self.model},
            )
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": str(e)}

    async def generate_text_async(
        self, prompt: str, 
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> str:
        """Compatibility method for code paths that call generate_text_async.

        The per-call ``model`` argument is honored (it was previously accepted
        and ignored, so a caller believed a requested model served while
        ``self.model`` was always used).
        """
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        return self._string_result(await self._call_with_model(prompt, model, kwargs))

    async def generate(
        self, prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """Compatibility method for LLM router's generate()."""
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        return self._string_result(
            await self._call_with_model(prompt, kwargs.pop("model", None), kwargs)
        )

    async def _call_with_model(
        self, prompt: str, model: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any]]:
        """Run one call, optionally against a requested model, stashing its
        telemetry so a failure is discoverable through the string API."""
        if model and model != self.model:
            original = self.model
            self.model = model
            try:
                result = await self.call(prompt, **kwargs)
            finally:
                self.model = original
        else:
            result = await self.call(prompt, **kwargs)
        success, text, meta = result
        provider_receipt = (meta or {}).get("provider_receipt")
        self._last_generation_metadata = {
            "provider": "gemini",
            "model": model or self.model,
            "ok": bool(success),
            **({"provider_receipt": dict(provider_receipt)} if isinstance(provider_receipt, dict) else {}),
            **({} if success else {"error": str((meta or {}).get("error", "gemini_call_failed"))}),
        }
        return result

    @staticmethod
    def _string_result(result: tuple[bool, str, dict[str, Any]]) -> str:
        success, text, _meta = result
        return text if success else ""

    async def think(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        **kwargs
    ) -> str | None:
        """
        Unified interface for non-chat callers.
        Returns the simplified string result from Gemini.
        """
        # Distinguish between prompt string and message list
        if not prompt and "messages" in kwargs:
            msgs = kwargs.get("messages", [])
            if msgs and isinstance(msgs, list):
                prompt = msgs[-1].get("content", "")
                if not system_prompt:
                    system_prompt = next((m["content"] for m in msgs if m["role"] == "system"), None)

        if not prompt:
            return None

        # Call generate with the prompt
        text = await self.generate(prompt, system_prompt=system_prompt, **kwargs)
        return text if text and text.strip() else None

    async def unload_models(self):
        """No-op for API models — nothing to unload."""
        return None
    
    async def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs):
        """Alias for generate_text_stream_async — matches the interface expected by LLMRouter.think_stream."""
        async for chunk in self.generate_text_stream_async(prompt, system_prompt=system_prompt, **kwargs):
            yield chunk

    def get_usage_stats(self) -> dict:
        """Return human-readable usage stats for the UI."""
        return self.rate_limiter.get_usage()
