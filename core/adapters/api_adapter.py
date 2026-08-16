"""core/adapters/api_adapter.py — Aura APIAdapter v1.0
=============================================
Unified multi-model API client.

Provides a single interface for all LLM backends:
  - Google Gemini  (api_deep/api_fast)
  - local          → Aura's managed on-device runtime

Config (reads from Aura's existing config / env):
  GEMINI_API_KEY      → enables Gemini

Usage:
    adapter = APIAdapter()
    await adapter.start()
    response = await adapter.generate(prompt, {"model_tier": "api_fast"})
"""

import asyncio
import contextvars
import hashlib
import inspect
import logging
import os
import re
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.adapters.prompt_boundary import split_prompt, structured_prompt
from core.adapters.provider_receipt import (
    digest,
    provider_receipt,
    reported_token_count,
)
from core.adapters.provider_tools import MAX_TOOLS_PER_REQUEST, admissible_tools
from core.brain.llm.cloud_errors import cloud_call_error_types
from core.runtime.errors import Severity, record_degradation

logger = logging.getLogger("Aura.APIAdapter")

#: Per-task generation provenance. See APIAdapter.__init__ for why this
#: is not an instance field.
_LAST_GENERATION: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "api_adapter_last_generation", default={}
)


def _record_api_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "api_adapter",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


def _load_dotenv_once() -> None:
    """Load .env at START, not at import.

    Importing this module used to search the filesystem for a dotenv file
    and merge it into ``os.environ`` for the whole process, before Aura's
    config had been built and before anything owned the secrets it
    injected. An import is not a place to acquire credentials: whoever
    imports the adapter for a type annotation changed the environment
    every later reader sees (CP126 ``58160fbd``).

    Called from ``start()``, once, and only when config did not already
    supply the key.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError as exc:
        logger.debug("python-dotenv not installed; skipping .env: %s", exc)


_DOTENV_LOADED = False

try:
    from core.schemas import ChatStreamEvent
except ImportError:
    ChatStreamEvent = Any


# ─── Model definitions ───────────────────────────────────────────────────────

# Typed admission bounds for caller-supplied generation controls. These
# values flow into paid provider requests and local decode budgets, so they
# are validated here rather than trusted from config.
_VALID_TIERS = {"local", "api_fast", "api_deep"}
_MAX_OUTPUT_TOKENS = 32768
_MAX_PROMPT_CHARS = 500_000


def _bounded_float(value: Any, *, default: float, low: float, high: float) -> float:
    """Finite float in [low, high]; NaN/inf/garbage fall back to default."""
    try:
        candidate = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    if candidate != candidate or candidate in (float("inf"), float("-inf")):
        return default
    return max(low, min(high, candidate))


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        candidate = int(default if value is None else value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, candidate))


#: Model per tier. The two entries were identical, so "deep" and "fast"
#: advertised a capability distinction that did not exist and nothing
#: checked (CP126 ``62c31a55``). They still may be identical — that is a
#: deployment fact, not a bug — but the adapter now MEASURES it and says so
#: in the metadata rather than implying a difference by naming one.
GEMINI_MODELS = {
    "api_deep": os.getenv("AURA_GEMINI_DEEP_MODEL", "gemini-2.0-flash"),
    "api_fast": os.getenv("AURA_GEMINI_FAST_MODEL", "gemini-2.0-flash"),
}


def gemini_tiers_are_distinct() -> bool:
    """Whether api_deep and api_fast actually resolve to different models."""
    return GEMINI_MODELS["api_deep"] != GEMINI_MODELS["api_fast"]


def resolve_gemini_model(tier: str) -> str:
    """The model for a tier, refusing a tier this adapter does not serve."""
    if tier not in GEMINI_MODELS:
        raise ValueError(f"api_adapter serves no Gemini model for tier {tier!r}")
    return GEMINI_MODELS[tier]

try:
    from core.brain.llm.mlx_client import get_mlx_client
    _HAS_LOCAL_RUNTIME = True
except ImportError:
    _HAS_LOCAL_RUNTIME = False


class _StreamFailed(RuntimeError):
    """A provider stream ended without completing.

    Raised by the provider legs so the ROUTER decides what the caller sees.
    The legs used to swallow their own failures and simply stop yielding,
    which is indistinguishable from a completed stream (CP126 ``88bb1083``).
    """


# ─── APIAdapter ──────────────────────────────────────────────────────────────

class APIAdapter:
    """
    Unified LLM client with automatic fallback.
    Integrates with Aura's existing config and ServiceContainer.
    """
    name = "api_adapter"

    # Bound on the cloud embedding round-trip; without it a stalled
    # provider held the calling task indefinitely.
    EMBED_TIMEOUT_S = 20.0
    #: Total deadline for one non-stream provider call. Every provider call
    #: was awaited directly, so a backend that accepted the request and then
    #: stopped answering held a conversation lane open with no bound at all
    #: (CP126 ``cf57b7f2``).
    GENERATE_TIMEOUT_S = 120.0
    #: Longest gap between two stream chunks before the stream is declared
    #: dead. A total deadline would cut off a long healthy answer; silence is
    #: the signal that matters.
    STREAM_INACTIVITY_TIMEOUT_S = 45.0

    def __init__(self):
        self._gemini_client     = None
        self._local_client      = None
        self._last_embed_space  = ""

        # Capability flags (set after start())
        self.has_gemini  = False
        self.has_local   = False

        # Usage tracking
        self._call_count: dict[str, int] = {"gemini": 0, "local": 0}
        self._error_count: dict[str, int] = {"gemini": 0, "local": 0}
        self._total_tokens: int = 0
        self._exact_token_reports: int = 0
        self._estimated_token_reports: int = 0
        self._last_boundary_provenance: str = ""
        self._gemini_backoff_until: float = 0.0
        self._last_gemini_error: str = ""
        # Provenance of the LAST generation, per execution context. One
        # shared dict meant a concurrent request overwrote it between a
        # caller's generate() and its get_last_generation_metadata(), so the
        # caller read another request's provider and fallback chain (CP126
        # ``63f2b817``). A contextvar follows the task that made the call.
        self._last_generation_metadata: dict[str, Any] = {}

        logger.info("APIAdapter constructed.")

    async def start(self):
        """Initialize clients from environment / Aura config."""
        # There used to be a shared aiohttp.ClientSession here, opened on
        # every start() with a 100-connection TCPConnector, "to prevent
        # connection pooling exhaustion". Nothing in this class ever made a
        # request with it — it was created, tracked, and closed, and that was
        # the whole of its life. Generation goes through the google.genai
        # client or the local backend. Removed rather than routed, because
        # routing a session nobody uses would have preserved the confusion.

        # Load config from Aura's config system
        gemini_key    = None

        try:
            from core.config import config
            # ISSUE #28 - gemini_api_key config precedence
            if hasattr(config, "llm") and hasattr(config.llm, "gemini_api_key") and config.llm.gemini_api_key:
                gemini_key = config.llm.gemini_api_key
            else:
                _load_dotenv_once()
                gemini_key = os.getenv("GEMINI_API_KEY")
        except (ImportError, AttributeError, RuntimeError):
            _load_dotenv_once()
            gemini_key = os.getenv("GEMINI_API_KEY")

        # Initialize Gemini
        if gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_key)
                self.has_gemini = True
                logger.info("✅ APIAdapter: Gemini enabled (%s)", GEMINI_MODELS["api_fast"])
            except ImportError:
                logger.warning("APIAdapter: 'google-genai' package not installed.")
            except (AttributeError, RuntimeError) as e:
                _record_api_degradation(
                    e,
                    action="disabled Gemini backend for this adapter instance; local runtime remains available for failover",
                    extra={"backend": "gemini", "phase": "start"},
                )
                logger.error("APIAdapter: Gemini init failed: %s", e)

        # Initialize Aura's local runtime
        if _HAS_LOCAL_RUNTIME:
            try:
                self._local_client = get_mlx_client()
                self.has_local = True
                logger.info("✅ APIAdapter: Local runtime enabled.")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_api_degradation(
                    e,
                    action="disabled local runtime backend for this adapter instance; API backend remains available for failover",
                    extra={"backend": "local", "phase": "start"},
                )
                logger.error("APIAdapter: local runtime init failed: %s", e)

        if not self.has_gemini and not self.has_local:
            logger.error("APIAdapter: NO LLM AVAILABLE. Set GEMINI_API_KEY or verify the local runtime.")

    async def setup_memory_facade(self):
        """Standard integration for MemoryFacade and AgencyFacade."""
        try:
            from core.agency.agency_facade import AgencyFacade
            from core.container import ServiceContainer
            if ServiceContainer.get("agency_facade", default=None) is None:
                fa = AgencyFacade()
                ServiceContainer.register("agency_facade", fa)
                logger.info("✅ AgencyFacade registered for MemoryFacade")
        except ImportError:
            logger.warning("⚠️ [BOOT] Early Facade registration deferred: AgencyFacade missing.")
        except (AttributeError, RuntimeError) as e:
            _record_api_degradation(
                e,
                action="deferred AgencyFacade registration; memory facade setup can retry after container boot",
                extra={"phase": "setup_memory_facade"},
            )
            logger.error("❌ [BOOT] AgencyFacade registration error: %s", e)

    async def stop(self):
        # A stopped adapter must not keep ADVERTISING generation capability.
        # stop() previously closed only the HTTP session, so has_gemini /
        # has_local stayed true and get_status() reported a live backend for
        # an adapter that had been shut down — routing and health both read
        # those flags.
        self.has_gemini = False
        self.has_local = False
        self._gemini_client = None
        self._local_client = None
        self._gemini_backoff_until = 0.0
        await self._close_http_session()
        logger.info("APIAdapter stopped. Calls: %s | Tokens: %d",
                    self._call_count, self._total_tokens)

    async def _close_http_session(self) -> None:
        """Close a shared HTTP session, if anything ever sets one.

        ``start()`` no longer opens one — see the note there; the old
        aiohttp session was created, tracked, closed, and never used to make
        a request, so it was removed rather than routed. What survived the
        removal was ``on_stop_async``'s docstring, still describing itself as
        "the shutdown hook for the shared HTTP session" while closing
        nothing.

        Kept as a real close rather than deleting the promise, because the
        next person to add a session will add it to ``start()`` and will not
        think to add a matching close — an aiohttp session that outlives
        shutdown holds its connector and its sockets. Now the hook does what
        it says, whether or not there is anything to do.
        """
        session = getattr(self, "_http_session", None)
        if session is None:
            return
        self._http_session = None
        close = getattr(session, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — shutdown must complete
            record_degradation(
                "api_adapter",
                exc,
                severity="warning",
                action="dropped the HTTP session reference after close failed",
            )

    async def on_stop_async(self) -> None:
        """ServiceContainer shutdown hook for the shared HTTP session."""
        await self.stop()

    # ─── Main API ────────────────────────────────────────────────────────────

    async def generate(self, prompt: str, config: dict[str, Any] | None = None) -> str:
        """
        Generate a response. Tier is specified in config["model_tier"].
        """
        config = config or {}
        tier        = config.get("model_tier", "local")
        purpose     = config.get("purpose", "general")

        start = time.monotonic()

        result_metadata = await self.generate_with_metadata(prompt, config)
        result = str(result_metadata.get("text") or "")

        # An all-backend failure must NOT come back as an empty string. This
        # is the exact error-versus-empty ambiguity the adapter layer exists
        # to prevent: callers could not distinguish "the model produced
        # nothing" from "every backend failed", and downstream code went on
        # to parse, store, or serve the emptiness as a real answer.
        if not result_metadata.get("ok", True) and not result:
            raise RuntimeError(
                "api_adapter_generation_failed:"
                f"{result_metadata.get('error') or 'unknown'}"
            )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug("APIAdapter.generate: tier=%s purpose=%s %.1fms len=%d",
                     tier, purpose, elapsed, len(result))
        return result

    def get_last_generation_metadata(self) -> dict[str, Any]:
        """Provenance of the last generation made BY THIS TASK.

        Read from a contextvar, not a shared field: two concurrent requests
        used to race here and a caller could be handed the other one's
        provider and fallback chain (CP126 ``63f2b817``). The instance field
        is kept in step for readers that still touch it directly, but the
        contextvar is the answer.
        """
        scoped = _LAST_GENERATION.get()
        if scoped:
            return dict(scoped)
        return dict(self._last_generation_metadata)

    def _publish_generation_metadata(self, result: dict[str, Any]) -> None:
        payload = dict(result)
        _LAST_GENERATION.set(payload)
        self._last_generation_metadata = payload

    async def generate_with_metadata(
        self,
        prompt: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text with truthful provider, model, and fallback provenance."""

        config = config or {}
        # Typed, finite, bounded admission. These values came straight from
        # caller config into provider requests: a NaN temperature, a string
        # where a number was expected, an unbounded max_tokens, or an
        # unknown tier could raise before routing or request pathological
        # amounts of work from a paid backend.
        tier = str(config.get("model_tier", "local") or "local").strip().lower()
        if tier not in _VALID_TIERS:
            logger.warning("APIAdapter: unknown model_tier %r; defaulting to local.", tier)
            tier = "local"
        temperature = _bounded_float(config.get("temperature"), default=0.7, low=0.0, high=2.0)
        max_tokens = _bounded_int(
            config.get("max_tokens"), default=800, low=1, high=_MAX_OUTPUT_TOKENS
        )
        prompt = str(prompt or "")
        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.warning(
                "APIAdapter: prompt of %d chars exceeds the %d-char ceiling; refusing.",
                len(prompt),
                _MAX_PROMPT_CHARS,
            )
            return {
                "ok": False,
                "text": "",
                "provider": "none",
                "model": "",
                "error": f"prompt_too_large:{len(prompt)}",
            }
        result = await self._route_generate_with_metadata(
            prompt,
            tier,
            temperature,
            max_tokens,
            config=config,
        )
        self._publish_generation_metadata(result)
        return dict(result)

    async def generate_stream(
        self, prompt: str, config: dict[str, Any] | None = None
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Streaming generation."""
        config = config or {}
        tier        = config.get("model_tier", "local")
        temperature = config.get("temperature", 0.7)
        max_tokens  = config.get("max_tokens", 800)

        async for chunk in self._route_stream(prompt, tier, temperature, max_tokens):
            yield chunk

    # ─── Routing ─────────────────────────────────────────────────────────────

    async def _route_generate(
        self, prompt: str, tier: str, temperature: float, max_tokens: int, config: dict[str, Any] | None = None
    ) -> str:
        """Route with automatic fallback chain."""
        result = await self._route_generate_with_metadata(
            prompt,
            tier,
            temperature,
            max_tokens,
            config=config,
        )
        self._publish_generation_metadata(result)
        return str(result.get("text") or "")

    async def _route_generate_with_metadata(
        self,
        prompt: str,
        tier: str,
        temperature: float,
        max_tokens: int,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route with an explicit cloud-only mode and structured provenance."""

        config = config or {}
        cloud_only = bool(config.get("cloud_only", False))
        fallback_chain: list[dict[str, str]] = []

        # Cloud chain (Gemini only)
        if tier in ("api_deep", "api_fast"):
            model_name = GEMINI_MODELS.get(tier, GEMINI_MODELS["api_fast"])
            if self.has_gemini and time.monotonic() >= self._gemini_backoff_until:
                self._last_gemini_error = ""
                # Native role separation. _gemini_generate has always
                # accepted system_instruction and nothing ever passed one, so
                # callers flattened the system prompt into the user string with
                # "User:"/"Aura:" labels — text the user can write themselves,
                # competing with the instructions it was pretending to be.
                system_instruction = str(config.get("system_instruction", "") or "") or None
                try:
                    result = await self._gemini_generate(
                        prompt,
                        tier,
                        temperature,
                        max_tokens,
                        system_instruction=system_instruction,
                        config=config,
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    *cloud_call_error_types(),
                ) as exc:
                    self._last_gemini_error = str(exc) or type(exc).__name__
                    _record_api_degradation(
                        exc,
                        action="continued APIAdapter fallback chain after Gemini provider failure",
                        extra={"backend": "gemini", "tier": tier},
                    )
                    err_text = self._last_gemini_error
                    if "429" in err_text or "quota" in err_text.lower():
                        self._gemini_backoff_until = time.monotonic() + 60.0
                    self._error_count["gemini"] += 1
                    result = None
                if result:
                    fallback_chain.append(
                        {"provider": "gemini", "model": model_name, "status": "success"}
                    )
                    receipt = provider_receipt(
                        provider="gemini",
                        model=model_name,
                        prompt=prompt,
                        response=str(result),
                        system_instruction=system_instruction,
                        transport="google.genai.aio.generate_content",
                    )
                    return {
                        "ok": True,
                        "text": str(result),
                        "endpoint": f"Gemini-APIAdapter:{model_name}",
                        "provider": "gemini",
                        "model": model_name,
                        "is_local": False,
                        # What was actually observed, rather than "the SDK
                        # returned". `provider_verified` stays for readers
                        # that check it, and it now means the receipt exists
                        # and matches the text being returned.
                        "provider_receipt": receipt,
                        # `_verified_cloud_generation_metadata` in the
                        # inference gate accepts "provider_receipt" as the
                        # stronger basis and records anything weaker. Its
                        # docstring says no adapter produces one; this one
                        # now does, for whatever a locally observed receipt
                        # is worth, and it names exactly that.
                        "provider_attribution": "provider_receipt",
                        "provider_verified": receipt["response_sha256"]
                        == digest(str(result)),
                        "role_separation": receipt.get(
                            "role_separation", "native" if system_instruction else "none"
                        ),
                        "prompt_boundary": self._last_boundary_provenance,
                        "tier_requested": tier,
                        "tiers_distinct": gemini_tiers_are_distinct(),
                        "fallback_chain": fallback_chain,
                        "error": "",
                    }
                failure_entry = {
                    "provider": "gemini",
                    "model": model_name,
                    "status": "error" if self._last_gemini_error else "no_text",
                }
                if self._last_gemini_error:
                    failure_entry["error"] = self._last_gemini_error[:240]
                fallback_chain.append(failure_entry)
            else:
                status = "backoff" if self.has_gemini else "unavailable"
                fallback_chain.append(
                    {"provider": "gemini", "model": model_name, "status": status}
                )

        if cloud_only:
            logger.error("APIAdapter: cloud-only generation failed for tier=%s", tier)
            return {
                "ok": False,
                "text": "",
                "endpoint": "APIAdapter-cloud-unavailable",
                "provider": "none",
                "model": "",
                "is_local": False,
                "fallback_chain": fallback_chain,
                "error": "cloud_only_backend_unavailable",
            }

        # Local fallback chain. A CLOUD tier answered locally is a quality
        # and provenance change the caller asked for the opposite of, and it
        # happened silently: `generate()` returns only text, so nothing
        # downstream could see it (CP126 ``27f97284``). It is recorded, it is
        # named in the result, and `strict_tier` refuses it outright.
        downgraded = tier in ("api_deep", "api_fast")
        if downgraded and bool(config.get("strict_tier", False)):
            return {
                "ok": False,
                "text": "",
                "endpoint": "APIAdapter-tier-unavailable",
                "provider": "none",
                "model": "",
                "is_local": False,
                "tier_requested": tier,
                "fallback_chain": fallback_chain,
                "error": f"strict_tier_unavailable:{tier}",
            }
        if self.has_local:
            result = await self._local_generate(prompt, temperature, max_tokens)
            if result:
                model_name = str(
                    getattr(self._local_client, "model_name", None)
                    or getattr(self._local_client, "model_path", None)
                    or "managed-local-runtime"
                )
                fallback_chain.append(
                    {"provider": "local", "model": model_name, "status": "success"}
                )
                if downgraded:
                    _record_api_degradation(
                        RuntimeError(f"{tier} request answered by the local runtime"),
                        severity="warning",
                        action=(
                            "served a cloud-tier request locally; pass "
                            "strict_tier=True to refuse instead"
                        ),
                        extra={"tier_requested": tier, "provider": "local"},
                    )
                receipt = provider_receipt(
                    provider="local",
                    model=model_name,
                    prompt=prompt,
                    response=str(result),
                    system_instruction=None,
                    transport="mlx_client.generate",
                )
                return {
                    "ok": True,
                    "text": str(result),
                    "endpoint": f"Local-APIAdapter:{model_name}",
                    "provider": "local",
                    "model": model_name,
                    "is_local": True,
                    # Present for local too. Its absence used to be the only
                    # thing distinguishing a local result from an unverified
                    # cloud one, which is not a distinction a reader can make.
                    "provider_receipt": receipt,
                    "provider_attribution": "provider_receipt",
                    "provider_verified": receipt["response_sha256"]
                    == digest(str(result)),
                    "tier_requested": tier,
                    "tier_downgraded": downgraded,
                    "fallback_chain": fallback_chain,
                    "error": "",
                }
            fallback_chain.append(
                {"provider": "local", "model": "managed-local-runtime", "status": "no_text"}
            )

        logger.error("APIAdapter: all backends failed for tier=%s", tier)
        return {
            "ok": False,
            "text": "",
            "endpoint": "APIAdapter-all-failed",
            "provider": "none",
            "model": "",
            "is_local": False,
            "fallback_chain": fallback_chain,
            "error": "all_backends_failed",
        }

    async def _route_stream(
        self, prompt: str, tier: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Stream from the first backend that works, and always terminate.

        Three findings meet here, and they are one shape — the router
        returned as soon as it had chosen a provider, so whatever the
        provider did was the whole outcome:

        * ``88bb1083`` a provider generator that failed mid-stream caught
          the error, stopped yielding, and emitted neither an error event
          nor an end event. The caller received a clipped stream that reads
          as unfinished rather than failed, with no way to tell them apart.
        * ``0f493981`` the cloud leg checked ``has_gemini`` and ignored the
          backoff deadline that the non-stream path honours, and it never
          fell back to local when the cloud produced nothing.
        * ``3616b6cc`` a request that asked for LOCAL was sent to Gemini
          when local was down, with a warning in a log the person never
          reads and nothing in the stream itself.

        So: the ROUTER owns the terminal event, exactly one of them; the
        provider generators yield tokens and raise. A tier=local request
        that leaves the device announces it in the stream, because a
        privacy decision the caller cannot observe is not one they made.
        """
        attempts: list[tuple[str, str]] = []
        cloud_ready = self.has_gemini and time.monotonic() >= self._gemini_backoff_until
        if self.has_gemini and not cloud_ready:
            attempts.append(("gemini_backoff", ""))

        if tier in ("api_fast", "api_deep"):
            if cloud_ready:
                attempts.append(("gemini", tier))
            if self.has_local:
                attempts.append(("local", tier))
        else:
            if self.has_local:
                attempts.append(("local", tier))
            if cloud_ready:
                attempts.append(("gemini_egress", "api_fast"))

        produced_any = False
        errors: list[str] = []
        for backend, backend_tier in attempts:
            if backend == "gemini_backoff":
                errors.append(
                    "gemini: in backoff until "
                    f"{max(0.0, self._gemini_backoff_until - time.monotonic()):.0f}s"
                )
                continue
            if backend == "gemini_egress":
                # The person asked for local and local is not there. Say so
                # IN the stream before a single token crosses the network.
                yield ChatStreamEvent(
                    type="provenance",
                    content=(
                        "local runtime unavailable; this request is being answered "
                        f"by {resolve_gemini_model(backend_tier)} in the cloud"
                    ),
                )
            try:
                source = (
                    self._local_stream(prompt, temperature, max_tokens)
                    if backend == "local"
                    else self._gemini_stream(prompt, backend_tier, temperature, max_tokens)
                )
                async for chunk in self._with_inactivity_deadline(source, backend):
                    produced_any = True
                    yield chunk
                if produced_any:
                    yield ChatStreamEvent(type="end")
                    return
                errors.append(f"{backend}: produced no tokens")
            except _StreamFailed as exc:
                errors.append(f"{backend}: {exc}")
                if produced_any:
                    # Tokens already reached the caller. Switching backends
                    # mid-answer would splice two different completions into
                    # one reply, so this ends honestly instead.
                    yield ChatStreamEvent(
                        type="error",
                        content=f"stream ended early after a {backend} failure: {exc}",
                    )
                    return

        logger.error("APIAdapter: all streams failed for tier=%s (%s)", tier, errors)
        yield ChatStreamEvent(
            type="error",
            content="No LLM backend produced a stream: " + "; ".join(errors[:4]),
        )

    async def _with_inactivity_deadline(
        self, source: AsyncGenerator[ChatStreamEvent, None], backend: str
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Fail a stream that has gone quiet rather than waiting forever.

        Both stream iterators were awaited with no deadline of any kind, so
        a backend that accepted the request and then stopped sending held
        the conversation lane open indefinitely (CP126 ``cf57b7f2``).
        """
        iterator = source.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    iterator.__anext__(), timeout=self.STREAM_INACTIVITY_TIMEOUT_S
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                await self._aclose_quietly(source)
                raise _StreamFailed(
                    f"no token for {self.STREAM_INACTIVITY_TIMEOUT_S:.0f}s"
                ) from exc
            yield chunk

    @staticmethod
    async def _aclose_quietly(source: Any) -> None:
        aclose = getattr(source, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except (RuntimeError, GeneratorExit, asyncio.CancelledError) as exc:
            logger.debug("APIAdapter: stream close raised: %s", exc)

    # ─── Gemini ──────────────────────────────────────────────────────────────

    def _screen_for_egress(
        self, prompt: str, system_instruction: str | None
    ) -> tuple[str, str | None] | None:
        """Read the prompt before the vendor SDK sends it, or send nothing.

        This adapter holds a ``google.genai`` client, which builds and sends
        its own HTTP. That is a second door out of the machine:
        ``NetworkGateway`` — and therefore governance, the defensive
        preflight, and the egress privacy filter — never sees these bytes.
        Screening here is what makes the boundary singular rather than
        merely present.

        Returns None to mean "do not send this to the cloud". The caller
        already treats None as a failed cloud leg and continues down its
        fallback chain to local inference, which is the correct outcome: the
        answer still gets produced, on this machine.
        """
        try:
            from core.security.egress_privacy import filter_model_prompt

            screened_prompt = filter_model_prompt(prompt, provider="gemini")
            screened_system = filter_model_prompt(
                system_instruction, provider="gemini"
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_api_degradation(
                exc,
                action="refused the Gemini call rather than send an unscreened prompt",
                extra={"backend": "gemini"},
            )
            self._last_gemini_error = f"egress_privacy_unavailable: {exc}"
            return None

        if not (screened_prompt.allowed and screened_system.allowed):
            reason = screened_prompt.reason or screened_system.reason
            logger.warning("APIAdapter: Gemini call refused by egress privacy — %s", reason)
            self._last_gemini_error = f"egress_privacy_refused: {reason}"
            return None

        return screened_prompt.text or "", screened_system.text

    #: Ceiling on what one request may declare. An unbounded tool list is a
    #: payload, and a schema deep enough to be interesting is deep enough to
    #: be a denial of service on the provider's parser.
    MAX_TOOLS_PER_REQUEST = 32
    MAX_TOOL_SCHEMA_CHARS = 20_000
    async def _gemini_generate(
        self, prompt: str, tier: str, temperature: float, max_tokens: int, system_instruction: str | None = None, config: dict[str, Any] | None = None
    ) -> str | None:
        config = config or {}
        if self._gemini_client and self.has_gemini:
            model_name = GEMINI_MODELS.get(tier, GEMINI_MODELS["api_fast"])
            self._last_gemini_error = ""
            sent = self._screen_for_egress(prompt, system_instruction)
            if sent is None:
                return None
            prompt, system_instruction = sent
            try:
                from google import genai

                # Non-stream sent the whole combined prompt as `contents`
                # with no system instruction, while the stream path split
                # it — one request type gave the model an instruction with
                # precedence and the other buried the same words inside the
                # user turn (CP126 ``9dcdf9fd``). Both split now, and both
                # record which way the boundary was established.
                if system_instruction is None:
                    system_text, prompt, boundary = structured_prompt(prompt, config)
                    system_instruction = system_text or None
                else:
                    boundary = "caller_supplied"
                self._last_boundary_provenance = boundary
                config_kwargs = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "system_instruction": system_instruction if system_instruction else None,
                }
                # Structural tools are forwarded only after they are checked
                # against Aura's own capability registry. A caller-supplied
                # definition used to be copied straight into the provider
                # request, so anything that could reach this adapter could
                # declare a tool the runtime does not have, does not govern,
                # and did not authorize (CP126 ``6e14ba27``).
                tools = admissible_tools(config.get("tools"))
                if tools:
                    config_kwargs["tools"] = tools

                gen_config = genai.types.GenerateContentConfig(**config_kwargs)
                response = await asyncio.wait_for(
                    self._gemini_client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=gen_config,
                    ),
                    timeout=self.GENERATE_TIMEOUT_S,
                )
                self._call_count["gemini"] += 1
                text = response.text or ""
                self._count_tokens(text, exact=reported_token_count(response))
                return text
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                asyncio.TimeoutError,
                *cloud_call_error_types(),
            ) as e:
                self._last_gemini_error = str(e) or type(e).__name__
                _record_api_degradation(
                    e,
                    action="backed off Gemini backend and allowed local generation fallback",
                    extra={"backend": "gemini", "tier": tier},
                )
                err_text = str(e)
                if "429" in err_text or "quota" in err_text.lower():
                    self._gemini_backoff_until = time.monotonic() + 60.0
                logger.warning("Gemini %s failed: %s", model_name, e)
                self._error_count["gemini"] += 1
        return None

    async def _gemini_stream(
        self, prompt: str, tier: str, temperature: float, max_tokens: int, system_instruction: str | None = None
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Yield tokens. Raise on failure. Never emit a terminal event.

        The router owns termination now, so this leg must not decide the
        stream is over — a leg that yields `end` on its own path cannot be
        failed over from.
        """
        if not self._gemini_client:
            raise _StreamFailed("no gemini client")
        model_name = resolve_gemini_model(tier)
        try:
            from google import genai
            system_text, user_text = split_prompt(prompt)
            # Same door, same screen. The stream path used to be the one that
            # got missed, which is how a boundary ends up with an exception
            # nobody remembers making.
            screened = self._screen_for_egress(
                user_text, system_instruction or system_text or None
            )
            if screened is None:
                return
            user_text, system_instruction = screened
            config = genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                # Not `system_instruction or system_text`: system_text is the
                # UNSCREENED original, and that fallback would put it back on
                # the wire the moment screening emptied the instruction.
                system_instruction=system_instruction,
            )
            async for chunk in self._gemini_client.aio.models.generate_content_stream(
                model=model_name,
                contents=user_text,
                config=config,
            ):
                if chunk.text:
                    self._count_tokens(chunk.text)
                    yield ChatStreamEvent(type="token", content=chunk.text)
            self._call_count["gemini"] += 1
        except (ImportError, AttributeError, RuntimeError, *cloud_call_error_types()) as e:
            self._error_count["gemini"] += 1
            self._last_gemini_error = str(e) or type(e).__name__
            if "429" in str(e) or "quota" in str(e).lower():
                self._gemini_backoff_until = time.monotonic() + 60.0
            _record_api_degradation(
                e,
                action="raised a typed stream failure so the router can fail over and terminate the stream",
                extra={"backend": "gemini", "tier": tier},
            )
            logger.warning("Gemini streaming failed: %s", e)
            raise _StreamFailed(str(e) or type(e).__name__) from e

    # ─── Local Runtime ───────────────────────────────────────────────────────

    async def _local_generate(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str | None:
        if not self._local_client:
            return None
        try:
            system_text, user_text = split_prompt(prompt)
            result = await asyncio.wait_for(
                self._local_client.generate(
                    user_text,
                    system_prompt=system_text,
                    temp=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self.GENERATE_TIMEOUT_S,
            )
            
            # Prevent hallucinated human turns from local models
            if result:
                stop_marker = "\nHuman:"
                idx = result.find(stop_marker)
                if idx != -1:
                    result = result[:idx].strip()
                    
            self._call_count["local"] += 1
            self._count_tokens(result or "")
            return result
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            # The MLX client raises ordinary RuntimeError for model admission,
            # decode, worker-state and lane failures, and TypeError/ValueError
            # /AttributeError for malformed client state. Catching only the
            # first three let those escape the fallback chain entirely, so a
            # recoverable local failure aborted the whole request instead of
            # failing over to cloud.
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            _record_api_degradation(
                e,
                action="incremented local error count and returned None so routing can fail over",
                extra={"backend": "local", "phase": "generate"},
            )
            logger.warning("Local runtime generate failed: %s", e)
            self._error_count["local"] += 1
        return None

    async def _local_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Yield tokens. Raise on failure. Never emit a terminal event."""
        if not self._local_client:
            raise _StreamFailed("no local client")
        try:
            system_text, user_text = split_prompt(prompt)
            buffer = ""
            async for chunk in self._local_client.generate_stream(
                user_text,
                system_prompt=system_text,
                temp=temperature,
                max_tokens=max_tokens
            ):
                content = chunk if isinstance(chunk, str) else chunk.content if hasattr(chunk, 'content') else str(chunk)
                buffer += content
                # ISSUE #11 - local_stream prefix match buffer holding on newlines
                stop_marker = "Human:"
                if stop_marker in buffer:
                    idx = buffer.find(stop_marker)
                    valid_part = buffer[:idx].rstrip()
                    if valid_part:
                        self._count_tokens(valid_part)
                        yield ChatStreamEvent(type="token", content=valid_part)
                    break
                else:
                    if any(buffer.endswith(stop_marker[:i]) for i in range(1, len(stop_marker) + 1)):
                        pass # keep in buffer
                    else:
                        self._count_tokens(buffer)
                        yield ChatStreamEvent(type="token", content=buffer)
                        buffer = ""
                        
            if buffer and "Human:" not in buffer:
                self._count_tokens(buffer)
                yield ChatStreamEvent(type="token", content=buffer)

            self._call_count["local"] += 1
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            self._error_count["local"] += 1
            _record_api_degradation(
                e,
                action="raised a typed stream failure so the router can fail over and terminate the stream",
                extra={"backend": "local", "phase": "stream"},
            )
            logger.warning("Local runtime stream failed: %s", e)
            raise _StreamFailed(str(e) or type(e).__name__) from e

    # ─── Embeddings ──────────────────────────────────────────────────────────

    # Identifies which vector space a returned embedding belongs to. A cloud
    # embedding and the lexical fallback are NOT comparable, so callers that
    # persist vectors must not mix them in one index.
    CLOUD_EMBED_SPACE = "gemini:text-embedding-004"
    LOCAL_EMBED_SPACE = "local:bow-hash-768"

    def last_embedding_space(self) -> str:
        """Vector space of the most recent embedding (see *_EMBED_SPACE)."""
        return getattr(self, "_last_embed_space", "")

    async def embed_async(self, text: str) -> list[float]:
        """Generate embeddings for text. Uses Gemini as primary, then a local shim."""
        if self.has_gemini:
            try:
                # Off-loop: models.embed_content is a SYNCHRONOUS network call.
                # Awaiting it directly on the event loop stalled every other
                # task for the provider's full round-trip.
                res = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._gemini_client.models.embed_content,
                        model="text-embedding-004",
                        contents=text,
                    ),
                    timeout=self.EMBED_TIMEOUT_S,
                )
                self._last_embed_space = self.CLOUD_EMBED_SPACE
                return res.embeddings[0].values
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                # SDK/runtime/quota failures surface as these; omitting them
                # made fallback reliability depend on exception-class accident.
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
                IndexError,
                KeyError,
            ) as e:
                _record_api_degradation(
                    e,
                    severity="warning",
                    action="used deterministic local bag-of-words embedding fallback",
                    extra={"backend": "gemini", "phase": "embedding"},
                )
                logger.debug("Gemini embedding failed: %s", e)

        # Deterministic LEXICAL fallback: bag-of-words hashing. It measures
        # token overlap only — it cannot represent synonymy, relations, or
        # context, so it is not a semantic embedding and must not be
        # described as one. Texts sharing literal words get non-zero cosine
        # similarity; paraphrases with no shared tokens get zero.
        self._last_embed_space = self.LOCAL_EMBED_SPACE
        return self._local_bow_embed(text)

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous wrapper for embeddings."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # A thread pool does not stop this blocking the caller: the
                # loop thread still sits on future.result() for the whole
                # round trip, which for the cloud path is a network call
                # (CP126 ``e8a9fd4e``). Nothing on the event loop may wait
                # on that, so the local embedding answers instead — same
                # vector space it would fall back to anyway, and the space
                # is recorded so no caller mixes the two in one index.
                self._last_embed_space = self.LOCAL_EMBED_SPACE
                _record_api_degradation(
                    RuntimeError("embed_sync called from a running event loop"),
                    severity="info",
                    action=(
                        "returned the local bag-of-words embedding; await "
                        "embed_async() from async code to reach the cloud space"
                    ),
                )
                return self._local_bow_embed(text)

            return asyncio.run(self.embed_async(text))
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_api_degradation(
                exc,
                severity="warning",
                action="returned deterministic local bag-of-words embedding from synchronous wrapper",
                extra={"phase": "embed_sync"},
            )
            logger.debug("Synchronous embedding wrapper failed; falling back to local bag-of-words embedding: %s", exc)
            # Fallback to local bag-of-words embedding
            return self._local_bow_embed(text)

    @staticmethod
    def _local_bow_embed(text: str, dim: int = 768) -> list[float]:
        """Bag-of-words hashing embedding that preserves semantic similarity.

        Each word is hashed to 3 positions in the vector and contributes a
        signed value. Texts sharing words will have proportional cosine
        similarity. IDF-like weighting is approximated by word length
        (longer words are rarer and contribute more). The result is
        L2-normalized to unit length.

        This is NOT as good as a real embedding model, but it makes semantic
        memory retrieval, consolidation, and deduplication actually work
        when cloud embeddings are unavailable — unlike random vectors which
        produce near-zero similarity for all pairs.
        """
        import hashlib

        import numpy as np

        vec = np.zeros(dim, dtype=np.float64)
        words = text.lower().split()
        if not words:
            # Empty text gets a zero vector
            return vec.tolist()

        for word in words:
            # Strip punctuation
            clean = ''.join(c for c in word if c.isalnum())
            if not clean:
                continue
            # IDF-like weight: longer words are rarer and matter more
            weight = 1.0 + min(len(clean), 12) * 0.15
            # Hash to 3 positions for better coverage and collision resistance
            for salt in (b"a", b"b", b"c"):
                h = hashlib.md5(salt + clean.encode()).digest()
                idx = int.from_bytes(h[:2], "big") % dim
                sign = 1.0 if h[2] & 1 else -1.0
                vec[idx] += sign * weight

        # L2 normalize to unit length
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec.tolist()

    # ─── Utilities ───────────────────────────────────────────────────────────

    #: The literal that separates instructions from the person's turn in a
    #: flat prompt. It is ordinary text, so anyone who can put text in the
    #: prompt can write it.
    ROLE_MARKER = "\nHuman:"
    #: Re-exported so a reader of this class can find the boundary rule
    #: without hunting for the module it moved to.
    ROLE_MARKER = "\nHuman:"
    MAX_TOOLS_PER_REQUEST = MAX_TOOLS_PER_REQUEST

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        """See :func:`core.adapters.prompt_boundary.split_prompt`."""
        return split_prompt(prompt)

    #: Characters per token, for the paths where a provider reports no usage
    #: count. An estimate that says it is one, rather than a zero that reads
    #: as a measurement.
    CHARS_PER_TOKEN_ESTIMATE = 4.0

    def _count_tokens(self, text: str, *, exact: int | None = None) -> None:
        """Advance the token counter that `get_status` reports.

        `total_tokens` was initialized, reported, and never incremented by
        any generation or stream path, so status published a permanent,
        technically valid zero (CP126 ``82ec3ab8``). An exact count is used
        when the provider gives one; otherwise this estimates and the status
        says which it is.
        """
        if exact is not None:
            self._total_tokens += max(0, int(exact))
            self._exact_token_reports += 1
            return
        chars = len(str(text or ""))
        if chars:
            self._total_tokens += max(1, int(chars / self.CHARS_PER_TOKEN_ESTIMATE))
            self._estimated_token_reports += 1

    def get_status(self) -> dict[str, Any]:
        # Copies, not references: the live counter dicts were handed out by
        # reference, so any consumer could mutate adapter telemetry without
        # going through the adapter.
        return {
            "gemini":       self.has_gemini,
            "local":        self.has_local,
            "calls":        dict(self._call_count),
            "errors":       dict(self._error_count),
            "total_tokens": self._total_tokens,
            # How the number was arrived at, so nobody reads an estimate as
            # a billing figure. Zero of both means nothing has generated yet
            # — which is a different fact from "the counter is broken", and
            # that is what this used to be unable to say.
            "token_accounting": {
                "exact_reports": self._exact_token_reports,
                "estimated_reports": self._estimated_token_reports,
                "chars_per_token_estimate": self.CHARS_PER_TOKEN_ESTIMATE,
            },
            "tiers_distinct": gemini_tiers_are_distinct(),
            "models": dict(GEMINI_MODELS),
            "embedding_space": self._last_embed_space,
        }

    def get_available_tiers(self) -> list[str]:
        tiers = ["local"] if self.has_local else []
        if self.has_gemini:
            tiers = ["api_fast", "api_deep"] + tiers
        return tiers


# ─── Singleton ───────────────────────────────────────────────────────────────

_adapter_instance: APIAdapter | None = None
_adapter_lock = threading.Lock()

def get_api_adapter() -> APIAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                _adapter_instance = APIAdapter()
    return _adapter_instance
