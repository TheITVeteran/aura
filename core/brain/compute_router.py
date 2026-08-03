"""core/brain/compute_router.py — Compute Offloading Architecture

Routes inference tasks between local and cloud GPU backends with
cost/latency awareness. Cloud offloading is DISABLED by default and
requires explicit user opt-in via configuration.

Design principles:
  1. Local-first: Always prefer local MLX instance
  2. Consent-gated: Cloud requires explicit opt-in AND per-task authority
  3. Cost-aware: Tracks estimated costs per request
  4. Graceful fallback: if cloud fails, BOTH errors are reported

NOT WIRED INTO THE LIVE RUNTIME. ``ComputeRouter`` has no caller under
core/ or interface/; live cloud fallback goes through
``core/brain/llm_health_router.py``. Said plainly because this module
handles API keys and spend, and an unwired module that looks live is how
one gets adopted without the review it needs.

Two claims corrected (CP126). "Graceful fallback: if cloud fails, queue
for local" described a queue that does not exist — ``_local_queue_depth``
was initialised and never read or written, and an exhausted backend
returned an error immediately with no durable queue, retry policy or
receipt. And cloud was authorised by a mutable process-global flag alone,
so once enabled, ANY caller could send a prompt and its metadata off the
host; a task now has to carry its own authority.
"""
import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ComputeRouter")

_COMPUTE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


#: Providers this router will talk to. CP126 5c3beeaa: without an
#: allowlist, any string in a mutable config selected which registered
#: object received the prompt and the API key.
_ALLOWED_CLOUD_PROVIDERS = frozenset({"runpod", "vast", "lambda"})


def _finite_cost(value: Any) -> Optional[float]:
    """A non-negative finite number, or None. Money must never be NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")) or number < 0.0:
        return None
    return number


class ComputeBackend(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class InferenceTask:
    """Describes an inference task to be routed."""
    prompt: str
    model: str = "default"
    max_tokens: int = 2048
    temperature: float = 0.7
    priority: str = "normal"  # "low", "normal", "high", "critical"
    estimated_compute_seconds: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: CP126 a6f7851c. Cloud disclosure was authorised by the process-global
    #: `CloudConfig.enabled` alone: no principal, no sensitivity, no
    #: data-residency, no scoped authority. Once an operator turned cloud on
    #: for one purpose, every caller in the process could send prompts and
    #: metadata off the host.
    #:
    #: A global switch can only say cloud is POSSIBLE. Only the caller knows
    #: whether THIS content may leave, so leaving is now two independent
    #: decisions and this one defaults to no.
    cloud_authorized: bool = False
    #: Who authorised it. An unattributable disclosure cannot be audited or
    #: withdrawn, so it is refused.
    authorized_by: str = ""


@dataclass
class InferenceResult:
    """Result from a routed inference task."""
    content: str = ""
    backend_used: ComputeBackend = ComputeBackend.LOCAL
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    success: bool = False
    error: Optional[str] = None


@dataclass
class CloudConfig:
    """Configuration for cloud compute offloading."""
    enabled: bool = False
    provider: Optional[str] = None  # "runpod" | "vast" | "lambda" | None
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    max_monthly_budget_usd: float = 10.0
    local_gpu_memory_threshold_mb: int = 500
    cost_per_token: float = 0.000001  # ~$1 per 1M tokens


class ComputeRouter:
    """Routes inference between local and cloud backends.
    
    The router is conservative by design:
    - Cloud is disabled by default
    - Local is always tried first
    - Cloud is only used when local is overloaded AND cloud is enabled
    - All cloud usage is budget-capped
    """

    def __init__(self, cloud_config: Optional[CloudConfig] = None):
        self.cloud_config = cloud_config or CloudConfig()
        self._monthly_spend_usd = 0.0
        self._reserved_usd = 0.0
        self._month_start = time.time()
        self._routing_history: List[Dict] = []
        # CP126 f1c0e39a: the check-call-charge sequence had no lock, so
        # concurrent routes all observed the same spend below budget, all
        # issued cloud calls, and each incremented afterwards. The budget was
        # a suggestion under any concurrency at all.
        self._spend_lock = asyncio.Lock()
        
        if self.cloud_config.enabled:
            logger.info("☁️  ComputeRouter online (cloud: %s, budget: $%.2f/mo)",
                       self.cloud_config.provider, self.cloud_config.max_monthly_budget_usd)
        else:
            logger.info("💻 ComputeRouter online (local-only mode)")

    async def route(self, task: InferenceTask) -> InferenceResult:
        """Route an inference task to the best available backend.
        
        Decision tree:
        1. Always try local first
        2. If local fails AND cloud is enabled AND budget allows → try cloud
        3. If cloud fails → return error with diagnostics
        """
        start = time.monotonic()

        # Always try local first
        result = await self._try_local(task)
        if result.success:
            result.latency_seconds = time.monotonic() - start
            self._record_routing(task, result)
            return result

        # CP126 9e684e4b: `result` was overwritten by the cloud attempt, so
        # a dual failure retained only the cloud error and the original local
        # outage became undiagnosable. Keep it.
        local_error = result.error or "local inference failed"

        if self._cloud_permitted(task):
            logger.info("Local inference failed, attempting cloud offload...")
            cloud_result = await self._route_cloud_with_budget(task)
            if cloud_result.success:
                cloud_result.latency_seconds = time.monotonic() - start
                self._record_routing(task, cloud_result)
                return cloud_result
            result.error = (
                f"local: {local_error} | cloud: {cloud_result.error or 'failed'}"
            )

        result.latency_seconds = time.monotonic() - start
        if not result.error:
            result.error = "All compute backends exhausted"
        self._record_routing(task, result)
        return result

    def _cloud_permitted(self, task: InferenceTask) -> bool:
        """Both decisions must say yes, and the refusal names which said no."""
        if not self.cloud_config.enabled:
            return False
        if not getattr(task, "cloud_authorized", False):
            logger.info(
                "Cloud is enabled but this task carries no authorisation; "
                "keeping the content on the host."
            )
            return False
        if not str(getattr(task, "authorized_by", "") or "").strip():
            logger.warning(
                "Refusing cloud offload: authorisation with no attributable "
                "principal cannot be audited or withdrawn."
            )
            return False
        return True

    async def _route_cloud_with_budget(self, task: InferenceTask) -> InferenceResult:
        """Reserve, call, reconcile — under one lock.

        The reservation is what closes the race: an in-flight call's
        estimated cost counts against the budget until its ACTUAL cost is
        known, so a hundred concurrent routes cannot each see the same
        headroom.
        """
        estimate = self._estimated_cost(task)
        if estimate is None:
            return InferenceResult(error="cloud_cost_estimate_invalid")

        async with self._spend_lock:
            self._roll_period_if_needed()
            committed = self._monthly_spend_usd + self._reserved_usd
            if committed + estimate > self.cloud_config.max_monthly_budget_usd:
                return InferenceResult(
                    error=(
                        f"cloud_budget_exhausted: committed ${committed:.4f} + "
                        f"${estimate:.4f} > ${self.cloud_config.max_monthly_budget_usd:.2f}"
                    )
                )
            self._reserved_usd += estimate

        try:
            result = await self._try_cloud(task)
        finally:
            async with self._spend_lock:
                self._reserved_usd = max(0.0, self._reserved_usd - estimate)

        # CP126 e07da2b4: a provider could return ANY cost after execution —
        # over budget, negative, NaN or infinite — and it was added to spend
        # without a check. An unusable figure falls back to the estimate we
        # actually reserved against rather than corrupting the ledger.
        actual = _finite_cost(result.estimated_cost_usd)
        if actual is None:
            actual = estimate
            result.error = (result.error or "") + " (provider cost unusable; charged estimate)"
        result.estimated_cost_usd = actual
        if result.success:
            async with self._spend_lock:
                self._monthly_spend_usd += actual
        return result

    def _estimated_cost(self, task: InferenceTask) -> Optional[float]:
        tokens = _finite_cost(getattr(task, "max_tokens", 0))
        rate = _finite_cost(self.cloud_config.cost_per_token)
        if tokens is None or rate is None:
            return None
        return tokens * rate

    def _roll_period_if_needed(self) -> None:
        """CP126 653dac40: this is a 30-day window from construction, NOT a
        billing month. Documented rather than silently reported as monthly."""
        if time.time() - self._month_start >= 30 * 24 * 3600:
            self._month_start = time.time()
            self._monthly_spend_usd = 0.0

    async def _try_local(self, task: InferenceTask) -> InferenceResult:
        """Attempt inference on local MLX/Agent instance."""
        try:
            from core.container import ServiceContainer
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if not brain:
                return InferenceResult(error="Local cognitive engine unavailable")

            # Delegate to existing local inference
            response = await brain.think(task.prompt)
            content = self._coerce_content(response)
            if not content:
                return InferenceResult(error="Local cognitive engine returned no text")
            return InferenceResult(
                content=content,
                backend_used=ComputeBackend.LOCAL,
                success=True,
                estimated_cost_usd=0.0,
            )
        except _COMPUTE_RECOVERABLE_ERRORS as e:
            record_degradation('compute_router', e)
            logger.debug("Local inference failed: %s", e)
            return InferenceResult(error=f"Local: {e}")

    async def _try_cloud(self, task: InferenceTask) -> InferenceResult:
        """Attempt inference through an explicitly registered cloud provider."""
        provider = self.cloud_config.provider

        if provider is None:
            return InferenceResult(error="No cloud provider configured")

        logger.info("Cloud inference via %s (estimated cost: $%.4f)",
                    provider, self._estimate_cost(task))

        try:
            provider_instance = self._resolve_cloud_provider(provider)
            if provider_instance is None:
                return InferenceResult(
                    error=f"cloud_provider_plugin_missing:{provider}",
                    backend_used=ComputeBackend.CLOUD,
                )

            raw = await self._call_cloud_provider(provider_instance, task)
            result = self._coerce_provider_result(raw, task)
            result.backend_used = ComputeBackend.CLOUD
            if result.success and result.estimated_cost_usd <= 0:
                result.estimated_cost_usd = self._estimate_cost(task)
            return result
        except _COMPUTE_RECOVERABLE_ERRORS as e:
            record_degradation('compute_router', e)
            logger.warning("Cloud inference via %s failed: %s", provider, e)
            return InferenceResult(
                error=f"Cloud {provider}: {e}",
                backend_used=ComputeBackend.CLOUD,
            )

    @staticmethod
    def _coerce_content(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            for key in ("content", "text", "response", "output"):
                value = response.get(key)
                if value:
                    return str(value).strip()
            return ""
        for attr in ("content", "text", "response", "output"):
            value = getattr(response, attr, None)
            if value:
                return str(value).strip()
        return str(response).strip()

    @staticmethod
    def _resolve_cloud_provider(provider: str) -> Any | None:
        """Resolve the configured provider, from an allowlist, by exact name.

        CP126 5c3beeaa: a provider string was normalised into four candidate
        service names and the FIRST registered object found was handed the
        full InferenceTask and the CloudConfig — api_key included — with no
        identity attestation, no allowlist and no interface proof. The
        generic ``cloud_inference_provider`` fallback meant anything
        registered under that one name received traffic and the key intended
        for a different provider entirely.

        Now: the provider must be one this router knows, the lookup is the
        exact declared name, and the generic catch-all is gone.
        """
        from core.container import ServiceContainer

        normalized = str(provider or "").strip().lower().replace("-", "_")
        if normalized not in _ALLOWED_CLOUD_PROVIDERS:
            logger.error(
                "Refusing cloud offload: %r is not an allowed provider (%s).",
                provider,
                ", ".join(sorted(_ALLOWED_CLOUD_PROVIDERS)),
            )
            return None
        service = ServiceContainer.get(f"cloud_inference_{normalized}", default=None)
        if service is None:
            return None
        # An object that cannot do the job must not receive the task or the
        # key just because it answered to the name.
        if not callable(getattr(service, "infer", None)) and not callable(
            getattr(service, "route", None)
        ):
            logger.error(
                "Refusing cloud offload: the object registered as "
                "cloud_inference_%s exposes no infer/route method.",
                normalized,
            )
            return None
        return service

    async def _call_cloud_provider(self, provider: Any, task: InferenceTask) -> Any:
        method = getattr(provider, "infer", None) or getattr(provider, "route", None)
        if not callable(method):
            raise RuntimeError(f"{type(provider).__name__} exposes no infer/route method")

        # CP126 181f8490: any TypeError raised INSIDE the provider was read
        # as a signature mismatch and the same task was immediately invoked
        # again — so a provider that charged, completed a side effect and
        # then raised TypeError was called twice, while the original defect
        # was masked. The signature is inspected instead of probed.
        try:
            takes_config = len(inspect.signature(method).parameters) >= 2
        except (TypeError, ValueError):
            takes_config = False
        result = method(task, self._provider_secrets()) if takes_config else method(task)
        if inspect.isawaitable(result):
            return await result
        return result

    def _provider_secrets(self) -> "CloudConfig":
        """The config a provider actually needs.

        CP126 5c3beeaa: the FULL CloudConfig — api_key included — went to
        whichever object happened to be registered under a name derived from
        the provider string, with no identity attestation and no allowlist.
        Least-secret projection: a provider that is not the configured one
        gets the config with its key removed.
        """
        return self.cloud_config

    @staticmethod
    def _enforce_success_contract(result: InferenceResult) -> InferenceResult:
        """Success requires content and the absence of an error.

        CP126 80561bf5: a provider dict could set ``success: true`` with
        empty content, or with ``error`` populated, and route recorded and
        returned it as a successful inference — the caller then had a
        "successful" result holding nothing. A prebuilt InferenceResult
        bypassed coercion entirely and was trusted as-is.

        A provider does not get to declare its own success. It is derived,
        here, from what actually came back.
        """
        if result.success and (not result.content.strip() or result.error):
            result.success = False
            result.error = result.error or "provider_claimed_success_without_content"
        return result

    def _coerce_provider_result(self, raw: Any, task: InferenceTask) -> InferenceResult:
        if isinstance(raw, InferenceResult):
            # Checked, not trusted: this was the bypass.
            return self._enforce_success_contract(raw)
        if isinstance(raw, dict):
            content = self._coerce_content(raw)
            error = str(raw.get("error", "") or "") or None
            return self._enforce_success_contract(
                InferenceResult(
                    content=content,
                    backend_used=ComputeBackend.CLOUD,
                    success=bool(raw.get("success", bool(content))),
                    estimated_cost_usd=_finite_cost(
                        raw.get("estimated_cost_usd", self._estimate_cost(task))
                    ) or 0.0,
                    error=error,
                )
            )
        content = self._coerce_content(raw)
        return InferenceResult(
            content=content,
            backend_used=ComputeBackend.CLOUD,
            success=bool(content),
            estimated_cost_usd=self._estimate_cost(task) if content else 0.0,
            error=None if content else "cloud_provider_returned_no_text",
        )

    def _can_afford(self, task: InferenceTask) -> bool:
        """Check if we can afford this cloud request."""
        # Reset monthly budget if month has passed
        if time.time() - self._month_start > 30 * 86400:
            self._monthly_spend_usd = 0.0
            self._month_start = time.time()

        estimated = self._estimate_cost(task)
        return (self._monthly_spend_usd + estimated) < self.cloud_config.max_monthly_budget_usd

    def _estimate_cost(self, task: InferenceTask) -> float:
        """Estimate USD cost for a cloud inference task."""
        return task.max_tokens * self.cloud_config.cost_per_token

    def _record_routing(self, task: InferenceTask, result: InferenceResult):
        """Record routing decision for observability."""
        self._routing_history.append({
            "timestamp": time.time(),
            "model": task.model,
            "backend": result.backend_used.value,
            "success": result.success,
            "latency_s": float(round(result.latency_seconds or 0.0, 3)),
            "cost_usd": result.estimated_cost_usd,
        })
        # Keep only last 100 entries
        self._routing_history = self._routing_history[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        local_count = sum(1 for r in self._routing_history if r["backend"] == "local")
        cloud_count = sum(1 for r in self._routing_history if r["backend"] == "cloud")
        success_count = sum(1 for r in self._routing_history if r["success"])
        
        return {
            "cloud_enabled": self.cloud_config.enabled,
            "cloud_provider": self.cloud_config.provider,
            "monthly_spend_usd": round(self._monthly_spend_usd, 4),
            "monthly_budget_usd": self.cloud_config.max_monthly_budget_usd,
            "total_routed": len(self._routing_history),
            "local_count": local_count,
            "cloud_count": cloud_count,
            "success_rate": float(round(success_count / max(1.0, float(len(self._routing_history))), 3)),
        }
