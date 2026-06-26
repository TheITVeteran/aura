"""core/brain/compute_router.py — Compute Offloading Architecture

Routes inference tasks between local and cloud GPU backends with
cost/latency awareness. Cloud offloading is DISABLED by default and
requires explicit user opt-in via configuration.

Design principles:
  1. Local-first: Always prefer local MLX instance
  2. Consent-gated: Cloud requires explicit opt-in
  3. Cost-aware: Tracks estimated costs per request
  4. Graceful fallback: If cloud fails, queue for local
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
        self._month_start = time.time()
        self._routing_history: List[Dict] = []
        self._local_queue_depth = 0
        
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

        # If local failed and cloud is available, try cloud
        if self.cloud_config.enabled and self._can_afford(task):
            logger.info("Local inference failed, attempting cloud offload...")
            result = await self._try_cloud(task)
            if result.success:
                result.latency_seconds = time.monotonic() - start
                self._monthly_spend_usd += result.estimated_cost_usd
                self._record_routing(task, result)
                return result

        # Both failed
        result.latency_seconds = time.monotonic() - start
        if not result.error:
            result.error = "All compute backends exhausted"
        self._record_routing(task, result)
        return result

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
        from core.container import ServiceContainer

        normalized = provider.strip().lower().replace("-", "_")
        service_names = (
            f"cloud_inference:{provider}",
            f"cloud_inference_{normalized}",
            f"compute_cloud_{normalized}",
            "cloud_inference_provider",
        )
        for service_name in service_names:
            service = ServiceContainer.get(service_name, default=None)
            if service is not None:
                return service
        return None

    async def _call_cloud_provider(self, provider: Any, task: InferenceTask) -> Any:
        method = getattr(provider, "infer", None) or getattr(provider, "route", None)
        if not callable(method):
            raise RuntimeError(f"{type(provider).__name__} exposes no infer/route method")

        try:
            result = method(task, self.cloud_config)
        except TypeError:
            result = method(task)
        if inspect.isawaitable(result):
            return await result
        return result

    def _coerce_provider_result(self, raw: Any, task: InferenceTask) -> InferenceResult:
        if isinstance(raw, InferenceResult):
            return raw
        if isinstance(raw, dict):
            content = self._coerce_content(raw)
            success = bool(raw.get("success", bool(content)))
            return InferenceResult(
                content=content,
                backend_used=ComputeBackend.CLOUD,
                success=success,
                estimated_cost_usd=float(raw.get("estimated_cost_usd", self._estimate_cost(task)) or 0.0),
                error=str(raw.get("error", "") or "") or None,
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
