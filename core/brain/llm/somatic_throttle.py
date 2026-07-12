"""core/brain/llm/somatic_throttle.py — Somatic Compute Sentinel

Enforces metabolic constraints directly inside the LLM token sampling loop.
Connects with DamasioV2 virtual physiology and hardware telemetry (RAM/CPU loads)
to dynamically scale down LLM generation parameters under heavy load to prevent
thermal throttling or out-of-memory (OOM) crashes.
"""
import logging
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_affect_engine

logger = logging.getLogger("Aura.Brain.SomaticThrottle")

_SOMATIC_THROTTLE_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_somatic_throttle_degradation(exc: BaseException, *, action: str) -> None:
    record_degradation(
        "somatic_throttle",
        exc,
        severity="warning",
        action=action,
    )


class SomaticComputeSentinel:
    """Natively enforces metabolic constraints directly inside the LLM token generation cycle."""

    def __init__(self):
        logger.info("🌡️ SomaticComputeSentinel initialized.")

    def adjust_generation_options(self, base_options: dict[str, Any]) -> dict[str, Any]:
        """Dynamically adjusts LLM sampling and length parameters based on metabolic and hardware stress."""
        # 1. Fetch virtual physiological stress (arousal)
        arousal = 0.0
        try:
            affect = resolve_affect_engine(default=None)
            if affect and hasattr(affect, "current"):
                current_state = affect.current
                if hasattr(current_state, "arousal"):
                    arousal = float(current_state.arousal)
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using neutral arousal for generation throttle")
            logger.debug("Failed to resolve affect engine arousal: %s", e)

        # 1b. Fetch governance token throttle factor
        gov_throttle = 1.0
        try:
            from research.protocols.resource_quotas import get_compute_governor
            gov = get_compute_governor()
            gov_throttle = gov.get_throttle_factor()
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using default compute governor throttle")
            logger.debug("Failed to resolve compute governor: %s", e)

        # 2. Fetch hardware stress metrics
        cpu_load = 0.0
        ram_pct = 0.0
        try:
            cpu_load = psutil.cpu_percent(interval=0) / 100.0
            ram_pct = psutil.virtual_memory().percent / 100.0
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using neutral hardware metrics for generation throttle")
            logger.debug("Failed to retrieve hardware metrics: %s", e)

        # 3. Determine if systemic overload is present.
        # Virtual arousal is an affective/context signal. It becomes a compute
        # throttling signal only when it is coupled to real resource pressure.
        arousal_resource_coupled = arousal > 0.9 and (ram_pct > 0.82 or cpu_load > 0.85)
        is_stressed = (
            arousal_resource_coupled
            or (ram_pct > 0.88)
            or (cpu_load > 0.9)
            or (gov_throttle <= 0.5)
        )
        is_critical = (
            (arousal > 0.97 and (ram_pct > 0.9 or cpu_load > 0.92))
            or (ram_pct > 0.93)
            or (cpu_load > 0.95)
            or (gov_throttle <= 0.2)
        )

        if gov_throttle == 0.0:
            # Token exhaustion: severe cap to block further consumption
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 8)
            base_options["temperature"] = 0.05
            logger.error("🚫 GOVERNANCE QUOTA EXHAUSTED: Token limit hit. Sampling capped to 8 tokens.")
        elif is_critical:
            # Force severe parameter cuts to prevent OOM/Thermal crash
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 128)
            base_options["temperature"] = 0.15
            # Throttle recurrent lane depth if supported by token generator
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.2
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.2
            logger.warning(
                "🔥 CRITICAL METABOLIC PANIC: Arousal=%.2f, RAM=%.1f%%, CPU=%.1f%%, GovThrottle=%.2f. Parameter throttle ENABLED (max_tokens capped at 128).",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle
            )
        elif is_stressed:
            # Moderate parameter cuts
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 256)
            base_options["temperature"] = 0.3
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.4
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.4
            logger.info(
                "⚠️ SYSTEMIC STRESS DETECTED: Arousal=%.2f, RAM=%.1f%%, CPU=%.1f%%, GovThrottle=%.2f. Parameter throttle activated (max_tokens capped at 256).",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle
            )

        return base_options
