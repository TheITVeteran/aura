"""core/brain/latent_cortex_service.py

Orchestrator-side facade for the Recursive Latent Cortex
(docs/RECURSIVE_LATENT_CORTEX.md). The engine itself runs inside the MLX
worker on the RESIDENT model; this service is the cognitive economy around
it — it decides how much latent computation a problem deserves and routes
the episode through the worker IPC.

The allocation policy is the spec's: thought (T, branches, budget) scales
with stakes and uncertainty, and is DAMPED by the body's real+anticipatory
pressure — a system heading toward crisis spends less on deep thought, which
is exactly what the allostasis seam is for.

Fail-honest: any refusal (kill switch, busy lane, no resident model, worker
error) returns ``{"ok": False, "reason": ...}`` so callers fall back to
ordinary generation EXPLICITLY. Nothing here fakes an answer.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentCortexService")


def _cortex_enabled() -> bool:
    return str(os.environ.get("AURA_LATENT_CORTEX", "1")).strip() != "0"


class LatentCortexService:
    """Budget allocation + IPC routing for latent-reasoning episodes."""

    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        self._episodes = 0
        self._ok_episodes = 0
        self._last_receipt: dict[str, Any] = {}
        self._last_refusal = ""
        logger.info("🧠 LatentCortexService initialized (Recursive Latent Cortex)")

    # ── Cognitive economy ───────────────────────────────────────────────
    def _body_pressure(self) -> float:
        """Total real+anticipatory body pressure in [0, 1]; 0 when unknown."""
        try:
            from core.being.aura_now import BodyState

            state = getattr(self.orchestrator, "state", None)
            return float(BodyState.from_aura_state(state).total_pressure())
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    def allocate(self, *, stakes: float, uncertainty: float) -> tuple[dict, dict]:
        """(config, budget) for one episode: the Will's thought allocation.

        More stakes/uncertainty ⇒ deeper recurrence, wider branches, bigger
        budget. Body pressure damps everything — deep thought is a luxury a
        strained body rations first.
        """
        stakes = min(1.0, max(0.0, float(stakes)))
        uncertainty = min(1.0, max(0.0, float(uncertainty)))
        pressure = min(1.0, max(0.0, self._body_pressure()))
        headroom = 1.0 - 0.7 * pressure

        max_steps = max(2, min(16, round((4 + 10 * uncertainty) * headroom)))
        n_branches = 1 if stakes < 0.3 else (3 if stakes > 0.75 and headroom > 0.6 else 2)
        config = {
            "n_slots": 16,
            "max_steps": max_steps,
            "min_steps": 2,
            "n_branches": n_branches,
            "alpha_schedule": "cosine",
            "decode_max_tokens": 512,
        }
        budget = {
            "max_layer_apps": int((2_000_000 + 8_000_000 * stakes) * headroom),
            "wall_clock_s": float(30.0 + 90.0 * stakes * headroom),
        }
        return config, budget

    # ── The episode ─────────────────────────────────────────────────────
    async def deep_reason(
        self,
        question: str | None = None,
        *,
        messages: list | None = None,
        stakes: float = 0.5,
        uncertainty: float = 0.5,
        domain: str = "general",
        config_overrides: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Run one latent-reasoning episode on the resident model."""
        if not _cortex_enabled():
            self._last_refusal = "disabled:AURA_LATENT_CORTEX=0"
            return {"ok": False, "reason": self._last_refusal}
        if not question and not messages:
            return {"ok": False, "reason": "empty_question"}

        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode: resident model client unavailable",
            )
            self._last_refusal = f"client_unavailable:{type(exc).__name__}"
            return {"ok": False, "reason": self._last_refusal}
        if client is None:
            self._last_refusal = "no_resident_model"
            return {"ok": False, "reason": self._last_refusal}

        config, budget = self.allocate(stakes=stakes, uncertainty=uncertainty)
        if config_overrides:
            config.update(dict(config_overrides))

        self._episodes += 1
        started = time.monotonic()
        result = await client.latent_reason_async(
            prompt=question,
            messages=messages,
            config=config,
            budget=budget,
            domain=domain,
            timeout_s=timeout_s,
        )
        elapsed = time.monotonic() - started
        if result.get("ok"):
            self._ok_episodes += 1
            self._last_receipt = dict(result.get("receipt") or {})
            logger.info(
                "🧠 Latent episode ok: %d steps, %d branches, halt=%s, %.1fs",
                int(self._last_receipt.get("steps_taken") or 0),
                int(self._last_receipt.get("n_branches") or 0),
                self._last_receipt.get("halting_reason"),
                elapsed,
            )
        else:
            self._last_refusal = str(result.get("reason") or "unknown")
            logger.info("🧠 Latent episode refused/failed: %s (%.1fs)", self._last_refusal, elapsed)
        return result

    # ── Health ──────────────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": _cortex_enabled(),
            "episodes": self._episodes,
            "ok_episodes": self._ok_episodes,
            "last_refusal": self._last_refusal,
            "last_receipt": {
                k: self._last_receipt.get(k)
                for k in (
                    "episode_id",
                    "steps_taken",
                    "halting_reason",
                    "n_branches",
                    "schedule_hash",
                    "params_unchanged",
                    "honest_flags",
                )
                if k in self._last_receipt
            },
            "healthy": True,
        }


_INSTANCE: LatentCortexService | None = None


def get_latent_cortex_service(orchestrator: Any = None) -> LatentCortexService:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LatentCortexService(orchestrator=orchestrator)
    return _INSTANCE


def register_latent_cortex(orchestrator: Any = None) -> LatentCortexService:
    from core.runtime.service_registry import get_runtime_service, register_runtime_service
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.LATENT_CORTEX, default=None) or get_latent_cortex_service(
        orchestrator
    )
    register_runtime_service(
        ServiceNames.LATENT_CORTEX,
        inst,
        required=False,
        owner="core/brain/latent_cortex_service.py",
        registered_by="register_latent_cortex",
    )
    return inst


__all__ = [
    "LatentCortexService",
    "get_latent_cortex_service",
    "register_latent_cortex",
]
