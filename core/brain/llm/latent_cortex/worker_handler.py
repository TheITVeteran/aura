"""Worker-side handler for the ``latent_reason`` IPC action.

Lives in its own module so ``mlx_worker.py`` gains one surgical elif and the
whole latent-reasoning surface stays independently testable. The handler is
synchronous and runs inside the worker process while the caller holds the
metal semaphore — the resident model is exclusively ours for the episode.

Job contract (all optional except the prompt source):
{
  "action": "latent_reason",
  "id": "...",
  "prompt": "..."            # or "messages": [...]
  "domain": "general",
  "config": {                # conservative defaults; hard caps in types.py
     "n_slots": 16, "n_branches": 2, "max_steps": 8,
     "latent_opt": false, "fast_weights": false,
     "decode_max_tokens": 512, "decode_temperature": 0.0,
     "schedule": {...}       # optional explicit program
  },
  "budget": {"max_layer_apps": ..., "wall_clock_s": ...}
}

Kill switch: AURA_LATENT_CORTEX=0 refuses every episode with an honest
reason — the caller falls back to ordinary generation, no silence.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.engine import LatentCortexEngine
from core.brain.llm.latent_cortex.schedules import ScheduleLibrary
from core.brain.llm.latent_cortex.types import (
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    FastWeightsConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

logger = logging.getLogger("Aura.LatentCortex.WorkerHandler")

_schedule_library: ScheduleLibrary | None = None


def cortex_enabled() -> bool:
    return str(os.environ.get("AURA_LATENT_CORTEX", "1")).strip() != "0"


def _library() -> ScheduleLibrary | None:
    """Process-wide schedule library, persisted under the data dir."""
    global _schedule_library
    if _schedule_library is None:
        try:
            from core.config import DATA_DIR

            path = Path(DATA_DIR) / "latent_cortex" / "schedule_library.json"
            _schedule_library = ScheduleLibrary(path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Schedule library unavailable (%s); using defaults.", exc)
            _schedule_library = ScheduleLibrary(None)
    return _schedule_library


def config_from_job(job_config: dict[str, Any] | None) -> CortexConfig:
    """Translate the wire config into a validated CortexConfig."""
    raw = dict(job_config or {})
    cfg = CortexConfig(
        workspace=WorkspaceConfig(
            n_slots=int(raw.get("n_slots", 16)),
            seed=int(raw.get("seed", 0)),
        ),
        recurrence=RecurrenceConfig(
            max_steps=int(raw.get("max_steps", 8)),
            min_steps=int(raw.get("min_steps", 2)),
            alpha=float(raw.get("alpha", 0.5)),
            alpha_schedule=str(raw.get("alpha_schedule", "cosine")),
        ),
        branches=BranchConfig(
            n_branches=int(raw.get("n_branches", 2)),
            exchange_interval=int(raw.get("exchange_interval", 4)),
        ),
        latent_opt=LatentOptConfig(
            enabled=bool(raw.get("latent_opt", False)),
            steps=int(raw.get("latent_opt_steps", 4)),
            control_mode=bool(raw.get("latent_opt_control", False)),
        ),
        fast_weights=FastWeightsConfig(
            enabled=bool(raw.get("fast_weights", False)),
            rank=int(raw.get("fast_weights_rank", 2)),
            target=str(raw.get("fast_weights_target", "o_proj")),
        ),
        schedule=raw.get("schedule"),
        decode_max_tokens=int(raw.get("decode_max_tokens", 512)),
        decode_temperature=float(raw.get("decode_temperature", 0.0)),
    )
    problems = cfg.validate()
    if problems:
        raise ValueError(f"latent_reason config rejected: {problems}")
    return cfg


def budget_from_job(job_budget: dict[str, Any] | None) -> ComputeBudget:
    raw = dict(job_budget or {})
    kwargs: dict[str, Any] = {}
    if "max_layer_apps" in raw:
        kwargs["max_layer_apps"] = int(raw["max_layer_apps"])
    if "wall_clock_s" in raw:
        kwargs["wall_clock_s"] = float(raw["wall_clock_s"])
    return ComputeBudget(**kwargs)


def handle_latent_reason(
    job: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    model_path: str,
) -> dict[str, Any]:
    """Run one latent-reasoning episode on the resident model.

    Returns the IPC response body. Never raises for episode-level failures —
    the engine's fail-honest contract puts them in the receipt; only truly
    malformed jobs surface as status=error.
    """
    if not cortex_enabled():
        return {
            "status": "error",
            "message": "latent_cortex_disabled:AURA_LATENT_CORTEX=0",
        }
    prompt = job.get("prompt")
    messages = job.get("messages")
    if not prompt and not messages:
        return {"status": "error", "message": "latent_reason requires prompt or messages"}

    config = config_from_job(job.get("config"))
    budget = budget_from_job(job.get("budget"))
    engine = LatentCortexEngine(
        model,
        tokenizer,
        config,
        model_path=model_path,
        schedule_library=_library(),
    )
    result = engine.reason(
        prompt=prompt if isinstance(prompt, str) else None,
        messages=messages if isinstance(messages, list) else None,
        budget=budget,
        domain=str(job.get("domain", "general")),
    )
    body = result.to_dict()
    body["status"] = "ok" if result.ok else "error"
    if not result.ok:
        body["message"] = result.reason
    # An unproven fast-weight erase means prompt caches computed before the
    # episode can no longer be trusted — the caller must clear them.
    body["requires_cache_clear"] = (
        result.receipt.fast_weights_applied
        and result.receipt.fast_weights_erased is not True
    )
    return body


__all__ = [
    "budget_from_job",
    "config_from_job",
    "cortex_enabled",
    "handle_latent_reason",
]
