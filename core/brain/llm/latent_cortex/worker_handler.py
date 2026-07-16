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
import uuid
from pathlib import Path
from typing import Any, Callable

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

_CONFIG_KEYS = {
    "alpha",
    "alpha_schedule",
    "allow_vanilla_fallback",
    "anchor_scale",
    "coda_frac",
    "collapse_cos_threshold",
    "comm_slot",
    "convergence_eps",
    "decode_max_tokens",
    "decode_temperature",
    "decode_top_p",
    "divergence_ratio",
    "exchange_gamma",
    "exchange_interval",
    "fast_weights",
    "fast_weights_lr",
    "fast_weights_max_layers",
    "fast_weights_opt_steps",
    "fast_weights_rank",
    "fast_weights_scale",
    "fast_weights_target",
    "jitter_scale",
    "input_context_max_chars",
    "latent_opt",
    "latent_opt_control",
    "latent_opt_lambda_manifold",
    "latent_opt_lambda_reconstruct",
    "latent_opt_lr",
    "latent_opt_max_grad_norm",
    "latent_opt_steps",
    "max_steps",
    "min_steps",
    "n_branches",
    "n_slots",
    "prelude_frac",
    "rms_clip_ratio",
    "schedule",
    "seed",
}


def _typed_value(raw: dict[str, Any], key: str, default: Any, expected: type) -> Any:
    value = raw.get(key, default)
    if expected is bool:
        if type(value) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ValueError(f"{key} must be a JSON integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a JSON number")
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite JSON number") from exc
    if expected is str:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value
    raise TypeError(f"unsupported wire type for {key}")


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
    if job_config is not None and not isinstance(job_config, dict):
        raise ValueError("latent_reason config must be a mapping")
    raw = dict(job_config or {})
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"latent_reason config contains unknown keys: {unknown}")
    cfg = CortexConfig(
        workspace=WorkspaceConfig(
            n_slots=_typed_value(raw, "n_slots", 16, int),
            seed=_typed_value(raw, "seed", 0, int),
            anchor_scale=_typed_value(raw, "anchor_scale", 0.05, float),
        ),
        recurrence=RecurrenceConfig(
            max_steps=_typed_value(raw, "max_steps", 8, int),
            min_steps=_typed_value(raw, "min_steps", 2, int),
            alpha=_typed_value(raw, "alpha", 0.5, float),
            alpha_schedule=_typed_value(raw, "alpha_schedule", "cosine", str),
            rms_clip_ratio=_typed_value(raw, "rms_clip_ratio", 3.0, float),
            convergence_eps=_typed_value(raw, "convergence_eps", 0.02, float),
            divergence_ratio=_typed_value(raw, "divergence_ratio", 10.0, float),
        ),
        branches=BranchConfig(
            n_branches=_typed_value(raw, "n_branches", 2, int),
            exchange_interval=_typed_value(raw, "exchange_interval", 4, int),
            exchange_gamma=_typed_value(raw, "exchange_gamma", 0.35, float),
            comm_slot=_typed_value(raw, "comm_slot", 0, int),
            collapse_cos_threshold=_typed_value(
                raw, "collapse_cos_threshold", 0.98, float
            ),
            jitter_scale=_typed_value(raw, "jitter_scale", 0.02, float),
        ),
        latent_opt=LatentOptConfig(
            enabled=_typed_value(raw, "latent_opt", False, bool),
            steps=_typed_value(raw, "latent_opt_steps", 4, int),
            lr=_typed_value(raw, "latent_opt_lr", 0.05, float),
            lambda_reconstruct=_typed_value(
                raw, "latent_opt_lambda_reconstruct", 1.0, float
            ),
            lambda_manifold=_typed_value(
                raw, "latent_opt_lambda_manifold", 0.5, float
            ),
            max_grad_norm=_typed_value(raw, "latent_opt_max_grad_norm", 1.0, float),
            control_mode=_typed_value(raw, "latent_opt_control", False, bool),
        ),
        fast_weights=FastWeightsConfig(
            enabled=_typed_value(raw, "fast_weights", False, bool),
            rank=_typed_value(raw, "fast_weights_rank", 2, int),
            scale=_typed_value(raw, "fast_weights_scale", 1.0, float),
            target=_typed_value(raw, "fast_weights_target", "o_proj", str),
            opt_steps=_typed_value(raw, "fast_weights_opt_steps", 4, int),
            lr=_typed_value(raw, "fast_weights_lr", 0.01, float),
            max_wrapped_layers=_typed_value(
                raw, "fast_weights_max_layers", 8, int
            ),
        ),
        prelude_frac=_typed_value(raw, "prelude_frac", 0.25, float),
        coda_frac=_typed_value(raw, "coda_frac", 0.25, float),
        schedule=raw.get("schedule"),
        decode_max_tokens=_typed_value(raw, "decode_max_tokens", 512, int),
        decode_temperature=_typed_value(raw, "decode_temperature", 0.0, float),
        decode_top_p=_typed_value(raw, "decode_top_p", 1.0, float),
        input_context_max_chars=_typed_value(
            raw, "input_context_max_chars", 0, int
        ),
        allow_vanilla_fallback=_typed_value(
            raw, "allow_vanilla_fallback", True, bool
        ),
    )
    problems = cfg.validate()
    if problems:
        raise ValueError(f"latent_reason config rejected: {problems}")
    return cfg


def budget_from_job(job_budget: dict[str, Any] | None) -> ComputeBudget:
    if job_budget is not None and not isinstance(job_budget, dict):
        raise ValueError("latent_reason budget must be a mapping")
    raw = dict(job_budget or {})
    unknown = sorted(set(raw) - {"max_layer_apps", "wall_clock_s"})
    if unknown:
        raise ValueError(f"latent_reason budget contains unknown keys: {unknown}")
    kwargs: dict[str, Any] = {}
    if "max_layer_apps" in raw:
        kwargs["max_layer_apps"] = _typed_value(raw, "max_layer_apps", 0, int)
    if "wall_clock_s" in raw:
        kwargs["wall_clock_s"] = _typed_value(raw, "wall_clock_s", 0.0, float)
    return ComputeBudget(**kwargs)


def handle_latent_reason(
    job: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    model_path: str,
    worker_identity: dict[str, Any] | None = None,
    surface_control_state: dict[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[dict], None] | None = None,
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
    episode_messages = messages if isinstance(messages, list) else None
    context_compaction: dict[str, Any] = {}
    if episode_messages is not None and config.input_context_max_chars:
        from core.brain.llm.latent_cortex.context_compaction import (
            compact_latent_messages,
        )

        episode_messages, context_compaction = compact_latent_messages(
            episode_messages,
            max_chars=config.input_context_max_chars,
        )
    result = engine.reason(
        prompt=prompt if isinstance(prompt, str) else None,
        messages=episode_messages,
        budget=budget,
        domain=str(job.get("domain", "general")),
        cancel_check=cancel_check,
        progress=progress,
    )
    if worker_identity is None:
        from core.brain.llm.latent_cortex.runtime_identity import build_worker_identity

        worker_identity = build_worker_identity(
            model,
            model_path=model_path,
            worker_boot_id=uuid.uuid4().hex,
            worker_source_path=Path(__file__).resolve().parents[1] / "mlx_worker.py",
        )
    receipt = result.receipt
    receipt.worker_boot_id = str(worker_identity.get("worker_boot_id") or "")
    receipt.worker_pid = int(worker_identity.get("worker_pid") or 0)
    receipt.worker_model_path = str(worker_identity.get("worker_model_path") or "")
    receipt.worker_model_parameter_count = int(
        worker_identity.get("worker_model_parameter_count") or 0
    )
    receipt.worker_model_stored_parameter_element_count = int(
        worker_identity.get("worker_model_stored_parameter_element_count") or 0
    )
    receipt.worker_model_parameter_count_basis = str(
        worker_identity.get("worker_model_parameter_count_basis") or ""
    )
    receipt.worker_source_sha256 = str(worker_identity.get("worker_source_sha256") or "")
    receipt.worker_affective_steering_active = bool(
        worker_identity.get("worker_affective_steering_active", False)
    )
    receipt.worker_affective_steering_alpha = float(
        worker_identity.get("worker_affective_steering_alpha") or 0.0
    )
    receipt.input_context_compaction = dict(context_compaction)
    control_state = dict(surface_control_state or {})
    applied_alpha = control_state.get("surface_alpha_applied")
    receipt.episode_affective_steering_applied = bool(
        receipt.worker_affective_steering_active
        and isinstance(applied_alpha, (int, float))
        and not isinstance(applied_alpha, bool)
    )
    receipt.episode_affective_steering_alpha = (
        float(applied_alpha)
        if receipt.episode_affective_steering_applied
        else 0.0
    )
    from core.brain.llm.latent_cortex.runtime_identity import (
        latent_request_payload_sha256,
    )

    receipt.request_payload_sha256 = latent_request_payload_sha256(
        prompt=prompt,
        messages=messages,
        domain=str(job.get("domain", "general")),
        config=job.get("config"),
        budget=job.get("budget"),
        runtime_controls=job.get("runtime_controls"),
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
    body["requires_worker_recycle"] = (
        result.receipt.params_unchanged is False
        or (
            result.receipt.fast_weights_applied
            and result.receipt.fast_weights_erased is not True
        )
    )
    return body


__all__ = [
    "budget_from_job",
    "config_from_job",
    "cortex_enabled",
    "handle_latent_reason",
]
