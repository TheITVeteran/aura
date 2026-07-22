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
  "response_contract": "{...}", # optional public shape DSL, no answer values
  "config": {                # conservative defaults; hard caps in types.py
     "n_slots": 16, "n_branches": 2, "max_steps": 8,
     "latent_opt": false, "fast_weights": false,
     "decode_max_tokens": 512, "decode_temperature": 0.0,
     "verifier_probe_max_tokens": 48,
     "verifier_accept_non_regression": false,
     "decode_bridge_policy": "none",
     "schedule": {...}       # optional explicit program
  },
  "budget": {"max_layer_apps": ..., "wall_clock_s": ...}
}

Kill switch: AURA_LATENT_CORTEX=0 refuses every episode with an honest
reason — the caller falls back to ordinary generation, no silence.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import Callable
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
    "decode_contract",
    "decode_contract_grace_tokens",
    "decode_min_tokens",
    "decode_bridge_policy",
    "decode_repetition_penalty",
    "decode_repetition_window",
    "decode_temperature",
    "decode_top_p",
    "divergence_ratio",
    "escape",
    "exchange_gamma",
    "exchange_interval",
    "fast_weights",
    "fast_weights_canary",
    "fast_weights_canary_max_delta_rms",
    "fast_weights_canary_max_drop",
    "fast_weights_canary_max_tokens",
    "fast_weights_canary_rescale_attempts",
    "fast_weights_lr",
    "fast_weights_max_layers",
    "fast_weights_export_candidates",
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
    "halting",
    "probe_cache",
    "telemetry",
    "verifier_accept_non_regression",
    "verifier_probe_max_tokens",
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
            export_candidates=_typed_value(
                raw, "fast_weights_export_candidates", False, bool
            ),
            canary_enabled=_typed_value(raw, "fast_weights_canary", True, bool),
            canary_max_logprob_drop=_typed_value(
                raw, "fast_weights_canary_max_drop", 0.5, float
            ),
            canary_max_effective_delta_rms=_typed_value(
                raw, "fast_weights_canary_max_delta_rms", 0.05, float
            ),
            canary_rescale_attempts=_typed_value(
                raw, "fast_weights_canary_rescale_attempts", 2, int
            ),
            canary_max_tokens=_typed_value(
                raw, "fast_weights_canary_max_tokens", 24, int
            ),
        ),
        prelude_frac=_typed_value(raw, "prelude_frac", 0.25, float),
        coda_frac=_typed_value(raw, "coda_frac", 0.25, float),
        schedule=raw.get("schedule"),
        decode_max_tokens=_typed_value(raw, "decode_max_tokens", 512, int),
        decode_contract=_typed_value(raw, "decode_contract", "none", str),
        decode_contract_grace_tokens=_typed_value(
            raw, "decode_contract_grace_tokens", 0, int
        ),
        decode_min_tokens=_typed_value(raw, "decode_min_tokens", 0, int),
        verifier_probe_max_tokens=_typed_value(
            raw, "verifier_probe_max_tokens", 48, int
        ),
        verifier_accept_non_regression=_typed_value(
            raw, "verifier_accept_non_regression", False, bool
        ),
        decode_temperature=_typed_value(raw, "decode_temperature", 0.0, float),
        decode_top_p=_typed_value(raw, "decode_top_p", 1.0, float),
        decode_repetition_penalty=_typed_value(
            raw, "decode_repetition_penalty", 1.0, float
        ),
        decode_repetition_window=_typed_value(
            raw, "decode_repetition_window", 72, int
        ),
        decode_bridge_policy=_typed_value(
            raw, "decode_bridge_policy", "none", str
        ),
        input_context_max_chars=_typed_value(
            raw, "input_context_max_chars", 0, int
        ),
        allow_vanilla_fallback=_typed_value(
            raw, "allow_vanilla_fallback", True, bool
        ),
        escape=raw.get("escape"),
        telemetry_enabled=_typed_value(raw, "telemetry", True, bool),
        probe_cache_enabled=_typed_value(raw, "probe_cache", True, bool),
        halting=raw.get("halting"),
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

    response_contract = job.get("response_contract")
    if response_contract is not None:
        if not isinstance(response_contract, str) or not response_contract.strip():
            return {
                "status": "error",
                "message": "latent_reason response_contract must be a non-empty string",
            }
        try:
            from core.brain.llm.latent_cortex.response_contracts import (
                parse_response_contract,
            )

            parse_response_contract(response_contract)
        except ValueError as exc:
            return {
                "status": "error",
                "message": f"latent_reason response_contract rejected: {exc}",
            }
        if tokenizer is None:
            return {
                "status": "error",
                "message": "latent_reason response_contract requires a tokenizer",
            }

    raw_config = job.get("config")
    if response_contract is not None:
        if raw_config is not None and not isinstance(raw_config, dict):
            return {"status": "error", "message": "latent_reason config must be a mapping"}
        raw_config = dict(raw_config or {})
        configured_contract = raw_config.get("decode_contract")
        if configured_contract not in (None, "final_answer_v1"):
            return {
                "status": "error",
                "message": (
                    "latent_reason response_contract conflicts with "
                    "config.decode_contract"
                ),
            }
        raw_config["decode_contract"] = "final_answer_v1"
        raw_config.setdefault(
            "decode_contract_grace_tokens",
            min(int(raw_config.get("decode_max_tokens", 512)), 512),
        )

    config = config_from_job(raw_config)
    budget = budget_from_job(job.get("budget"))
    cognitive_context = job.get("cognitive_context")
    if cognitive_context is not None:
        if not isinstance(cognitive_context, list) or len(cognitive_context) > 6:
            return {
                "status": "error",
                "message": "latent_reason cognitive_context must be a list of at most 6 items",
            }
        for entry in cognitive_context:
            basic_invalid = (
                not isinstance(entry, dict)
                or not isinstance(entry.get("source"), str)
                or not entry["source"].strip()
                or not isinstance(entry.get("text"), str)
                or not entry["text"].strip()
                or len(entry["text"]) > 400
            )
            if basic_invalid:
                return {
                    "status": "error",
                    "message": (
                        "latent_reason cognitive_context entries require source and text<=400 chars"
                    ),
                }
            memory_fields = {
                "context_role",
                "instruction_authority",
                "evidence_id",
                "content_sha256",
                "scope_sha256",
                "retrieval_receipt_sha256",
                "epistemic_state_sha256",
                "memory_tier",
            }
            if entry.get("context_role") == "memory_observation":
                digests = (
                    entry.get("content_sha256"),
                    entry.get("scope_sha256"),
                    entry.get("retrieval_receipt_sha256"),
                    entry.get("epistemic_state_sha256"),
                )
                if (
                    set(entry) != {"source", "text", *memory_fields}
                    or entry.get("instruction_authority") is not False
                    or not isinstance(entry.get("evidence_id"), str)
                    or not entry["evidence_id"].startswith("memory-")
                    or not isinstance(entry.get("memory_tier"), str)
                    or not (
                        entry["source"] == "memory"
                        or entry["source"].startswith(f"memory.{entry['memory_tier']}.")
                    )
                    or any(
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)
                        for digest in digests
                    )
                    or hashlib.sha256(entry["text"].strip().encode("utf-8")).hexdigest()
                    != entry["content_sha256"]
                ):
                    return {
                        "status": "error",
                        "message": "latent_reason memory context authority is invalid",
                    }
            elif set(entry) != {"source", "text"}:
                return {
                    "status": "error",
                    "message": "latent_reason non-memory context carries reserved fields",
                }
    operation_authority = job.get("operation_authority")
    if operation_authority is not None:
        try:
            from core.brain.llm.latent_cortex.epistemic_runtime import (
                validate_runtime_operation_authority,
            )

            operation_authority = validate_runtime_operation_authority(
                operation_authority,
                prompt=prompt if isinstance(prompt, str) else None,
                messages=messages if isinstance(messages, list) else None,
                config=dict(job.get("config") or {}),
                budget=dict(job.get("budget") or {}),
                cognitive_context=cognitive_context,
            )
        except (ImportError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason operation authority rejected: {exc}",
            }
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
    # Verifier guidance: when the caller asks for it, candidate branches and
    # latent-opt proposals are scored by deterministic task-typed checks
    # (arithmetic recomputation, code syntax, facet coverage, grounding) —
    # the winner is picked because its answer CHECKS OUT, not because its
    # trajectory converged prettier. Tokenizer required: verification reads
    # decoded probe text.
    task_verifier = None
    verifier_requested = bool(job.get("verifier_guidance")) or response_contract is not None
    if verifier_requested and tokenizer is None:
        # A REQUESTED verifier that cannot be built must not vanish. Without
        # a tokenizer the guidance was skipped silently, so the episode ran
        # with no task verifier while the caller — which had asked for one,
        # or supplied a response contract that implies one — received a
        # receipt that simply omitted it. Absence of guidance then read as
        # "no guidance was wanted" rather than "guidance was lost".
        from core.runtime.errors import record_degradation

        record_degradation(
            "latent_cortex_worker_handler",
            RuntimeError("verifier_guidance_requested_without_tokenizer"),
            severity="error",
            action="ran latent episode without the requested task verifier because no tokenizer was available",
        )
        logger.error(
            "Latent episode requested verifier guidance but no tokenizer is "
            "available; the episode runs UNVERIFIED and the receipt records it."
        )
    if verifier_requested and tokenizer is not None:
        from core.brain.llm.latent_cortex.task_verifiers import (
            _ANSWER_FACET_HINTS,
            EpisodeTaskVerifier,
        )

        facet_reliability = job.get("facet_reliability")
        if facet_reliability is not None:
            if (
                not isinstance(facet_reliability, dict)
                or len(facet_reliability) > 8
                or any(
                    not isinstance(name, str)
                    or name not in _ANSWER_FACET_HINTS
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                    for name, value in facet_reliability.items()
                )
            ):
                return {
                    "status": "error",
                    "message": (
                        "latent_reason facet_reliability must map known facet "
                        "names to floats in [0, 1]"
                    ),
                }
        objective = prompt if isinstance(prompt, str) else ""
        if not objective and isinstance(episode_messages, list):
            for message in reversed(episode_messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        objective = content
                        break
        task_verifier = EpisodeTaskVerifier(
            objective,
            facet_reliability=facet_reliability,
            response_contract=str(response_contract or ""),
        )
    result = engine.reason(
        prompt=prompt if isinstance(prompt, str) else None,
        messages=episode_messages,
        budget=budget,
        domain=str(job.get("domain", "general")),
        verifier=task_verifier,
        cognitive_context=cognitive_context,
        cancel_check=cancel_check,
        progress=progress,
    )
    if task_verifier is not None:
        result.receipt.verifier_guidance = task_verifier.to_receipt()
    elif verifier_requested:
        # Legible in the receipt: downstream must be able to tell "no verifier
        # was wanted" from "a verifier was wanted and could not be built".
        result.receipt.verifier_guidance = {
            "requested": True,
            "available": False,
            "reason": "tokenizer_unavailable",
        }
    if worker_identity is None:
        from core.brain.llm.latent_cortex.runtime_identity import build_worker_identity

        worker_identity = build_worker_identity(
            model,
            model_path=model_path,
            worker_boot_id=uuid.uuid4().hex,
            worker_source_path=Path(__file__).resolve().parents[1] / "mlx_worker.py",
        )
    receipt = result.receipt
    receipt.runtime_operation_authority = dict(operation_authority or {})
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
        cognitive_context=cognitive_context,
        operation_authority=operation_authority,
        response_contract=response_contract,
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
    # MISSING PROOF IS NOT PROOF OF SAFETY. This tested `is False` only, so a
    # parameter check that FAILED to run — or was skipped entirely — left
    # params_unchanged as None and the worker kept serving with weights whose
    # integrity had never been established. The absent case is exactly when a
    # recycle matters most: it is the case where nothing can vouch for the
    # resident parameters. Only an explicit True (the check ran and the
    # parameters were unchanged) avoids the recycle.
    body["requires_worker_recycle"] = (
        result.receipt.params_unchanged is not True
        or (
            result.receipt.fast_weights_applied
            and result.receipt.fast_weights_erased is not True
        )
    )
    if result.receipt.params_unchanged is None:
        logger.warning(
            "Latent episode returned no parameter-integrity proof; recycling the "
            "worker rather than trusting unverified resident weights."
        )
    return body


__all__ = [
    "budget_from_job",
    "config_from_job",
    "cortex_enabled",
    "handle_latent_reason",
]
