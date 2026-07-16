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

import asyncio
import logging
import math
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
        self._failure_streak = 0
        self._last_attempt_at = 0.0
        self._last_success_at = 0.0
        self._last_latency_s = 0.0
        self._last_allocation: dict[str, Any] = {}
        logger.info("🧠 LatentCortexService initialized (Recursive Latent Cortex)")

    # ── Cognitive economy ───────────────────────────────────────────────
    @staticmethod
    def _unit_signal(value: Any, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return min(1.0, max(0.0, number))

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
        stakes = self._unit_signal(stakes, name="stakes")
        uncertainty = self._unit_signal(uncertainty, name="uncertainty")
        try:
            pressure = self._unit_signal(self._body_pressure(), name="body_pressure")
        except ValueError:
            pressure = 0.0
        headroom = 1.0 - 0.7 * pressure

        max_steps = max(2, min(16, round((4 + 10 * uncertainty) * headroom)))
        n_branches = 1 if stakes < 0.3 else (3 if stakes > 0.75 and headroom > 0.6 else 2)
        intensity = max(stakes, uncertainty)
        latent_opt_steps = max(1, min(4, round((1 + 3 * intensity) * headroom)))
        fast_weight_steps = max(1, min(3, round((1 + 2 * intensity) * headroom)))
        fast_weight_layers = max(2, min(8, round((2 + 6 * intensity) * headroom)))
        config = {
            "n_slots": 16,
            "max_steps": max_steps,
            "min_steps": 2,
            "n_branches": n_branches,
            "alpha_schedule": "cosine",
            # The production service exercises the complete machine. Ablation
            # arms belong to the falsification harness, never the live default.
            "latent_opt": True,
            "latent_opt_steps": latent_opt_steps,
            "latent_opt_lr": 0.03,
            "fast_weights": True,
            "fast_weights_rank": 2,
            "fast_weights_opt_steps": fast_weight_steps,
            "fast_weights_lr": 0.005,
            "fast_weights_max_layers": fast_weight_layers,
            "decode_max_tokens": 512,
        }
        budget = {
            "max_layer_apps": int((2_000_000 + 8_000_000 * stakes) * headroom),
            "wall_clock_s": float(30.0 + 90.0 * stakes * headroom),
        }
        self._last_allocation = {
            "stakes": stakes,
            "uncertainty": uncertainty,
            "body_pressure": pressure,
            "headroom": headroom,
            "config": dict(config),
            "budget": dict(budget),
        }
        return config, budget

    @staticmethod
    def _receipt_contract_errors(
        receipt: Any,
        config: dict[str, Any],
        runtime_controls: dict[str, Any] | None = None,
    ) -> list[str]:
        if not isinstance(receipt, dict):
            return ["receipt_not_mapping"]
        errors: list[str] = []

        def positive_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] > 0

        def nonnegative_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] >= 0

        def sha256(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        def git_oid(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) in {40, 64}
                and all(character in "0123456789abcdef" for character in value)
            )

        if not str(receipt.get("episode_id") or ""):
            errors.append("missing_episode_id")
        if receipt.get("params_unchanged") is not True:
            errors.append("checkpoint_invariant_unproven")
        if (
            not sha256(receipt.get("checkpoint_fingerprint"))
            or receipt.get("checkpoint_fingerprint_method") != "sha256"
            or not positive_int(receipt, "checkpoint_file_count")
        ):
            errors.append("exact_checkpoint_identity_unproven")
        from core.brain.llm.latent_cortex.runtime_identity import worker_identity_errors

        errors.extend(worker_identity_errors(receipt))
        controls = dict(runtime_controls or {})
        if controls:
            expected_alpha = controls.get("clean_user_surface_steering_alpha")
            expected_loops = controls.get("clean_user_surface_recurrent_loops")
            if receipt.get("worker_affective_steering_active") is not True:
                errors.append("affective_steering_inactive")
            if receipt.get("episode_affective_steering_applied") is not True:
                errors.append("episode_affective_steering_unapplied")
            if (
                isinstance(expected_alpha, bool)
                or not isinstance(expected_alpha, (int, float))
                or isinstance(receipt.get("episode_affective_steering_alpha"), bool)
                or not isinstance(
                    receipt.get("episode_affective_steering_alpha"), (int, float)
                )
                or not math.isclose(
                    float(receipt["episode_affective_steering_alpha"]),
                    float(expected_alpha),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                errors.append("affective_steering_alpha_mismatch")
            if (
                type(expected_loops) is not int
                or expected_loops <= 0
                or not positive_int(receipt, "steps_taken")
                or receipt["steps_taken"] < expected_loops
            ):
                errors.append("live_recurrence_depth_unproven")
        if not sha256(receipt.get("request_payload_sha256")):
            errors.append("request_payload_identity_unproven")
        if not sha256(receipt.get("input_tokens_sha256")) or not positive_int(
            receipt, "input_token_count"
        ):
            errors.append("tokenized_input_identity_unproven")
        runtime_identity = receipt.get("runtime_identity")
        if not isinstance(runtime_identity, dict):
            errors.append("runtime_identity_missing")
        else:
            if runtime_identity.get("identity_bound") is not True:
                errors.append("runtime_identity_unbound")
            if not git_oid(runtime_identity.get("source_commit")):
                errors.append("runtime_source_commit_unproven")
            if not sha256(runtime_identity.get("workspace_state_sha256")):
                errors.append("runtime_workspace_identity_unproven")
            if not sha256(runtime_identity.get("shell_assets_sha256")):
                errors.append("runtime_shell_identity_unproven")
            if (
                runtime_identity.get("installed_app_required") is True
                and runtime_identity.get("installed_app_verified") is not True
            ):
                errors.append("installed_app_identity_unproven")
        if not sha256(receipt.get("schedule_hash")):
            errors.append("invalid_schedule_hash")
        if not positive_int(receipt, "steps_taken"):
            errors.append("no_recurrent_steps")
        if type(config.get("n_slots")) is not int or receipt.get("n_slots") != config.get(
            "n_slots"
        ):
            errors.append("workspace_cardinality_mismatch")
        if type(config.get("n_branches")) is not int or receipt.get(
            "n_branches"
        ) != config.get("n_branches"):
            errors.append("branch_cardinality_mismatch")
        budget = receipt.get("budget")
        if not isinstance(budget, dict) or not positive_int(budget, "spent_layer_apps"):
            errors.append("missing_compute_receipt")
        elif (
            not positive_int(budget, "max_layer_apps")
            or budget["spent_layer_apps"] > budget["max_layer_apps"]
            or budget.get("exhausted") is not False
        ):
            errors.append("incomplete_or_exhausted_compute_receipt")
        if not positive_int(receipt, "decode_requested_tokens") or receipt.get(
            "decode_requested_tokens"
        ) != config.get("decode_max_tokens"):
            errors.append("decode_request_mismatch")
        if not positive_int(receipt, "decode_generated_tokens"):
            errors.append("decode_output_empty")
        if receipt.get("decode_termination") not in {"eos", "token_limit"}:
            errors.append("decode_incomplete")
        configured_temperature = config.get("decode_temperature", 0.0)
        configured_top_p = config.get("decode_top_p", 1.0)
        if (
            isinstance(configured_temperature, bool)
            or not isinstance(configured_temperature, (int, float))
            or isinstance(receipt.get("decode_temperature"), bool)
            or not isinstance(receipt.get("decode_temperature"), (int, float))
            or not math.isclose(
                float(receipt["decode_temperature"]),
                float(configured_temperature),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append("decode_temperature_mismatch")
        if (
            isinstance(configured_top_p, bool)
            or not isinstance(configured_top_p, (int, float))
            or isinstance(receipt.get("decode_top_p"), bool)
            or not isinstance(receipt.get("decode_top_p"), (int, float))
            or not math.isclose(
                float(receipt["decode_top_p"]),
                float(configured_top_p),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append("decode_top_p_mismatch")
        raw_flags = receipt.get("honest_flags")
        if not isinstance(raw_flags, list) or any(
            not isinstance(flag, str) for flag in raw_flags
        ):
            errors.append("invalid_honest_flags")
            flags: list[str] = []
        else:
            flags = raw_flags
        if any(flag.startswith("fallback_vanilla") for flag in flags):
            errors.append("vanilla_fallback")
        if config.get("latent_opt") is True:
            if receipt.get("latent_opt_applied") is not True:
                errors.append("latent_optimization_not_applied")
            if receipt.get("latent_opt_mode") != "gradient":
                errors.append("latent_optimization_wrong_mode")
            if not positive_int(receipt, "latent_opt_attempts"):
                errors.append("latent_optimization_not_attempted")
            if not positive_int(receipt, "latent_opt_steps"):
                errors.append("latent_optimization_no_accepted_steps")
            if not nonnegative_int(receipt, "latent_opt_rejected"):
                errors.append("latent_optimization_rejection_count_invalid")
            elif (
                positive_int(receipt, "latent_opt_attempts")
                and positive_int(receipt, "latent_opt_steps")
                and (
                    receipt["latent_opt_attempts"]
                    != receipt["latent_opt_steps"] + receipt["latent_opt_rejected"]
                )
            ):
                errors.append("latent_optimization_accounting_mismatch")
            if receipt.get("latent_opt_budget_exhausted") is not False:
                errors.append("latent_optimization_budget_exhausted")
        if config.get("fast_weights") is True:
            if receipt.get("fast_weights_applied") is not True:
                errors.append("fast_weights_not_applied")
            if receipt.get("fast_weights_erased") is not True:
                errors.append("fast_weight_erase_unproven")
            if not positive_int(receipt, "fast_weights_layers"):
                errors.append("fast_weights_no_layers")
            if not positive_int(receipt, "fast_weight_optimization_attempts"):
                errors.append("fast_weight_optimization_not_attempted")
            if not positive_int(receipt, "fast_weight_optimized_steps"):
                errors.append("fast_weight_optimization_no_accepted_steps")
            if not nonnegative_int(receipt, "fast_weight_rejected_steps"):
                errors.append("fast_weight_rejection_count_invalid")
            elif (
                positive_int(receipt, "fast_weight_optimization_attempts")
                and positive_int(receipt, "fast_weight_optimized_steps")
                and (
                    receipt["fast_weight_optimization_attempts"]
                    != receipt["fast_weight_optimized_steps"]
                    + receipt["fast_weight_rejected_steps"]
                )
            ):
                errors.append("fast_weight_optimization_accounting_mismatch")
            if receipt.get("fast_weight_budget_exhausted") is not False:
                errors.append("fast_weight_optimization_budget_exhausted")
        return errors

    @staticmethod
    def select_foreground_episode(
        *,
        foreground: bool,
        desktop_required: bool,
        cognitive_mode: str,
        prompt_shape: dict[str, Any] | None,
        compact_contract: bool,
        strict_output_contract: bool,
        incompatible_contract: bool,
        proof_or_benchmark: bool,
        explicitly_required: bool = False,
    ) -> dict[str, Any]:
        """Return a deterministic, auditable decision for live latent routing."""

        shape = dict(prompt_shape or {})
        question_parts = shape.get("question_parts", 0)
        question_parts = question_parts if type(question_parts) is int else 0
        extended = bool(shape.get("prefers_extended_answer"))
        single_reply_coverage = bool(shape.get("requires_single_reply_coverage"))
        mode = str(cognitive_mode or "").strip().lower()

        exclusion = ""
        if not foreground:
            exclusion = "not_foreground"
        elif not desktop_required:
            exclusion = "desktop_cognitive_engine_not_required"
        elif compact_contract:
            exclusion = "compact_contract"
        elif strict_output_contract:
            exclusion = "strict_output_contract"
        elif incompatible_contract:
            exclusion = "incompatible_contract"
        elif proof_or_benchmark and not explicitly_required:
            exclusion = "proof_lane_not_explicitly_opted_in"
        depth_worthy = bool(
            explicitly_required
            or mode == "deliberate"
            or extended
            or single_reply_coverage
            or question_parts > 1
        )
        selected = bool(not exclusion and depth_worthy)
        reason = (
            "explicit_requirement"
            if selected and explicitly_required
            else "deliberate_cognitive_mode"
            if selected and mode == "deliberate"
            else "multipart_or_extended_prompt"
            if selected
            else exclusion or "depth_threshold_not_met"
        )
        depth_signal = min(1.0, 0.55 + 0.10 * min(3, question_parts))
        if extended or single_reply_coverage:
            depth_signal = max(depth_signal, 0.75)
        if mode == "deliberate":
            depth_signal = max(depth_signal, 0.80)
        if explicitly_required:
            depth_signal = max(depth_signal, 0.90)
        return {
            "latent_cortex_selected": selected,
            "latent_cortex_selection_reason": reason,
            "latent_cortex_depth_worthy": depth_worthy,
            "stakes": round(max(0.55, depth_signal - 0.05), 3),
            "uncertainty": round(depth_signal, 3),
        }

    def _record_failure(self, reason: str) -> dict[str, Any]:
        self._failure_streak += 1
        self._last_refusal = str(reason or "unknown")
        return {"ok": False, "reason": self._last_refusal}

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
        runtime_controls: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
        require_full_stack: bool = True,
        foreground_request: bool = True,
    ) -> dict[str, Any]:
        """Run one latent-reasoning episode on the resident model."""
        if not _cortex_enabled():
            return self._record_failure("disabled:AURA_LATENT_CORTEX=0")
        if question is not None and not isinstance(question, str):
            return self._record_failure("invalid_question")
        if messages is not None and not isinstance(messages, list):
            return self._record_failure("invalid_messages")
        if not (isinstance(question, str) and question.strip()) and not messages:
            return self._record_failure("empty_question")
        if config_overrides is not None and not isinstance(config_overrides, dict):
            return self._record_failure("invalid_config_overrides")
        if runtime_controls is not None and not isinstance(runtime_controls, dict):
            return self._record_failure("invalid_runtime_controls")
        if runtime_controls is not None:
            expected_control_keys = {
                "clean_user_surface_recurrent_loops",
                "clean_user_surface_steering_alpha",
            }
            recurrent_loops = runtime_controls.get(
                "clean_user_surface_recurrent_loops"
            )
            steering_alpha = runtime_controls.get(
                "clean_user_surface_steering_alpha"
            )
            if (
                set(runtime_controls) != expected_control_keys
                or type(recurrent_loops) is not int
                or not 1 <= recurrent_loops <= 2
                or isinstance(steering_alpha, bool)
                or not isinstance(steering_alpha, (int, float))
                or not math.isfinite(float(steering_alpha))
                or not 0.01 <= float(steering_alpha) <= 1.0
            ):
                return self._record_failure("invalid_runtime_controls")
        if type(require_full_stack) is not bool:
            return self._record_failure("invalid_require_full_stack")
        if type(foreground_request) is not bool:
            return self._record_failure("invalid_foreground_request")
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_timeout")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            return self._record_failure("invalid_timeout")
        try:
            config, budget = self.allocate(stakes=stakes, uncertainty=uncertainty)
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_cognitive_economy")
        if config_overrides is not None:
            config.update(dict(config_overrides))
        if require_full_stack:
            config["latent_opt"] = True
            config["latent_opt_control"] = False
            config["fast_weights"] = True

        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode: resident model client unavailable",
            )
            return self._record_failure(f"client_unavailable:{type(exc).__name__}")
        if client is None:
            return self._record_failure("no_resident_model")

        self._episodes += 1
        self._last_attempt_at = time.time()
        started = time.monotonic()
        try:
            result = await client.latent_reason_async(
                prompt=question,
                messages=messages,
                config=config,
                budget=budget,
                domain=domain,
                runtime_controls=runtime_controls,
                timeout_s=timeout_s,
                foreground_request=foreground_request,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="contained resident latent episode failure and preserved caller fallback",
                severity="warning",
            )
            self._last_latency_s = time.monotonic() - started
            return self._record_failure(f"client_error:{type(exc).__name__}")
        elapsed = time.monotonic() - started
        self._last_latency_s = elapsed
        if not isinstance(result, dict):
            return self._record_failure("invalid_client_response")
        if result.get("ok"):
            raw_receipt = result.get("receipt")
            receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else {}
            contract_errors = self._receipt_contract_errors(
                raw_receipt,
                config,
                runtime_controls,
            )
            if contract_errors:
                reason = "receipt_contract_failed:" + ",".join(contract_errors)
                record_degradation(
                    "latent_cortex",
                    RuntimeError(reason),
                    action="refused to count incomplete latent episode as successful",
                    severity="degraded",
                )
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = receipt
                return failed
            self._ok_episodes += 1
            self._failure_streak = 0
            self._last_refusal = ""
            self._last_success_at = time.time()
            self._last_receipt = receipt
            logger.info(
                "🧠 Latent episode ok: %d steps, %d branches, halt=%s, %.1fs",
                int(self._last_receipt.get("steps_taken") or 0),
                int(self._last_receipt.get("n_branches") or 0),
                self._last_receipt.get("halting_reason"),
                elapsed,
            )
        else:
            self._record_failure(str(result.get("reason") or "unknown"))
            logger.info("🧠 Latent episode refused/failed: %s (%.1fs)", self._last_refusal, elapsed)
        return result

    # ── Health ──────────────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        enabled = _cortex_enabled()
        state = (
            "disabled"
            if not enabled
            else "degraded"
            if self._failure_streak >= 3
            else "operational"
            if self._last_success_at > 0.0
            else "idle_unproven"
        )
        return {
            "enabled": enabled,
            "state": state,
            "episodes": self._episodes,
            "ok_episodes": self._ok_episodes,
            "failure_streak": self._failure_streak,
            "last_refusal": self._last_refusal,
            "last_attempt_at": self._last_attempt_at,
            "last_success_at": self._last_success_at,
            "last_latency_s": round(self._last_latency_s, 3),
            "last_allocation": dict(self._last_allocation),
            "last_receipt": {
                k: self._last_receipt.get(k)
                for k in (
                    "episode_id",
                    "steps_taken",
                    "halting_reason",
                    "n_branches",
                    "schedule_hash",
                    "checkpoint_fingerprint",
                    "checkpoint_fingerprint_method",
                    "checkpoint_file_count",
                    "worker_boot_id",
                    "worker_pid",
                    "worker_model_path",
                    "worker_model_parameter_count",
                    "worker_source_sha256",
                    "worker_affective_steering_active",
                    "worker_affective_steering_alpha",
                    "episode_affective_steering_applied",
                    "episode_affective_steering_alpha",
                    "request_payload_sha256",
                    "input_tokens_sha256",
                    "input_token_count",
                    "runtime_identity",
                    "params_unchanged",
                    "latent_opt_applied",
                    "latent_opt_attempts",
                    "latent_opt_steps",
                    "latent_opt_budget_exhausted",
                    "fast_weights_applied",
                    "fast_weight_optimization_attempts",
                    "fast_weight_optimized_steps",
                    "fast_weight_budget_exhausted",
                    "fast_weights_erased",
                    "honest_flags",
                )
                if k in self._last_receipt
            },
            "healthy": state == "operational",
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
