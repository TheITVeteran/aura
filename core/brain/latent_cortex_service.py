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
error) returns ``{"ok": False, "reason": ...}`` with bounded evidence so the
caller can decide whether no model work ran or the single model owner was
already exhausted. Nothing here fakes an answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.output_quality import evaluate_latent_output
from core.runtime.errors import record_degradation
from core.runtime.structured_input import analyze_prompt_shape

logger = logging.getLogger("Aura.LatentCortexService")

#: Depth multipliers for the ontogeny effort choice, mirrored here so the
#: allocation path costs no import. Kept in step with
#: ``core.ontogeny.control_points.EFFORT_MULTIPLIER``, which is the contract.
_EFFORT_MULTIPLIER = {"lean": 0.75, "standard": 1.0, "deep": 1.3}

# Explicit flag vocabularies. Anything outside both is a configuration
# error, not a silent activation (CP126 d9a04e05).
_TRUTHY_FLAG_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSEY_FLAG_VALUES = frozenset({"0", "false", "no", "off", "disabled", ""})

_CONTROLLER_SURFACE_OVERRIDE_KEYS = {
    "decode_max_tokens",
    "decode_temperature",
    "decode_top_p",
}


def _controller_accepts_overrides(overrides: dict[str, Any] | None) -> bool:
    """Keep live answer-surface tuning from disabling cognitive control.

    Structural overrides still opt out because experiment and lesion callers
    must receive the exact recurrence/branch/schedule arm they requested.
    """

    return overrides is None or set(overrides) <= _CONTROLLER_SURFACE_OVERRIDE_KEYS


def _cortex_enabled() -> bool:
    # CP126 d9a04e05. This enabled the cortex for every value except the
    # exact string "0", so "false", "no", "disabled", "off" and any typo all
    # ACTIVATED resident latent execution — the opposite of what the operator
    # wrote. A flag that ignores what it was told is worse than no flag.
    raw = str(os.environ.get("AURA_LATENT_CORTEX", "1")).strip().lower()
    if raw in _TRUTHY_FLAG_VALUES:
        return True
    if raw in _FALSEY_FLAG_VALUES:
        return False
    # Neither: the operator meant something we do not understand. Default on
    # (this subsystem is on by default) but say so, rather than silently
    # reinterpreting the instruction.
    record_degradation(
        "latent_cortex_service",
        ValueError(f"unrecognised AURA_LATENT_CORTEX value {raw!r}"),
        severity="warning",
        action="kept the latent cortex at its default state after an unreadable flag",
    )
    return True


def _integrity_verdict(receipt: Any, claim: str) -> str:
    """ "proven" | "refuted" | "unproven" for one weight-integrity claim.

    WHERE STRICTNESS BELONGS. A verdict of "refuted" — evidence that
    contradicts the claim — is fatal everywhere, immediately: it means
    weights really did change, or an adaptation really is still resident.
    Requiring "proven" is a different matter, because the digests that
    would prove it are not yet produced by the episode path; demanding them
    here would fail every real episode and take the latent cortex down
    rather than make it honest.

    So the requirement is placed where an unbacked claim actually causes
    harm: frontier certification, which publishes capability claims, must
    see "proven". Routine episodes carry their verdict in the receipt, so
    nothing downstream can mistake an assertion for a measurement, and the
    remaining work is the PRODUCER (a cheap canary digest around the
    episode), not another gate.

    Reads the receipt's digest evidence rather than its boolean. A receipt
    that predates the proof schema, or whose proof is malformed, yields
    "unproven" — which callers must treat as unsafe, because the absence of
    a check is not a passed check.
    """
    if not isinstance(receipt, dict):
        return "unproven"
    verdicts = receipt.get("integrity_verdicts")
    if isinstance(verdicts, dict):
        entry = verdicts.get(claim)
        if isinstance(entry, dict):
            verdict = str(entry.get("verdict") or "")
            if verdict in {"proven", "refuted", "unproven"}:
                return verdict
    # Fall back to recomputing from the raw proof, so a receipt carrying
    # digests but no precomputed verdicts is still judged on its evidence.
    try:
        from core.brain.llm.latent_cortex.types import WeightIntegrityProof

        proof = WeightIntegrityProof.from_dict(receipt.get("weight_integrity"))
    except (ImportError, AttributeError, TypeError, ValueError):
        return "unproven"
    proven = (
        proof.params_unchanged_proven
        if claim == "params_unchanged"
        else proof.fast_weights_erased_proven
    )
    if proven is None:
        return "unproven"
    return "proven" if proven else "refuted"


def _erased_layers_declared(receipt: Any) -> bool:
    """Whether the teardown enumerated the layers it claims to have cleared."""
    if not isinstance(receipt, dict):
        return False
    proof = receipt.get("weight_integrity")
    if not isinstance(proof, dict):
        return False
    layers = proof.get("erased_layer_ids")
    return bool(isinstance(layers, (list, tuple)) and layers)


def _controller_outcome(
    verifier_evidence: Any,
) -> tuple[float, bool, bool, str]:
    """Extract only independently graded task outcomes for bandit learning."""

    if not isinstance(verifier_evidence, dict):
        return 0.0, False, False, "verifier_evidence_missing"
    raw_best = verifier_evidence.get("best_score")
    best_score = 0.0
    if (
        isinstance(raw_best, (int, float))
        and not isinstance(raw_best, bool)
        and math.isfinite(float(raw_best))
    ):
        best_score = max(0.0, min(1.0, float(raw_best)))
    checked = verifier_evidence.get("outcome_checked") is True
    passed = checked and verifier_evidence.get("outcome_passed") is True
    reason = (
        "independent_grade"
        if checked
        else str(verifier_evidence.get("outcome_reason") or "task_ground_truth_unavailable")
    )
    return best_score, checked, passed, reason


# What to assume when the body cannot be read. 0.0 means "maximum headroom"
# on this scale, so unknown must never map to it. High enough to damp heavy
# allocation, low enough that a body which never reports does not freeze the
# service outright.
_UNKNOWN_BODY_PRESSURE = 0.6


class LatentCortexService:
    """Budget allocation + IPC routing for latent-reasoning episodes."""

    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        self._episodes = 0
        self._ok_episodes = 0
        self._last_receipt: dict[str, Any] = {}
        self._last_failure_receipt: dict[str, Any] = {}
        self._last_progress: dict[str, Any] = {}
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
        """Total real+anticipatory body pressure in [0, 1].

        CP126 9bc8e55e. Unknown used to mean 0.0 — which in this scale is
        "fully healthy, maximum headroom". So an absent, incompatible,
        stale or failing body was rewarded with UNDAMPED compute, silently,
        with no degradation receipt. The signal disappeared in exactly the
        state where it was most needed.

        Unknown now means conservatively pressured: enough to damp heavy
        allocation, not so much that a body which never reports freezes the
        service. The failure is recorded, so a persistent unknown is visible
        rather than indistinguishable from a calm body.
        """
        try:
            from core.being.aura_now import BodyState

            state = getattr(self.orchestrator, "state", None)
            # total_pressure is a @property, not a method. Calling it raised
            # TypeError: 'float' object is not callable on EVERY invocation,
            # so this function has never once returned a real reading — it
            # always fell into the handler below, which used to answer 0.0
            # and hand the latent cortex undamped compute while reporting
            # nothing. Found in a live log only after that handler was made
            # to record itself.
            return float(BodyState.from_aura_state(state).total_pressure)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            if not getattr(self, "_body_pressure_unknown_reported", False):
                self._body_pressure_unknown_reported = True
                record_degradation(
                    "latent_cortex_service",
                    exc,
                    severity="warning",
                    action=(
                        "body pressure unobservable; damping allocation at "
                        f"{_UNKNOWN_BODY_PRESSURE} instead of assuming full headroom"
                    ),
                )
            return _UNKNOWN_BODY_PRESSURE

    def _runtime_pressure_snapshot(self) -> dict[str, Any]:
        """Read the canonical admission signal without creating new policy."""

        try:
            from core.runtime.control_plane import get_runtime_control_plane

            snapshot = get_runtime_control_plane().admission.pressure_snapshot()
            payload = snapshot.to_dict()
            if not isinstance(payload, dict):
                raise TypeError("runtime pressure snapshot is not a mapping")
            return payload
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if not getattr(self, "_runtime_pressure_unknown_reported", False):
                self._runtime_pressure_unknown_reported = True
                record_degradation(
                    "latent_cortex_service",
                    exc,
                    severity="warning",
                    action=(
                        "runtime pressure unavailable; adaptive compute uses "
                        "conservative unknown-resource headroom"
                    ),
                )
            return {
                "observation_source": "unavailable",
                "resource_observation_available": False,
                "red_zones": ["pressure_provider_unavailable"],
            }

    #: How much above-average novelty may add to effective uncertainty.
    #:
    #: Bounded on purpose. Novelty is a *measurement* — how far the current
    #: state sits from the centre of the distribution Aura has actually lived
    #: — not a model's prediction, so it needs no earned authority to be acted
    #: on. But it is one signal among several, and a situation being unfamiliar
    #: is a reason to think a little harder, never a reason to abandon the
    #: stakes and uncertainty the caller actually measured.
    _NOVELTY_EFFORT_WEIGHT = 0.2

    def _novelty_adjusted_uncertainty(self, uncertainty: float) -> tuple[float, float | None]:
        """Let an unfamiliar situation buy a little more thought.

        The allocator has always scaled depth with stakes and uncertainty, both
        of which describe the *task*. Neither notices that Aura has never been
        anywhere like this before, which is exactly when the surface of a
        problem is least informative about its difficulty.

        Only above-average novelty counts. An ordinary moment is left exactly
        as the caller measured it — a signal that moves every allocation is a
        signal that has stopped saying anything.
        """
        try:
            from core.ontogeny.service import get_ontogeny

            novelty = float(get_ontogeny().novelty())
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation(
                "latent_cortex_service", exc, severity="debug",
                action="novelty unavailable; allocation uses the caller's uncertainty alone",
            )
            return uncertainty, None
        excess = max(0.0, novelty - 0.5) * 2.0
        adjusted = min(1.0, uncertainty + self._NOVELTY_EFFORT_WEIGHT * excess)
        return adjusted, novelty

    def _effort_choice(
        self, *, stakes: float, uncertainty: float, novelty: float | None,
        foreground: bool, model_parameter_count: int,
    ) -> tuple[str, str | None]:
        """Ask the organ how hard to think. Returns (effort, episode_id).

        Unlike novelty — which is a measurement and needs no permission — this
        is a learned *choice*, so it goes through the full ladder and returns
        the incumbent's "standard" until a head has earned otherwise. It is
        also the control point that is hardest to grade honestly: whether a
        depth was right is only answerable once a verifier has graded the
        answer, which happens elsewhere and often not at all. The resolver
        refuses every proxy for that, so this may sit unpromoted indefinitely.
        That is the design working, not the design failing.
        """
        try:
            import math as _math

            from core.ontogeny.control_points import COGNITION_EFFORT
            from core.ontogeny.service import get_ontogeny

            angle = 2.0 * _math.pi * (time.localtime().tm_hour / 24.0)
            verdict = get_ontogeny().consider(
                COGNITION_EFFORT,
                {
                    "stakes": float(stakes),
                    "uncertainty": float(uncertainty),
                    "novelty": float(novelty if novelty is not None else 0.5),
                    "body_pressure": float(self._body_pressure()),
                    "foreground": 1.0 if foreground else 0.0,
                    "resident_scale": 1.0 if model_parameter_count >= 20_000_000_000 else 0.0,
                    "hour_of_day_sin": _math.sin(angle),
                    "hour_of_day_cos": _math.cos(angle),
                },
                incumbent_choice="standard",
                seed=f"effort:{stakes:.3f}:{uncertainty:.3f}:{time.time():.3f}",
                # High-stakes thinking is never explored with. The exploration
                # slice buys evidence with latency, not with the quality of an
                # answer somebody is waiting on.
                stakes=max(float(stakes), 0.75 if foreground else 0.0),
            )
            return verdict.choice, verdict.episode_id
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
            record_degradation(
                "latent_cortex_service", exc, severity="debug",
                action="effort left at the allocator's own depth",
            )
            return "standard", None

    def allocate(
        self,
        *,
        stakes: float,
        uncertainty: float,
        objective: str = "",
        model_parameter_count: int = 0,
        foreground_request: bool = False,
        timeout_s: float | None = None,
    ) -> tuple[dict, dict]:
        """(config, budget) for one episode: the Will's thought allocation.

        More stakes/uncertainty ⇒ deeper recurrence, wider branches, bigger
        budget. Body pressure damps everything — deep thought is a luxury a
        strained body rations first.
        """
        stakes = self._unit_signal(stakes, name="stakes")
        uncertainty = self._unit_signal(uncertainty, name="uncertainty")
        uncertainty, novelty = self._novelty_adjusted_uncertainty(uncertainty)
        effort, effort_episode = self._effort_choice(
            stakes=stakes, uncertainty=uncertainty, novelty=novelty,
            foreground=foreground_request, model_parameter_count=model_parameter_count,
        )
        try:
            pressure = self._unit_signal(self._body_pressure(), name="body_pressure")
        except ValueError:
            # Same reasoning: an invalid reading is not evidence of headroom.
            pressure = _UNKNOWN_BODY_PRESSURE
        if (
            isinstance(model_parameter_count, bool)
            or not isinstance(model_parameter_count, int)
            or model_parameter_count < 0
        ):
            raise ValueError("model_parameter_count must be a non-negative integer")
        if type(foreground_request) is not bool:
            raise ValueError("foreground_request must be a boolean")
        if not isinstance(objective, str):
            raise ValueError("objective must be text")
        owner_timeout_s: float | None = None
        if timeout_s is not None:
            try:
                owner_timeout_s = float(timeout_s)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("timeout_s must be finite and positive") from exc
            if not math.isfinite(owner_timeout_s) or owner_timeout_s <= 0.0:
                raise ValueError("timeout_s must be finite and positive")
        headroom = 1.0 - 0.7 * pressure

        # The organ's effort choice scales depth inside the same hard bounds
        # the allocator has always enforced. It may tune how hard she thinks;
        # it may not remove the floor or raise the ceiling.
        effort_scale = _EFFORT_MULTIPLIER.get(effort, 1.0)
        max_steps = max(2, min(16, round((4 + 10 * uncertainty) * headroom * effort_scale)))
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
            "isolation_steps": 2,
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
            "fast_weights_canary_max_delta_rms": 0.05,
            "decode_max_tokens": 512,
            "verifier_probe_max_tokens": 48,
            "verifier_accept_non_regression": False,
            "generative_verifier_enabled": True,
            "generative_verifier_max_atoms": 1,
            "generative_verifier_max_tokens": 160,
            "counterfactual_verifier_enabled": True,
            "counterfactual_verifier_max_atoms": 1,
            "counterfactual_verifier_max_interventions": 2,
            "counterfactual_verifier_max_tokens": 128,
            "prefix_stability_enabled": True,
            "prefix_stability_samples": 3,
            "prefix_stability_max_tokens": 128,
            "prefix_stability_temperature": 0.35,
            "prefix_stability_top_p": 0.9,
            "prefix_stability_seed": 104_729,
            "prefix_stability_calibrator": None,
        }
        budget = {
            "max_layer_apps": int((2_000_000 + 8_000_000 * stakes) * headroom),
            "wall_clock_s": float(30.0 + 90.0 * stakes * headroom),
            # Recorded so an allocation can be explained after the fact: a
            # deeper-than-usual episode should be traceable to the reason it
            # was deeper, not just observed to have been.
            "effective_uncertainty": round(uncertainty, 4),
            "novelty": round(novelty, 4) if novelty is not None else None,
            "effort": effort,
            "ontogeny_episode": effort_episode,
        }
        allocation_profile = "general_full_stack_v1"
        if foreground_request and model_parameter_count >= 20_000_000_000:
            # Interactive resident-scale profile: every production mechanism
            # remains causal, but the 32B lane receives a bounded amount of
            # virtual width and optimizer work instead of a small-model lab
            # schedule that cannot meet the desktop deadline.
            allocation_profile = "resident_32b_interactive_full_stack_v2"
            config.update(
                {
                    # Nine positions preserve one mailbox, six organ evidence
                    # rows, an optional one-shot continuation fragment, and a
                    # persistent private hypothesis. The prior four-slot
                    # profile silently dropped most admitted evidence.
                    "n_slots": 9,
                    "max_steps": 2,
                    "min_steps": 2,
                    "n_branches": 2 if stakes >= 0.3 else 1,
                    "exchange_interval": 1,
                    "latent_opt_steps": 1,
                    "fast_weights_opt_steps": 1,
                    "fast_weights_max_layers": 2,
                    # Mechanically-clean episode synapses become durable
                    # learning CANDIDATES (consumer + compounding gates
                    # decide; nothing consolidates from inside an episode).
                    "fast_weights_export_candidates": True,
                    "decode_max_tokens": 256,
                    "decode_bridge_policy": "assistant_answer_v1",
                    # Product probes are previews used for branch/adaptation
                    # arbitration, not user-facing drafts. CP120 measured five
                    # 48-token probes consuming ~68s; 24 tokens plus verified
                    # branch-baseline reuse preserves the answer budget while
                    # keeping every arbitration mechanism causal and receipted.
                    "verifier_probe_max_tokens": 24,
                    "verifier_accept_non_regression": True,
                    "generative_verifier_enabled": True,
                    "generative_verifier_max_atoms": 1,
                    # The strict JSON contract includes a 64-character claim
                    # commitment; shorter budgets truncate before the witness.
                    "generative_verifier_max_tokens": 128,
                    "counterfactual_verifier_enabled": True,
                    "counterfactual_verifier_max_atoms": 1,
                    "counterfactual_verifier_max_interventions": 2,
                    "counterfactual_verifier_max_tokens": 96,
                    "prefix_stability_enabled": True,
                    "prefix_stability_samples": 3,
                    "prefix_stability_max_tokens": 128,
                    "prefix_stability_temperature": 0.35,
                    "prefix_stability_top_p": 0.9,
                    "prefix_stability_seed": 104_729,
                    "prefix_stability_calibrator": None,
                    "input_context_max_chars": 9000,
                    "allow_vanilla_fallback": False,
                }
            )
            if owner_timeout_s is not None:
                budget["wall_clock_s"] = min(
                    max(105.0, budget["wall_clock_s"]),
                    max(15.0, owner_timeout_s - 8.0),
                )
        from core.brain.llm.latent_cortex.adaptive_compute import (
            apply_adaptive_compute_plan,
            build_adaptive_compute_plan,
        )

        adaptive_plan = build_adaptive_compute_plan(
            objective=objective,
            stakes=stakes,
            uncertainty=uncertainty,
            body_pressure=pressure,
            deadline_s=(
                owner_timeout_s
                if owner_timeout_s is not None
                else float(budget["wall_clock_s"])
            ),
            resource_snapshot=self._runtime_pressure_snapshot(),
            foreground_request=foreground_request,
            model_parameter_count=model_parameter_count,
            requested_decode_tokens=int(config["decode_max_tokens"]),
        )
        config, budget = apply_adaptive_compute_plan(config, budget, adaptive_plan)
        self._last_allocation = {
            "stakes": stakes,
            "uncertainty": uncertainty,
            "body_pressure": pressure,
            "headroom": headroom,
            "allocation_profile": allocation_profile,
            "model_parameter_count": model_parameter_count,
            "adaptive_compute": adaptive_plan,
            "config": dict(config),
            "budget": dict(budget),
        }
        return config, budget

    @staticmethod
    def _receipt_contract_errors(
        receipt: Any,
        config: dict[str, Any],
        runtime_controls: dict[str, Any] | None = None,
        expected_worker_identity: dict[str, Any] | None = None,
        output_tokens: Any = ...,
        expected_domain: str = "general",
        output_text: Any = ...,
        answer_replacement_private: Any = None,
        expected_objective: str = "",
    ) -> list[str]:
        if not isinstance(receipt, dict):
            return ["receipt_not_mapping"]
        errors: list[str] = []

        def positive_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] > 0

        def nonnegative_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] >= 0

        def finite_number_list(value: Any) -> bool:
            return isinstance(value, list) and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in value
            )

        def finite_number(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )

        def verifier_arbitration_valid(
            arbitration: Any,
            *,
            attempts: int,
            accepted_steps: int,
        ) -> bool:
            """Independently replay one non-regression arbitration receipt."""
            if not isinstance(arbitration, dict):
                return False
            score_accepts = arbitration.get("score_improvement_accepts")
            proxy_accepts = arbitration.get("proxy_nonregression_accepts")
            decisions = arbitration.get("decisions")
            score_trail = arbitration.get("score_trail")
            tolerance = arbitration.get("score_tolerance")
            proxy_scale = arbitration.get("proxy_tolerance_scale")
            if (
                arbitration.get("policy") != "task_score_nonregression_with_proxy_descent_v1"
                or arbitration.get("baseline_source")
                not in {"caller_reused_verified_branch", "decoded_state_probe"}
                or type(score_accepts) is not int
                or score_accepts < 0
                or type(proxy_accepts) is not int
                or proxy_accepts < 0
                or score_accepts + proxy_accepts != accepted_steps
                or not finite_number(tolerance)
                or not 0.0 <= float(tolerance) <= 1e-3
                or not finite_number(proxy_scale)
                or not 0.0 < float(proxy_scale) <= 1e-3
                or not isinstance(decisions, list)
                or len(decisions) != attempts
                or not finite_number_list(score_trail)
                or len(score_trail) != len(decisions) + 1
            ):
                return False

            score_tolerance = float(tolerance)
            proxy_tolerance_scale = float(proxy_scale)
            receipt_epsilon = 2e-12
            observed_score_accepts = 0
            observed_proxy_accepts = 0
            allowed_decisions = {
                "accepted_task_score_improvement",
                "accepted_task_score_nonregression_with_proxy_descent",
                "rejected_task_score_regression",
                "rejected_proxy_non_descent",
                "rejected_nonfinite_task_score",
                "rejected_nonfinite_proxy_loss",
            }
            for index, row in enumerate(decisions):
                if not isinstance(row, dict) or row.get("proposal") != index:
                    return False
                decision = row.get("decision")
                if decision not in allowed_decisions:
                    return False
                baseline = row.get("baseline_score")
                current_proxy = row.get("current_proxy_loss")
                required_delta = row.get("proxy_required_delta")
                if (
                    not finite_number(baseline)
                    or not finite_number(current_proxy)
                    or not finite_number(required_delta)
                    or float(required_delta) < 0.0
                    or not math.isclose(
                        float(baseline),
                        float(score_trail[index]),
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    )
                    or not math.isclose(
                        float(required_delta),
                        proxy_tolerance_scale * max(1.0, abs(float(current_proxy))),
                        rel_tol=1e-6,
                        abs_tol=receipt_epsilon,
                    )
                ):
                    return False
                baseline_score = float(baseline)
                next_score = float(score_trail[index + 1])
                raw_candidate = row.get("candidate_score")
                candidate_proxy = row.get("candidate_proxy_loss")
                if decision == "rejected_nonfinite_task_score":
                    if raw_candidate != "nonfinite" or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                    continue
                if not finite_number(raw_candidate):
                    return False
                candidate_score = float(raw_candidate)
                score_improved = (
                    candidate_score > baseline_score + score_tolerance + receipt_epsilon
                )
                score_nonregressing = (
                    candidate_score >= baseline_score - score_tolerance - receipt_epsilon
                )
                proxy_finite = finite_number(candidate_proxy)
                proxy_improved = bool(
                    proxy_finite
                    and float(candidate_proxy)
                    < float(current_proxy) - float(required_delta) - receipt_epsilon
                )
                if decision == "rejected_nonfinite_proxy_loss":
                    if proxy_finite or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                elif decision == "accepted_task_score_improvement":
                    if (
                        not proxy_finite
                        or not score_improved
                        or not math.isclose(
                            next_score,
                            candidate_score,
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
                    observed_score_accepts += 1
                elif decision == ("accepted_task_score_nonregression_with_proxy_descent"):
                    if (
                        score_improved
                        or not score_nonregressing
                        or not proxy_improved
                        or not math.isclose(
                            next_score,
                            max(baseline_score, candidate_score),
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
                    observed_proxy_accepts += 1
                elif decision == "rejected_task_score_regression":
                    if score_nonregressing or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                elif decision == "rejected_proxy_non_descent":
                    if (
                        not score_nonregressing
                        or score_improved
                        or not proxy_finite
                        or proxy_improved
                        or not math.isclose(
                            next_score,
                            baseline_score,
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
            return (
                observed_score_accepts == score_accepts and observed_proxy_accepts == proxy_accepts
            )

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

        resource_accounting: dict[str, Any] | None = None
        information_accounting: dict[str, Any] | None = None
        budget_receipt = receipt.get("budget")
        try:
            from core.brain.llm.latent_cortex.resource_accounting import (
                validate_information_receipt,
                validate_resource_receipt,
            )

            if not isinstance(budget_receipt, dict):
                raise ValueError("episode budget receipt is absent")
            resource_accounting = validate_resource_receipt(
                budget_receipt.get("resource_accounting")
            )
            information_accounting = validate_information_receipt(
                budget_receipt.get("information_accounting")
            )
            if resource_accounting["accounting_complete"] is not True:
                errors.append("resource_accounting_incomplete")
            if information_accounting["accounting_complete"] is not True:
                errors.append("information_accounting_incomplete")
            n_layers = receipt.get("n_layers")
            if (
                type(n_layers) is int
                and n_layers > 0
                and resource_accounting["model_profile"]["num_hidden_layers"] != n_layers
            ):
                errors.append("resource_model_profile_mismatch")
        except (ImportError, TypeError, ValueError):
            errors.append("resource_accounting_unproven")
            errors.append("information_accounting_unproven")

        if not str(receipt.get("episode_id") or ""):
            errors.append("missing_episode_id")
        # CP126 e93ffe9f. A bare params_unchanged=True was accepted as proof
        # that resident weights survived the episode. It is an assertion, and
        # this gate decides whether the lane may keep serving on those
        # weights — exactly where an unbacked claim must not pass. The
        # digests are now required, and a claim its own evidence REFUTES is
        # reported separately from one that was merely never measured.
        params_verdict = _integrity_verdict(receipt, "params_unchanged")
        if params_verdict == "refuted":
            # Evidence CONTRADICTS the claim: weights changed. Always fatal.
            errors.append("checkpoint_invariant_refuted")
        elif params_verdict == "unproven" and receipt.get("params_unchanged") is not True:
            # No evidence AND no claim.
            errors.append("checkpoint_invariant_unproven")
        if (
            not sha256(receipt.get("checkpoint_fingerprint"))
            or receipt.get("checkpoint_fingerprint_method") != "sha256"
            or not positive_int(receipt, "checkpoint_file_count")
        ):
            errors.append("exact_checkpoint_identity_unproven")
        from core.brain.llm.latent_cortex.runtime_identity import worker_identity_errors

        errors.extend(worker_identity_errors(receipt))
        guidance = receipt.get("verifier_guidance")
        if (
            isinstance(guidance, dict)
            and guidance.get("schema")
            in {"aura.latent_task_verifier.v3", "aura.latent_task_verifier.v4"}
            and type(guidance.get("evaluations")) is int
            and guidance["evaluations"] > 0
        ):
            try:
                from core.brain.llm.latent_cortex.atomic_decomposition import (
                    validate_atomic_decomposition_envelope,
                )

                atomic = validate_atomic_decomposition_envelope(
                    guidance.get("atomic_decomposition")
                )
                if (
                    atomic["grade_admissible"] is not True
                    or guidance.get("grade_admissible") is not True
                ):
                    raise ValueError("atomic decomposition denied grading authority")
                if guidance.get("schema") == "aura.latent_task_verifier.v4":
                    from core.brain.llm.latent_cortex.deterministic_verifier_router import (
                        validate_deterministic_router_envelope,
                    )

                    routed = validate_deterministic_router_envelope(
                        guidance.get("deterministic_router"),
                        atomic_receipt=atomic,
                    )
                    if routed["hard_pass"] is not True:
                        raise ValueError("deterministic verifier refuted candidate")
            except (ImportError, TypeError, ValueError):
                errors.append("atomic_decomposition_unproven")
        generative = receipt.get("generative_verifier")
        verified_generation: dict[str, Any] | None = None
        if config.get("generative_verifier_enabled") is True:
            if (
                isinstance(generative, dict)
                and generative.get("schema") == "aura.rlc.generative_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.generative_verifier import (
                        validate_generative_verifier_envelope,
                    )

                    verified_generation = validate_generative_verifier_envelope(generative)
                    effect = verified_generation["selection_effect"]
                    if effect == "winner_replaced" and receipt.get("selected_branch") != (
                        verified_generation["replacement_branch"]
                    ):
                        raise ValueError("generative verifier replacement was not selected")
                    if verified_generation["causal_refutation"] and effect == "none":
                        raise ValueError("generative refutation was not applied to selection")
                    if effect == "no_alternative":
                        raise ValueError("generative verifier refuted the only branch")
                except (ImportError, KeyError, TypeError, ValueError):
                    errors.append("generative_verifier_unproven")
                    verified_generation = None
            elif not (
                isinstance(generative, dict)
                and generative.get("requested") is True
                and generative.get("available") is False
                and generative.get("selection_effect") == "none"
                and isinstance(generative.get("reason"), str)
                and generative.get("reason")
            ):
                errors.append("generative_verifier_unreceipted")
        counterfactual = receipt.get("counterfactual_verifier")
        verified_counterfactual: dict[str, Any] | None = None
        if config.get("counterfactual_verifier_enabled") is True:
            if (
                isinstance(counterfactual, dict)
                and counterfactual.get("schema") == "aura.rlc.counterfactual_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.counterfactual_verifier import (
                        validate_counterfactual_verifier_envelope,
                    )

                    verified_counterfactual = (
                        validate_counterfactual_verifier_envelope(counterfactual)
                    )
                    blind_review = receipt.get("blind_review")
                    blind_rows = (
                        blind_review.get("rows")
                        if isinstance(blind_review, dict)
                        else None
                    )
                    if not isinstance(blind_rows, list):
                        raise ValueError("counterfactual verifier lacks blind score evidence")
                    blind_scores = {
                        int(row["branch"]): float(row["score"])
                        for row in blind_rows
                        if isinstance(row, dict)
                    }
                    expected_scores = {
                        str(branch): round(score, 6)
                        for branch, score in blind_scores.items()
                    }
                    if (
                        len(expected_scores) != len(blind_rows)
                        or verified_counterfactual["task_scores"] != expected_scores
                    ):
                        raise ValueError("counterfactual task scores differ from blind review")
                    source_selected = verified_counterfactual["source_selected_branch"]
                    if source_selected != max(
                        range(len(blind_scores)),
                        key=lambda branch: blind_scores[branch],
                    ):
                        raise ValueError("counterfactual source winner differs from task scores")
                    generated_effect = (
                        verified_generation.get("selection_effect")
                        if isinstance(verified_generation, dict)
                        else "none"
                    )
                    if generated_effect == "winner_replaced":
                        if (
                            verified_generation.get("vetoed_branch")
                            != verified_counterfactual["selected_branch"]
                        ):
                            raise ValueError(
                                "generative verifier did not follow counterfactual selection"
                            )
                    elif (
                        verified_counterfactual["selection_authority_admitted"] is True
                        and receipt.get("selected_branch")
                        != verified_counterfactual["selected_branch"]
                    ):
                        raise ValueError(
                            "counterfactual verifier selection was not applied"
                        )
                except (ImportError, KeyError, TypeError, ValueError):
                    errors.append("counterfactual_verifier_unproven")
                    verified_counterfactual = None
            elif not (
                isinstance(counterfactual, dict)
                and counterfactual.get("requested") is True
                and counterfactual.get("available") is False
                and counterfactual.get("selection_effect") == "none"
                and isinstance(counterfactual.get("reason"), str)
                and counterfactual.get("reason")
            ):
                errors.append("counterfactual_verifier_unreceipted")
        prefix_stability = receipt.get("prefix_stability")
        if config.get("prefix_stability_enabled") is True:
            if (
                isinstance(prefix_stability, dict)
                and prefix_stability.get("schema")
                == "aura.rlc.prefix_stability_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.prefix_stability import (
                        validate_prefix_stability_envelope,
                    )

                    validate_prefix_stability_envelope(
                        prefix_stability,
                        expected_calibrator_config=config.get(
                            "prefix_stability_calibrator"
                        ),
                    )
                except (ImportError, KeyError, OSError, TypeError, ValueError):
                    errors.append("prefix_stability_unproven")
            elif not (
                isinstance(prefix_stability, dict)
                and set(prefix_stability)
                == {
                    "requested",
                    "available",
                    "reason",
                    "selection_effect",
                    "correctness_effect",
                }
                and prefix_stability.get("requested") is True
                and prefix_stability.get("available") is False
                and prefix_stability.get("selection_effect") == "none"
                and prefix_stability.get("correctness_effect") == "none"
                and isinstance(prefix_stability.get("reason"), str)
                and prefix_stability.get("reason")
            ):
                errors.append("prefix_stability_unreceipted")
        if config.get("critic_blind_spot_evidence") is not None:
            try:
                from core.brain.llm.latent_cortex.critic_identity import (
                    validate_critic_identity,
                    validate_shared_blind_spot_evidence,
                )

                identity = validate_critic_identity(
                    receipt.get("critic_identity"),
                    worker_identity=(expected_worker_identity or receipt),
                )
                blind_spots = validate_shared_blind_spot_evidence(
                    receipt.get("shared_blind_spots"),
                    generator_function_sha256=identity["generator_identity"]["function_sha256"],
                    critic_function_sha256=identity["critic_function_sha256"],
                )
                if blind_spots != config.get("critic_blind_spot_evidence"):
                    raise ValueError("worker critic evidence differs from service snapshot")
                if blind_spots["critic_reliability_admitted"] is not True:
                    raise ValueError("critic reliability gate did not admit authority")
                if (
                    not isinstance(guidance, dict)
                    or guidance.get("requested") is not True
                    or guidance.get("available") is not True
                ):
                    raise ValueError("admitted critic was not causally used")
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                errors.append("disjoint_critic_authority_unproven")
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
                or not isinstance(receipt.get("episode_affective_steering_alpha"), (int, float))
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
        elif information_accounting is not None:
            rendered_inputs = [
                source
                for source in information_accounting["sources"]
                if source.get("source_id") == "rendered_model_input"
                and source.get("kind") == "model_input_tokens"
            ]
            if (
                len(rendered_inputs) != 1
                or rendered_inputs[0].get("content_sha256") != receipt.get("input_tokens_sha256")
                or rendered_inputs[0].get("token_count") != receipt.get("input_token_count")
            ):
                errors.append("input_information_binding_unproven")
        input_context_max_chars = config.get("input_context_max_chars", 0)
        if type(input_context_max_chars) is int and input_context_max_chars > 0:
            compaction = receipt.get("input_context_compaction")
            if not isinstance(compaction, dict):
                errors.append("input_context_compaction_missing")
            elif (
                compaction.get("schema") != "aura.latent_context_compaction.v1"
                or compaction.get("policy") != "resident_latent_salience_v1"
                or compaction.get("max_chars") != input_context_max_chars
                or not sha256(compaction.get("original_sha256"))
                or not sha256(compaction.get("compacted_sha256"))
                or not positive_int(compaction, "original_message_count")
                or not positive_int(compaction, "compacted_message_count")
                or not positive_int(compaction, "original_char_count")
                or not positive_int(compaction, "compacted_char_count")
                or compaction["compacted_char_count"] > input_context_max_chars
                or compaction["original_char_count"] < compaction["compacted_char_count"]
                or compaction["compacted_message_count"] > compaction["original_message_count"]
                or type(compaction.get("applied")) is not bool
                or type(compaction.get("omitted_char_count")) is not int
                or compaction["omitted_char_count"] < 0
                or compaction["omitted_char_count"]
                != compaction["original_char_count"] - compaction["compacted_char_count"]
                or (
                    compaction["applied"]
                    and (
                        compaction["original_sha256"] == compaction["compacted_sha256"]
                        or compaction["omitted_char_count"] == 0
                    )
                )
                or (
                    not compaction["applied"]
                    and (
                        compaction["original_sha256"] != compaction["compacted_sha256"]
                        or compaction["omitted_char_count"] != 0
                    )
                )
            ):
                errors.append("input_context_compaction_invalid")
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
        if type(config.get("n_branches")) is not int or receipt.get("n_branches") != config.get(
            "n_branches"
        ):
            errors.append("branch_cardinality_mismatch")
        try:
            from core.brain.llm.latent_cortex.recurrent_grounding import (
                validate_recurrent_grounding_receipt,
            )

            validate_recurrent_grounding_receipt(
                receipt.get("recurrent_grounding"),
                input_tokens_sha256=str(receipt.get("input_tokens_sha256") or ""),
                input_token_count=int(receipt.get("input_token_count") or 0),
                cognitive_slots=list(receipt.get("cognitive_slots") or []),
                n_slots=int(config.get("n_slots") or 0),
                n_branches=int(config.get("n_branches") or 0),
                selected_branch=int(receipt.get("selected_branch") or 0),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("recurrent_grounding_unproven")
        try:
            from core.brain.llm.latent_cortex.loop_core import (
                build_loop_core_contract,
            )
            from core.brain.llm.latent_cortex.loop_stability import (
                validate_loop_stability_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            prelude_end = receipt.get("prelude_end")
            coda_start = receipt.get("coda_start")
            if type(prelude_end) is not int or type(coda_start) is not int:
                raise ValueError("loop boundaries must be integers")
            expected_loop_core = build_loop_core_contract(
                prelude_end=prelude_end,
                coda_start=coda_start,
                max_steps=executed_config.recurrence.max_steps,
                min_steps=executed_config.recurrence.min_steps,
                alpha=executed_config.recurrence.alpha,
                alpha_schedule=executed_config.recurrence.alpha_schedule,
                rms_clip_ratio=executed_config.recurrence.rms_clip_ratio,
                convergence_eps=executed_config.recurrence.convergence_eps,
                divergence_ratio=executed_config.recurrence.divergence_ratio,
                fixed_depth=executed_config.recurrence.fixed_depth,
            )
            validate_loop_stability_receipt(
                receipt.get("loop_stability"),
                recurrent_grounding=receipt.get("recurrent_grounding"),
                expected_loop_core=expected_loop_core,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("loop_stability_unproven")
        try:
            from core.brain.llm.latent_cortex.kv_state_tree import (
                validate_kv_state_tree_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            validate_kv_state_tree_receipt(
                receipt.get("kv_state_tree"),
                episode_id=str(receipt.get("episode_id") or ""),
                input_tokens_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_layers=int(receipt.get("n_layers") or 0),
                expected_n_branches=executed_config.branches.n_branches,
                require_final=True,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("kv_state_tree_unproven")
        try:
            from core.brain.llm.latent_cortex.update_gate import (
                UpdateGateRuntime,
                validate_update_gate_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_update_gate = UpdateGateRuntime.from_config(executed_config.update_gate)
            validate_update_gate_receipt(
                receipt.get("update_acceptance"),
                expected_gate=expected_update_gate,
                recurrent_grounding=receipt.get("recurrent_grounding"),
                loop_stability=receipt.get("loop_stability"),
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("update_acceptance_unproven")
        try:
            from core.brain.llm.latent_cortex.bidirectional_reflector import (
                validate_bidirectional_reflector_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)

            validate_bidirectional_reflector_receipt(
                receipt.get("bidirectional_reflector"),
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("bidirectional_reflector_unproven")
        try:
            from core.brain.llm.latent_cortex.contradiction_tensor import (
                ContradictionTensorRuntime,
                validate_contradiction_tensor_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_contradiction = ContradictionTensorRuntime.from_config(
                executed_config.contradiction_head
            )
            validate_contradiction_tensor_receipt(
                receipt.get("contradiction_tensor"),
                expected_runtime=expected_contradiction,
                reflector=receipt.get("bidirectional_reflector"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("contradiction_tensor_unproven")
        try:
            from core.brain.llm.latent_cortex.contradiction_perturber import (
                ContradictionPerturberConfig,
                validate_contradiction_perturbation_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            cognitive_slots = receipt.get("cognitive_slots")
            protected_positions = (
                sorted(
                    {
                        int(row["slot"])
                        for row in cognitive_slots
                        if (isinstance(row, dict) and type(row.get("slot")) is int)
                    }
                )
                if isinstance(cognitive_slots, list)
                else []
            )
            validate_contradiction_perturbation_receipt(
                receipt.get("contradiction_perturbation"),
                expected_config=ContradictionPerturberConfig.from_value(
                    executed_config.contradiction_perturber
                ),
                contradiction_tensor=receipt.get("contradiction_tensor"),
                expected_selected_branch=int(receipt.get("selected_branch", -1)),
                expected_protected_positions=protected_positions,
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("contradiction_perturbation_unproven")
        try:
            from core.brain.llm.latent_cortex.neural_uncertainty import (
                NeuralUncertaintyRuntime,
                validate_neural_uncertainty_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_uncertainty = NeuralUncertaintyRuntime.from_config(
                executed_config.uncertainty_head
            )
            validate_neural_uncertainty_receipt(
                receipt.get("neural_uncertainty"),
                expected_runtime=expected_uncertainty,
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("neural_uncertainty_unproven")
        try:
            from core.brain.llm.latent_cortex.local_exploration import (
                LocalExplorationConfig,
                validate_local_exploration_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            cognitive_slots = receipt.get("cognitive_slots")
            protected_positions = (
                sorted(
                    {
                        int(row["slot"])
                        for row in cognitive_slots
                        if (isinstance(row, dict) and type(row.get("slot")) is int)
                    }
                )
                if isinstance(cognitive_slots, list)
                else []
            )
            validate_local_exploration_receipt(
                receipt.get("local_exploration"),
                expected_config=LocalExplorationConfig.from_value(
                    executed_config.local_exploration
                ),
                contradiction_tensor=receipt.get("contradiction_tensor"),
                contradiction_perturbation=receipt.get("contradiction_perturbation"),
                neural_uncertainty=receipt.get("neural_uncertainty"),
                expected_selected_branch=int(receipt.get("selected_branch", -1)),
                expected_protected_positions=protected_positions,
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("local_exploration_unproven")
        try:
            from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                HeterogeneousIntegrationConfig,
                validate_heterogeneous_integration_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            validate_heterogeneous_integration_receipt(
                receipt.get("heterogeneous_integration"),
                expected_config=HeterogeneousIntegrationConfig.from_value(
                    executed_config.heterogeneous_integration
                ),
                contradiction_perturbation=receipt.get("contradiction_perturbation"),
                local_exploration=receipt.get("local_exploration"),
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("heterogeneous_integration_unproven")
        try:
            from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                validate_heterogeneous_decode_receipt,
            )

            replacement_applied = (
                isinstance(receipt.get("answer_replacement"), dict)
                and receipt["answer_replacement"].get("decision") == "replace"
            )
            if replacement_applied:
                if (
                    not isinstance(answer_replacement_private, dict)
                    or not isinstance(
                        answer_replacement_private.get("baseline_tokens"),
                        list,
                    )
                ):
                    raise ValueError("replacement baseline tokens are unavailable")
                validate_heterogeneous_decode_receipt(
                    receipt.get("heterogeneous_decode"),
                    integration=receipt.get("heterogeneous_integration"),
                    expected_output_tokens=answer_replacement_private[
                        "baseline_tokens"
                    ],
                )
            else:
                validate_heterogeneous_decode_receipt(
                    receipt.get("heterogeneous_decode"),
                    integration=receipt.get("heterogeneous_integration"),
                    expected_output_tokens=output_tokens,
                )
        except (ImportError, TypeError, ValueError):
            errors.append("heterogeneous_decode_unproven")
        try:
            from core.brain.llm.latent_cortex.mistake_locator import (
                MistakeLocatorRuntime,
                validate_mistake_locator_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_locator = MistakeLocatorRuntime.from_config(executed_config.mistake_locator)
            validate_mistake_locator_receipt(
                receipt.get("mistake_locator"),
                expected_runtime=expected_locator,
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
                expected_domain=expected_domain,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("mistake_locator_unproven")
        try:
            from core.brain.llm.latent_cortex.verifier_fusion import (
                validate_verifier_fusion_receipt,
            )

            validate_verifier_fusion_receipt(
                receipt.get("verifier_fusion"),
                blind_review=receipt.get("blind_review"),
                decoy_verification=receipt.get("decoy_verification"),
                generative_verifier=receipt.get("generative_verifier"),
                counterfactual_verifier=receipt.get("counterfactual_verifier"),
                prefix_stability=receipt.get("prefix_stability"),
                neural_uncertainty=receipt.get("neural_uncertainty"),
                mistake_locator=receipt.get("mistake_locator"),
                selected_branch=receipt.get("selected_branch"),
                evidence=config.get("verifier_fusion_evidence"),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("verifier_fusion_unproven")
        try:
            from core.brain.llm.latent_cortex.stop_gate import (
                StopGateRuntime,
                validate_stop_gate_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_stop_gate = StopGateRuntime.from_config(executed_config.halting)
            validate_stop_gate_receipt(
                receipt.get("halting"),
                expected_gate=expected_stop_gate,
                expected_n_branches=executed_config.branches.n_branches,
                update_acceptance=receipt.get("update_acceptance"),
                loop_stability=receipt.get("loop_stability"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("halting_unproven")
        if receipt.get("last_stage") != "action_state_captured":
            try:
                from core.brain.llm.latent_cortex.terminal_disposition import (
                    validate_terminal_disposition_receipt,
                )

                if not isinstance(output_text, str) or not isinstance(output_tokens, list):
                    raise ValueError("terminal output is unavailable")
                validate_terminal_disposition_receipt(
                    receipt.get("terminal_disposition"),
                    halting_reason=receipt.get("halting_reason"),
                    halting=receipt.get("halting"),
                    loop_stability=receipt.get("loop_stability"),
                    cognitive_action_trace=receipt.get("cognitive_action_trace"),
                    budget=receipt.get("budget"),
                    output_tokens=output_tokens,
                    output_text=output_text,
                    full_bridge_tokens_sha256=receipt.get(
                        "decode_bridge_tokens_sha256"
                    ),
                )
                language = receipt["terminal_disposition"]["language"]
                if (
                    language.get("source")
                    not in {"resident_model_decode", "resident_model_repair"}
                    or language.get("instruction_applied") is not True
                ):
                    raise ValueError("terminal language was not resident-model generated")
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("terminal_disposition_unproven")
            try:
                from core.brain.llm.latent_cortex.causal_receipt import (
                    validate_causal_receipt,
                )

                validate_causal_receipt(
                    receipt.get("causal_receipt"),
                    worker_receipt=receipt,
                    require_complete=True,
                )
            except (ImportError, TypeError, ValueError):
                errors.append("causal_receipt_unproven")
        try:
            from core.brain.llm.latent_cortex.verified_best import (
                validate_verified_best_receipt,
            )

            validate_verified_best_receipt(
                receipt.get("verified_best_state"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                loop_stability=receipt.get("loop_stability"),
                expected_n_branches=int(config.get("n_branches") or 0),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("verified_best_state_unproven")
        try:
            from core.brain.llm.latent_cortex.transient_constraints import (
                TransientConstraintConfig,
                validate_transient_constraint_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            cognitive_slots = receipt.get("cognitive_slots")
            protected = (
                tuple(
                    sorted(
                        {
                            int(row["slot"])
                            for row in cognitive_slots
                            if (isinstance(row, dict) and type(row.get("slot")) is int)
                        }
                    )
                )
                if isinstance(cognitive_slots, list)
                else ()
            )
            executed_config = config_from_job(config)
            expected_branches = executed_config.branches.n_branches
            validate_transient_constraint_receipt(
                receipt.get("transient_negative_constraints"),
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_branches=expected_branches,
                protected_positions={index: protected for index in range(expected_branches)},
                expected_config=TransientConstraintConfig.from_value(
                    executed_config.transient_negative_constraints
                ),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                verifier_preflight=receipt.get("verifier_preflight"),
                information_accounting=information_accounting,
                resource_accounting=resource_accounting,
                kv_state_tree=receipt.get("kv_state_tree"),
                verified_best_state=receipt.get("verified_best_state"),
                loop_stability=receipt.get("loop_stability"),
                require_verified_best_binding=True,
                require_external_bindings=True,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("transient_negative_constraints_unproven")
        try:
            from core.brain.llm.latent_cortex.virtual_quanta import (
                VirtualQuantaConfig,
                validate_virtual_quanta_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            virtual_receipt = receipt.get("virtual_quanta")
            validate_virtual_quanta_receipt(
                virtual_receipt,
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_branches=executed_config.branches.n_branches,
                expected_config=VirtualQuantaConfig.from_value(executed_config.virtual_quanta),
                cognitive_slots=receipt.get("cognitive_slots"),
                verifier_preflight=receipt.get("verifier_preflight"),
                information_accounting=information_accounting,
                resource_accounting=resource_accounting,
                kv_state_tree=receipt.get("kv_state_tree"),
                require_external_bindings=bool(
                    isinstance(virtual_receipt, dict) and virtual_receipt.get("arms")
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("virtual_quanta_unproven")
        try:
            from core.brain.llm.latent_cortex.latent_tree_search import (
                LatentTreeSearchConfig,
                validate_latent_tree_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            tree_receipt = receipt.get("latent_tree_search")
            validate_latent_tree_receipt(
                tree_receipt,
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                expected_config=LatentTreeSearchConfig.from_value(
                    executed_config.latent_tree_search
                ),
                kv_state_tree=receipt.get("kv_state_tree"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                resource_accounting=resource_accounting,
                loop_stability=receipt.get("loop_stability"),
                require_external_bindings=bool(
                    isinstance(tree_receipt, dict) and tree_receipt.get("transactions")
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("latent_tree_search_unproven")
        one_shot_slots = [
            row
            for row in (receipt.get("cognitive_slots") or [])
            if isinstance(row, dict)
            and row.get("knowledge_class") == "one_shot_nonparametric_memory"
        ]
        one_shot_receipt = receipt.get("nonparametric_memory")
        try:
            from core.brain.llm.latent_cortex.nonparametric_context import (
                validate_receipt as validate_nonparametric_receipt,
            )

            if one_shot_receipt:
                validated_one_shot = validate_nonparametric_receipt(one_shot_receipt)
                if validated_one_shot["applied"]:
                    if (
                        len(one_shot_slots) != 1
                        or one_shot_slots[0].get("instruction_authority") is not False
                        or one_shot_slots[0].get("text_sha256")
                        != validated_one_shot["observation_sha256"]
                    ):
                        raise ValueError("admitted one-shot evidence is not bound to its slot")
                elif one_shot_slots:
                    raise ValueError("one-shot evidence slot exists without admitted retrieval")
                expected_resource = validated_one_shot["resource_accounting"]
                operation = (
                    resource_accounting.get("operations", {}).get("nonparametric_memory_retrieval")
                    if resource_accounting is not None
                    else None
                )
                if not isinstance(operation, dict) or any(
                    operation.get(resource_name) != expected_resource[receipt_name]
                    for resource_name, receipt_name in (
                        ("tensor_element_reads", "tensor_element_reads"),
                        ("tensor_element_writes", "tensor_element_writes"),
                        ("tensor_scalar_ops", "tensor_scalar_ops"),
                        ("host_scalar_ops", "host_scalar_ops"),
                    )
                ):
                    raise ValueError("one-shot retrieval work differs from resource ledger")
                if information_accounting is None:
                    raise ValueError("one-shot information accounting is absent")
                from core.brain.llm.latent_cortex.resource_accounting import (
                    policy_sha256,
                )

                source_identity = validated_one_shot["source_identity"]
                store_sources = [
                    row
                    for row in information_accounting["sources"]
                    if row.get("source_id") == "one_shot_nonparametric_memory"
                ]
                if source_identity:
                    if (
                        len(store_sources) != 1
                        or store_sources[0].get("kind") != "local_nonparametric_memory_store"
                        or store_sources[0].get("content_sha256")
                        != source_identity["content_sha256"]
                        or store_sources[0].get("byte_count") != source_identity["source_bytes"]
                    ):
                        raise ValueError("one-shot store differs from information ledger")
                elif store_sources:
                    raise ValueError("information ledger claims an unavailable one-shot store")
                expected_policy = policy_sha256(
                    {
                        "policy": "context_only_prompt_tail_recall_v1",
                        "active_source_receipt_sha256": source_identity.get(
                            "receipt_sha256", "none"
                        ),
                    }
                )
                if (
                    information_accounting["policies"].get("nonparametric_memory")
                    != expected_policy
                ):
                    raise ValueError("one-shot retrieval policy is not bound")
                context_sources = [
                    row
                    for row in information_accounting["sources"]
                    if str(row.get("source_id") or "").endswith(":one_shot_memory")
                ]
                if validated_one_shot["applied"]:
                    slot = one_shot_slots[0]
                    expected_source_id = (
                        f"cognitive_context:{slot.get('context_index')}:one_shot_memory"
                    )
                    if (
                        len(context_sources) != 1
                        or context_sources[0].get("source_id") != expected_source_id
                        or context_sources[0].get("kind") != "typed_cognitive_context"
                        or context_sources[0].get("content_sha256")
                        != validated_one_shot["observation_sha256"]
                    ):
                        raise ValueError("one-shot observation differs from information ledger")
                elif context_sources:
                    raise ValueError("information ledger claims an unadmitted one-shot observation")
            elif one_shot_slots:
                raise ValueError("one-shot evidence slot has no retrieval receipt")
        except (ImportError, TypeError, ValueError):
            errors.append("nonparametric_memory_binding_unproven")
        isolation_steps = config.get("isolation_steps")
        if type(isolation_steps) is int:
            isolation = receipt.get("branch_isolation")
            isolation_valid = isinstance(isolation, dict)
            if isolation_valid:
                candidates = isolation.get("candidates")
                cache_discipline = isolation.get("cache_discipline")
                branch_count = config.get("n_branches")
                candidate_rows_valid = (
                    isinstance(candidates, list)
                    and type(branch_count) is int
                    and len(candidates) == branch_count
                    and all(
                        isinstance(row, dict)
                        and row.get("index") == index
                        and isinstance(row.get("role"), str)
                        and bool(row["role"])
                        and sha256(row.get("context_sha256"))
                        and sha256(row.get("rng_stream_sha256"))
                        and sha256(row.get("seed_sha256"))
                        and sha256(row.get("candidate_sha256"))
                        and type(row.get("candidate_step")) is int
                        and row["candidate_step"] >= isolation_steps
                        for index, row in enumerate(candidates)
                    )
                )
                unique_commitments = bool(candidate_rows_valid) and all(
                    len({row[key] for row in candidates}) == len(candidates)
                    for key in (
                        "rng_stream_sha256",
                        "seed_sha256",
                        "candidate_sha256",
                    )
                )
                one_context = (
                    bool(candidate_rows_valid)
                    and len({row["context_sha256"] for row in candidates}) == 1
                )
                cache_valid = (
                    isinstance(cache_discipline, dict)
                    and set(cache_discipline)
                    == {
                        "schema",
                        "nonpersistent_calls",
                        "restored_calls",
                        "restore_failures",
                        "all_restored",
                    }
                    and cache_discipline.get("schema") == "aura.rlc.cache_discipline.v1"
                    and positive_int(cache_discipline, "nonpersistent_calls")
                    and cache_discipline.get("restored_calls")
                    == cache_discipline.get("nonpersistent_calls")
                    and cache_discipline.get("restore_failures") == 0
                    and cache_discipline.get("all_restored") is True
                )
                exchanges = receipt.get("exchanges")
                first_exchange_step = isolation.get("first_exchange_step")
                exposure_valid = (
                    nonnegative_int(isolation, "blocked_cross_exposures")
                    and (
                        (exchanges == 0 and first_exchange_step is None)
                        or (
                            type(exchanges) is int
                            and exchanges > 0
                            and type(first_exchange_step) is int
                            and first_exchange_step >= isolation_steps
                        )
                    )
                    and isolation.get("cross_exposure_started")
                    is (type(exchanges) is int and exchanges > 0)
                )
                isolation_valid = (
                    set(isolation)
                    == {
                        "schema",
                        "n_branches",
                        "required_steps",
                        "sealed",
                        "certified",
                        "reason",
                        "configured_role_lesion",
                        "seed_alias_free",
                        "seed_states_unique",
                        "rng_streams_unique",
                        "cross_exposure_started",
                        "first_exchange_step",
                        "blocked_cross_exposures",
                        "candidates",
                        "cache_discipline",
                    }
                    and isolation.get("schema") == "aura.rlc.branch_isolation.v1"
                    and isolation.get("n_branches") == branch_count
                    and isolation.get("required_steps") == isolation_steps
                    and isolation.get("sealed") is True
                    and isolation.get("certified") is True
                    and isolation.get("reason") == "certified"
                    and isolation.get("configured_role_lesion") is False
                    and isolation.get("seed_alias_free") is True
                    and isolation.get("seed_states_unique") is True
                    and isolation.get("rng_streams_unique") is True
                    and candidate_rows_valid
                    and unique_commitments
                    and one_context
                    and cache_valid
                    and exposure_valid
                )
            if not isolation_valid:
                errors.append("branch_isolation_unproven")
        exchanges = receipt.get("exchanges")
        if type(exchanges) is int and exchanges > 0:
            try:
                from core.brain.llm.latent_cortex.branch_exchange import (
                    validate_branch_exchange_trace,
                )

                exchange_trace = validate_branch_exchange_trace(
                    receipt.get("branch_exchange"),
                    exchange_count=exchanges,
                    n_branches=int(config.get("n_branches")),
                    n_slots=int(config.get("n_slots")),
                    comm_slot=int(config.get("comm_slot", 0)),
                    exchange_gamma=float(config.get("exchange_gamma", 0.35)),
                    branch_isolation=receipt.get("branch_isolation"),
                    cognitive_slots=receipt.get("cognitive_slots"),
                    exchange_interval=int(config.get("exchange_interval", 4)),
                    schedule_hash=str(receipt.get("schedule_hash") or ""),
                    bytecode_events=receipt.get("bytecode_events"),
                    cognitive_action_trace=receipt.get("cognitive_action_trace"),
                )
                expected_reads = 0
                expected_writes = 0
                expected_scalar_ops = 0
                for exchange_row in exchange_trace["exchanges"]:
                    accounting = exchange_row["tensor_accounting"]
                    expected_reads += accounting["source_elements_read"]
                    expected_writes += (
                        accounting["message_elements_emitted"]
                        + accounting["consensus_elements_written"]
                    )
                    expected_scalar_ops += accounting["tensor_scalar_ops"]
                operation = (
                    resource_accounting.get("operations", {}).get("branch_exchange")
                    if resource_accounting is not None
                    else None
                )
                if (
                    not isinstance(operation, dict)
                    or operation.get("tensor_element_reads") != expected_reads
                    or operation.get("tensor_element_writes") != expected_writes
                    or operation.get("tensor_scalar_ops") != expected_scalar_ops
                    or any(
                        operation.get(name) != 0
                        for name in operation
                        if name
                        not in {
                            "tensor_element_reads",
                            "tensor_element_writes",
                            "tensor_scalar_ops",
                        }
                    )
                ):
                    errors.append("branch_exchange_resource_binding_unproven")
            except (ImportError, TypeError, ValueError):
                errors.append("branch_exchange_provenance_unproven")
        elif receipt.get("branch_exchange") not in ({}, None):
            errors.append("unexpected_branch_exchange_trace")
        if (
            not (type(exchanges) is int and exchanges > 0)
            and resource_accounting is not None
            and "branch_exchange" in resource_accounting.get("operations", {})
        ):
            errors.append("branch_exchange_resource_binding_unproven")
        raw_action_trace = receipt.get("cognitive_action_trace")
        if isinstance(raw_action_trace, list) and raw_action_trace:
            validating_context_focus = False
            try:
                from core.brain.llm.latent_cortex.cognitive_operators import (
                    validate_operator_receipt,
                )

                neural_actions = {
                    "decompose",
                    "blind_resolve",
                    "branch",
                    "search_memory",
                    "retrieve_evidence",
                    "simulate",
                    "falsify",
                    "check_assumption",
                    "regenerate_from_prefix",
                    "formalize",
                }
                raw_operator_trace = receipt.get("cognitive_operator_trace")
                if not isinstance(raw_operator_trace, list) or not raw_operator_trace:
                    raise ValueError("cognitive operator trace is absent")
                operator_rows = [validate_operator_receipt(row) for row in raw_operator_trace]
                expected_operator_work: dict[str, dict[str, int]] = {}
                for row in operator_rows:
                    operation_name = f"cognitive_operator:{row['operator']}"
                    expected = expected_operator_work.setdefault(
                        operation_name,
                        {
                            "tensor_element_reads": 0,
                            "tensor_element_writes": 0,
                            "tensor_scalar_ops": 0,
                            "host_scalar_ops": 0,
                        },
                    )
                    accounting = row["tensor_accounting"]
                    expected["tensor_element_reads"] += accounting["element_reads"]
                    expected["tensor_element_writes"] += accounting["element_writes"]
                    expected["tensor_scalar_ops"] += accounting["tensor_scalar_ops"]
                    expected["host_scalar_ops"] += accounting["commitment_host_ops"]
                operations = (
                    resource_accounting.get("operations", {})
                    if resource_accounting is not None
                    else {}
                )
                observed_operator_names = {
                    name for name in operations if name.startswith("cognitive_operator:")
                }
                if observed_operator_names != set(expected_operator_work):
                    raise ValueError("cognitive operator resource coverage differs")
                for operation_name, expected in expected_operator_work.items():
                    operation = operations.get(operation_name)
                    if not isinstance(operation, dict) or any(
                        operation.get(name) != value for name, value in expected.items()
                    ):
                        raise ValueError("cognitive operator resource totals differ")
                    if any(operation.get(name) != 0 for name in operation if name not in expected):
                        raise ValueError("cognitive operator resource kind differs")
                by_step: dict[int, list[dict[str, Any]]] = {}
                for row in operator_rows:
                    by_step.setdefault(row["action_step"], []).append(row)
                action_rows_by_step: dict[int, dict[str, Any]] = {}
                for action_row in raw_action_trace:
                    if not isinstance(action_row, dict):
                        raise ValueError("cognitive action trace row is invalid")
                    transition = action_row.get("transition")
                    signal = action_row.get("state_signal")
                    step = transition.get("step_index") if isinstance(transition, dict) else None
                    if (
                        type(step) is not int
                        or step < 0
                        or step in action_rows_by_step
                        or not isinstance(signal, dict)
                    ):
                        raise ValueError("cognitive action trace row is incomplete")
                    action_rows_by_step[step] = action_row
                if set(by_step) - set(action_rows_by_step):
                    raise ValueError("cognitive operator step is orphaned")
                for step, action_row in sorted(action_rows_by_step.items()):
                    transition = action_row["transition"]
                    signal = action_row["state_signal"]
                    action = transition.get("action")
                    rows = by_step.get(step, [])
                    if action not in neural_actions:
                        if rows:
                            raise ValueError("structural action claimed neural operators")
                        continue
                    active_branches = signal.get("active_branches")
                    if (
                        type(active_branches) is not int
                        or active_branches <= 0
                        or len(rows) != active_branches
                        or {row["action"] for row in rows} != {action}
                        or {row["action_step"] for row in rows} != {step}
                        or len({row["branch_index"] for row in rows}) != len(rows)
                        or len({row["operator"] for row in rows}) != len(rows)
                    ):
                        raise ValueError("cognitive operator coverage is invalid")
                from core.brain.llm.latent_cortex.context_focus import (
                    CONTEXT_FOCUS_ACTIONS,
                    validate_context_focus_receipt,
                )
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )

                validating_context_focus = True
                raw_focus_trace = receipt.get("context_focus_trace")
                if not isinstance(raw_focus_trace, list):
                    raise ValueError("context focus trace is invalid")
                cognitive_slots = receipt.get("cognitive_slots")
                if not isinstance(cognitive_slots, list):
                    raise ValueError("cognitive slot inventory is invalid")
                focus_rows = [
                    validate_context_focus_receipt(
                        row,
                        cognitive_slots=cognitive_slots,
                    )
                    for row in raw_focus_trace
                ]
                expected_focus_work: dict[str, dict[str, int]] = {}
                for row in focus_rows:
                    operation_name = f"context_focus:{row['action']}"
                    expected = expected_focus_work.setdefault(
                        operation_name,
                        {
                            "tensor_element_reads": 0,
                            "tensor_element_writes": 0,
                            "tensor_scalar_ops": 0,
                            "host_scalar_ops": 0,
                        },
                    )
                    accounting = row["tensor_accounting"]
                    expected["tensor_element_reads"] += accounting["element_reads"]
                    expected["tensor_element_writes"] += accounting["element_writes"]
                    expected["tensor_scalar_ops"] += accounting["tensor_scalar_ops"]
                    expected["host_scalar_ops"] += accounting["commitment_host_ops"]
                observed_focus_names = {
                    name for name in operations if name.startswith("context_focus:")
                }
                if observed_focus_names != set(expected_focus_work):
                    raise ValueError("context focus resource coverage differs")
                for operation_name, expected in expected_focus_work.items():
                    operation = operations.get(operation_name)
                    if not isinstance(operation, dict) or any(
                        operation.get(name) != amount
                        for name, amount in expected.items()
                    ):
                        raise ValueError("context focus resource totals differ")
                    if any(
                        operation.get(name) != 0
                        for name in operation
                        if name not in expected
                    ):
                        raise ValueError("context focus resource kind differs")
                focus_by_step: dict[int, list[dict[str, Any]]] = {}
                for row in focus_rows:
                    focus_by_step.setdefault(row["action_step"], []).append(row)
                if set(focus_by_step) - set(action_rows_by_step):
                    raise ValueError("context focus step is orphaned")
                operator_by_step_branch = {
                    (row["action_step"], row["branch_index"]): row
                    for row in operator_rows
                }
                for step, action_row in sorted(action_rows_by_step.items()):
                    transition = action_row["transition"]
                    signal = action_row["state_signal"]
                    action = OperationKind(transition["action"])
                    rows = focus_by_step.get(step, [])
                    if action not in CONTEXT_FOCUS_ACTIONS:
                        if rows:
                            raise ValueError(
                                "non-context action claimed context focus"
                            )
                        continue
                    active_branches = signal["active_branches"]
                    if (
                        len(rows) != active_branches
                        or {row["action"] for row in rows} != {action.value}
                        or len({row["branch_index"] for row in rows}) != len(rows)
                    ):
                        raise ValueError("context focus coverage is invalid")
                    for row in rows:
                        operator_row = operator_by_step_branch.get(
                            (step, row["branch_index"])
                        )
                        if (
                            operator_row is None
                            or operator_row["action"] != action.value
                            or operator_row["input_sha256"]
                            != row["output_sha256"]
                        ):
                            raise ValueError(
                                "context focus did not feed the cognitive operator"
                            )
            except (ImportError, TypeError, ValueError):
                errors.append(
                    "context_focus_execution_unproven"
                    if validating_context_focus
                    else "cognitive_operator_execution_unproven"
                )
            try:
                from core.brain.llm.latent_cortex.structural_diversity import (
                    validate_structural_diversity_receipt,
                )

                validate_structural_diversity_receipt(
                    receipt.get("structural_diversity"),
                    n_branches=int(receipt.get("n_branches")),
                    cognitive_slots=receipt.get("cognitive_slots"),
                    operator_trace=receipt.get("cognitive_operator_trace"),
                    action_trace=raw_action_trace,
                    branch_isolation=receipt.get("branch_isolation"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("structural_diversity_unproven")
            try:
                from core.brain.llm.latent_cortex.disagreement_graph import (
                    validate_disagreement_graph_receipt,
                )

                validate_disagreement_graph_receipt(
                    receipt.get("disagreement_graph"),
                    n_branches=int(receipt.get("n_branches")),
                    operator_trace=receipt.get("cognitive_operator_trace"),
                    action_trace=raw_action_trace,
                    structural_diversity=receipt.get("structural_diversity"),
                    blind_review=receipt.get("blind_review"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("disagreement_graph_unproven")
            try:
                from core.brain.llm.latent_cortex.diagnostic_action_selector import (
                    validate_diagnostic_action_selector_receipt,
                )

                validate_diagnostic_action_selector_receipt(
                    receipt.get("diagnostic_action_selection"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    value_policy=receipt.get("value_of_computation"),
                    action_trace=raw_action_trace,
                )
            except (ImportError, TypeError, ValueError):
                errors.append("diagnostic_action_selection_unproven")
            try:
                from core.brain.llm.latent_cortex.local_repair import (
                    validate_local_repair_receipt,
                )

                validate_local_repair_receipt(
                    receipt.get("local_repair"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    diagnostic_selection=receipt.get(
                        "diagnostic_action_selection"
                    ),
                )
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("local_repair_unproven")
            try:
                from core.brain.llm.latent_cortex.answer_replacement import (
                    validate_answer_replacement_receipt,
                )
                from core.brain.llm.latent_cortex.worker_handler import (
                    config_from_job,
                )

                executed_config = config_from_job(config)
                from core.brain.llm.latent_cortex.answer_replacement import (
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                )

                replacement_output_limit = min(
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                    int(executed_config.decode_max_tokens)
                    + (
                        int(executed_config.decode_contract_grace_tokens)
                        if executed_config.decode_contract == "final_answer_v1"
                        else 48
                    ),
                )
                validate_answer_replacement_receipt(
                    receipt.get("answer_replacement"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    diagnostic_selection=receipt.get(
                        "diagnostic_action_selection"
                    ),
                    local_repair=receipt.get("local_repair"),
                    private_evidence=answer_replacement_private,
                    expected_objective=expected_objective,
                    expected_selected_branch=int(receipt.get("selected_branch")),
                    expected_enabled=executed_config.answer_replacement_enabled,
                    expected_margin=executed_config.answer_replacement_margin,
                    expected_max_output_tokens=replacement_output_limit,
                    expected_output_text=(
                        output_text if isinstance(output_text, str) else None
                    ),
                    expected_output_tokens=(
                        output_tokens if isinstance(output_tokens, list) else None
                    ),
                )
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("answer_replacement_unproven")
            try:
                from core.brain.llm.latent_cortex.correlated_support import (
                    validate_correlated_support_receipt,
                )

                validate_correlated_support_receipt(
                    receipt.get("correlated_support"),
                    structural_diversity=receipt.get("structural_diversity"),
                    correlation_evidence=config.get("branch_correlation_evidence"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("correlated_support_unproven")
        elif resource_accounting is not None and any(
            name.startswith(("cognitive_operator:", "context_focus:"))
            for name in resource_accounting.get("operations", {})
        ):
            errors.append("cognitive_operator_execution_unproven")
        preflight: dict[str, Any] | None = None
        if receipt.get("verifier_preflight"):
            try:
                from core.brain.llm.latent_cortex.blind_review import (
                    validate_decoy_preflight_receipt,
                )

                preflight = validate_decoy_preflight_receipt(
                    receipt.get("verifier_preflight"),
                    episode_id=receipt.get("episode_id"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                )
                if preflight[
                    "verifier_admitted"
                ] is False and "verifier_preflight_decoy_calibration_failed" not in (
                    receipt.get("honest_flags") or []
                ):
                    raise ValueError("decoy preflight rejection was not disclosed")
            except (ImportError, TypeError, ValueError):
                errors.append("decoy_verifier_preflight_unproven")
        if receipt.get("branch_contract"):
            try:
                from core.brain.llm.latent_cortex.blind_review import (
                    validate_blind_review_receipt,
                    validate_decoy_review_receipt,
                )

                decoy = validate_decoy_review_receipt(
                    receipt.get("decoy_verification"),
                    blind_receipt=receipt.get("blind_review"),
                    episode_id=receipt.get("episode_id"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                )
                review_selected_branch = receipt.get("selected_branch")
                if isinstance(verified_counterfactual, dict):
                    review_selected_branch = verified_counterfactual[
                        "source_selected_branch"
                    ]
                elif (
                    isinstance(verified_generation, dict)
                    and verified_generation.get("selection_effect")
                    == "winner_replaced"
                ):
                    review_selected_branch = verified_generation["vetoed_branch"]
                validate_blind_review_receipt(
                    receipt.get("blind_review"),
                    n_branches=int(receipt.get("n_branches")),
                    branch_scores=receipt.get("branch_scores"),
                    isolation_receipt=receipt.get("branch_isolation"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                    episode_id=receipt.get("episode_id"),
                    selected_branch=review_selected_branch,
                    decoy_receipt=receipt.get("decoy_verification"),
                )
                honest_flags = receipt.get("honest_flags")
                if not isinstance(honest_flags, list):
                    raise ValueError("decoy-review honest flags are invalid")
                if (
                    decoy["selection_admitted"] is False
                    and "branch_verifier_decoy_calibration_failed" not in honest_flags
                ):
                    raise ValueError("decoy selection rejection was not disclosed")
            except (ImportError, TypeError, ValueError):
                errors.append("blind_or_decoy_branch_review_unproven")
        exchange_interval = config.get("exchange_interval")
        if (
            type(exchange_interval) is int
            and exchange_interval > 0
            and type(config.get("n_branches")) is int
            and config["n_branches"] > 1
            and positive_int(receipt, "steps_taken")
            and receipt["steps_taken"] >= exchange_interval
            and not positive_int(receipt, "exchanges")
        ):
            errors.append("branch_exchange_unproven")
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
        decode_contract = config.get("decode_contract", "none")
        contract_required = decode_contract == "final_answer_v1"
        configured_contract_grace = config.get(
            "decode_contract_grace_tokens",
            0,
        )
        if contract_required:
            if receipt.get("decode_contract_required") is not True:
                errors.append("decode_contract_requirement_unreceipted")
            if receipt.get("decode_contract_satisfied") is not True:
                errors.append("decode_contract_unsatisfied")
            if receipt.get("decode_termination") not in {
                "contract_complete",
                "confidence_bound_replacement",
            }:
                errors.append("decode_contract_termination_mismatch")
            if (
                type(configured_contract_grace) is not int
                or configured_contract_grace < 0
                or receipt.get("decode_contract_grace_tokens") != configured_contract_grace
            ):
                errors.append("decode_contract_grace_mismatch")
            grace_used = receipt.get("decode_contract_grace_used_tokens")
            expected_grace_used = max(
                0,
                int(receipt.get("decode_generated_tokens") or 0)
                - int(receipt.get("decode_requested_tokens") or 0),
            )
            if (
                type(grace_used) is not int
                or not 0 <= grace_used <= configured_contract_grace
                or grace_used != expected_grace_used
            ):
                errors.append("decode_contract_grace_accounting_invalid")
        elif receipt.get("decode_contract_required") is True:
            errors.append("unexpected_decode_contract")
        configured_probe_tokens = config.get("verifier_probe_max_tokens", 48)
        if (
            type(configured_probe_tokens) is not int
            or receipt.get("verifier_probe_max_tokens") != configured_probe_tokens
        ):
            errors.append("verifier_probe_profile_mismatch")
        if receipt.get("decode_termination") not in {
            "eos",
            # The public answer contract completed (one FINAL_ANSWER JSON
            # object closed and parsed) — a complete answer by construction.
            "contract_complete",
            "token_limit",
            # Sentence grace: the limit landed mid-sentence and sampling
            # continued a few model-chosen tokens to the natural boundary.
            "token_limit_sentence_grace",
            # Wall-clock analogues: time pressure ended decoding, ideally at
            # a sentence boundary (wind-down). The output-quality gate is
            # the completeness judge either way.
            "wall_reserve_sentence_grace",
            "wall_reserve",
            "confidence_bound_replacement",
        }:
            errors.append("decode_incomplete")
        decode_bridge_policy = config.get("decode_bridge_policy", "none")
        if decode_bridge_policy in {
            "assistant_answer_v1",
            "assistant_answer_v2",
            "assistant_answer_v3",
        }:
            if receipt.get("decode_bridge_applied") is not True:
                errors.append("decode_bridge_unapplied")
            if receipt.get("decode_bridge_policy") != decode_bridge_policy:
                errors.append("decode_bridge_policy_mismatch")
            if not positive_int(receipt, "decode_bridge_token_count"):
                errors.append("decode_bridge_tokens_missing")
            if not sha256(receipt.get("decode_bridge_tokens_sha256")):
                errors.append("decode_bridge_token_identity_unproven")
            if not sha256(receipt.get("decode_bridge_logits_digest")):
                errors.append("decode_bridge_logits_unproven")
        if not nonnegative_int(receipt, "decode_newline_suppressions"):
            errors.append("decode_newline_discipline_unreceipted")
        configured_repetition = config.get("decode_repetition_penalty", 1.0)
        applied_repetition = receipt.get("decode_repetition_penalty_applied")
        if (
            isinstance(applied_repetition, bool)
            or not isinstance(applied_repetition, (int, float))
            or not isinstance(configured_repetition, (int, float))
            or isinstance(configured_repetition, bool)
            or abs(float(applied_repetition) - float(configured_repetition)) > 1e-9
        ):
            errors.append("decode_repetition_guard_unproven")
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
        if not isinstance(raw_flags, list) or any(not isinstance(flag, str) for flag in raw_flags):
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
            # Under verifier guidance, zero ACCEPTED steps is a legitimate
            # verified outcome (every proposal was checked and declined) —
            # the verifier evidence must exist to earn that exemption.
            verifier_evidence = receipt.get("verifier_guidance")
            verifier_ran = (
                isinstance(verifier_evidence, dict)
                and int(verifier_evidence.get("evaluations") or 0) > 0
            )
            if not positive_int(receipt, "latent_opt_steps") and not verifier_ran:
                errors.append("latent_optimization_no_accepted_steps")
            if not nonnegative_int(receipt, "latent_opt_rejected"):
                errors.append("latent_optimization_rejection_count_invalid")
            elif (
                positive_int(receipt, "latent_opt_attempts")
                and nonnegative_int(receipt, "latent_opt_steps")
                and (
                    receipt["latent_opt_attempts"]
                    != receipt["latent_opt_steps"] + receipt["latent_opt_rejected"]
                )
            ):
                errors.append("latent_optimization_accounting_mismatch")
            if receipt.get("latent_opt_budget_exhausted") is not False:
                errors.append("latent_optimization_budget_exhausted")
            if config.get("verifier_accept_non_regression") is True:
                arbitration = receipt.get("latent_opt_verifier")
                if not verifier_arbitration_valid(
                    arbitration,
                    attempts=int(receipt.get("latent_opt_attempts") or 0),
                    accepted_steps=int(receipt.get("latent_opt_steps") or 0),
                ):
                    errors.append("latent_optimization_verifier_receipt_invalid")
        if config.get("fast_weights") is True:
            if receipt.get("fast_weights_applied") is not True:
                errors.append("fast_weights_not_applied")
            erase_verdict = _integrity_verdict(receipt, "fast_weights_erased")
            if erase_verdict == "refuted":
                # The canary did not return to baseline: an adaptation is
                # still resident. This is the case that must never be
                # confused with "we did not look".
                errors.append("fast_weight_erase_refuted")
            elif erase_verdict == "unproven" and receipt.get("fast_weights_erased") is not True:
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
            loss_trail = receipt.get("fast_weight_loss_trail")
            gradient_trail = receipt.get("fast_weight_gradient_norm_trail")
            step_sizes = receipt.get("fast_weight_accepted_step_sizes")
            if receipt.get("fast_weight_optimizer") != ("rms_normalized_sgd_backtracking_v1"):
                errors.append("fast_weight_optimizer_unproven")
            if (
                not finite_number_list(loss_trail)
                or len(loss_trail) != receipt.get("fast_weight_optimized_steps", 0) + 1
                or any(
                    later >= earlier
                    for earlier, later in zip(loss_trail, loss_trail[1:], strict=False)
                )
            ):
                errors.append("fast_weight_loss_descent_unproven")
            if (
                not finite_number_list(gradient_trail)
                or len(gradient_trail) != receipt.get("fast_weight_optimization_attempts", 0)
                or any(float(value) <= 0.0 for value in gradient_trail)
            ):
                errors.append("fast_weight_gradient_evidence_invalid")
            if (
                not finite_number_list(step_sizes)
                or len(step_sizes) != receipt.get("fast_weight_optimized_steps", 0)
                or any(float(value) <= 0.0 for value in step_sizes)
            ):
                errors.append("fast_weight_step_evidence_invalid")
            if not nonnegative_int(receipt, "fast_weight_line_search_backtracks"):
                errors.append("fast_weight_line_search_evidence_invalid")
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
        visible_objective: str | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic, auditable decision for live latent routing."""

        shape = dict(prompt_shape or {})
        analyzed_shape = analyze_prompt_shape(visible_objective).to_dict()
        for key in (
            "question_parts",
            "explicit_question_marks",
            "question_like_lines",
            "connector_parts",
            "repeated_clause_parts",
            "numbered_parts",
            "imperative_parts",
        ):
            supplied = shape.get(key)
            supplied = supplied if type(supplied) is int else 0
            shape[key] = max(supplied, int(analyzed_shape.get(key) or 0))
        for key in ("prefers_extended_answer", "requires_single_reply_coverage"):
            shape[key] = bool(shape.get(key) or analyzed_shape.get(key))
        question_parts = shape.get("question_parts", 0)
        question_parts = question_parts if type(question_parts) is int else 0
        extended = bool(shape.get("prefers_extended_answer"))
        single_reply_coverage = bool(shape.get("requires_single_reply_coverage"))
        mode = str(cognitive_mode or "").strip().lower()

        depth_worthy = bool(
            explicitly_required
            or mode == "deliberate"
            or extended
            or single_reply_coverage
            or question_parts > 1
        )
        exclusion = ""
        if not foreground:
            exclusion = "not_foreground"
        elif not desktop_required:
            exclusion = "desktop_cognitive_engine_not_required"
        elif compact_contract and not depth_worthy:
            exclusion = "compact_contract"
        elif strict_output_contract:
            exclusion = "strict_output_contract"
        elif incompatible_contract:
            exclusion = "incompatible_contract"
        elif proof_or_benchmark and not explicitly_required:
            exclusion = "proof_lane_not_explicitly_opted_in"
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
            "latent_cortex_prompt_shape": shape,
            "stakes": round(max(0.55, depth_signal - 0.05), 3),
            "uncertainty": round(depth_signal, 3),
        }

    def _record_failure(self, reason: str) -> dict[str, Any]:
        self._failure_streak += 1
        self._last_refusal = str(reason or "unknown")
        return {"ok": False, "reason": self._last_refusal}

    @staticmethod
    def _visible_objective(question: str | None, messages: list | None) -> str:
        if isinstance(question, str) and question.strip():
            return question.strip()
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [
                    str(item.get("text") or "").strip()
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                rendered = "\n".join(part for part in parts if part).strip()
                if rendered:
                    return rendered
        return ""

    @staticmethod
    def _facet_reliability_weights(domain: str) -> dict[str, float] | None:
        """Foundry-calibrated facet weights; None until any facet is measured.

        weight_for stays 1.0 below 10 graded verdicts, so this returns None
        (wire unchanged) until an operator has actually graded facet
        judgments — behavior never shifts on ungraded speculation.
        """
        try:
            from core.brain.llm.latent_cortex.task_verifiers import (
                _ANSWER_FACET_HINTS,
            )
            from core.brain.verifiers.foundry import get_verifier_foundry

            foundry = get_verifier_foundry()
            weights = {
                name: float(foundry.weight_for(f"latent_facet_{name}", str(domain)))
                for name in _ANSWER_FACET_HINTS
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if all(value == 1.0 for value in weights.values()):
            return None
        return weights

    def _record_facet_judgments(self, receipt: dict[str, Any], domain: str, objective: str) -> None:
        """Feed the episode's facet judgments to the Foundry grade queue.

        Each judgment (facet, satisfied, excerpt) becomes an ungraded
        verdict an operator can grade against the excerpt — the held-out
        loop that keeps 'because'-without-explaining from ever paying.
        """
        guidance = receipt.get("verifier_guidance")
        if not isinstance(guidance, dict):
            return
        judgments = guidance.get("facet_judgments")
        if not isinstance(judgments, list) or not judgments:
            return
        try:
            import hashlib

            from core.brain.verifiers.foundry import get_verifier_foundry

            foundry = get_verifier_foundry()
            task_key = hashlib.sha256(str(objective or "").encode("utf-8")).hexdigest()[:16]
            for row in judgments[:8]:
                facet = row.get("facet") if isinstance(row, dict) else None
                if not isinstance(facet, str) or not facet:
                    continue
                satisfied = bool(row.get("satisfied"))
                foundry.record_verdict(
                    verifier=f"latent_facet_{facet}",
                    domain=str(domain),
                    hard_pass=satisfied,
                    score=1.0 if satisfied else 0.0,
                    checked=True,
                    task_key=task_key,
                    meta={"excerpt": str(row.get("excerpt") or "")[:200]},
                )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
        ) as exc:
            logger.debug("Facet judgment recording skipped: %s", exc)

    @staticmethod
    async def _broadcast_conclusion(
        result: dict[str, Any],
        *,
        objective: str,
        stakes: float,
    ) -> None:
        """Publish exactly the conclusion returned by a foreground call."""

        receipt = result.get("receipt")
        if result.get("ok") is not True or not isinstance(receipt, dict):
            return
        try:
            from core.brain.gwt_rlc_coupling import broadcast_episode_conclusion

            receipt["workspace_broadcast"] = await broadcast_episode_conclusion(
                objective,
                str(result.get("text") or ""),
                receipt,
                stakes=stakes,
            )
            result["receipt"] = receipt
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Workspace broadcast of conclusion skipped: %s", exc)

    async def deep_reason_with_acquisition(
        self,
        question: str | None = None,
        *,
        messages: list | None = None,
        orchestrator: Any = None,
        tenant_id: str = "local",
        user_id: str = "owner",
        session_id: str = "local",
        **reason_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one episode plus at most one governed retrieval continuation."""

        started = time.monotonic()
        reason_kwargs = dict(reason_kwargs)
        reason_kwargs.pop("publish_workspace_conclusion", None)
        if reason_kwargs.get("external_execution_offer") is not None:
            return self._record_failure(
                "external_execution_requires_single_episode"
            )
        foreground = reason_kwargs.get("foreground_request", True)
        first = await self.deep_reason(
            question,
            messages=messages,
            publish_workspace_conclusion=False,
            **reason_kwargs,
        )
        if first.get("ok") is not True:
            return first

        objective = self._visible_objective(question, messages)
        first_receipt = first.get("receipt")
        original_context = reason_kwargs.get("cognitive_context")
        actual_context = (
            first_receipt.get("cognitive_slots")
            if isinstance(first_receipt, dict)
            and isinstance(first_receipt.get("cognitive_slots"), list)
            and first_receipt["cognitive_slots"]
            else original_context
        )
        try:
            from core.brain.llm.latent_cortex.cognitive_acquisition import (
                build_acquisition_receipt,
                build_acquisition_request,
                build_continuation_receipt,
                validate_acquisition_receipt,
                validate_continuation_receipt,
            )

            request = build_acquisition_request(
                objective=objective,
                first_text=str(first.get("text") or ""),
                first_receipt=first_receipt,
                cognitive_context=actual_context,
            )
        except (ImportError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex.cognitive_acquisition",
                exc,
                action="retained the first proven answer after acquisition request validation failed",
                severity="warning",
            )
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first
        if request is None:
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        adaptive_acquisition: dict[str, Any] | None = None
        if isinstance(first_receipt, dict):
            execution = first_receipt.get("adaptive_compute")
            plan = execution.get("plan") if isinstance(execution, dict) else None
            if isinstance(plan, dict):
                try:
                    from core.brain.llm.latent_cortex.adaptive_compute import (
                        build_adaptive_acquisition_receipt,
                        validate_adaptive_acquisition_receipt,
                    )

                    tools = plan.get("routing", {}).get("tools", {})
                    authorized = tools.get("max_acquisitions") == 1
                    adaptive_acquisition = build_adaptive_acquisition_receipt(
                        plan=plan,
                        request_sha256=str(request["request_sha256"]),
                        attempted=authorized,
                    )
                    validate_adaptive_acquisition_receipt(adaptive_acquisition)
                    first_receipt["adaptive_acquisition"] = adaptive_acquisition
                    if not authorized:
                        self._last_receipt = first_receipt
                        if foreground is True:
                            await self._broadcast_conclusion(
                                first,
                                objective=objective,
                                stakes=float(reason_kwargs.get("stakes", 0.5)),
                            )
                        return first
                except (ImportError, AttributeError, TypeError, ValueError):
                    return self._record_failure(
                        "adaptive_acquisition_authority_invalid"
                    )

        acquisition_started = time.monotonic()
        try:
            from core.brain.cognitive_ingress import (
                assemble_cognitive_ingress_async,
                cognitive_context_items,
            )

            ingress = await assemble_cognitive_ingress_async(
                self.orchestrator if orchestrator is None else orchestrator,
                objective,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                retrieval_query=str(request["retrieval_query"]),
                acquisition_source=(
                    "memory"
                    if request["action"] == "search_memory"
                    else "reference"
                ),
            )
            acquired_context = cognitive_context_items(ingress) or None
            acquisition = build_acquisition_receipt(
                request,
                acquired_context=acquired_context,
                ingress_receipt=ingress.to_receipt(),
                elapsed_s=time.monotonic() - acquisition_started,
            )
            validate_acquisition_receipt(acquisition, request=request)
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            record_degradation(
                "latent_cortex.cognitive_acquisition",
                exc,
                action="retained the first proven answer after the bounded acquisition failed",
                severity="warning",
            )
            acquisition = build_acquisition_receipt(
                request,
                acquired_context=None,
                ingress_receipt={"status": "failed", "error_type": type(exc).__name__},
                elapsed_s=min(30.0, time.monotonic() - acquisition_started),
                error_code=f"acquisition_{type(exc).__name__.lower()}",
            )
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="acquisition_failed",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        if acquisition["status"] != "completed_new_context":
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="no_new_context",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        timeout_s = float(reason_kwargs.get("timeout_s", 300.0))
        remaining_s = timeout_s - (time.monotonic() - started)
        if remaining_s < 15.0:
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="budget_insufficient",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        second_kwargs = dict(reason_kwargs)
        second_kwargs.update(
            {
                "stakes": max(
                    float(reason_kwargs.get("stakes", 0.5)),
                    float(ingress.stakes),
                ),
                "uncertainty": float(ingress.uncertainty),
                "timeout_s": remaining_s,
                "cognitive_context": acquired_context,
                "epistemic_genesis": ingress.epistemic_genesis,
                "epistemic_state": ingress.epistemic_state,
                "selective_memory_result": ingress.memory_result,
            }
        )
        second = await self.deep_reason(
            question,
            messages=messages,
            publish_workspace_conclusion=bool(foreground),
            **second_kwargs,
        )
        returned_round = 2 if second.get("ok") is True else 1
        returned = second if returned_round == 2 else first
        continuation = build_continuation_receipt(
            request,
            acquisition,
            first_result=first,
            second_result=second,
            returned_round=returned_round,
            continuation_reason=(
                "second_episode_succeeded"
                if returned_round == 2
                else "second_episode_failed"
            ),
        )
        validate_continuation_receipt(continuation)
        returned_receipt = returned.get("receipt")
        if isinstance(returned_receipt, dict):
            returned_receipt["cognitive_acquisition"] = continuation
            if adaptive_acquisition is not None:
                returned_receipt["adaptive_acquisition"] = adaptive_acquisition
            self._last_receipt = returned_receipt
        if returned_round == 1 and foreground is True:
            await self._broadcast_conclusion(
                first,
                objective=objective,
                stakes=float(reason_kwargs.get("stakes", 0.5)),
            )
        return returned

    async def run_action_state_episode(
        self,
        *,
        prompt: str | None = None,
        messages: list | None = None,
        domain: str,
        config: dict[str, Any],
        budget: dict[str, Any],
        cognitive_context: list | None,
        action_policy_evidence: dict[str, Any],
        action_state_runtime: dict[str, Any],
        action_intervention: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Run the claim-grade first-action lane without product-policy drift.

        The campaign runner supplies the complete signed public experiment
        contract.  The service deliberately does not synthesize body, Will,
        memory, or controller inputs on this lane because doing so after the
        runner froze its request would change the paired prestate.
        """

        if not isinstance(action_state_runtime, dict):
            return self._record_failure("invalid_action_state_runtime")
        if action_intervention is not None and not isinstance(
            action_intervention, dict
        ):
            return self._record_failure("invalid_action_intervention")
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            result = await client.latent_reason_async(
                prompt=prompt,
                messages=messages,
                config=config,
                budget=budget,
                domain=domain,
                timeout_s=timeout_s,
                foreground_request=False,
                cognitive_context=cognitive_context,
                action_policy_evidence=action_policy_evidence,
                action_intervention=action_intervention,
                action_state_runtime=action_state_runtime,
                external_execution_offer=external_execution_offer,
                verifier_guidance=True,
            )
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex.action_state_lane",
                exc,
                action="contained claim-grade action-state lane failure",
                severity="error",
            )
            return self._record_failure(
                f"action_state_runtime_failed:{type(exc).__name__}"
            )
        if result.get("ok") is not True:
            self._last_failure_receipt = dict(result.get("receipt") or {})
            return self._record_failure(
                str(result.get("reason") or "action_state_runtime_failed")
            )
        self._last_receipt = dict(result.get("receipt") or {})
        self._last_progress = dict(result.get("progress") or {})
        return result

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
        cognitive_context: list | None = None,
        epistemic_genesis: Any | None = None,
        epistemic_state: Any | None = None,
        selective_memory_result: Any | None = None,
        publish_workspace_conclusion: bool = True,
        external_execution_offer: dict[str, Any] | None = None,
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
            recurrent_loops = runtime_controls.get("clean_user_surface_recurrent_loops")
            steering_alpha = runtime_controls.get("clean_user_surface_steering_alpha")
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
        if type(publish_workspace_conclusion) is not bool:
            return self._record_failure("invalid_publish_workspace_conclusion")
        try:
            from core.brain.llm.latent_cortex.cognitive_context import (
                normalize_cognitive_context,
            )

            cognitive_context = normalize_cognitive_context(cognitive_context) or None
        except (TypeError, ValueError):
            return self._record_failure("invalid_cognitive_context")
        try:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            external_execution_offer = (
                validate_external_execution_offer(external_execution_offer)
                if external_execution_offer is not None
                else None
            )
        except (ImportError, TypeError, ValueError):
            return self._record_failure("invalid_external_execution_offer")
        memory_context_present = any(
            isinstance(entry, dict) and entry.get("context_role") == "memory_observation"
            for entry in (cognitive_context or [])
        )
        if (
            memory_context_present
            or epistemic_genesis is not None
            or epistemic_state is not None
            or selective_memory_result is not None
        ):
            try:
                from core.brain.llm.latent_cortex.epistemic_memory import (
                    SelectiveMemoryResult,
                    validate_memory_context_items,
                )
                from core.brain.llm.latent_cortex.epistemic_state import EpistemicState

                if (
                    not isinstance(epistemic_genesis, EpistemicState)
                    or epistemic_genesis.version != 0
                    or not isinstance(epistemic_state, EpistemicState)
                    or not isinstance(selective_memory_result, SelectiveMemoryResult)
                    or epistemic_state.episode_id != epistemic_genesis.episode_id
                    or epistemic_state.problem != epistemic_genesis.problem
                    or epistemic_state.parent_sha256 != epistemic_genesis.state_sha256
                ):
                    return self._record_failure("invalid_epistemic_memory_authority")
                visible_objective = self._visible_objective(question, messages)
                if (
                    epistemic_state.problem.objective_sha256
                    != hashlib.sha256(visible_objective.encode("utf-8")).hexdigest()
                ):
                    return self._record_failure("epistemic_objective_mismatch")
                validate_memory_context_items(
                    epistemic_state,
                    selective_memory_result,
                    cognitive_context or [],
                )
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.warning("Epistemic memory authority rejected: %s", exc)
                return self._record_failure("invalid_epistemic_memory_authority")
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_timeout")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            return self._record_failure("invalid_timeout")
        try:
            stakes = self._unit_signal(stakes, name="stakes")
            uncertainty = self._unit_signal(uncertainty, name="uncertainty")
        except ValueError:
            return self._record_failure("invalid_cognitive_economy")
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.runtime.errors import DependencyUnavailable, ModelUnavailable

            client = get_mlx_client()
        except (
            ImportError,
            DependencyUnavailable,
            ModelUnavailable,
            OSError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode: resident model client unavailable",
            )
            return self._record_failure(f"client_unavailable:{type(exc).__name__}")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action=(
                    "refused latent episode whose resident model client "
                    "failed an integrity check"
                ),
                severity="degraded",
            )
            return self._record_failure(
                f"client_integrity_failure:{type(exc).__name__}"
            )
        if client is None:
            return self._record_failure("no_resident_model")
        worker_identity: dict[str, Any] = {}
        identity_getter = getattr(client, "get_worker_identity_snapshot", None)
        if callable(identity_getter):
            try:
                candidate_identity = identity_getter()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                candidate_identity = {}
            if isinstance(candidate_identity, dict):
                worker_identity = dict(candidate_identity)
        try:
            model_parameter_count = int(worker_identity.get("worker_model_parameter_count") or 0)
        except (TypeError, ValueError, OverflowError):
            model_parameter_count = 0
        try:
            visible_objective = self._visible_objective(question, messages)
            config, budget = self.allocate(
                stakes=stakes,
                uncertainty=uncertainty,
                objective=visible_objective,
                model_parameter_count=max(0, model_parameter_count),
                foreground_request=foreground_request,
                timeout_s=timeout_s,
            )
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_cognitive_economy")
        allocation_profile = str(self._last_allocation.get("allocation_profile") or "")
        adaptive_plan: dict[str, Any] | None = dict(
            self._last_allocation.get("adaptive_compute") or {}
        )
        if config_overrides is not None:
            config.update(dict(config_overrides))
        if not _controller_accepts_overrides(config_overrides):
            adaptive_plan = None
            self._last_allocation["adaptive_compute_execution"] = (
                "explicit_structural_override"
            )
        # Learned execution controller: evidence-gated arm selection over
        # the base allocation (deeper recurrence / wider branches /
        # probe-guided bytecode / lean ΔW). Exploits only after Wilson
        # separation on graded outcomes in this context bucket; explores
        # sparsely; never touches explicit operator overrides.
        controller_decision: dict[str, Any] | None = None
        action_policy_evidence: dict[str, Any] | None = None
        if foreground_request and _controller_accepts_overrides(config_overrides):
            try:
                from core.brain.llm.latent_cortex.execution_controller import (
                    controller_enabled,
                    get_execution_controller,
                )

                if not controller_enabled():
                    raise RuntimeError("execution controller disabled")
                controller = get_execution_controller()
                controller_decision = controller.choose(
                    objective=self._visible_objective(question, messages),
                    domain=domain,
                    stakes=stakes,
                    uncertainty=uncertainty,
                )
                snapshot_builder = getattr(
                    controller,
                    "action_evidence_snapshot",
                    None,
                )
                if callable(snapshot_builder):
                    action_policy_evidence = snapshot_builder(
                        bucket=str(controller_decision["bucket"])
                    )
                else:
                    from core.brain.llm.latent_cortex.value_of_computation import (
                        build_evidence_snapshot,
                    )

                    action_policy_evidence = build_evidence_snapshot(
                        bucket=str(controller_decision["bucket"]),
                        cells={},
                    )
                if controller_decision["arm"] != "base":
                    # CP126 ea828a97: apply_arm could silently leave the config
                    # unchanged (e.g. probe-guided bytecode with no recurrent
                    # region) while the decision still named the treatment, so
                    # the outcome was credited to bytecode that never ran. Take
                    # the application RECEIPT and fall the decision back to
                    # base when the arm did not actually apply.
                    application = controller.apply_arm_receipt(
                        controller_decision["arm"],
                        config,
                        recurrent_region=(
                            (16, 48)
                            if allocation_profile == "resident_32b_interactive_full_stack_v2"
                            else None
                        ),
                    )
                    config = application["config"]
                    controller_decision["applied"] = bool(application["applied"])
                    controller_decision["application_reason"] = str(application.get("reason") or "")
                    if not application["applied"]:
                        controller_decision["arm"] = str(application.get("effective_arm") or "base")
                self._last_allocation["execution_controller"] = controller_decision
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
            ) as exc:
                logger.debug("Execution controller unavailable: %s", exc)
                controller_decision = None
                action_policy_evidence = None
        if adaptive_plan is not None:
            try:
                from core.brain.llm.latent_cortex.adaptive_compute import (
                    enforce_adaptive_compute_limits,
                )

                config = enforce_adaptive_compute_limits(config, adaptive_plan)
                self._last_allocation["adaptive_compute_execution"] = "enforced"
            except (ImportError, TypeError, ValueError, OverflowError):
                return self._record_failure("adaptive_compute_enforcement_failed")
        if external_execution_offer is not None and (
            controller_decision is None or action_policy_evidence is None
        ):
            return self._record_failure(
                "external_execution_controller_unavailable"
            )
        try:
            from core.brain.llm.latent_cortex.branches import BRANCH_ROLES
            from core.brain.llm.latent_cortex.correlated_support import (
                get_branch_correlation_ledger,
            )
            from core.brain.llm.latent_cortex.execution_controller import context_bucket

            branch_count = int(config.get("n_branches") or 1)
            correlation_roles = list(BRANCH_ROLES[:branch_count])
            correlation_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            correlation_ledger = get_branch_correlation_ledger()
            config["branch_correlation_evidence"] = correlation_ledger.evidence(
                bucket=correlation_bucket,
                roles=correlation_roles,
            )
            self._last_allocation["correlated_support"] = {
                **correlation_ledger.status(),
                "bucket": correlation_bucket,
                "roles": correlation_roles,
                "evidence_state": config["branch_correlation_evidence"]["evidence_state"],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Branch correlation evidence unavailable: %s", exc)
            config["branch_correlation_evidence"] = None
        try:
            from core.brain.llm.latent_cortex.execution_controller import context_bucket
            from core.brain.llm.latent_cortex.verifier_fusion import (
                get_verifier_fusion_ledger,
            )

            verifier_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            verifier_ledger = get_verifier_fusion_ledger()
            config["verifier_fusion_evidence"] = verifier_ledger.evidence(
                bucket=verifier_bucket
            )
            self._last_allocation["verifier_fusion"] = {
                **verifier_ledger.status(),
                "bucket": verifier_bucket,
                "evidence_state": config["verifier_fusion_evidence"][
                    "evidence_state"
                ],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Verifier fusion evidence unavailable: %s", exc)
            config["verifier_fusion_evidence"] = None
        if allocation_profile == "resident_32b_interactive_full_stack_v2":
            try:
                from core.brain.llm.latent_cortex.critic_identity import (
                    build_critic_source_identity,
                    build_generator_function_identity,
                    get_critic_blind_spot_ledger,
                )
                from core.brain.llm.latent_cortex.execution_controller import (
                    context_bucket,
                )

                generator_identity = build_generator_function_identity(worker_identity)
                critic_source = build_critic_source_identity()
                critic_ledger = get_critic_blind_spot_ledger()
                critic_bucket = (
                    str(controller_decision.get("bucket") or "")
                    if controller_decision is not None
                    else context_bucket(
                        self._visible_objective(question, messages),
                        domain,
                        stakes,
                        uncertainty,
                    )
                )
                config["critic_blind_spot_evidence"] = critic_ledger.evidence(
                    bucket=critic_bucket,
                    generator_function_sha256=generator_identity["function_sha256"],
                    critic_function_sha256=critic_source["source_closure_sha256"],
                )
                self._last_allocation["critic_blind_spots"] = {
                    **critic_ledger.status(),
                    "bucket": critic_bucket,
                    "evidence_state": config["critic_blind_spot_evidence"]["evidence_state"],
                    "critic_reliability_admitted": config["critic_blind_spot_evidence"][
                        "critic_reliability_admitted"
                    ],
                }
            except (
                ImportError,
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.error("Critic identity/evidence unavailable: %s", exc)
                return self._record_failure(
                    f"critic_identity_evidence_unavailable:{type(exc).__name__}"
                )
        if allocation_profile == "resident_32b_interactive_full_stack_v2":
            try:
                requested_decode_tokens = int(config.get("decode_max_tokens") or 256)
            except (TypeError, ValueError, OverflowError):
                return self._record_failure("invalid_decode_token_override")
            # Compound-aware answer surface: the SAME facet definition the
            # product-quality gate judges by decides how much room and how
            # much sampling discipline the answer gets. CP103's live turn
            # proved a 4-facet request cannot earn the gate inside 256 tokens
            # at persona-lane temperature — the episode was mechanically
            # complete and the ANSWER SURFACE was the only failing stage.
            from core.brain.llm.latent_cortex.output_quality import request_facets

            objective_facets = request_facets(self._visible_objective(question, messages))
            compound_objective = len(objective_facets) >= 2
            if compound_objective:
                config["decode_max_tokens"] = max(
                    256,
                    min(384, max(requested_decode_tokens, 320)),
                )
                config["decode_bridge_policy"] = "assistant_answer_v3"
                # EOS floor: a compound answer abandoned 16 tokens in is
                # sampling variance, not a decision (CP116 live evidence).
                config["decode_min_tokens"] = 96
                # Coverage determinism: compound answers must satisfy every
                # requested facet inside a bounded budget; persona-lane
                # temperature (0.58) makes that a coin flip (CP113-117 each
                # failed on a different sampling tail). CP105's lesson was
                # that a temperature clamp WITHOUT a degeneration guard
                # loops; the repetition penalty, EOS floor, and newline
                # discipline now make low-temperature decoding safe.
                try:
                    requested_temperature = float(config.get("decode_temperature") or 0.0)
                except (TypeError, ValueError, OverflowError):
                    return self._record_failure("invalid_decode_temperature_override")
                config["decode_temperature"] = min(0.3, max(0.0, requested_temperature))
            else:
                config["decode_max_tokens"] = max(
                    64,
                    min(256, requested_decode_tokens),
                )
                config["decode_bridge_policy"] = "assistant_answer_v1"
                config["decode_min_tokens"] = 48
            # Degeneration guard for every resident answer: CP105's live turn
            # proved a repetition loop survives temperature tuning (one line
            # ~80 times at t=0.35, trigram diversity 0.012). The persona
            # lane's temperature is kept; the CTRL-style penalty is what
            # actually prevents loops.
            config["decode_repetition_penalty"] = 1.25
            config["decode_repetition_window"] = 72
            if compound_objective:
                # The 105s wall-clock floor was tuned for 256-token answers.
                # A 384-token compound answer measures ≈116s on the resident
                # 32B; grant up to 150s but never exceed what the owner's
                # timeout can actually wait for.
                budget["wall_clock_s"] = min(
                    max(150.0, float(budget.get("wall_clock_s") or 0.0)),
                    max(15.0, float(timeout_s) - 8.0),
                )
                # Fit the answer surface to the wall clock actually granted.
                # CP120 measured ~100s before final decode because five
                # 48-token verifier previews dominated the episode. The v2
                # interactive profile shortens those previews and reuses the
                # verified winner score; a conservative 65s still reserves
                # prefill, all causal mechanisms, bridge, and cleanup. Never
                # promise a 384-token surface the owner cannot finish.
                affordable_tokens = int((float(budget["wall_clock_s"]) - 65.0) / 0.26)
                config["decode_max_tokens"] = max(
                    128, min(int(config["decode_max_tokens"]), affordable_tokens)
                )
            self._last_allocation["objective_facets"] = list(objective_facets)
            self._last_allocation["compound_objective"] = compound_objective
        if require_full_stack:
            config["latent_opt"] = True
            config["latent_opt_control"] = False
            config["fast_weights"] = True
        self._last_allocation["config"] = dict(config)
        self._last_allocation["budget"] = dict(budget)
        if epistemic_state is not None:
            self._last_allocation["epistemic_state"] = {
                "schema": epistemic_state.schema,
                "episode_id": epistemic_state.episode_id,
                "version": epistemic_state.version,
                "state_sha256": epistemic_state.state_sha256,
                "memory_result_sha256": selective_memory_result.result_sha256,
            }

        try:
            from core.brain.llm_health_router import (
                acquire_external_generation_gate_lease,
                release_external_generation_gate_lease,
            )
            from core.runtime.errors import DependencyUnavailable

            generation_lease_id = await acquire_external_generation_gate_lease(
                owner=(
                    "latent_cortex_foreground:episode"
                    if foreground_request
                    else "latent_cortex_lab:episode"
                ),
                timeout_s=timeout_s + 10.0,
                wait_s=min(5.0, timeout_s),
            )
        except (
            ImportError,
            DependencyUnavailable,
            OSError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode whose process-wide generation lease was unavailable",
                severity="warning",
            )
            return self._record_failure(f"generation_lease_unavailable:{type(exc).__name__}")
        except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action=(
                    "refused latent episode whose generation lease path "
                    "failed an integrity check"
                ),
                severity="degraded",
            )
            return self._record_failure(
                f"generation_lease_integrity_failure:{type(exc).__name__}"
            )
        if generation_lease_id is None:
            return self._record_failure("generation_gate_busy")

        # GWT→RLC coupling: the mind's live broadcast and its strongest
        # competing coalitions seed identifiable thought slots alongside the
        # organ items — deliberation runs over what consciousness is actually
        # about, not just the prompt. Organ items keep priority; coalitions
        # only fill remaining slots. Lab/background episodes stay decoupled:
        # live workspace state would confound controlled experiments.
        if foreground_request:
            try:
                from core.brain.gwt_rlc_coupling import merge_cognitive_context

                cognitive_context = merge_cognitive_context(cognitive_context)
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.debug("Workspace coalition merge skipped: %s", exc)

        operation_lease = None
        operation_authority: dict[str, Any] | None = None
        operation_cost_receipt: dict[str, Any] = {}
        action_transitions: list[dict[str, Any]] = []

        async def complete_runtime_operation(
            outcome: Any,
            *,
            worker_receipt: Any,
            failure_code: str = "",
        ) -> dict[str, Any] | None:
            nonlocal operation_cost_receipt
            if operation_lease is None:
                return None
            from core.brain.llm.latent_cortex.epistemic_runtime import (
                measured_operation_cost,
            )
            from core.brain.llm.latent_cortex.epistemic_state import (
                OperationOutcome,
            )

            terminal_outcome = outcome
            terminal_failure = failure_code
            journal_action_transitions: tuple[dict[str, Any], ...] = tuple(action_transitions)
            action_costs: tuple[float, ...] = ()
            try:
                cost, operation_cost_receipt = measured_operation_cost(
                    worker_receipt,
                    requested_budget=budget,
                    state=operation_lease.state,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                terminal_outcome = OperationOutcome.FAILED
                terminal_failure = "compute_receipt_invalid"
                cost = 0.0
                operation_cost_receipt = {
                    "basis": "invalid_worker_compute_receipt",
                    "error_type": type(exc).__name__,
                }
                journal_action_transitions = ()
            if journal_action_transitions:
                remaining_state_budget = max(
                    0.0,
                    operation_lease.state.budget.total - operation_lease.state.budget.used,
                )
                mutable_action_costs = [
                    float(row["metrics"]["cost"]) * remaining_state_budget
                    for row in journal_action_transitions
                ]
                action_cost_total = math.fsum(mutable_action_costs)
                rounding_tolerance = max(1e-10, remaining_state_budget * 1e-7)
                if action_cost_total > cost + rounding_tolerance:
                    terminal_outcome = OperationOutcome.FAILED
                    terminal_failure = "compute_receipt_invalid"
                    operation_cost_receipt["action_cost_error"] = (
                        "action_transition_cost_exceeds_worker_total"
                    )
                    journal_action_transitions = ()
                    mutable_action_costs = []
                elif action_cost_total > cost and mutable_action_costs:
                    mutable_action_costs[-1] = max(
                        0.0,
                        mutable_action_costs[-1] - (action_cost_total - cost),
                    )
                action_costs = tuple(mutable_action_costs)
                operation_cost_receipt["action_state_cost"] = round(math.fsum(action_costs), 12)
                operation_cost_receipt["action_operation_count"] = len(journal_action_transitions)
            if terminal_outcome is not OperationOutcome.SUCCEEDED and not terminal_failure:
                terminal_failure = "worker_operation_failed"
            detail = (
                f"worker outcome={terminal_outcome.value}; "
                f"cost basis={operation_cost_receipt.get('basis', 'unmeasured')}"
            )
            await asyncio.to_thread(
                operation_lease.complete,
                outcome=terminal_outcome,
                cost=cost,
                action_transitions=journal_action_transitions,
                action_costs=action_costs,
                failure_code=terminal_failure,
                detail=detail,
            )
            receipt = operation_lease.to_receipt()
            receipt["compute"] = dict(operation_cost_receipt)
            return receipt

        self._episodes += 1
        self._last_attempt_at = time.time()
        self._last_progress = {}
        self._last_failure_receipt = {}
        started = time.monotonic()
        try:
            if controller_decision is not None:
                try:
                    from core.brain.llm.latent_cortex.epistemic_memory import (
                        validate_memory_context_items,
                    )
                    from core.brain.llm.latent_cortex.epistemic_runtime import (
                        RuntimeOperationLease,
                    )
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        ComputeBudgetState,
                        EpistemicState,
                        ProblemFrame,
                    )
                    from core.config import DATA_DIR

                    if epistemic_state is None:
                        objective = self._visible_objective(question, messages)
                        epistemic_genesis = EpistemicState.genesis(
                            episode_id=f"rlc-runtime-{uuid.uuid4().hex[:24]}",
                            problem=ProblemFrame.create(objective),
                            budget=ComputeBudgetState(total=1.0),
                        )
                        epistemic_state = epistemic_genesis

                    operation_lease = await asyncio.to_thread(
                        RuntimeOperationLease.begin,
                        genesis=epistemic_genesis,
                        state=epistemic_state,
                        decision=controller_decision,
                        config=config,
                        budget=budget,
                        action_policy_evidence=action_policy_evidence,
                        external_execution_offer=external_execution_offer,
                        root=Path(DATA_DIR) / "latent_cortex" / "epistemic_runtime",
                    )
                    operation_authority = dict(operation_lease.authority)
                    epistemic_state = operation_lease.state
                    rebound_context = []
                    for entry in cognitive_context or []:
                        if (
                            isinstance(entry, dict)
                            and entry.get("context_role") == "memory_observation"
                        ):
                            rebound_context.append(
                                {
                                    **entry,
                                    "epistemic_state_sha256": epistemic_state.state_sha256,
                                }
                            )
                        else:
                            rebound_context.append(entry)
                    cognitive_context = rebound_context or None
                    if selective_memory_result is not None:
                        validate_memory_context_items(
                            epistemic_state,
                            selective_memory_result,
                            cognitive_context or [],
                        )
                    self._last_allocation["runtime_operation"] = {
                        "schema": operation_authority["schema"],
                        "operation_id": operation_authority["operation_id"],
                        "operation_kind": operation_authority["operation_kind"],
                        "attempt_sha256": operation_authority["attempt_sha256"],
                        "admitted_state_sha256": operation_authority["admitted_state_sha256"],
                    }
                except (
                    ImportError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    record_degradation(
                        "latent_cortex.operation_admission",
                        exc,
                        action=(
                            "refused recurrent compute whose operation could not "
                            "be admitted and journaled"
                        ),
                        severity="error",
                    )
                    return self._record_failure(
                        f"runtime_operation_admission_failed:{type(exc).__name__}"
                    )
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
                    # Typed cognitive-slot ingress: organ content (memory,
                    # goals, world model, interoception, self-model) seeds
                    # identifiable workspace slots inside the episode.
                    cognitive_context=cognitive_context,
                    operation_authority=operation_authority,
                    action_policy_evidence=action_policy_evidence,
                    external_execution_offer=external_execution_offer,
                    # Foreground resident episodes select branches and accept
                    # latent-opt proposals by deterministic task-typed checks
                    # (arithmetic recomputation, code syntax, facet coverage,
                    # grounding) — verified correctness, not convergence.
                    verifier_guidance=(
                        allocation_profile == "resident_32b_interactive_full_stack_v2"
                    ),
                    # Held-out calibration: facets whose cue-detectors humans
                    # keep overruling (Foundry grades) are muted inside the
                    # episode's verifier. None until grading evidence exists.
                    facet_reliability=self._facet_reliability_weights(domain),
                )
            except asyncio.CancelledError:
                if operation_lease is not None:
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        OperationOutcome,
                    )

                    await complete_runtime_operation(
                        OperationOutcome.CANCELLED,
                        worker_receipt={},
                        failure_code="caller_cancelled",
                    )
                raise
            except (
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
                TimeoutError,
            ) as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="contained resident latent episode failure and preserved caller fallback",
                    severity="warning",
                )
                self._last_latency_s = time.monotonic() - started
                if operation_lease is not None:
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        OperationOutcome,
                    )

                    await complete_runtime_operation(
                        OperationOutcome.FAILED,
                        worker_receipt={},
                        failure_code="client_exception",
                    )
                return self._record_failure(f"client_error:{type(exc).__name__}")
        finally:
            release_external_generation_gate_lease(generation_lease_id)
        elapsed = time.monotonic() - started
        self._last_latency_s = elapsed
        if not isinstance(result, dict):
            if operation_lease is not None:
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationOutcome,
                )

                await complete_runtime_operation(
                    OperationOutcome.FAILED,
                    worker_receipt={},
                    failure_code="invalid_client_response",
                )
            return self._record_failure("invalid_client_response")
        raw_receipt = result.get("receipt")
        result_receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else {}
        action_policy_matches = action_policy_evidence is None
        if action_policy_evidence is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_action_trace,
                )

                policy_receipt = result_receipt.get("value_of_computation")
                raw_trace = result_receipt.get("cognitive_action_trace")
                policy_fields = {
                    "schema",
                    "bucket",
                    "snapshot_sha256",
                    "active",
                    "executors",
                    "actions_selected",
                    "checked_transitions",
                    "selected_actions",
                }
                if (
                    not isinstance(policy_receipt, dict)
                    or set(policy_receipt) != policy_fields
                    or policy_receipt.get("schema") != action_policy_evidence["schema"]
                    or policy_receipt.get("snapshot_sha256")
                    != action_policy_evidence["snapshot_sha256"]
                    or policy_receipt.get("bucket") != action_policy_evidence["bucket"]
                    or policy_receipt.get("active") is not True
                    or not isinstance(raw_trace, list)
                    or not raw_trace
                    or policy_receipt.get("actions_selected") != len(raw_trace)
                ):
                    raise ValueError("worker action policy receipt is incomplete")
                raw_executors = policy_receipt.get("executors")
                if (
                    not isinstance(raw_executors, list)
                    or not raw_executors
                    or len(raw_executors) > len(OperationKind)
                ):
                    raise ValueError("worker action executor inventory is invalid")
                try:
                    executors = tuple(OperationKind(item) for item in raw_executors)
                except (TypeError, ValueError) as exc:
                    raise ValueError("worker action executor inventory is invalid") from exc
                execute_advertised = OperationKind.EXECUTE in executors
                if (
                    len(set(executors)) != len(executors)
                    or execute_advertised != (external_execution_offer is not None)
                ):
                    raise ValueError("worker action executor inventory is invalid")
                validated_trace = validate_action_trace(
                    raw_trace,
                    evidence_snapshot=action_policy_evidence,
                    executors=executors,
                )
                for validated_row in validated_trace["rows"]:
                    decision = validated_row["decision"]
                    transition = validated_row["transition"]
                    if (
                        transition["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                        or transition["bucket"] != action_policy_evidence["bucket"]
                        or decision["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                        or decision["bucket"] != action_policy_evidence["bucket"]
                    ):
                        raise ValueError("worker action transition authority differs")
                    action_transitions.append(transition)
                selected_actions = [row["action"] for row in action_transitions]
                checked_transitions = sum(int(row["checked"]) for row in action_transitions)
                if (
                    validated_trace["selected_actions"] != selected_actions
                    or policy_receipt.get("selected_actions") != selected_actions
                    or policy_receipt.get("checked_transitions") != checked_transitions
                ):
                    raise ValueError("worker action policy summary differs from trace")
                raw_handoff = result_receipt.get("external_execution_handoff")
                if external_execution_offer is not None:
                    from core.brain.llm.latent_cortex.external_execution import (
                        validate_external_execution_handoff,
                    )

                    validate_external_execution_handoff(
                        raw_handoff,
                        offer=external_execution_offer,
                        cognitive_action_trace=raw_trace,
                    )
                elif raw_handoff not in ({}, None):
                    raise ValueError("worker emitted unoffered external execution handoff")
                action_policy_matches = True
            except (ImportError, TypeError, ValueError):
                action_transitions.clear()
                action_policy_matches = False
        contract_errors: list[str] = []
        quality_receipt: dict[str, Any] | None = None
        private_answer_replacement = result.pop(
            "answer_replacement_private",
            None,
        )
        visible_objective = self._visible_objective(question, messages)
        if result.get("ok") is True:
            contract_errors = self._receipt_contract_errors(
                raw_receipt,
                config,
                runtime_controls,
                worker_identity,
                result.get("tokens"),
                domain,
                output_text=result.get("text"),
                answer_replacement_private=private_answer_replacement,
                expected_objective=visible_objective,
            )
            if not contract_errors and adaptive_plan is not None:
                try:
                    from core.brain.llm.latent_cortex.adaptive_compute import (
                        build_adaptive_execution_receipt,
                        validate_adaptive_execution_receipt,
                    )

                    adaptive_execution = build_adaptive_execution_receipt(
                            plan=adaptive_plan,
                            config=config,
                            budget=budget,
                            worker_receipt=result_receipt,
                    )
                    validate_adaptive_execution_receipt(adaptive_execution)
                    result_receipt["adaptive_compute"] = adaptive_execution
                except (ImportError, TypeError, ValueError, OverflowError):
                    contract_errors.append("adaptive_compute_execution_unproven")
            if not contract_errors:
                quality_receipt = evaluate_latent_output(
                    result.get("text"),
                    generated_tokens=result_receipt.get("decode_generated_tokens"),
                    termination=result_receipt.get("decode_termination"),
                    objective=visible_objective,
                )
                result_receipt["output_quality"] = quality_receipt
                result["receipt"] = result_receipt
        if operation_lease is not None:
            from core.brain.llm.latent_cortex.epistemic_state import (
                OperationOutcome,
            )

            worker_authority = result_receipt.get("runtime_operation_authority")
            authority_matches = worker_authority == operation_authority
            worker_succeeded = (
                result.get("ok") is True
                and authority_matches
                and action_policy_matches
                and not contract_errors
                and quality_receipt is not None
                and quality_receipt.get("passed") is True
            )
            try:
                operation_receipt = await complete_runtime_operation(
                    (OperationOutcome.SUCCEEDED if worker_succeeded else OperationOutcome.FAILED),
                    worker_receipt=result_receipt,
                    failure_code=(
                        ""
                        if worker_succeeded
                        else (
                            "operation_authority_mismatch"
                            if not authority_matches
                            else (
                                "action_policy_receipt_mismatch"
                                if result.get("ok") is True and not action_policy_matches
                                else (
                                    "worker_receipt_contract_failed"
                                    if contract_errors
                                    else (
                                        "output_quality_failed"
                                        if quality_receipt is not None
                                        and quality_receipt.get("passed") is not True
                                        else "worker_operation_failed"
                                    )
                                )
                            )
                        )
                    ),
                )
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "latent_cortex.operation_completion",
                    exc,
                    action=(
                        "refused recurrent output whose terminal operation state "
                        "could not be journaled"
                    ),
                    severity="error",
                )
                return self._record_failure(
                    f"runtime_operation_completion_failed:{type(exc).__name__}"
                )
            epistemic_state = operation_lease.state
            result_receipt["epistemic_operation"] = operation_receipt
            result["receipt"] = result_receipt
            if not authority_matches:
                reason = "runtime_operation_authority_mismatch"
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            if result.get("ok") is True and not action_policy_matches:
                reason = "runtime_action_policy_receipt_mismatch"
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
        if action_policy_matches and action_policy_evidence is not None:
            result_receipt["host_action_policy_evidence"] = dict(
                action_policy_evidence
            )
            result["receipt"] = result_receipt
        if epistemic_state is not None:
            epistemic_state_receipt = {
                "schema": epistemic_state.schema,
                "episode_id": epistemic_state.episode_id,
                "version": epistemic_state.version,
                "state_sha256": epistemic_state.state_sha256,
            }
            if selective_memory_result is not None:
                epistemic_state_receipt.update(
                    {
                        "memory_result_sha256": selective_memory_result.result_sha256,
                        "memory_evidence_ids": [
                            candidate.evidence_id
                            for candidate in selective_memory_result.candidates
                        ],
                    }
                )
            result_receipt["epistemic_state"] = epistemic_state_receipt
        raw_progress = result.get("progress")
        self._last_progress = dict(raw_progress) if isinstance(raw_progress, dict) else {}
        if result.get("ok"):
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
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            if quality_receipt is None or quality_receipt.get("passed") is not True:
                reasons = (
                    quality_receipt.get("reasons")
                    if quality_receipt is not None
                    else ["missing_quality_receipt"]
                )
                reason = "output_quality_failed:" + ",".join(
                    str(item) for item in reasons or ["unknown"]
                )
                record_degradation(
                    "latent_cortex.output_quality",
                    RuntimeError(reason),
                    action=(
                        "refused a mechanically complete latent episode whose visible answer did not satisfy the product contract"
                    ),
                    severity="degraded",
                )
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            self._ok_episodes += 1
            self._failure_streak = 0
            self._last_refusal = ""
            self._last_success_at = time.time()
            # The grading path for cognition.effort. Until this existed the
            # control point was registered, recording, and permanently
            # unpromotable: nothing ever called note_grade(), so every effort
            # decision resolved UNOBSERVED. It reports the SAME independently
            # graded outcome the bandit is allowed to learn from — a verifier's
            # judgement of the answer — and nothing else. If no verifier graded
            # this episode (outcome_checked is False), nothing is reported and
            # the decision stays honestly UNOBSERVED rather than being taught
            # from latency, convergence, or the answer's own confidence.
            #
            # Deliberately outside the controller branch below: the effort
            # choice is made on every episode, so it is graded on every episode
            # a verifier actually graded, not only on the ones that also took
            # the execution-controller path.
            effort_episode_id = str(budget.get("ontogeny_episode") or "")
            if effort_episode_id:
                try:
                    effort_score, effort_checked, _passed, _reason = _controller_outcome(
                        result_receipt.get("verifier_guidance")
                    )
                    if effort_checked:
                        from core.ontogeny.control_points import get_effort_resolver

                        get_effort_resolver().note_grade(
                            effort_episode_id, verified_score=effort_score
                        )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "latent_cortex_service", exc, severity="debug",
                        action="effort decision left ungraded for this episode",
                    )
            # Controller learning accepts only an independently graded task
            # outcome. Candidate-local arithmetic, syntax, facet, and
            # grounding scores still steer this episode, but cannot become a
            # Wilson trial or teach the bandit that the whole answer was right.
            if controller_decision is not None:
                try:
                    from core.brain.llm.latent_cortex.execution_controller import (
                        get_execution_controller,
                    )

                    verifier_evidence = result_receipt.get("verifier_guidance")
                    (
                        best_score,
                        outcome_checked,
                        outcome_passed,
                        outcome_reason,
                    ) = _controller_outcome(verifier_evidence)
                    # CP126 3b3d44e8: the outcome must be bound to the DECISION
                    # that produced it — a caller-asserted bucket/arm could
                    # credit any arm, including recording a base execution as
                    # a treatment.
                    outcome_recorded = get_execution_controller().record_outcome(
                        bucket=str(controller_decision.get("bucket") or ""),
                        arm=str(controller_decision.get("arm") or "base"),
                        verified_score=best_score,
                        success=outcome_passed,
                        checked=outcome_checked,
                        wall_clock_s=time.monotonic() - started,
                        decision_id=str(controller_decision.get("decision_id") or ""),
                    )
                    checked_action_transitions = [
                        row for row in action_transitions if row["checked"] is True
                    ]
                    action_outcomes_recorded = (
                        get_execution_controller().record_action_transitions(
                            checked_action_transitions
                        )
                        if checked_action_transitions
                        else False
                    )
                    result_receipt["execution_controller"] = {
                        **controller_decision,
                        "outcome_recorded": outcome_recorded,
                        "outcome_checked": outcome_checked,
                        "outcome_passed": (outcome_passed if outcome_checked else None),
                        "outcome_reason": outcome_reason,
                        "action_transitions_checked": len(checked_action_transitions),
                        "action_outcomes_recorded": action_outcomes_recorded,
                    }
                    result["receipt"] = result_receipt
                except (
                    ImportError,
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                ) as exc:
                    logger.debug("Controller outcome not recorded: %s", exc)
            # Identity consistency: the canonical self verifies the
            # conclusion (persona displacement, forbidden intentions, core
            # values). The verdict PRICES the broadcast — an inconsistent
            # thought must outcompete honestly, never silently erased.
            try:
                from core.self.identity_consistency import (
                    check_identity_consistency,
                )

                result_receipt["identity_consistency"] = check_identity_consistency(
                    str(result.get("text") or "")
                )
                result["receipt"] = result_receipt
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Identity consistency check skipped: %s", exc)
            # Held-out grading queue: every facet judgment on the winning
            # candidate becomes a gradeable Foundry verdict (excerpt
            # attached), so facet-cue reliability is measured against human
            # ground truth instead of trusted forever.
            self._record_facet_judgments(
                result_receipt,
                domain,
                self._visible_objective(question, messages),
            )
            # RLC→GWT coupling: the conclusion returns to the workspace as a
            # competing coalition (priced by how it was earned) BEFORE any
            # action path consumes it — deliberation revises the broadcast,
            # the broadcast reaches the Will through the normal competition.
            # Lab/background episodes never write into the live mind.
            if foreground_request and publish_workspace_conclusion:
                await self._broadcast_conclusion(
                    result,
                    objective=self._visible_objective(question, messages),
                    stakes=stakes,
                )
                result_receipt = result.get("receipt", result_receipt)
            self._last_receipt = result_receipt
            self._last_failure_receipt = {}
            logger.info(
                "🧠 Latent episode ok: %d steps, %d branches, halt=%s, %.1fs",
                int(self._last_receipt.get("steps_taken") or 0),
                int(self._last_receipt.get("n_branches") or 0),
                self._last_receipt.get("halting_reason"),
                elapsed,
            )
        else:
            self._last_failure_receipt = result_receipt
            self._record_failure(str(result.get("reason") or "unknown"))
            logger.info(
                "🧠 Latent episode refused/failed: %s (%.1fs) stage=%s "
                "input_tokens=%s timings=%s progress=%s",
                self._last_refusal,
                elapsed,
                result_receipt.get("last_stage") or self._last_progress.get("stage") or "unknown",
                result_receipt.get("input_token_count")
                or self._last_progress.get("input_tokens")
                or "unknown",
                result_receipt.get("stage_timings_s") or {},
                self._last_progress,
            )
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
            "last_progress": dict(self._last_progress),
            "last_failure_receipt": {
                key: self._last_failure_receipt.get(key)
                for key in (
                    "episode_id",
                    "input_token_count",
                    "params_unchanged",
                    "fast_weights_applied",
                    "fast_weights_erased",
                    "fast_weight_optimizer",
                    "fast_weight_loss_trail",
                    "fast_weight_gradient_norm_trail",
                    "fast_weight_accepted_step_sizes",
                    "fast_weight_line_search_backtracks",
                    "verifier_probe_max_tokens",
                    "latent_opt_verifier",
                    "last_stage",
                    "stage_timings_s",
                    "honest_flags",
                    "epistemic_state",
                    "adaptive_compute",
                    "adaptive_acquisition",
                )
                if key in self._last_failure_receipt
            },
            "last_receipt": {
                k: self._last_receipt.get(k)
                for k in (
                    "episode_id",
                    "steps_taken",
                    "halting_reason",
                    "terminal_disposition",
                    "causal_receipt",
                    "n_slots",
                    "n_branches",
                    "exchanges",
                    "schedule_hash",
                    "checkpoint_fingerprint",
                    "checkpoint_fingerprint_method",
                    "checkpoint_file_count",
                    "worker_boot_id",
                    "worker_pid",
                    "worker_model_path",
                    "worker_model_parameter_count",
                    "worker_model_stored_parameter_element_count",
                    "worker_model_parameter_count_basis",
                    "worker_source_sha256",
                    "worker_affective_steering_active",
                    "worker_affective_steering_alpha",
                    "episode_affective_steering_applied",
                    "episode_affective_steering_alpha",
                    "request_payload_sha256",
                    "input_tokens_sha256",
                    "input_token_count",
                    "input_context_compaction",
                    "runtime_identity",
                    "params_unchanged",
                    "latent_opt_applied",
                    "latent_opt_attempts",
                    "latent_opt_steps",
                    "latent_opt_budget_exhausted",
                    "verifier_probe_max_tokens",
                    "latent_opt_verifier",
                    "fast_weights_applied",
                    "fast_weight_optimization_attempts",
                    "fast_weight_optimized_steps",
                    "fast_weight_budget_exhausted",
                    "fast_weights_erased",
                    "decode_requested_tokens",
                    "decode_generated_tokens",
                    "decode_termination",
                    "last_stage",
                    "stage_timings_s",
                    "honest_flags",
                    "epistemic_state",
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

    inst = get_runtime_service(
        ServiceNames.LATENT_CORTEX, default=None
    ) or get_latent_cortex_service(orchestrator)
    register_runtime_service(
        ServiceNames.LATENT_CORTEX,
        inst,
        required=False,
        owner="core/brain/latent_cortex_service.py",
        registered_by="register_latent_cortex",
    )
    # The selective-memory bridge resolves organs through the same runtime
    # registry as every other cognitive signal. Register the existing
    # playbook and reasoning-reflection stores as named organs; this does not
    # create new memory databases or duplicate their ownership.
    for service_name, getter in (
        (
            "procedural_memory",
            lambda: __import__(
                "core.brain.procedural_memory",
                fromlist=["get_procedural_memory"],
            ).get_procedural_memory(),
        ),
        (
            "reasoning_memory",
            lambda: __import__(
                "core.brain.reasoning_memory",
                fromlist=["get_reasoning_memory"],
            ).get_reasoning_memory(),
        ),
    ):
        if get_runtime_service(service_name, default=None) is not None:
            continue
        try:
            register_runtime_service(
                service_name,
                getter(),
                required=False,
                owner="core/brain/latent_cortex_service.py",
                registered_by="register_latent_cortex",
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Selective-memory organ registration skipped (%s): %s", service_name, exc)
    return inst


__all__ = [
    "LatentCortexService",
    "get_latent_cortex_service",
    "register_latent_cortex",
]
