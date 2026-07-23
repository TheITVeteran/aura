"""Public fixed-point, anchor, KV, and train/live parity evidence."""

from __future__ import annotations

import math
from typing import Any

from core.brain.llm.latent_cortex.loop_core import (
    UPDATE_IMPLEMENTATION,
    alpha_for_step,
    canonical_sha256,
    validate_kv_bound_receipt,
    validate_loop_core_contract,
)

SCHEMA = "aura.rlc.loop_stability.v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _branch_summary(branch: Any) -> dict[str, Any]:
    trace = [dict(row) for row in branch.loop_stability_trace]
    residuals = [float(row["residual"]) for row in trace]
    return {
        "branch_index": int(branch.index),
        "role": str(branch.role),
        "anchor_sha256": trace[0]["anchor_sha256"],
        "transition_count": len(trace),
        "initial_residual": round(residuals[0], 8),
        "final_residual": round(residuals[-1], 8),
        "max_anchor_rms_ratio": round(
            max(float(row["anchor_rms_ratio"]) for row in trace),
            8,
        ),
        "contracting_steps": sum(row["contracting"] is True for row in trace),
        "expanding_steps": sum(row["contracting"] is False for row in trace),
        "oscillating_steps": sum(row["oscillating"] is True for row in trace),
        "fixed_point_steps": sum(
            row["fixed_point_candidate"] is True for row in trace
        ),
        "contained_divergences": sum(
            row["disposition"] == "contained_divergence" for row in trace
        ),
        "transitions": trace,
    }


def build_loop_stability_receipt(
    *,
    branches: list[Any],
    selected_branch: int,
    loop_core: dict[str, Any],
    kv_bound: dict[str, Any],
    recurrent_grounding: dict[str, Any],
) -> dict[str, Any]:
    """Build a receipt over public dynamics, not private latent contents."""

    core = validate_loop_core_contract(loop_core)
    kv = validate_kv_bound_receipt(kv_bound)
    branch_rows = [_branch_summary(branch) for branch in branches]
    clip = float(core["rms_clip_ratio"])
    payload = {
        "schema": SCHEMA,
        "loop_core": core,
        "kv_bound": kv,
        "selected_branch": selected_branch,
        "branches": branch_rows,
        "all_finite": all(
            row["all_finite"] is True
            for branch in branch_rows
            for row in branch["transitions"]
        ),
        "all_anchor_bounded": all(
            (1.0 / clip) - 1e-5
            <= float(row["anchor_rms_ratio"])
            <= clip + 1e-5
            for branch in branch_rows
            for row in branch["transitions"]
        ),
        "all_accepted_states_anchor_bounded": all(
            row["disposition"] == "contained_divergence"
            or (
                (1.0 / clip) - 1e-5
                <= float(row["anchor_rms_ratio"])
                <= clip + 1e-5
            )
            for branch in branch_rows
            for row in branch["transitions"]
        ),
        "contained_divergences": sum(
            branch["contained_divergences"] for branch in branch_rows
        ),
        "fixed_point_diagnostics_complete": all(
            branch["transition_count"] > 0 for branch in branch_rows
        ),
        "shared_train_inference_core": (
            core["update_implementation"] == UPDATE_IMPLEMENTATION
        ),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_loop_stability_receipt(
        receipt,
        recurrent_grounding=recurrent_grounding,
        expected_loop_core=core,
    )


def validate_loop_stability_receipt(
    value: Any,
    *,
    recurrent_grounding: dict[str, Any],
    expected_loop_core: dict[str, Any],
) -> dict[str, Any]:
    """Independently reconstruct fixed-point summaries and state links."""

    fields = {
        "schema",
        "loop_core",
        "kv_bound",
        "selected_branch",
        "branches",
        "all_finite",
        "all_anchor_bounded",
        "all_accepted_states_anchor_bounded",
        "contained_divergences",
        "fixed_point_diagnostics_complete",
        "shared_train_inference_core",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("loop-stability receipt fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("loop-stability receipt commitment mismatch")
    core = validate_loop_core_contract(
        value["loop_core"],
        expected=expected_loop_core,
    )
    validate_kv_bound_receipt(value["kv_bound"])
    grounding_branches = recurrent_grounding.get("branches")
    branches = value["branches"]
    if (
        value["schema"] != SCHEMA
        or not isinstance(branches, list)
        or not branches
        or not isinstance(grounding_branches, list)
        or len(branches) != len(grounding_branches)
        or type(value["selected_branch"]) is not int
        or value["selected_branch"] not in range(len(branches))
        or value["selected_branch"] != recurrent_grounding.get("selected_branch")
    ):
        raise ValueError("loop-stability topology is invalid")
    branch_fields = {
        "branch_index",
        "role",
        "anchor_sha256",
        "transition_count",
        "initial_residual",
        "final_residual",
        "max_anchor_rms_ratio",
        "contracting_steps",
        "expanding_steps",
        "oscillating_steps",
        "fixed_point_steps",
        "contained_divergences",
        "transitions",
    }
    transition_fields = {
        "ordinal",
        "branch_step",
        "window_start",
        "window_end",
        "hypothesis_pre_sha256",
        "hypothesis_post_sha256",
        "reasoning_pre_sha256",
        "reasoning_post_sha256",
        "anchor_sha256",
        "continuous_from_previous",
        "disposition",
        "divergence_reason",
        "containment_action",
        "alpha",
        "input_mean_rms",
        "output_mean_rms",
        "anchor_mean_rms",
        "anchor_rms_ratio",
        "residual",
        "contraction_ratio",
        "delta_cosine",
        "contracting",
        "oscillating",
        "fixed_point_candidate",
        "all_finite",
    }
    clip = float(core["rms_clip_ratio"])
    reconstructed: list[dict[str, Any]] = []
    for index, (branch, grounding) in enumerate(
        zip(branches, grounding_branches, strict=True)
    ):
        if (
            not isinstance(branch, dict)
            or set(branch) != branch_fields
            or branch["branch_index"] != index
            or grounding.get("branch_index") != index
            or branch["role"] != grounding.get("role")
            or not isinstance(branch["role"], str)
            or not branch["role"]
            or not _is_sha256(branch["anchor_sha256"])
            or not isinstance(branch["transitions"], list)
            or not branch["transitions"]
            or branch["transition_count"] != len(branch["transitions"])
            or len(branch["transitions"]) != len(grounding.get("transitions") or [])
        ):
            raise ValueError("loop-stability branch evidence is invalid")
        residuals: list[float] = []
        anchor_ratios: list[float] = []
        contracting = 0
        expanding = 0
        oscillating = 0
        fixed_points = 0
        contained_divergences = 0
        previous_reasoning_post = ""
        previous_residual: float | None = None
        previous_output_rms: float | None = None
        anchor_rms: float | None = None
        for ordinal, (row, grounded) in enumerate(
            zip(
                branch["transitions"],
                grounding["transitions"],
                strict=True,
            )
        ):
            finite_fields = (
                "alpha",
                "input_mean_rms",
                "output_mean_rms",
                "anchor_mean_rms",
                "anchor_rms_ratio",
                "residual",
            )
            if (
                not isinstance(row, dict)
                or set(row) != transition_fields
                or row["ordinal"] != ordinal
                or row["branch_step"] != grounded.get("branch_step")
                or row["window_start"] != grounded.get("window_start")
                or row["window_end"] != grounded.get("window_end")
                or row["hypothesis_pre_sha256"]
                != grounded.get("hypothesis_pre_sha256")
                or row["hypothesis_post_sha256"]
                != grounded.get("hypothesis_post_sha256")
                or not _is_sha256(row["hypothesis_pre_sha256"])
                or not _is_sha256(row["hypothesis_post_sha256"])
                or not _is_sha256(row["reasoning_pre_sha256"])
                or not _is_sha256(row["reasoning_post_sha256"])
                or row["anchor_sha256"] != branch["anchor_sha256"]
                or type(row["continuous_from_previous"]) is not bool
                or row["disposition"]
                not in {"accepted", "contained_divergence"}
                or not isinstance(row["divergence_reason"], str)
                or not isinstance(row["containment_action"], str)
                or any(not _finite(row[name]) for name in finite_fields)
                or not 0.0 < float(row["alpha"]) <= 1.0
                or min(
                    float(row["input_mean_rms"]),
                    float(row["output_mean_rms"]),
                    float(row["anchor_mean_rms"]),
                    float(row["anchor_rms_ratio"]),
                    float(row["residual"]),
                )
                < 0.0
                or row["all_finite"] is not True
                or type(row["oscillating"]) is not bool
                or type(row["fixed_point_candidate"]) is not bool
            ):
                raise ValueError("loop-stability transition evidence is invalid")
            if row["disposition"] == "accepted":
                if row["divergence_reason"] or row["containment_action"]:
                    raise ValueError("accepted transition claims containment")
            elif (
                not row["divergence_reason"].startswith("diverged")
                or (
                    row["containment_action"] != "escaped"
                    and row["containment_action"] != "halt_revert"
                    and not row["containment_action"].startswith("halt:")
                )
            ):
                raise ValueError("divergent transition has no bounded disposition")
            if (
                type(row["branch_step"]) is not int
                or not 0 <= row["branch_step"] < int(core["max_steps"])
                or row["window_start"] != core["prelude_end"]
                or row["window_end"] != core["coda_start"]
            ):
                raise ValueError("loop-stability transition differs from loop core")
            expected_alpha = alpha_for_step(
                alpha=float(core["alpha"]),
                schedule=str(core["alpha_schedule"]),
                max_steps=int(core["max_steps"]),
                step=int(row["branch_step"]),
            )
            if not math.isclose(
                float(row["alpha"]),
                expected_alpha,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("loop-stability alpha differs from loop core")
            continuous = bool(
                ordinal > 0
                and previous_reasoning_post == row["reasoning_pre_sha256"]
            )
            if row["continuous_from_previous"] is not continuous:
                raise ValueError("loop-stability continuity commitment is invalid")
            contraction = row["contraction_ratio"]
            cosine = row["delta_cosine"]
            if not continuous:
                if (
                    contraction is not None
                    or cosine is not None
                    or row["contracting"] is not None
                    or row["oscillating"] is not False
                ):
                    raise ValueError(
                        "discontinuous transition has prior-step diagnostics"
                    )
            elif (
                not _finite(contraction)
                or float(contraction) < 0.0
                or not _finite(cosine)
                or not -1.0 <= float(cosine) <= 1.0
                or row["contracting"] is not (float(contraction) < 1.0)
                or row["oscillating"] is not (float(cosine) < -0.5)
            ):
                raise ValueError("loop-stability derivative diagnostics are invalid")
            if continuous and (
                previous_residual is None
                or previous_output_rms is None
                or not math.isclose(
                    float(contraction),
                    float(row["residual"]) / max(previous_residual, 1e-9),
                    rel_tol=0.0,
                    abs_tol=2e-8,
                )
                or not math.isclose(
                    float(row["input_mean_rms"]),
                    previous_output_rms,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError("loop-stability transition chain is inconsistent")
            if anchor_rms is None:
                anchor_rms = float(row["anchor_mean_rms"])
            if (
                not math.isclose(
                    float(row["anchor_mean_rms"]),
                    anchor_rms,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                or not math.isclose(
                    float(row["anchor_rms_ratio"]),
                    float(row["output_mean_rms"])
                    / max(float(row["anchor_mean_rms"]), 1e-6),
                    rel_tol=0.0,
                    abs_tol=2e-8,
                )
            ):
                raise ValueError("loop-stability anchor diagnostics are inconsistent")
            if row["fixed_point_candidate"] is not (
                float(row["residual"]) < float(core["convergence_eps"])
            ):
                raise ValueError("fixed-point classification differs from contract")
            residuals.append(float(row["residual"]))
            anchor_ratios.append(float(row["anchor_rms_ratio"]))
            contracting += row["contracting"] is True
            expanding += row["contracting"] is False
            oscillating += row["oscillating"] is True
            fixed_points += row["fixed_point_candidate"] is True
            contained_divergences += row["disposition"] == "contained_divergence"
            previous_reasoning_post = row["reasoning_post_sha256"]
            previous_residual = float(row["residual"])
            previous_output_rms = float(row["output_mean_rms"])
        summary = {
            "branch_index": index,
            "role": branch["role"],
            "anchor_sha256": branch["anchor_sha256"],
            "transition_count": len(residuals),
            "initial_residual": round(residuals[0], 8),
            "final_residual": round(residuals[-1], 8),
            "max_anchor_rms_ratio": round(max(anchor_ratios), 8),
            "contracting_steps": contracting,
            "expanding_steps": expanding,
            "oscillating_steps": oscillating,
            "fixed_point_steps": fixed_points,
            "contained_divergences": contained_divergences,
            "transitions": branch["transitions"],
        }
        if branch != summary:
            raise ValueError("loop-stability branch summary mismatch")
        reconstructed.append(summary)
    all_anchor_bounded = all(
        (1.0 / clip) - 1e-5 <= ratio <= clip + 1e-5
        for branch in reconstructed
        for ratio in (
            float(row["anchor_rms_ratio"])
            for row in branch["transitions"]
        )
    )
    all_accepted_states_anchor_bounded = all(
        row["disposition"] == "contained_divergence"
        or (1.0 / clip) - 1e-5
        <= float(row["anchor_rms_ratio"])
        <= clip + 1e-5
        for branch in reconstructed
        for row in branch["transitions"]
    )
    contained_divergences = sum(
        branch["contained_divergences"] for branch in reconstructed
    )
    transition_count = sum(
        branch["transition_count"] for branch in reconstructed
    )
    recurrent_kv_calls = [
        row
        for row in value["kv_bound"]["calls"]
        if row["persist"] is False
        and row["start"] == core["prelude_end"]
        and row["end"] == core["coda_start"]
    ]
    if (
        value["all_finite"] is not True
        or value["all_anchor_bounded"] is not all_anchor_bounded
        or value["all_accepted_states_anchor_bounded"]
        is not all_accepted_states_anchor_bounded
        or value["all_accepted_states_anchor_bounded"] is not True
        or value["contained_divergences"] != contained_divergences
        or value["fixed_point_diagnostics_complete"] is not True
        or value["shared_train_inference_core"] is not True
        or core["update_implementation"] != UPDATE_IMPLEMENTATION
        or len(recurrent_kv_calls) != transition_count
    ):
        raise ValueError("loop-stability acceptance summary is invalid")
    return dict(value)


__all__ = [
    "SCHEMA",
    "build_loop_stability_receipt",
    "validate_loop_stability_receipt",
]
