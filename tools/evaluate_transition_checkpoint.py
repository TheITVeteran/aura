#!/usr/bin/env python3
"""Re-evaluate frozen transition tissue without a model or legacy transition head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from core.learning.recurrent_action_schema import action_targets_from_program  # noqa: E402
from core.learning.recurrent_state_schema import state_targets_from_trace  # noqa: E402
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    structured_state_trajectory_diagnostics,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES,
    TRANSITION_MEMORY_PARAMETER_NAMES,
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_MODES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
    TRANSITION_REPLAY_PARAMETER_NAMES,
    TRANSITION_TAPE_READER_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from core.runtime.atomic_writer import atomic_write_json  # noqa: E402
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    _load_frozen_dataset,
    _load_latest_checkpoint,
)

REPORT_SCHEMA = "aura.unified_intrinsic.authoritative_checkpoint_evaluation.v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load_controller(checkpoint_dir: Path) -> tuple[UnifiedRecurrentController, dict[str, Any]]:
    loaded = _load_latest_checkpoint(checkpoint_dir, required=True)
    if loaded is None:  # pragma: no cover - required=True is authoritative
        raise RuntimeError("transition checkpoint is unavailable")
    receipt, weights_path = loaded
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("transition checkpoint identity is unavailable")
    tensors = mx.load(str(weights_path))
    memory_input = tensors.get("bundle.controller.transition_memory_input")
    if memory_input is None or len(memory_input.shape) != 3:
        raise RuntimeError("transition checkpoint model geometry is unavailable")
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=int(memory_input.shape[1]),
            correction_rank=int(identity["controller_rank"]),
            state_slots=int(identity.get("state_slots", 5)),
            minimum_iterations=1,
            initialization_seed=int(identity["init_seed"]),
        )
    )
    optional_cross_register = {"transition_processor_state_cross_projection"}
    required_names = (
        set(TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES)
        | set(TRANSITION_MEMORY_PARAMETER_NAMES)
        | (set(TRANSITION_PROCESSOR_PARAMETER_NAMES) - optional_cross_register)
        | {"transition_processor_opcode_output"}
    )
    extension_groups = (
        set(TRANSITION_TAPE_READER_PARAMETER_NAMES),
        {
            "transition_processor_opcode_interaction_up",
            "transition_processor_opcode_interaction_down",
        },
        {"transition_processor_opcode_hidden"},
        set(TRANSITION_REPLAY_PARAMETER_NAMES),
        optional_cross_register,
    )
    available = {
        name.removeprefix("bundle.controller."): value
        for name, value in tensors.items()
        if name.startswith("bundle.controller.transition_")
        or name.removeprefix("bundle.controller.")
        in TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES
    }
    missing = required_names - set(available)
    if missing:
        raise RuntimeError(
            "transition checkpoint tensor inventory is incomplete: "
            + ",".join(sorted(missing))
        )
    processor_identity = identity.get("direct_transition_processor", {})
    if (
        isinstance(processor_identity, dict)
        and processor_identity.get("mode") in {"copy_write", "masked_copy_write"}
        and not optional_cross_register <= set(available)
    ):
        raise RuntimeError("copy-write checkpoint has no cross-register tissue")
    for group in extension_groups:
        present = group & set(available)
        if present and present != group:
            raise RuntimeError(
                "transition checkpoint extension inventory is partial: "
                + ",".join(sorted(group - present))
            )
    loaded_names = required_names | {
        name for group in extension_groups for name in group if name in available
    }
    for name in loaded_names:
        expected = getattr(controller, name)
        observed = available[name]
        if tuple(expected.shape) != tuple(observed.shape):
            raise RuntimeError(f"transition checkpoint tensor shape differs: {name}")
        setattr(controller, name, observed)
    mx.eval(*(getattr(controller, name) for name in loaded_names))
    extension_names = set(TRANSITION_TAPE_READER_PARAMETER_NAMES) | (
        set(TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES)
        - {"transition_processor_opcode_output"}
    ) | set(TRANSITION_REPLAY_PARAMETER_NAMES)
    extension = sorted(extension_names - set(available))
    return controller, {
        "checkpoint_file": str(weights_path.resolve()),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_receipt_sha256": receipt["receipt_sha256"],
        "checkpoint_step": int(receipt["step"]),
        "checkpoint_identity_sha256": identity["identity_sha256"],
        "loaded_transition_tensor_names": sorted(loaded_names),
        "zero_attached_extensions": extension,
    }


def _evaluate_task(
    controller: UnifiedRecurrentController,
    task: Any,
    depth: int,
    *,
    routing: str,
    transition_processor_mode: str,
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    replay_mode: str = "disabled",
    state_history_arm: str = "intact",
) -> dict[str, Any]:
    trace = task.transition_trace
    program = task.transition_program
    if trace is None or program is None:
        raise ValueError("transition evaluation task has no process evidence")
    targets = state_targets_from_trace(
        trace,
        depth,
        state_slots=controller.config.state_slots,
    )
    actions = action_targets_from_program(program, depth).values
    state = controller.exact_probabilities(
        targets.initial_values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    initial_state = state
    if state_history_arm not in {
        "intact",
        "state_lesion",
        "history_lesion",
        "combined_lesion",
    }:
        raise ValueError("transition state/history arm differs")
    action_history: list[Any] = []
    logits: list[Any] = []
    for action_values in actions:
        current_values = mx.argmax(state, axis=-1)
        terminal = bool((current_values[0, -1] == 1).item())
        action = controller.exact_probabilities(
            action_values,
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        action_history.append(action)
        if terminal:
            decision = mx.log(mx.maximum(state, 1e-6))
        else:
            processor_state = (
                initial_state
                if state_history_arm in {"state_lesion", "combined_lesion"}
                else state
            )
            memory = (
                mx.zeros(
                    (
                        int(state.shape[0]),
                        controller.config.state_slots,
                        controller.config.correction_rank,
                    ),
                    dtype=mx.float32,
                )
                if state_history_arm in {"history_lesion", "combined_lesion"}
                else controller._typed_transition_memory(
                    action_history,
                    state_probabilities=processor_state,
                    action_probabilities=action,
                )
            )
            decision = controller.resolve_transition_processor_logits(
                None,
                processor_state,
                action,
                memory,
                transition_processor_mode=transition_processor_mode,
                opcode_expert_routing=routing,
                transition_copy_prior_logit_bias=(
                    transition_copy_prior_logit_bias
                ),
            )
            decision, _replay_candidate, _replay_gate = (
                controller.typed_transition_replay_logits(
                    decision,
                    action_history,
                    action_probabilities=action,
                    replay_mode=replay_mode,
                    opcode_expert_routing=routing,
                )
            )
        mx.eval(decision)
        logits.append(decision)
        state = controller.straight_through_probabilities(decision)
    return structured_state_trajectory_diagnostics(
        logits,
        targets,
        active_steps=min(int(trace.depth), depth),
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("transition checkpoint evaluation has no rows")
    active = tuple(
        "active_state_exact_accuracy active_value_exact_accuracy "
        "active_trajectory_exact final_active_state_exact first_error_fraction"
    .split())
    report = {name: sum(float(row[name]) for row in rows) / len(rows) for name in active}
    report["first_error_histogram"] = dict(
        sorted(
            {
                str(step): sum(row["first_error_step"] == step for row in rows)
                for step in {row["first_error_step"] for row in rows}
            }.items(),
            key=lambda item: (item[0] != "None", item[0]),
        )
    )
    counts = {
        name: sum(int(row["conditional_transition_counts"][name]) for row in rows)
        for name in rows[0]["conditional_transition_counts"]
    }
    report["conditional_transition_counts"] = counts
    report["p_correct_given_previous_correct"] = (
        counts["correct_after_correct"] / counts["correct_predecessors"]
        if counts["correct_predecessors"]
        else None
    )
    report["p_correct_given_previous_wrong"] = (
        counts["correct_after_wrong"] / counts["wrong_predecessors"]
        if counts["wrong_predecessors"]
        else None
    )
    recovery_rows = [row for row in rows if row["recovery_observable"]]
    report["recovery_observation_count"] = len(recovery_rows)
    report["recovered_after_first_error"] = (
        sum(bool(row["recovered_after_first_error"]) for row in recovery_rows)
        / len(recovery_rows)
        if recovery_rows
        else None
    )
    report["sustained_recovery_after_first_error"] = (
        sum(
            bool(row["sustained_recovery_after_first_error"])
            for row in recovery_rows
        )
        / len(recovery_rows)
        if recovery_rows
        else None
    )
    terminal_rows = [row for row in rows if row["terminal_stability_observable"]]
    report["terminal_stability_observation_count"] = len(terminal_rows)
    report["terminal_correct_stability"] = (
        sum(bool(row["terminal_correct_stable"]) for row in terminal_rows)
        / len(terminal_rows)
        if terminal_rows
        else None
    )
    report["terminal_self_stability"] = (
        sum(bool(row["terminal_self_stable"]) for row in terminal_rows)
        / len(terminal_rows)
        if terminal_rows
        else None
    )
    register_names = tuple(rows[0]["per_register_accuracy"])
    register_accuracy = {}
    for name in register_names:
        observed = [
            float(row["per_register_accuracy"][name])
            for row in rows
            if row["per_register_accuracy"][name] is not None
        ]
        register_accuracy[name] = sum(observed) / len(observed) if observed else None
    report["per_register_accuracy"] = register_accuracy
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--depths", default="1,3,5,9,10,12,16")
    parser.add_argument("--routings", default="opcode,uniform,lesion")
    parser.add_argument(
        "--transition-processor-mode",
        choices=tuple(
            mode for mode in TRANSITION_PROCESSOR_MODES if mode != "residual"
        ),
        default="authoritative",
    )
    parser.add_argument(
        "--transition-copy-prior-logit-bias",
        type=float,
        default=TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    )
    parser.add_argument("--replay-modes", default="disabled")
    parser.add_argument("--state-history-arms", default="intact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 0.0 <= args.transition_copy_prior_logit_bias <= 8.0
        or not math.isfinite(args.transition_copy_prior_logit_bias)
    ):
        raise ValueError("transition copy prior logit bias differs")

    controller, checkpoint = _load_controller(args.checkpoint_dir.resolve(strict=True))
    _train, holdout = _load_frozen_dataset(args.dataset.resolve(strict=True))
    depths = tuple(int(value) for value in args.depths.split(",") if value)
    routings = tuple(value for value in args.routings.split(",") if value)
    replay_modes = tuple(value for value in args.replay_modes.split(",") if value)
    state_history_arms = tuple(
        value for value in args.state_history_arms.split(",") if value
    )
    arms: dict[str, Any] = {}
    for routing in routings:
        for replay_mode in replay_modes:
            for state_history_arm in state_history_arms:
                arm_name = (
                    routing
                    if replay_modes == ("disabled",)
                    and state_history_arms == ("intact",)
                    else f"{routing}:{replay_mode}:{state_history_arm}"
                )
                depth_reports = {}
                for depth in depths:
                    task_rows = [
                        {
                            "family": task.family,
                            "task_id": task.task_id,
                            **_evaluate_task(
                                controller,
                                task,
                                depth,
                                routing=routing,
                                transition_processor_mode=(
                                    args.transition_processor_mode
                                ),
                                transition_copy_prior_logit_bias=(
                                    args.transition_copy_prior_logit_bias
                                ),
                                replay_mode=replay_mode,
                                state_history_arm=state_history_arm,
                            ),
                        }
                        for task in holdout
                    ]
                    depth_reports[f"T{depth}"] = {
                        **_aggregate(task_rows),
                        "families": {
                            family: _aggregate(
                                [row for row in task_rows if row["family"] == family]
                            )
                            for family in sorted({row["family"] for row in task_rows})
                        },
                    }
                arms[arm_name] = depth_reports
    body = {
        "schema": REPORT_SCHEMA,
        "checkpoint": checkpoint,
        "dataset": str(args.dataset.resolve()),
        "holdout_tasks": len(holdout),
        "depths": list(depths),
        "routings": list(routings),
        "replay_modes": list(replay_modes),
        "state_history_arms": list(state_history_arms),
        "runtime_contract": {
            "transition_processor_mode": args.transition_processor_mode,
            "transition_copy_prior_logit_bias": (
                args.transition_copy_prior_logit_bias
            ),
            "legacy_transition_logits_available": False,
            "exact_microcode_available": False,
            "initial_state_authority": "verified_public_initial_state",
            "complete_public_action_prefix_visible": True,
            "public_tape_query_uses_current_state_and_action": True,
            "future_public_action_visible": False,
            "private_transition_trace_visible": False,
            "public_actions_are_correctness_authority": False,
            "replay_candidate_reads_recurrent_state": False,
            "replay_candidate_reads_public_prefix": True,
            "terminal_state_structurally_preserved": True,
        },
        "arms": arms,
        "claim_not_supported": [
            "prompt_to_initial_state_accuracy",
            "prompt_to_public_action_accuracy",
            "decoded_answer_gain",
            "resident_32b_transfer",
            "wow_signal",
        ],
    }
    report = {**body, "report_sha256": _sha(body)}
    atomic_write_json(
        args.output,
        report,
        schema_version=1,
        schema_name=REPORT_SCHEMA,
    )
    primary_arm = "opcode" if "opcode" in arms else next(iter(arms))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "primary_arm": primary_arm,
                "primary_active_value_exact_accuracy": {
                    depth: values["active_value_exact_accuracy"]
                    for depth, values in arms[primary_arm].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
