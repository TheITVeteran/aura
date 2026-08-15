#!/usr/bin/env python3
"""Re-evaluate frozen transition tissue without a model or legacy transition head."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    TRANSITION_MEMORY_PARAMETER_NAMES,
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
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
            minimum_iterations=1,
            initialization_seed=int(identity["init_seed"]),
        )
    )
    required_names = (
        set(TRANSITION_MEMORY_PARAMETER_NAMES)
        | set(TRANSITION_PROCESSOR_PARAMETER_NAMES)
        | {"transition_processor_opcode_output"}
    )
    extension_groups = (
        set(TRANSITION_TAPE_READER_PARAMETER_NAMES),
        {
            "transition_processor_opcode_interaction_up",
            "transition_processor_opcode_interaction_down",
        },
        {"transition_processor_opcode_hidden"},
    )
    available = {
        name.removeprefix("bundle.controller."): value
        for name, value in tensors.items()
        if name.startswith("bundle.controller.transition_")
    }
    missing = required_names - set(available)
    if missing:
        raise RuntimeError(
            "transition checkpoint tensor inventory is incomplete: "
            + ",".join(sorted(missing))
        )
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
    )
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
) -> dict[str, Any]:
    trace = task.transition_trace
    program = task.transition_program
    if trace is None or program is None:
        raise ValueError("transition evaluation task has no process evidence")
    targets = state_targets_from_trace(trace, depth)
    actions = action_targets_from_program(program, depth).values
    state = controller.exact_probabilities(
        targets.initial_values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
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
            memory = controller._typed_transition_memory(
                action_history,
                state_probabilities=state,
                action_probabilities=action,
            )
            decision = controller.resolve_transition_processor_logits(
                None,
                state,
                action,
                memory,
                transition_processor_mode="authoritative",
                opcode_expert_routing=routing,
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
        "active_trajectory_exact first_error_fraction"
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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--depths", default="1,3,5,9,10,12,16")
    parser.add_argument("--routings", default="opcode,uniform,lesion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller, checkpoint = _load_controller(args.checkpoint_dir.resolve(strict=True))
    _train, holdout = _load_frozen_dataset(args.dataset.resolve(strict=True))
    depths = tuple(int(value) for value in args.depths.split(",") if value)
    routings = tuple(value for value in args.routings.split(",") if value)
    arms: dict[str, Any] = {}
    for routing in routings:
        depth_reports = {}
        for depth in depths:
            task_rows = [
                {
                    "family": task.family,
                    "task_id": task.task_id,
                    **_evaluate_task(controller, task, depth, routing=routing),
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
        arms[routing] = depth_reports
    body = {
        "schema": REPORT_SCHEMA,
        "checkpoint": checkpoint,
        "dataset": str(args.dataset.resolve()),
        "holdout_tasks": len(holdout),
        "depths": list(depths),
        "routings": list(routings),
        "runtime_contract": {
            "transition_processor_mode": "authoritative",
            "legacy_transition_logits_available": False,
            "exact_microcode_available": False,
            "initial_state_authority": "verified_public_initial_state",
            "complete_public_action_prefix_visible": True,
            "public_tape_query_uses_current_state_and_action": True,
            "future_public_action_visible": False,
            "private_transition_trace_visible": False,
            "public_actions_are_correctness_authority": False,
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
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "opcode": {
                    depth: values["active_value_exact_accuracy"]
                    for depth, values in arms["opcode"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
