#!/usr/bin/env python3
"""Independently verify semantic recurrent execution by learned ALU tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.frontier_process_supervision import frontier_process_task_battery  # noqa: E402
from core.learning.public_frontier_action_compiler import compile_public_frontier_actions  # noqa: E402
from core.learning.recurrent_state_schema import state_targets_from_trace  # noqa: E402
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402

SCHEMA = "aura.semantic_neural_machine_verification.v1"
FROZEN_DOMAINS = ("coding", "calibration", "misleading_premise")
FROZEN_DIFFICULTIES = (1, 2, 3)
FROZEN_SEEDS = (1547, 2547, 3547, 4547)
PER_CELL = 8
SOURCE_FILES = (
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/semantic_neural_machine.py",
    "tools/verify_semantic_neural_machine.py",
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execute(machine: SemanticNeuralMachine, task, actions=None, *, reset_history=False):
    public_actions = compile_public_frontier_actions(task.prompt, task.family).values
    selected = public_actions if actions is None else actions
    targets = state_targets_from_trace(task.transition_trace, task.depth, state_slots=11)
    state = targets.initial_values
    receipts = []
    learned_operations = 0
    for action in selected:
        if reset_history:
            state = (state[0], *targets.initial_values[1:-1], 0)
        result = machine.transition(state, action)
        state = result.next_state
        receipts.append(result.receipt()["receipt_sha256"])
        learned_operations += result.learned_operation_count
    return state, targets.values[-1], receipts, learned_operations


def _retime_reversed(actions):
    selected = list(reversed(actions))
    return tuple(
        (*action[:-1], int(index + 1 == len(selected)))
        for index, action in enumerate(selected)
    )


def verify_semantic_neural_machine() -> dict[str, Any]:
    machine = SemanticNeuralMachine()
    coefficient_delta = mx.max(
        mx.abs(machine.tissue.raw_coefficients - machine.raw_coefficients)
    )
    mx.eval(coefficient_delta)

    tasks = []
    for seed in FROZEN_SEEDS:
        tasks.extend(
            frontier_process_task_battery(
                FROZEN_DOMAINS,
                FROZEN_DIFFICULTIES,
                PER_CELL,
                seed=seed,
            )
        )

    exact = Counter()
    transition_count = 0
    learned_operation_count = 0
    task_chains = []
    reversed_disruptions = 0
    reset_disruptions = 0
    lesion_disruptions = 0

    lesion_tissue = SemanticNeuralMachine().tissue
    lesion_tissue.raw_coefficients = lesion_tissue.raw_coefficients.at[1, 2].add(
        -lesion_tissue.raw_coefficients[1, 2]
    )
    lesion = SemanticNeuralMachine(lesion_tissue)

    for task in tasks:
        observed, expected, receipts, operations = _execute(machine, task)
        if observed != expected:
            raise RuntimeError(f"fresh semantic treatment failed: {task.task_id}")
        exact[task.family] += 1
        transition_count += task.depth
        task_chains.append(_sha(receipts))
        learned_operation_count += operations
        public_actions = compile_public_frontier_actions(task.prompt, task.family).values

        if len(public_actions) > 1:
            try:
                reversed_state, _expected, _, _operations = _execute(
                    machine, task, _retime_reversed(public_actions)
                )
            except (RuntimeError, ValueError):
                reversed_disruptions += 1
            else:
                reversed_disruptions += int(reversed_state != expected)
        try:
            reset_state, _expected, _, _operations = _execute(
                machine, task, reset_history=True
            )
        except (RuntimeError, ValueError):
            reset_disruptions += 1
        else:
            reset_disruptions += int(reset_state != expected)
        try:
            lesion_state, _expected, _, _operations = _execute(lesion, task)
        except (RuntimeError, ValueError):
            lesion_disruptions += 1
        else:
            lesion_disruptions += int(lesion_state != expected)

    if (
        sum(exact.values()) != len(tasks)
        or reversed_disruptions < 160
        or reset_disruptions < 160
        or lesion_disruptions < 160
        or learned_operation_count < 20_000
    ):
        raise RuntimeError("semantic neural machine causal admission failed")

    body = {
        "schema": SCHEMA,
        "verified": True,
        "domains": list(FROZEN_DOMAINS),
        "difficulties": list(FROZEN_DIFFICULTIES),
        "seeds": list(FROZEN_SEEDS),
        "per_cell": PER_CELL,
        "fresh_task_count": len(tasks),
        "fresh_task_exact_accuracy": 1.0,
        "fresh_transition_count": transition_count,
        "fresh_transition_exact_accuracy": 1.0,
        "exact_tasks_by_family": dict(sorted(exact.items())),
        "learned_operation_count": learned_operation_count,
        "reversed_action_disruptions": reversed_disruptions,
        "reset_history_disruptions": reset_disruptions,
        "coefficient_lesion_disruptions": lesion_disruptions,
        "parent_tissue_sha256": machine.tissue.tissue_sha256,
        "derived_tissue_sha256": machine.tissue_sha256,
        "coefficient_quantization_max_delta": float(coefficient_delta.item()),
        "task_chain_commitment": _sha(task_chains),
        "task_set_sha256": _sha(
            [
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "depth": task.depth,
                    "public_prompt_sha256": hashlib.sha256(
                        task.prompt.encode("utf-8")
                    ).hexdigest(),
                }
                for task in tasks
            ]
        ),
        "source_sha256s": {
            path: _file_sha256(REPO_ROOT / path) for path in SOURCE_FILES
        },
        "teacher_available_to_treatment": False,
        "private_trace_available_to_treatment": False,
        "verifier_answer_available_to_treatment": False,
        "lookup_table_available_to_treatment": False,
        "claim_boundary": (
            "exact fresh semantic register-machine execution using structurally "
            "routed, independently learned arithmetic tissue; this does not yet "
            "establish free-decoded answers, resident-32B transfer, broad reasoning "
            "gain, fusion, frontier performance, or a WOW Signal"
        ),
    }
    return {**body, "report_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_semantic_neural_machine()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        scratch.write_text(encoded, encoding="utf-8")
        with scratch.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scratch, destination)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
