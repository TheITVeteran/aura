#!/usr/bin/env python3
"""Adjudicate the frozen resident decoded-transfer canary.

The decode evaluator emits observations, not a scientific conclusion.  This
tool freezes the bounded canary decision rule independently of model execution
and refuses to promote ceiling, incomplete, regressive, or lesion-insensitive
results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from tools import launch_unified_intrinsic_resident_evaluation as launcher  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)

VERDICT_SCHEMA: Final = "aura.unified_intrinsic.resident_transfer_verdict.v1"
REPORT_SCHEMAS: Final = frozenset(
    {
        "aura.unified_intrinsic_decode_evaluation.v1",
        "aura.unified_intrinsic_decode_evaluation.v2",
    }
)
CONTROL_SCHEMAS: Final = {
    "aura.unified_intrinsic.matched_control_binding.v1": "campaign_episode_initial",
    "aura.unified_intrinsic.root_control_binding.v1": "deterministic_pretraining_root",
}
SUPPORTED = "supported_bounded_resident_transfer"
INCONCLUSIVE_CEILING = "inconclusive_control_ceiling"
INCONCLUSIVE_INSTRUMENT = "inconclusive_instrument"
REFUTED = "refuted_bounded_resident_transfer"


class ResidentTransferAdjudicationError(RuntimeError):
    """Stable failure boundary for malformed or incomplete evidence."""


def _fail(message: str) -> Never:
    raise ResidentTransferAdjudicationError(message)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"resident transfer {name} is invalid")
    return value


def _arm(report: Mapping[str, Any], name: str, tasks: int) -> dict[str, Any]:
    arms = report.get("arm_results")
    row = arms.get(name) if isinstance(arms, dict) else None
    if not isinstance(row, dict):
        _fail(f"resident transfer arm is missing: {name}")
    correct = _integer(row.get("correct"), f"{name}.correct")
    observed_tasks = _integer(row.get("tasks"), f"{name}.tasks", minimum=1)
    accuracy = row.get("accuracy")
    if (
        observed_tasks != tasks
        or correct > tasks
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or abs(float(accuracy) - correct / tasks) > 1e-12
    ):
        _fail(f"resident transfer arm summary differs: {name}")
    return row


def _candidate_matrix(
    report: Mapping[str, Any],
    *,
    arm_names: set[str],
    task_count: int,
) -> dict[str, dict[str, bool]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != task_count * len(arm_names):
        _fail("resident transfer candidate count differs")
    matrix: dict[str, dict[str, bool]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            _fail("resident transfer candidate is invalid")
        task_id = row.get("task_id")
        arm = row.get("arm")
        correct = row.get("correct")
        if (
            not isinstance(task_id, str)
            or not task_id
            or arm not in arm_names
            or type(correct) is not bool
        ):
            _fail("resident transfer candidate identity differs")
        task = matrix.setdefault(task_id, {})
        if arm in task:
            _fail("resident transfer candidate arm is duplicated")
        task[arm] = correct
    if len(matrix) != task_count or any(set(row) != arm_names for row in matrix.values()):
        _fail("resident transfer candidate matrix is incomplete")
    return matrix


def _validate_matched_control(report: Mapping[str, Any]) -> None:
    """Require an authenticated control identity for current evidence reports."""

    if report.get("schema") == "aura.unified_intrinsic_decode_evaluation.v1":
        return
    binding = report.get("matched_control")
    if not isinstance(binding, dict):
        _fail("resident transfer matched control is missing")
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    expected_mode = CONTROL_SCHEMAS.get(binding.get("schema"))
    required_strings = {
        "campaign_root",
        "controller_sha256",
        "campaign_identity_sha256",
    }
    if expected_mode == "deterministic_pretraining_root":
        required_strings.update(
            {
                "stem",
                "checkpoint_sha256",
                "checkpoint_receipt_sha256",
            }
        )
    if (
        expected_mode is None
        or binding.get("mode") != expected_mode
        or binding.get("binding_sha256") != canonical_sha256(body)
        or any(
            not isinstance(binding.get(key), str) or not binding[key] for key in required_strings
        )
        or any(len(binding[key]) != 64 for key in required_strings if key.endswith("sha256"))
        or (
            expected_mode == "deterministic_pretraining_root"
            and type(binding.get("checkpoint_step")) is not int
        )
    ):
        _fail("resident transfer matched control commitment differs")


def adjudicate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return one bounded, machine-checkable resident-transfer verdict."""

    if report.get("schema") not in REPORT_SCHEMAS:
        _fail("resident transfer report schema differs")
    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != canonical_sha256(report_body):
        _fail("resident transfer report commitment differs")
    _validate_matched_control(report)

    task_count = _integer(report.get("task_count"), "task_count", minimum=1)
    task_depths = report.get("task_depths")
    recurrence_depths = report.get("recurrence_depths")
    if (
        not isinstance(task_depths, list)
        or not task_depths
        or any(type(value) is not int or value < 1 for value in task_depths)
        or not isinstance(recurrence_depths, list)
        or len(recurrence_depths) != 1
        or type(recurrence_depths[0]) is not int
        or recurrence_depths[0] < 2
    ):
        _fail("resident transfer depth contract differs")
    depth = recurrence_depths[0]
    treatment_name = f"trained_t{depth}"
    control_name = f"untrained_t{depth}"
    grammar_name = f"grammar_lesion_t{depth}"
    pointer_name = f"pointer_lesion_t{depth}"
    compiled_name = f"compiled_t{depth}"

    arm_names = {
        "base_t1",
        "untrained_t1",
        "trained_t1",
        control_name,
        treatment_name,
        grammar_name,
        pointer_name,
        compiled_name,
    }
    matrix = _candidate_matrix(
        report,
        arm_names=arm_names,
        task_count=task_count,
    )

    base = _arm(report, "base_t1", task_count)
    trained_t1 = _arm(report, "trained_t1", task_count)
    control = _arm(report, control_name, task_count)
    treatment = _arm(report, treatment_name, task_count)
    grammar = _arm(report, grammar_name, task_count)
    pointer = _arm(report, pointer_name, task_count)
    compiled = _arm(report, compiled_name, task_count)
    summarized = {
        "base_t1": base,
        "untrained_t1": _arm(report, "untrained_t1", task_count),
        "trained_t1": trained_t1,
        control_name: control,
        treatment_name: treatment,
        grammar_name: grammar,
        pointer_name: pointer,
        compiled_name: compiled,
    }
    if any(
        sum(row[arm] for row in matrix.values()) != summary["correct"]
        for arm, summary in summarized.items()
    ):
        _fail("resident transfer arm summary differs from candidates")

    effects = report.get("paired_training_effects")
    effect = effects.get(str(depth)) if isinstance(effects, dict) else None
    if not isinstance(effect, dict):
        _fail("resident transfer paired effect is missing")
    expected_effect = {
        "tasks": task_count,
        "control_arm": control_name,
        "trained_arm": treatment_name,
        "untrained_correct": control["correct"],
        "trained_correct": treatment["correct"],
    }
    if any(effect.get(key) != value for key, value in expected_effect.items()):
        _fail("resident transfer paired effect differs from arm summaries")
    wrong_to_right = _integer(effect.get("wrong_to_right"), "wrong_to_right")
    right_to_wrong = _integer(effect.get("right_to_wrong"), "right_to_wrong")
    net_gain = _integer(
        effect.get("net_correct_gain"),
        "net_correct_gain",
        minimum=-task_count,
    )
    if (
        wrong_to_right > task_count
        or right_to_wrong > task_count
        or net_gain != wrong_to_right - right_to_wrong
        or net_gain != treatment["correct"] - control["correct"]
    ):
        _fail("resident transfer transition accounting differs")
    reconstructed_wrong_to_right = sum(
        not row[control_name] and row[treatment_name] for row in matrix.values()
    )
    reconstructed_right_to_wrong = sum(
        row[control_name] and not row[treatment_name] for row in matrix.values()
    )
    if (
        wrong_to_right != reconstructed_wrong_to_right
        or right_to_wrong != reconstructed_right_to_wrong
    ):
        _fail("resident transfer transitions differ from candidates")

    checks = {
        "control_has_headroom": control["correct"] < task_count,
        "compiled_instrument_exact": compiled["correct"] == task_count,
        "wrong_to_right_present": wrong_to_right > 0,
        "zero_right_to_wrong": right_to_wrong == 0,
        "positive_matched_control_gain": net_gain > 0,
        "recurrence_beats_trained_t1": treatment["correct"] > trained_t1["correct"],
        "treatment_beats_base": treatment["correct"] > base["correct"],
        "grammar_lesion_loses": treatment["correct"] > grammar["correct"],
        "pointer_lesion_loses": treatment["correct"] > pointer["correct"],
    }
    if not checks["control_has_headroom"]:
        verdict = INCONCLUSIVE_CEILING
    elif not checks["compiled_instrument_exact"]:
        verdict = INCONCLUSIVE_INSTRUMENT
    elif all(checks.values()):
        verdict = SUPPORTED
    else:
        verdict = REFUTED

    body = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "supported": verdict == SUPPORTED,
        "report_sha256": report["report_sha256"],
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "evaluation_seed": report.get("evaluation_seed"),
        "task_count": task_count,
        "task_depths": list(task_depths),
        "recurrence_depth": depth,
        "arms": {
            "base_t1": base["correct"],
            "trained_t1": trained_t1["correct"],
            control_name: control["correct"],
            treatment_name: treatment["correct"],
            grammar_name: grammar["correct"],
            pointer_name: pointer["correct"],
            compiled_name: compiled["correct"],
        },
        "paired_effect": dict(effect),
        "checks": checks,
        "claim_boundary": (
            "A supported verdict proves bounded prompt-disjoint resident-32B "
            "neural transfer on the typed recurrent task battery, relative to "
            "an initialization-matched control and mechanism lesions. It does "
            "not prove broad reasoning, frontier performance, production "
            "fusion, or a WOW Signal."
        ),
    }
    return {**body, "verdict_sha256": canonical_sha256(body)}


def _completed_report(
    campaign: Path,
    evaluation_root: Path | None = None,
) -> dict[str, Any]:
    arguments = argparse.Namespace(campaign=campaign, output=evaluation_root)
    plan = launcher._existing_plan(arguments)  # noqa: SLF001
    if plan is None:
        _fail("resident evaluation plan is unavailable")
    status = launcher.status(arguments)
    if status.get("state") != "completed" or not isinstance(status.get("report"), dict):
        _fail("resident evaluation is not complete")
    return dict(status["report"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = _completed_report(
            arguments.campaign.expanduser().resolve(strict=True),
            arguments.evaluation_root.expanduser().absolute()
            if arguments.evaluation_root is not None
            else None,
        )
        verdict = adjudicate_report(report)
        if arguments.output is not None:
            atomic_write_bytes(
                arguments.output.expanduser().resolve(),
                canonical_bytes(verdict) + b"\n",
                mode=0o400,
            )
    except (OSError, ValueError, ResidentTransferAdjudicationError) as exc:
        print(f"resident transfer adjudication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["supported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
