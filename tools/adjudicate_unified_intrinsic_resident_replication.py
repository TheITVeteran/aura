#!/usr/bin/env python3
"""Freeze and adjudicate a powered resident-transfer replication."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.exact_paired_statistics import (  # noqa: E402
    exact_paired_binomial_tail,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from tools import adjudicate_unified_intrinsic_resident_transfer as single  # noqa: E402
from tools import launch_unified_intrinsic_resident_evaluation as launcher  # noqa: E402
from tools import run_unified_intrinsic_resident_campaign as resident  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)

PLAN_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_plan.v1"
VERDICT_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_verdict.v1"
SUPPORTED: Final = "supported_powered_resident_replication"
REFUTED: Final = "refuted_powered_resident_replication"
DEFAULT_SEEDS: Final = (20260811261, 20260811262, 20260811263)


class ResidentReplicationError(RuntimeError):
    """Replication evidence is incomplete, malformed or inconsistent."""


def _fail(message: str) -> Never:
    raise ResidentReplicationError(message)


def _csv_unique_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(parsed) < 2 or len(parsed) != len(set(parsed)) or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("at least two unique non-negative seeds are required")
    return parsed


def _read_canonical(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResidentReplicationError(f"replication document is unreadable: {path}") from exc
    if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
        _fail(f"replication document is not canonical: {path}")
    return value


def _campaign(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = arguments.campaign.expanduser().resolve(strict=True)
    config = resident._load_config(root / "campaign.json")  # noqa: SLF001
    if Path(config["paths"]["campaign_root"]).resolve(strict=True) != root:
        _fail("replication campaign root differs")
    return root, config


def _replication_root(arguments: argparse.Namespace, campaign: Path) -> Path:
    root = (
        arguments.output.expanduser().absolute()
        if arguments.output is not None
        else campaign / "resident-replication"
    )
    if root == campaign or not root.is_relative_to(campaign):
        _fail("replication output must be a strict campaign child")
    return root


def _plan_path(arguments: argparse.Namespace, campaign: Path) -> Path:
    return _replication_root(arguments, campaign) / "replication-plan.json"


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, config = _campaign(arguments)
    root = _replication_root(arguments, campaign)
    ensure_private_directory(root)
    seeds = tuple(arguments.seeds)
    task_depths = tuple(arguments.task_depths)
    recurrence_depths = tuple(arguments.recurrence_depths)
    task_count_per_seed = (
        len(str(config["training"]["families"]).split(",")) * len(task_depths) * arguments.per_cell
    )
    evaluations = [
        {
            "seed": seed,
            "output": str(root / f"seed-{seed}"),
        }
        for seed in seeds
    ]
    body = {
        "schema": PLAN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_root": str(campaign),
        "campaign_config_sha256": config["config_sha256"],
        "source_commit": config["source"]["git"]["commit"],
        "checkpoint_contract": "exact_terminal_answer_bridge_admitted_checkpoint",
        "seeds": list(seeds),
        "per_cell": arguments.per_cell,
        "task_depths": list(task_depths),
        "recurrence_depths": list(recurrence_depths),
        "max_tokens": arguments.max_tokens,
        "task_count_per_seed": task_count_per_seed,
        "total_tasks": task_count_per_seed * len(seeds),
        "total_candidates": task_count_per_seed * len(seeds) * 8,
        "evaluations": evaluations,
        "decision_rule": {
            "alpha_numerator": 1,
            "alpha_denominator": 100,
            "minimum_pooled_effect_numerator": 1,
            "minimum_pooled_effect_denominator": 5,
            "each_seed_positive_matched_control_gain": True,
            "zero_pooled_right_to_wrong": True,
            "compiled_exact_every_seed": True,
            "strict_aggregate_grammar_lesion_loss": True,
            "strict_aggregate_pointer_lesion_loss": True,
            "strict_aggregate_base_loss": True,
            "strict_aggregate_trained_t1_loss": True,
            "task_and_prompt_identity_disjoint_across_seeds": True,
        },
        "claim_boundary": (
            "A supported verdict proves powered multi-seed resident-32B neural "
            "transfer only on the typed recurrent task battery. It does not "
            "prove broad reasoning, frontier performance, production fusion, "
            "or a WOW Signal."
        ),
    }
    plan = {**body, "plan_sha256": canonical_sha256(body)}
    path = root / "replication-plan.json"
    payload = canonical_bytes(plan) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != plan:
            _fail("replication plan already differs")
    return plan


def _load_plan(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    campaign, config = _campaign(arguments)
    plan = _read_canonical(_plan_path(arguments, campaign))
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != canonical_sha256(body)
        or plan.get("campaign_root") != str(campaign)
        or plan.get("campaign_id") != config["campaign_id"]
        or plan.get("campaign_config_sha256") != config["config_sha256"]
        or plan.get("source_commit") != config["source"]["git"]["commit"]
    ):
        _fail("replication plan identity differs")
    return campaign, config, plan


def _evaluation_arguments(
    campaign: Path, plan: Mapping[str, Any], row: Mapping[str, Any]
) -> argparse.Namespace:
    return argparse.Namespace(
        action="status",
        campaign=campaign,
        output=Path(str(row["output"])),
        stem="checkpoint_answer_bridge_admitted",
        per_cell=int(plan["per_cell"]),
        evaluation_seed=int(row["seed"]),
        max_tokens=int(plan["max_tokens"]),
        task_depths=tuple(plan["task_depths"]),
        recurrence_depths=tuple(plan["recurrence_depths"]),
        memory_limit_gb=40.0,
        cache_limit_gb=2.0,
        wired_limit_gb=48.0,
        startup_lethal_mb=launcher.DEFAULT_STARTUP_LETHAL_MB,
        steady_lethal_mb=launcher.DEFAULT_STEADY_LETHAL_MB,
        timeout=4 * 60 * 60,
    )


def status(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, _config, plan = _load_plan(arguments)
    rows: list[dict[str, Any]] = []
    for evaluation in plan["evaluations"]:
        output = Path(evaluation["output"])
        state = "pending"
        detail: dict[str, Any] | None = None
        if (output / "evaluation-plan.json").exists():
            detail = launcher.status(_evaluation_arguments(campaign, plan, evaluation))
            state = str(detail["state"])
        rows.append({"seed": evaluation["seed"], "state": state, "detail": detail})
    return {
        "schema": "aura.unified_intrinsic.resident_replication_status.v1",
        "plan_sha256": plan["plan_sha256"],
        "evaluations": rows,
        "complete": all(row["state"] == "completed" for row in rows),
    }


def launch_next(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, _config, plan = _load_plan(arguments)
    observed = status(arguments)
    for index, row in enumerate(observed["evaluations"]):
        if row["state"] == "failed":
            _fail(f"replication evaluation failed: seed={row['seed']}")
        if row["state"] == "running":
            return {"state": "running", "seed": row["seed"], "status": row["detail"]}
        if row["state"] == "pending":
            evaluation = plan["evaluations"][index]
            launch_arguments = _evaluation_arguments(campaign, plan, evaluation)
            launch_arguments.action = "launch"
            return {
                "state": "launched",
                "seed": evaluation["seed"],
                "launch": launcher.launch(launch_arguments),
            }
    return {"state": "completed", "status": observed}


def _arm(report: Mapping[str, Any], name: str) -> int:
    arms = report.get("arm_results")
    row = arms.get(name) if isinstance(arms, dict) else None
    if not isinstance(row, dict) or type(row.get("correct")) is not int:
        _fail(f"replication arm is missing: {name}")
    return int(row["correct"])


def adjudicate(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, config, plan = _load_plan(arguments)
    completion = launcher._read_document(campaign / "completion-receipt.json")  # noqa: SLF001
    checkpoint = completion.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("complete") is not True:
        _fail("replication terminal checkpoint is unavailable")
    checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    depth = int(plan["recurrence_depths"][0])
    names = {
        "base": "base_t1",
        "trained_t1": "trained_t1",
        "control": f"untrained_t{depth}",
        "treatment": f"trained_t{depth}",
        "grammar": f"grammar_lesion_t{depth}",
        "pointer": f"pointer_lesion_t{depth}",
        "compiled": f"compiled_t{depth}",
    }
    totals = {name: 0 for name in names}
    wrong_to_right = 0
    right_to_wrong = 0
    reports: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    prompt_sha256s: set[str] = set()
    every_seed_positive = True
    instruments_exact = True
    for evaluation in plan["evaluations"]:
        evaluation_status = launcher.status(_evaluation_arguments(campaign, plan, evaluation))
        report = evaluation_status.get("report")
        if evaluation_status.get("state") != "completed" or not isinstance(report, dict):
            _fail(f"replication evaluation is incomplete: seed={evaluation['seed']}")
        report_body = {key: value for key, value in report.items() if key != "report_sha256"}
        if (
            report.get("report_sha256") != canonical_sha256(report_body)
            or report.get("checkpoint_sha256") != checkpoint_sha256
            or report.get("evaluation_seed") != evaluation["seed"]
            or report.get("per_cell") != plan["per_cell"]
            or report.get("task_depths") != plan["task_depths"]
            or report.get("recurrence_depths") != plan["recurrence_depths"]
            or report.get("task_count") != plan["task_count_per_seed"]
        ):
            _fail(f"replication report identity differs: seed={evaluation['seed']}")
        try:
            single_verdict = single.adjudicate_report(report)
        except single.ResidentTransferAdjudicationError as exc:
            raise ResidentReplicationError(
                f"replication seed evidence is malformed: seed={evaluation['seed']}"
            ) from exc
        if single_verdict.get("supported") is not True:
            _fail(f"replication seed verdict is not supported: seed={evaluation['seed']}")
        candidates = report.get("candidates")
        if not isinstance(candidates, list):
            _fail("replication candidates are unavailable")
        current_task_ids = {
            row.get("task_id")
            for row in candidates
            if isinstance(row, dict) and isinstance(row.get("task_id"), str)
        }
        current_prompts = {
            row.get("prompt_sha256")
            for row in candidates
            if isinstance(row, dict)
            and isinstance(row.get("prompt_sha256"), str)
            and len(row["prompt_sha256"]) == 64
            and all(character in "0123456789abcdef" for character in row["prompt_sha256"])
        }
        if len(current_task_ids) != int(report["task_count"]) or len(current_prompts) != int(
            report["task_count"]
        ):
            _fail("replication task or prompt identity is malformed")
        if task_ids & current_task_ids or prompt_sha256s & current_prompts:
            _fail("replication tasks or prompts overlap across seeds")
        task_ids.update(current_task_ids)
        prompt_sha256s.update(current_prompts)
        for role, arm_name in names.items():
            totals[role] += _arm(report, arm_name)
        effect = report.get("paired_training_effects", {}).get(str(depth), {})
        if not isinstance(effect, dict):
            _fail("replication paired effect is unavailable")
        wins = effect.get("wrong_to_right")
        losses = effect.get("right_to_wrong")
        net = effect.get("net_correct_gain")
        if any(type(value) is not int for value in (wins, losses, net)):
            _fail("replication paired effect is malformed")
        wrong_to_right += int(wins)
        right_to_wrong += int(losses)
        every_seed_positive &= int(net) > 0
        instruments_exact &= _arm(report, names["compiled"]) == int(report["task_count"])
        reports.append(
            {
                "seed": evaluation["seed"],
                "report_sha256": report["report_sha256"],
                "single_verdict_sha256": single_verdict["verdict_sha256"],
                "wrong_to_right": wins,
                "right_to_wrong": losses,
                "net_correct_gain": net,
            }
        )
    total_tasks = int(plan["total_tasks"])
    tail = exact_paired_binomial_tail(wrong_to_right, right_to_wrong)
    pvalue = Fraction(tail.numerator, tail.denominator)
    alpha = Fraction(1, 100)
    effect = Fraction(totals["treatment"] - totals["control"], total_tasks)
    minimum_effect = Fraction(1, 5)
    checks = {
        "all_evaluations_present": len(reports) == len(plan["seeds"]),
        "compiled_instrument_exact": instruments_exact,
        "every_seed_positive_matched_control_gain": every_seed_positive,
        "zero_right_to_wrong": right_to_wrong == 0,
        "pooled_exact_p_at_most_one_percent": pvalue <= alpha,
        "pooled_effect_at_least_twenty_percent": effect >= minimum_effect,
        "treatment_beats_base": totals["treatment"] > totals["base"],
        "recurrence_beats_trained_t1": totals["treatment"] > totals["trained_t1"],
        "grammar_lesion_loses": totals["treatment"] > totals["grammar"],
        "pointer_lesion_loses": totals["treatment"] > totals["pointer"],
    }
    supported = all(checks.values())
    body = {
        "schema": VERDICT_SCHEMA,
        "verdict": SUPPORTED if supported else REFUTED,
        "supported": supported,
        "plan_sha256": plan["plan_sha256"],
        "campaign_config_sha256": config["config_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "reports": reports,
        "total_tasks": total_tasks,
        "arms": totals,
        "paired_effect": {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "net_correct_gain": totals["treatment"] - totals["control"],
            "effect_numerator": effect.numerator,
            "effect_denominator": effect.denominator,
            "one_sided_exact_p_numerator": pvalue.numerator,
            "one_sided_exact_p_denominator": pvalue.denominator,
        },
        "checks": checks,
        "claim_boundary": plan["claim_boundary"],
    }
    verdict = {**body, "verdict_sha256": canonical_sha256(body)}
    if arguments.verdict_output is not None:
        atomic_write_bytes(
            arguments.verdict_output.expanduser().absolute(),
            canonical_bytes(verdict) + b"\n",
            mode=0o400,
        )
    return verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "launch-next", "adjudicate"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verdict-output", type=Path)
    parser.add_argument(
        "--seeds",
        type=_csv_unique_ints,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--task-depths",
        type=lambda value: launcher._csv_positive_ints(value, minimum=1),  # noqa: SLF001
        default=(1, 2, 4),
    )
    parser.add_argument(
        "--recurrence-depths",
        type=lambda value: launcher._csv_positive_ints(value, minimum=2),  # noqa: SLF001
        default=(4,),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.per_cell < 1 or arguments.max_tokens < 1 or len(arguments.recurrence_depths) != 1:
        parser.error("replication numeric contract is invalid")
    try:
        result = {
            "prepare": prepare,
            "status": status,
            "launch-next": launch_next,
            "adjudicate": adjudicate,
        }[arguments.action](arguments)
    except (OSError, ValueError, ResidentReplicationError) as exc:
        print(f"resident replication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.action == "adjudicate" and result.get("supported") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
