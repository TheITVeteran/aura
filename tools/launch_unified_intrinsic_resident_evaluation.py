#!/usr/bin/env python3
"""Launch a frozen resident unified-recurrence decode evaluation safely.

Training and evaluation both load the resident model, so they require the same
detached custody, sleep inhibition, preload memory sentinel, and immutable
scientific inputs.  This launcher adds those operational guarantees without
changing the evaluator committed into the campaign's source capsule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.mlx_memory_guard import host_pressure  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools import run_unified_intrinsic_resident_campaign as resident  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_preload_barrier import (  # noqa: E402
    command_sha256,
    publish_release,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)

PLAN_SCHEMA: Final = "aura.unified_intrinsic.resident_evaluation_plan.v1"
STATUS_SCHEMA: Final = "aura.unified_intrinsic.resident_evaluation_status.v1"
DEFAULT_STARTUP_LETHAL_MB: Final = 54.0 * 1024.0
DEFAULT_STEADY_LETHAL_MB: Final = 48.0 * 1024.0
PRELOAD_TIMEOUT_S: Final = 300.0


class ResidentEvaluationLaunchError(RuntimeError):
    """Stable failure boundary for source-bound resident evaluation."""


def _fail(message: str) -> Never:
    raise ResidentEvaluationLaunchError(message)


def _read_document(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidentEvaluationLaunchError(
            f"resident evaluation document is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
        _fail(f"resident evaluation document is not canonical: {path}")
    return value


def _read_report_document(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read old pretty reports without weakening semantic report custody."""

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ResidentEvaluationLaunchError(
            f"resident evaluation report is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"resident evaluation report is not an object: {path}")
    canonical = canonical_bytes(value) + b"\n"
    return value, {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_bytes": raw == canonical,
        "size_bytes": len(raw),
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    ensure_private_directory(path.parent)
    payload = canonical_bytes(dict(value)) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        current = _read_document(path)
        if current != dict(value):
            _fail(f"resident evaluation artifact already differs: {path}")


def _csv_positive_ints(value: str, *, minimum: int) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from exc
    if not parsed or len(parsed) != len(set(parsed)) or any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(
            f"depths must be unique integers greater than or equal to {minimum}"
        )
    return parsed


def _terminal_campaign(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = resident._load_config(config_path)  # noqa: SLF001
    root = Path(config["paths"]["campaign_root"])
    completion = _read_document(root / "completion-receipt.json")
    body = {key: value for key, value in completion.items() if key != "completion_sha256"}
    checkpoint = resident._checkpoint_snapshot(config)  # noqa: SLF001
    if (
        completion.get("schema") != resident.COMPLETION_SCHEMA
        or completion.get("config_sha256") != config["config_sha256"]
        or completion.get("completion_sha256") != canonical_sha256(body)
        or completion.get("resident_training_complete") is not True
        or completion.get("checkpoint") != checkpoint
        or checkpoint.get("complete") is not True
    ):
        _fail("resident evaluation requires the exact terminal campaign checkpoint")
    return config, completion


def _selected_checkpoint(
    config: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    stem: str,
) -> dict[str, Any]:
    """Resolve the exact checkpoint named by an evaluation before model load."""

    output = Path(str(config["paths"]["training_output"]))
    try:
        selected = resolve_checkpoint_generation(output, stem=stem, required=True)
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise ResidentEvaluationLaunchError(
            f"resident evaluation checkpoint is unavailable: {stem}"
        ) from exc
    if selected is None:  # pragma: no cover - required=True is authoritative
        _fail(f"resident evaluation checkpoint is unavailable: {stem}")
    receipt = selected.receipt
    checkpoint = completion.get("checkpoint")
    if not isinstance(checkpoint, dict):
        _fail("resident evaluation completion checkpoint is malformed")
    if stem == "checkpoint_answer_bridge_admitted":
        training_receipt = checkpoint.get("training_receipt")
        training_receipt_path = output / "training_receipt.json"
        persisted = _read_document(training_receipt_path)
        admission = persisted.get("answer_bridge_admission")
        if (
            not isinstance(training_receipt, dict)
            or training_receipt.get("binding") != "authoritative_checkpoint"
            or persisted.get("receipt_sha256") != training_receipt.get("receipt_sha256")
            or persisted.get("verdict") != "answer_bridge_admitted"
            or not isinstance(admission, dict)
            or admission.get("admitted") is not True
            or admission.get("exact") != admission.get("tasks")
            or type(admission.get("tasks")) is not int
            or int(admission["tasks"]) < 1
            or receipt.get("step") != persisted.get("steps")
        ):
            _fail("resident evaluation requires an admitted answer-bridge checkpoint")
    return {
        "stem": stem,
        "step": receipt["step"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "identity_sha256": receipt["identity"]["identity_sha256"],
    }


def _evaluation_root(arguments: argparse.Namespace) -> tuple[Path, Path]:
    campaign_root = arguments.campaign.expanduser().resolve(strict=True)
    evaluation_root = (
        arguments.output.expanduser().absolute()
        if arguments.output is not None
        else campaign_root / "resident-evaluation"
    )
    if evaluation_root == campaign_root or not evaluation_root.is_relative_to(campaign_root):
        _fail("resident evaluation output must be a strict campaign child")
    return campaign_root, evaluation_root


def _existing_plan(arguments: argparse.Namespace) -> dict[str, Any] | None:
    campaign_root, evaluation_root = _evaluation_root(arguments)
    path = evaluation_root / "evaluation-plan.json"
    if not path.exists():
        return None
    plan = _read_document(path)
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != canonical_sha256(body)
        or plan.get("campaign_root") != str(campaign_root)
        or plan.get("evaluation_root") != str(evaluation_root)
    ):
        _fail("stored resident evaluation plan is invalid")
    return plan


def _build_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    requested_campaign_root, evaluation_root = _evaluation_root(arguments)
    config_path = requested_campaign_root / "campaign.json"
    config, completion = _terminal_campaign(config_path)
    selected_checkpoint = _selected_checkpoint(
        config,
        completion,
        stem=arguments.stem,
    )
    campaign_root = Path(config["paths"]["campaign_root"])
    source_root = Path(config["source"]["git"]["root"]).resolve(strict=True)
    python = Path(config["runtime"]["interpreter"]["executable"])
    evaluator = source_root / "tools/evaluate_unified_intrinsic_decoding.py"
    barrier = source_root / "tools/unified_intrinsic_preload_barrier.py"
    sentinel = source_root / "tools/memory_sentinel.py"
    for path in (python, evaluator, barrier, sentinel):
        if not path.is_file():
            _fail(f"resident evaluation executable input is unavailable: {path}")

    if campaign_root != requested_campaign_root:
        _fail("resident evaluation campaign root differs")
    ensure_private_directory(evaluation_root)
    report = evaluation_root / "decode-report.json"
    progress_dir = evaluation_root / "decode-progress"
    ready = evaluation_root / "preload-ready-{pid}.json"
    release = evaluation_root / "preload-release-{pid}.json"
    resource_stage = evaluation_root / "resource-stage-{pid}.json"
    task_depths = tuple(arguments.task_depths)
    recurrence_depths = tuple(arguments.recurrence_depths)
    scientific = {
        "campaign_config_sha256": config["config_sha256"],
        "campaign_completion_sha256": completion["completion_sha256"],
        "checkpoint_sha256": selected_checkpoint["checkpoint_sha256"],
        "checkpoint": selected_checkpoint,
        "evaluator": str(evaluator),
        "stem": arguments.stem,
        "per_cell": arguments.per_cell,
        "evaluation_seed": arguments.evaluation_seed,
        "max_tokens": arguments.max_tokens,
        "task_depths": list(task_depths),
        "recurrence_depths": list(recurrence_depths),
    }
    evaluation_identity_sha256 = canonical_sha256(scientific)
    evaluator_command = [
        str(python),
        str(evaluator),
        str(campaign_root),
        "--stem",
        arguments.stem,
        "--per-cell",
        str(arguments.per_cell),
        "--evaluation-seed",
        str(arguments.evaluation_seed),
        "--max-tokens",
        str(arguments.max_tokens),
        "--task-depths",
        ",".join(str(value) for value in task_depths),
        "--recurrence-depths",
        ",".join(str(value) for value in recurrence_depths),
        "--report",
        str(report),
        "--progress-dir",
        str(progress_dir),
        "--memory-limit-gb",
        str(arguments.memory_limit_gb),
        "--cache-limit-gb",
        str(arguments.cache_limit_gb),
        "--wired-limit-gb",
        str(arguments.wired_limit_gb),
        "--resource-stage-path",
        str(resource_stage),
        "--resource-startup-lethal-mb",
        str(arguments.startup_lethal_mb),
        "--resource-steady-lethal-mb",
        str(arguments.steady_lethal_mb),
        "--preload-ready-path",
        str(ready),
        "--preload-release-path",
        str(release),
        "--preload-key-path",
        str(config["paths"]["heartbeat_key"]),
        "--preload-config-sha256",
        evaluation_identity_sha256,
    ]
    command = [
        str(python),
        str(barrier),
        "--ready",
        str(ready),
        "--release",
        str(release),
        "--key",
        str(config["paths"]["heartbeat_key"]),
        "--config-sha256",
        evaluation_identity_sha256,
        "--timeout",
        str(PRELOAD_TIMEOUT_S),
        "--",
        *evaluator_command,
    ]
    body = {
        "schema": PLAN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scientific": scientific,
        "evaluation_identity_sha256": evaluation_identity_sha256,
        "campaign_root": str(campaign_root),
        "evaluation_root": str(evaluation_root),
        "report": str(report),
        "progress_dir": str(progress_dir),
        "ready": str(ready),
        "release": str(release),
        "resource_stage": str(resource_stage),
        "heartbeat_key": str(config["paths"]["heartbeat_key"]),
        "python": str(python),
        "sentinel": str(sentinel),
        "command": command,
        "evaluator_command": evaluator_command,
        "timeout_s": arguments.timeout,
        "startup_lethal_mb": arguments.startup_lethal_mb,
        "steady_lethal_mb": arguments.steady_lethal_mb,
        "claims_not_supported": [
            "broad_reasoning_gain",
            "frontier_performance",
            "production_fusion",
            "wow_signal",
        ],
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    plan = _build_plan(arguments)
    _write_once(Path(plan["evaluation_root"]) / "evaluation-plan.json", plan)
    return plan


def _invoke_detached(argv: list[str]) -> dict[str, Any]:
    parser = detached.build_parser()
    parsed = parser.parse_args(argv)
    return detached._launch(parsed, parser)  # noqa: SLF001


def _wait_for_target(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + PRELOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            status = detached._status(run_dir)  # noqa: SLF001
        except (OSError, ValueError, detached.DetachedStepError):
            time.sleep(0.1)
            continue
        if status.get("terminal") is True:
            _fail("resident evaluator exited before preload release")
        if int(status.get("child_pid") or 0) > 1:
            return status
        time.sleep(0.1)
    _fail("resident evaluator target identity timed out")


def _evaluation_run_dirs(root: Path) -> list[Path]:
    legacy = root / "detached"
    attempts_root = root / "detached-attempts"
    attempts = sorted(
        path
        for path in attempts_root.glob("attempt-*")
        if path.is_dir() and not path.is_symlink()
    )
    result = ([legacy] if (legacy / detached.PLAN_FILE).exists() else []) + attempts
    offset = 1 if result and result[0] == legacy else 0
    for index, path in enumerate(result[offset:], start=offset + 1):
        if path.name != f"attempt-{index:04d}":
            _fail("resident evaluation attempt sequence is not contiguous")
    return result


def _next_attempt_run_dir(root: Path) -> tuple[int, Path]:
    existing = _evaluation_run_dirs(root)
    for run_dir in existing:
        inspection = detached._status(run_dir)  # noqa: SLF001
        if inspection.get("terminal") is not True:
            _fail("resident evaluation already has a live or indeterminate attempt")
        receipt = inspection.get("receipt")
        if (
            isinstance(receipt, dict)
            and receipt.get("passed") is True
            and (root / "decode-report.json").exists()
        ):
            _fail("resident evaluation is already complete")
    attempt = len(existing) + 1
    run_dir = root / "detached-attempts" / f"attempt-{attempt:04d}"
    if run_dir.exists():
        _fail("resident evaluation attempt allocation collided")
    return attempt, run_dir


def launch(arguments: argparse.Namespace) -> dict[str, Any]:
    plan = prepare(arguments)
    config, completion = _terminal_campaign(Path(plan["campaign_root"]) / "campaign.json")
    selected = _selected_checkpoint(config, completion, stem=str(plan["scientific"]["stem"]))
    if selected != plan["scientific"].get("checkpoint"):
        _fail("resident evaluation selected checkpoint differs from its frozen plan")
    root = Path(plan["evaluation_root"])
    attempt, run_dir = _next_attempt_run_dir(root)
    detached_result = _invoke_detached(
        [
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            f"{plan['campaign_id']}-resident-evaluation",
            "--cwd",
            plan["campaign_root"],
            "--timeout",
            str(plan["timeout_s"]),
            "--resume-contract",
            "none",
            "--execution-output-root",
            str(root),
            "--",
            *plan["command"],
        ]
    )
    status = _wait_for_target(run_dir)
    target_pid, target_token, supervisor_pid = resident._target_identity(status)  # noqa: SLF001
    sentinel_dir = root / f"sentinel-{attempt:04d}-{target_pid}"
    sentinel_ring = sentinel_dir / "ring.jsonl"
    resource_stage = plan["resource_stage"].replace("{pid}", str(target_pid))
    _invoke_detached(
        [
            "launch",
            "--run-dir",
            str(sentinel_dir),
            "--name",
            f"{plan['campaign_id']}-evaluation-sentinel-{target_pid}",
            "--cwd",
            plan["campaign_root"],
            "--timeout",
            str(float(plan["timeout_s"]) + 300.0),
            "--",
            plan["python"],
            plan["sentinel"],
            "--pid",
            str(target_pid),
            "--lethal-mb",
            str(plan["steady_lethal_mb"]),
            "--startup-lethal-mb",
            str(plan["startup_lethal_mb"]),
            "--steady-marker",
            resource_stage,
            "--interval",
            "1.0",
            "--ring",
            str(sentinel_ring),
            "--ring-window-seconds",
            "7200",
            "--tombstone-dir",
            str(sentinel_dir / "tombstones"),
        ]
    )
    deadline = time.monotonic() + PRELOAD_TIMEOUT_S
    sentinel_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            sentinel_status = detached._status(sentinel_dir)  # noqa: SLF001
        except (OSError, ValueError, detached.DetachedStepError):
            time.sleep(0.1)
            continue
        if sentinel_status.get("terminal") is True:
            _fail("resident evaluation sentinel exited before preload release")
        if sentinel_ring.exists() and int(sentinel_status.get("child_pid") or 0) > 1:
            break
        time.sleep(0.1)
    if sentinel_status is None:
        _fail("resident evaluation sentinel startup timed out")
    sentinel_pid = int(sentinel_status.get("child_pid") or 0)
    sentinel_token = str(sentinel_status.get("child_start_token") or "")
    if detached._identity_state(sentinel_pid, sentinel_token) != "alive":  # noqa: SLF001
        _fail("resident evaluation sentinel identity is not live")
    ring_entry, ring_sha256 = resident._stable_last_ring_entry(  # noqa: SLF001
        sentinel_ring,
        required_stage=None,
    )
    if (
        ring_entry.get("guard_stage") != "startup"
        or float(ring_entry.get("active_lethal_mb") or 0.0)
        != float(plan["startup_lethal_mb"])
    ):
        _fail("resident evaluation sentinel startup evidence differs")
    caffeinate = resident._verify_caffeinate(target_pid, supervisor_pid)  # noqa: SLF001
    pressure = host_pressure()
    if pressure.get("available") is not True or pressure.get("under_pressure") is not False:
        _fail("host pressure denied resident evaluation preload")
    evaluator_command = [
        value.replace("{pid}", str(target_pid)) for value in plan["evaluator_command"]
    ]
    release_path = Path(plan["release"].replace("{pid}", str(target_pid)))
    release = publish_release(
        release_path,
        ready_path=Path(plan["ready"].replace("{pid}", str(target_pid))),
        key_path=Path(plan["heartbeat_key"]),
        sentinel_pid=sentinel_pid,
        sentinel_start_token=sentinel_token,
        sentinel_ring_entry_sha256=ring_sha256,
        host_pressure=pressure,
        expected_target_pid=target_pid,
        expected_target_start_token=target_token,
        expected_command_sha256=command_sha256(evaluator_command),
    )
    return {
        "schema": STATUS_SCHEMA,
        "state": "running",
        "attempt": attempt,
        "run_dir": str(run_dir),
        "plan_sha256": plan["plan_sha256"],
        "detached": detached_result,
        "target_pid": target_pid,
        "target_start_token": target_token,
        "sentinel_pid": sentinel_pid,
        "sentinel_start_token": sentinel_token,
        "sentinel_ring_entry": ring_entry,
        "caffeinate": caffeinate,
        "release": release,
    }


def status(arguments: argparse.Namespace) -> dict[str, Any]:
    plan = _existing_plan(arguments) or prepare(arguments)
    root = Path(plan["evaluation_root"])
    run_dirs = _evaluation_run_dirs(root)
    if not run_dirs:
        return {"schema": STATUS_SCHEMA, "state": "not_launched", "plan": plan}
    run_dir = run_dirs[-1]
    inspection = detached._status(run_dir)  # noqa: SLF001
    report_path = Path(plan["report"])
    report: dict[str, Any] | None = None
    report_transport: dict[str, Any] | None = None
    if report_path.exists():
        report, report_transport = _read_report_document(report_path)
        report_body = {key: value for key, value in report.items() if key != "report_sha256"}
        if report.get("report_sha256") != canonical_sha256(report_body):
            _fail("resident evaluation report hash is invalid")
        scientific = plan["scientific"]
        if (
            report.get("checkpoint_sha256") != scientific["checkpoint_sha256"]
            or report.get("evaluation_seed") != scientific["evaluation_seed"]
            or report.get("per_cell") != scientific["per_cell"]
            or report.get("max_tokens") != scientific["max_tokens"]
            or report.get("task_depths") != scientific["task_depths"]
            or report.get("recurrence_depths") != scientific["recurrence_depths"]
        ):
            _fail("resident evaluation report differs from its frozen plan")
    receipt = inspection.get("receipt")
    passed = isinstance(receipt, dict) and receipt.get("passed") is True
    state = (
        "completed"
        if inspection.get("terminal") is True and report is not None and passed
        else "running"
    )
    if inspection.get("terminal") is True and (report is None or not passed):
        state = "failed"
    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "attempt": len(run_dirs),
        "run_dir": str(run_dir),
        "attempt_count": len(run_dirs),
        "plan_sha256": plan["plan_sha256"],
        "detached": inspection,
        "report": report,
        "report_transport": report_transport,
    }


def stop(arguments: argparse.Namespace) -> dict[str, Any]:
    plan = _existing_plan(arguments) or prepare(arguments)
    root = Path(plan["evaluation_root"])
    stopped: list[dict[str, Any]] = []
    for run_dir in sorted(root.glob("sentinel-*")):
        if (run_dir / detached.PLAN_FILE).exists():
            stopped.append(detached._stop(run_dir))  # noqa: SLF001
    for run_dir in reversed(_evaluation_run_dirs(root)):
        stopped.append(detached._stop(run_dir))  # noqa: SLF001
    return {"schema": STATUS_SCHEMA, "state": "stop_requested", "results": stopped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "status", "stop"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stem", default="checkpoint_answer_bridge_admitted")
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20260811241)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--task-depths",
        type=lambda value: _csv_positive_ints(value, minimum=1),
        default=(1, 2, 4),
    )
    parser.add_argument(
        "--recurrence-depths",
        type=lambda value: _csv_positive_ints(value, minimum=2),
        default=(4,),
    )
    parser.add_argument("--memory-limit-gb", type=float, default=40.0)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float, default=48.0)
    parser.add_argument("--startup-lethal-mb", type=float, default=DEFAULT_STARTUP_LETHAL_MB)
    parser.add_argument("--steady-lethal-mb", type=float, default=DEFAULT_STEADY_LETHAL_MB)
    parser.add_argument("--timeout", type=float, default=4 * 60 * 60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.per_cell < 1
        or arguments.max_tokens < 1
        or arguments.timeout <= PRELOAD_TIMEOUT_S
        or any(
            value <= 0.0
            for value in (
                arguments.memory_limit_gb,
                arguments.cache_limit_gb,
                arguments.wired_limit_gb,
                arguments.startup_lethal_mb,
                arguments.steady_lethal_mb,
            )
        )
        or arguments.steady_lethal_mb > arguments.startup_lethal_mb
    ):
        parser.error("resident evaluation numeric contract is invalid")
    try:
        result = {
            "prepare": prepare,
            "launch": launch,
            "status": status,
            "stop": stop,
        }[arguments.action](arguments)
    except (OSError, ValueError, ResidentEvaluationLaunchError) as exc:
        print(f"resident evaluation launcher failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
