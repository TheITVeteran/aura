#!/usr/bin/env python3
"""Freeze and adjudicate a powered resident-transfer replication."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import plistlib
import re
import subprocess
import sys
import time
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

PLAN_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_plan.v4"
VERDICT_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_verdict.v2"
CONTROLLER_STATUS_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_controller_status.v1"
LAUNCH_INTENT_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_launch_intent.v1"
LAUNCH_RECEIPT_SCHEMA: Final = "aura.unified_intrinsic.resident_replication_launchd.v1"
SUPPORTED: Final = "supported_powered_resident_replication"
REFUTED: Final = "refuted_powered_resident_replication"
DEFAULT_SEEDS: Final = (20260811261, 20260811262, 20260811263)
LAUNCH_AGENTS_ROOT: Final = Path.home() / "Library/LaunchAgents"


class ResidentReplicationError(RuntimeError):
    """Replication evidence is incomplete, malformed or inconsistent."""


class ResidentReplicationIncompleteError(ResidentReplicationError):
    """Replication is incomplete and no preregistered conclusion is yet forced."""


def _fail(message: str) -> Never:
    raise ResidentReplicationError(message)


def _incomplete(message: str) -> Never:
    raise ResidentReplicationIncompleteError(message)


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


def _controller_status_path(arguments: argparse.Namespace, campaign: Path) -> Path:
    return _replication_root(arguments, campaign) / "controller-status.json"


def _launch_label(config: Mapping[str, Any]) -> str:
    campaign_id = str(config.get("campaign_id") or "")
    if not campaign_id or re.fullmatch(r"[A-Za-z0-9._-]+", campaign_id) is None:
        _fail("replication launchd campaign id is invalid")
    return f"com.aura.unified-intrinsic-replication.{campaign_id}"


def _runtime_python(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    runtime = config.get("runtime")
    interpreter = runtime.get("interpreter") if isinstance(runtime, Mapping) else None
    if not isinstance(interpreter, Mapping):
        _fail("replication runtime interpreter attestation is unavailable")
    executable_value = interpreter.get("executable")
    prefix_value = interpreter.get("sys_prefix")
    real_value = interpreter.get("real_executable")
    sha256_value = interpreter.get("sha256")
    size_value = interpreter.get("size_bytes")
    if not all(isinstance(value, str) and value for value in (executable_value, prefix_value)):
        _fail("replication runtime interpreter path is malformed")
    executable = Path(executable_value)
    prefix = Path(prefix_value)
    if not executable.is_absolute() or not prefix.is_absolute():
        _fail("replication runtime interpreter path is not absolute")
    if executable != prefix / "bin/python":
        _fail("replication runtime interpreter is not the attested virtualenv python")
    try:
        stat = executable.stat()
        real_executable = executable.resolve(strict=True)
        digest = hashlib.sha256(real_executable.read_bytes()).hexdigest()
    except OSError as exc:
        raise ResidentReplicationError("replication runtime interpreter is unavailable") from exc
    if (
        not isinstance(real_value, str)
        or real_executable != Path(real_value)
        or not isinstance(sha256_value, str)
        or digest != sha256_value
        or type(size_value) is not int
        or stat.st_size != size_value
    ):
        _fail("replication runtime interpreter attestation differs")
    return executable, {
        "executable": str(executable),
        "real_executable": str(real_executable),
        "sys_prefix": str(prefix),
        "sha256": digest,
        "size_bytes": stat.st_size,
    }


def _probe_runtime_python(config: Mapping[str, Any]) -> dict[str, Any]:
    python, identity = _runtime_python(config)
    probe_source = (
        "import json,sys; import mlx.core as mx; "
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'mlx_module':mx.__file__},sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", probe_source],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:300]
        _fail(f"replication runtime MLX preflight failed: {detail}")
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ResidentReplicationError(
            "replication runtime MLX preflight output is malformed"
        ) from exc
    if (
        not isinstance(observed, dict)
        or observed.get("executable") != identity["executable"]
        or observed.get("prefix") != identity["sys_prefix"]
        or not isinstance(observed.get("mlx_module"), str)
        or not observed["mlx_module"]
    ):
        _fail("replication runtime MLX preflight identity differs")
    body = {
        "interpreter": identity,
        "mlx_module": observed["mlx_module"],
    }
    return {**body, "probe_sha256": canonical_sha256(body)}


def _launch_contract(
    arguments: argparse.Namespace,
    campaign: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[Path, bytes, dict[str, Any]]:
    root = _replication_root(arguments, campaign)
    source_root = Path(__file__).resolve(strict=True).parent.parent
    script = (source_root / "tools/adjudicate_unified_intrinsic_resident_replication.py").resolve(
        strict=True
    )
    python, interpreter = _runtime_python(config)
    label = _launch_label(config)
    command = [
        str(python),
        str(script),
        "run",
        str(campaign),
        "--controller-timeout",
        str(float(arguments.controller_timeout)),
        "--poll-interval",
        str(float(arguments.poll_interval)),
        "--launchd-supervised",
    ]
    if arguments.output is not None:
        command.extend(["--output", str(root)])
    payload = {
        "Label": label,
        "ProgramArguments": command,
        "WorkingDirectory": str(campaign),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "EnvironmentVariables": {"VIRTUAL_ENV": interpreter["sys_prefix"]},
        "StandardOutPath": str(root / "controller-launchd.log"),
        "StandardErrorPath": str(root / "controller-launchd.log"),
    }
    plist = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    plist_path = LAUNCH_AGENTS_ROOT / f"{label}.plist"
    body = {
        "schema": LAUNCH_INTENT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "label": label,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(plist).hexdigest(),
        "program_arguments": command,
        "working_directory": str(campaign),
        "controller_source": str(script),
        "controller_source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "interpreter": interpreter,
    }
    return plist_path, plist, {**body, "intent_sha256": canonical_sha256(body)}


def _launchd_job(label: str) -> dict[str, Any]:
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("replication launchd job is unavailable")
    pid = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                pid = int(stripped.removeprefix("pid = "))
            except ValueError:
                _fail("replication launchd pid is invalid")
            break
    if pid <= 1:
        _fail("replication launchd pid is unavailable")
    return {"target": target, "pid": pid}


def _verify_launchd_supervision(
    arguments: argparse.Namespace,
    campaign: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if getattr(arguments, "launchd_supervised", False) is not True:
        _fail("replication launchd supervision is required")
    plist_path, expected_plist, expected_intent = _launch_contract(
        arguments, campaign, config, plan
    )
    intent = _read_canonical(_replication_root(arguments, campaign) / "launch-intent.json")
    if intent != expected_intent or plist_path.read_bytes() != expected_plist:
        _fail("replication launchd intent differs")
    job = _launchd_job(_launch_label(config))
    if job["pid"] != os.getpid():
        _fail("replication launchd controller pid differs")
    return {
        "target": job["target"],
        "controller_pid": job["pid"],
        "controller_start_token": launcher.detached._process_start_token(job["pid"]),  # noqa: SLF001
        "intent_sha256": intent["intent_sha256"],
        "plist_sha256": intent["plist_sha256"],
    }


def install_launchd(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, config, plan = _load_plan(arguments)
    root = _replication_root(arguments, campaign)
    ensure_private_directory(root)
    runtime_probe = _probe_runtime_python(config)
    plist_path, plist, intent = _launch_contract(arguments, campaign, config, plan)
    intent_path = root / "launch-intent.json"
    payload = canonical_bytes(intent) + b"\n"
    if not atomic_write_bytes_if_absent(intent_path, payload, mode=0o400):
        if _read_canonical(intent_path) != intent:
            _fail("replication launch intent already differs")
    ensure_private_directory(LAUNCH_AGENTS_ROOT)
    atomic_write_bytes(plist_path, plist, mode=0o600)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    started = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if started.returncode != 0:
        _fail(f"replication launchd bootstrap failed: {started.stderr.strip()[:300]}")
    deadline = time.monotonic() + 20.0
    job: dict[str, Any] | None = None
    controller: dict[str, Any] | None = None
    authenticated = False
    while time.monotonic() < deadline:
        try:
            job = _launchd_job(_launch_label(config))
            controller = _read_controller_status(arguments, campaign, config, plan)
            if (
                controller is not None
                and controller.get("controller_pid") == job["pid"]
                and launcher.detached._identity_state(  # noqa: SLF001
                    job["pid"], str(controller.get("controller_start_token") or "")
                )
                == "alive"
            ):
                authenticated = True
                break
        except ResidentReplicationError:
            job = None
        time.sleep(0.25)
    if not authenticated or job is None or controller is None:
        subprocess.run(
            ["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{_launch_label(config)}"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        _fail("replication launchd authenticated start timed out")
    body = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "target": job["target"],
        "pid": job["pid"],
        "start_token": launcher.detached._process_start_token(job["pid"]),  # noqa: SLF001
        "plist_path": str(plist_path),
        "plist_sha256": intent["plist_sha256"],
        "runtime_probe": runtime_probe,
        "initial_controller_status_sha256": canonical_sha256(controller),
        "installed_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "launch_sha256": canonical_sha256(body)}
    atomic_write_bytes(
        root / "launchd-receipt.json",
        canonical_bytes(receipt) + b"\n",
        mode=0o600,
    )
    return receipt


def _controller_signature(body: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_bytes(dict(body)), hashlib.sha256).hexdigest()


def _controller_key(config: Mapping[str, Any]) -> bytes:
    return resident._key(  # noqa: SLF001
        Path(config["paths"]["heartbeat_key"]),
        expected_sha256=str(config["heartbeat_key_sha256"]),
    )


def _read_controller_status(
    arguments: argparse.Namespace,
    campaign: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _controller_status_path(arguments, campaign)
    if not path.exists():
        return None
    status = _read_canonical(path)
    body = {key: value for key, value in status.items() if key != "hmac_sha256"}
    signature = status.get("hmac_sha256")
    if (
        status.get("schema") != CONTROLLER_STATUS_SCHEMA
        or status.get("plan_sha256") != plan["plan_sha256"]
        or status.get("campaign_config_sha256") != config["config_sha256"]
        or type(status.get("sequence")) is not int
        or status["sequence"] < 1
        or not isinstance(signature, str)
        or not hmac.compare_digest(
            signature,
            _controller_signature(body, _controller_key(config)),
        )
    ):
        _fail("replication controller status authentication failed")
    return status


def _publish_controller_status(
    arguments: argparse.Namespace,
    campaign: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    previous = _read_controller_status(arguments, campaign, config, plan)
    body = {
        "schema": CONTROLLER_STATUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
        "state": state,
        "controller_pid": os.getpid(),
        "controller_start_token": launcher.detached._process_start_token(  # noqa: SLF001
            os.getpid()
        ),
        "heartbeat_at": time.time(),
        "details": dict(details),
    }
    status = {
        **body,
        "hmac_sha256": _controller_signature(body, _controller_key(config)),
    }
    atomic_write_bytes(
        _controller_status_path(arguments, campaign),
        canonical_bytes(status) + b"\n",
        mode=0o600,
    )
    return status


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
    evaluator_source_root = (
        arguments.evaluator_source_root.expanduser().resolve(strict=True)
        if arguments.evaluator_source_root is not None
        else Path(config["source"]["git"]["root"]).resolve(strict=True)
    )
    matched_control = (
        launcher.root_control_binding(
            arguments.matched_control_campaign,
            stem=arguments.matched_control_stem,
        )
        if arguments.matched_control_campaign is not None
        else None
    )
    body = {
        "schema": PLAN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_root": str(campaign),
        "campaign_config_sha256": config["config_sha256"],
        "source_commit": config["source"]["git"]["commit"],
        "evaluator_source_root": str(evaluator_source_root),
        "evaluator_source_sha256s": launcher._evaluator_source_sha256s(  # noqa: SLF001
            evaluator_source_root
        ),
        "matched_control": matched_control,
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
    evaluator_source_root = Path(str(plan.get("evaluator_source_root") or ""))
    try:
        observed_evaluator_source = launcher._evaluator_source_sha256s(  # noqa: SLF001
            evaluator_source_root
        )
    except launcher.ResidentEvaluationLaunchError as exc:
        raise ResidentReplicationError("replication evaluator source is unavailable") from exc
    if not evaluator_source_root.is_absolute() or observed_evaluator_source != plan.get(
        "evaluator_source_sha256s"
    ):
        _fail("replication evaluator source differs")
    matched_control = plan.get("matched_control")
    if matched_control is not None:
        if not isinstance(matched_control, dict):
            _fail("replication matched control is malformed")
        try:
            observed_control = launcher.root_control_binding(
                Path(str(matched_control.get("campaign_root") or "")),
                stem=str(matched_control.get("stem") or ""),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ResidentReplicationError("replication matched control is unavailable") from exc
        if observed_control != matched_control:
            _fail("replication matched control differs")
    return campaign, config, plan


def _evaluation_arguments(
    campaign: Path, plan: Mapping[str, Any], row: Mapping[str, Any]
) -> argparse.Namespace:
    return argparse.Namespace(
        action="status",
        campaign=campaign,
        output=Path(str(row["output"])),
        evaluator_source_root=Path(str(plan["evaluator_source_root"])),
        matched_control_campaign=(
            Path(str(plan["matched_control"]["campaign_root"]))
            if plan.get("matched_control") is not None
            else None
        ),
        matched_control_stem=(
            str(plan["matched_control"]["stem"])
            if plan.get("matched_control") is not None
            else "checkpoint_latest"
        ),
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
    campaign, config, plan = _load_plan(arguments)
    rows: list[dict[str, Any]] = []
    for evaluation in plan["evaluations"]:
        output = Path(evaluation["output"])
        state = "pending"
        detail: dict[str, Any] | None = None
        if (output / "evaluation-plan.json").exists():
            detail = launcher.status(_evaluation_arguments(campaign, plan, evaluation))
            state = str(detail["state"])
        rows.append({"seed": evaluation["seed"], "state": state, "detail": detail})
    controller = _read_controller_status(arguments, campaign, config, plan)
    controller_liveness = None
    if controller is not None:
        controller_liveness = launcher.detached._identity_state(  # noqa: SLF001
            int(controller.get("controller_pid") or 0),
            str(controller.get("controller_start_token") or ""),
        )
    return {
        "schema": "aura.unified_intrinsic.resident_replication_status.v1",
        "plan_sha256": plan["plan_sha256"],
        "evaluations": rows,
        "complete": all(row["state"] == "completed" for row in rows),
        "controller": controller,
        "controller_liveness": controller_liveness,
    }


def launch_next(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, _config, plan = _load_plan(arguments)
    observed = status(arguments)
    for index, row in enumerate(observed["evaluations"]):
        if row["state"] in {"failed", "stopped"}:
            _fail(
                "replication evaluation terminated before completion: "
                f"seed={row['seed']} state={row['state']}"
            )
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
        if row["state"] != "completed":
            _fail(
                "replication evaluation state is unsupported: "
                f"seed={row['seed']} state={row['state']}"
            )
    return {"state": "completed", "status": observed}


def _arm(report: Mapping[str, Any], name: str) -> int:
    arms = report.get("arm_results")
    row = arms.get(name) if isinstance(arms, dict) else None
    if not isinstance(row, dict) or type(row.get("correct")) is not int:
        _fail(f"replication arm is missing: {name}")
    return int(row["correct"])


def _write_verdict(arguments: argparse.Namespace, verdict: Mapping[str, Any]) -> None:
    if arguments.verdict_output is None:
        return
    path = arguments.verdict_output.expanduser().absolute()
    payload = canonical_bytes(dict(verdict)) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != verdict:
            _fail("replication verdict already differs")


def _collect_evidence(
    campaign: Path,
    plan: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    completion = launcher._read_document(campaign / "completion-receipt.json")  # noqa: SLF001
    checkpoint = completion.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("complete") is not True:
        _fail("replication terminal checkpoint is unavailable")
    if checkpoint.get("checkpoint_sha256") != checkpoint_sha256:
        _fail("replication terminal checkpoint identity differs")
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
    evaluation_states: list[dict[str, Any]] = []
    for evaluation in plan["evaluations"]:
        output = Path(str(evaluation["output"]))
        if (output / "evaluation-plan.json").exists():
            evaluation_status = launcher.status(_evaluation_arguments(campaign, plan, evaluation))
        else:
            evaluation_status = {"state": "pending"}
        state = str(evaluation_status.get("state") or "unknown")
        evaluation_states.append({"seed": evaluation["seed"], "state": state})
        report = evaluation_status.get("report")
        if state != "completed":
            if report is not None:
                _fail(
                    "non-completed replication evaluation exposed a report: "
                    f"seed={evaluation['seed']} state={state}"
                )
            continue
        if not isinstance(report, dict):
            _fail(f"completed replication report is unavailable: seed={evaluation['seed']}")
        report_body = {key: value for key, value in report.items() if key != "report_sha256"}
        if (
            report.get("report_sha256") != canonical_sha256(report_body)
            or report.get("checkpoint_sha256") != checkpoint_sha256
            or report.get("evaluation_seed") != evaluation["seed"]
            or report.get("per_cell") != plan["per_cell"]
            or report.get("task_depths") != plan["task_depths"]
            or report.get("recurrence_depths") != plan["recurrence_depths"]
            or report.get("task_count") != plan["task_count_per_seed"]
            or report.get("evaluation_source_sha256s") != plan["evaluator_source_sha256s"]
            or (
                plan.get("matched_control") is not None
                and report.get("matched_control") != plan["matched_control"]
            )
        ):
            _fail(f"replication report identity differs: seed={evaluation['seed']}")
        try:
            single_verdict = single.adjudicate_report(report)
        except single.ResidentTransferAdjudicationError as exc:
            raise ResidentReplicationError(
                f"replication seed evidence is malformed: seed={evaluation['seed']}"
            ) from exc
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
    return {
        "totals": totals,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "reports": reports,
        "every_seed_positive": every_seed_positive,
        "instruments_exact": instruments_exact,
        "evaluation_states": evaluation_states,
    }


def adjudicate(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, config, plan = _load_plan(arguments)
    completion = launcher._read_document(campaign / "completion-receipt.json")  # noqa: SLF001
    checkpoint = completion.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("complete") is not True:
        _fail("replication terminal checkpoint is unavailable")
    checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str):
        _fail("replication terminal checkpoint identity is unavailable")
    evidence = _collect_evidence(campaign, plan, checkpoint_sha256)
    reports = evidence["reports"]
    if not reports:
        _incomplete("replication has no completed evaluation evidence")
    totals = evidence["totals"]
    wrong_to_right = int(evidence["wrong_to_right"])
    right_to_wrong = int(evidence["right_to_wrong"])
    observed_tasks = len(reports) * int(plan["task_count_per_seed"])
    planned_tasks = int(plan["total_tasks"])
    all_present = len(reports) == len(plan["seeds"])
    decision_rule = plan.get("decision_rule")
    if not isinstance(decision_rule, Mapping):
        _fail("replication decision rule is unavailable")
    irreversible_failures: list[str] = []
    if (
        decision_rule.get("each_seed_positive_matched_control_gain") is True
        and evidence["every_seed_positive"] is not True
    ):
        irreversible_failures.append("every_seed_positive_matched_control_gain")
    if decision_rule.get("zero_pooled_right_to_wrong") is True and right_to_wrong > 0:
        irreversible_failures.append("zero_right_to_wrong")
    if (
        decision_rule.get("compiled_exact_every_seed") is True
        and evidence["instruments_exact"] is not True
    ):
        irreversible_failures.append("compiled_instrument_exact")
    if not all_present and not irreversible_failures:
        _incomplete(
            "replication is incomplete and no preregistered support condition "
            "has been irreversibly refuted"
        )

    tail = exact_paired_binomial_tail(wrong_to_right, right_to_wrong)
    pvalue = Fraction(tail.numerator, tail.denominator)
    alpha = Fraction(
        int(decision_rule["alpha_numerator"]),
        int(decision_rule["alpha_denominator"]),
    )
    effect = Fraction(totals["treatment"] - totals["control"], observed_tasks)
    minimum_effect = Fraction(
        int(decision_rule["minimum_pooled_effect_numerator"]),
        int(decision_rule["minimum_pooled_effect_denominator"]),
    )
    checks = {
        "all_evaluations_present": all_present,
        "compiled_instrument_exact": evidence["instruments_exact"],
        "every_seed_positive_matched_control_gain": evidence["every_seed_positive"],
        "zero_right_to_wrong": right_to_wrong == 0,
        "pooled_exact_p_at_most_one_percent": pvalue <= alpha if all_present else None,
        "pooled_effect_at_least_twenty_percent": effect >= minimum_effect if all_present else None,
        "treatment_beats_base": totals["treatment"] > totals["base"] if all_present else None,
        "recurrence_beats_trained_t1": totals["treatment"] > totals["trained_t1"]
        if all_present
        else None,
        "grammar_lesion_loses": totals["treatment"] > totals["grammar"] if all_present else None,
        "pointer_lesion_loses": totals["treatment"] > totals["pointer"] if all_present else None,
    }
    supported = all_present and all(value is True for value in checks.values())
    body = {
        "schema": VERDICT_SCHEMA,
        "verdict": SUPPORTED if supported else REFUTED,
        "supported": supported,
        "adjudication_scope": "complete" if all_present else "decisive_early_refutation",
        "plan_sha256": plan["plan_sha256"],
        "campaign_config_sha256": config["config_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "reports": reports,
        "evaluation_states": evidence["evaluation_states"],
        "evaluations_observed": len(reports),
        "evaluations_planned": len(plan["seeds"]),
        "total_tasks": observed_tasks,
        "planned_total_tasks": planned_tasks,
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
        "irreversible_failures": irreversible_failures,
        "claim_boundary": plan["claim_boundary"],
        "adjudication_boundary": (
            "An early refutation proves only that the exact preregistered "
            "conjunctive support rule cannot be satisfied after the observed "
            "completed evaluations. Undecidable aggregate checks remain null; "
            "this is not a powered effect estimate."
            if not all_present
            else "All preregistered evaluations were present and adjudicated."
        ),
    }
    verdict = {**body, "verdict_sha256": canonical_sha256(body)}
    _write_verdict(arguments, verdict)
    return verdict


def _campaign_is_terminal(
    campaign: Path,
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    completion_path = campaign / "completion-receipt.json"
    if completion_path.exists():
        _verified_config, completion = launcher._terminal_campaign(  # noqa: SLF001
            campaign / "campaign.json"
        )
        try:
            selected = launcher._selected_checkpoint(  # noqa: SLF001
                config,
                completion,
                stem="checkpoint_answer_bridge_admitted",
            )
        except launcher.ResidentEvaluationLaunchError as exc:
            return True, {
                "completion_sha256": completion["completion_sha256"],
                "answer_bridge_admitted": False,
                "admission_error": str(exc),
            }
        return True, {
            "completion_sha256": completion["completion_sha256"],
            "answer_bridge_admitted": True,
            "admitted_checkpoint": selected,
        }
    training = resident._inspect_status(config)  # noqa: SLF001
    if training.get("effective_state") == "failed":
        _fail("resident training failed before replication")
    return False, {
        "training_state": training.get("effective_state"),
        "training_controller_liveness": training.get("controller_liveness"),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign, config, plan = _load_plan(arguments)
    launchd = _verify_launchd_supervision(arguments, campaign, config, plan)
    deadline = time.monotonic() + float(arguments.controller_timeout)
    try:
        while time.monotonic() < deadline:
            terminal, training = _campaign_is_terminal(campaign, config)
            if not terminal:
                _publish_controller_status(
                    arguments,
                    campaign,
                    config,
                    plan,
                    "waiting_for_training",
                    training,
                )
                time.sleep(float(arguments.poll_interval))
                continue

            if training.get("answer_bridge_admitted") is not True:
                controller = _publish_controller_status(
                    arguments,
                    campaign,
                    config,
                    plan,
                    "not_admitted",
                    {**training, "launchd": launchd},
                )
                return {
                    "state": "not_admitted",
                    "supported": False,
                    "reason": training.get("admission_error"),
                    "controller": controller,
                }

            _publish_controller_status(
                arguments,
                campaign,
                config,
                plan,
                "admitted",
                {**training, "launchd": launchd},
            )
            observed = status(arguments)
            if observed["complete"] is True or any(
                row["state"] == "completed" for row in observed["evaluations"]
            ):
                verdict_arguments = argparse.Namespace(**vars(arguments))
                verdict_arguments.verdict_output = (
                    _replication_root(arguments, campaign) / "replication-verdict.json"
                )
                try:
                    verdict = adjudicate(verdict_arguments)
                except ResidentReplicationIncompleteError:
                    verdict = None
                if verdict is not None:
                    final_state = "completed" if verdict["supported"] else "refuted"
                    controller = _publish_controller_status(
                        arguments,
                        campaign,
                        config,
                        plan,
                        final_state,
                        {
                            **training,
                            "verdict": verdict["verdict"],
                            "verdict_sha256": verdict["verdict_sha256"],
                            "adjudication_scope": verdict.get("adjudication_scope", "complete"),
                        },
                    )
                    return {
                        "state": final_state,
                        "supported": verdict["supported"],
                        "verdict": verdict,
                        "controller": controller,
                    }

            next_result = launch_next(arguments)
            _publish_controller_status(
                arguments,
                campaign,
                config,
                plan,
                str(next_result["state"]),
                {
                    **training,
                    "seed": next_result.get("seed"),
                    "evaluations": [
                        {"seed": row["seed"], "state": row["state"]}
                        for row in observed["evaluations"]
                    ],
                },
            )
            if next_result["state"] == "completed":
                continue
            time.sleep(float(arguments.poll_interval))
        _fail("replication controller timed out")
    except Exception as exc:
        _publish_controller_status(
            arguments,
            campaign,
            config,
            plan,
            "failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "status", "launch-next", "adjudicate", "run", "install-launchd"),
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluator-source-root", type=Path)
    parser.add_argument("--matched-control-campaign", type=Path)
    parser.add_argument("--matched-control-stem", default="checkpoint_latest")
    parser.add_argument("--verdict-output", type=Path)
    parser.add_argument(
        "--seeds",
        type=_csv_unique_ints,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--controller-timeout", type=float, default=16 * 60 * 60)
    parser.add_argument("--launchd-supervised", action="store_true")
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
    if (
        arguments.per_cell < 1
        or arguments.max_tokens < 1
        or arguments.poll_interval <= 0.0
        or arguments.controller_timeout <= arguments.poll_interval
        or len(arguments.recurrence_depths) != 1
    ):
        parser.error("replication numeric contract is invalid")
    try:
        result = {
            "prepare": prepare,
            "status": status,
            "launch-next": launch_next,
            "adjudicate": adjudicate,
            "run": run,
            "install-launchd": install_launchd,
        }[arguments.action](arguments)
    except (OSError, ValueError, ResidentReplicationError) as exc:
        print(f"resident replication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.action == "adjudicate" and result.get("supported") is not True:
        return 1
    if arguments.action == "run" and result.get("state") not in {
        "completed",
        "refuted",
        "not_admitted",
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
