#!/usr/bin/env python
"""Fail-closed unattended handoff for the resident-32B WOW pilot.

The reconciliation controller owns one campaign.  This coordinator owns the
scientific boundary between the control calibration and the seed-disjoint
complete-system closed-book pilot. It never grades a WOW claim, promotes an adapter, or
changes either campaign.  Its only authority is to launch the frozen pilot
after the frozen calibration proves that the battery is complete and neither
at floor nor ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import plistlib
import secrets
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import run_rlc_reconciliation_controller as campaign_controller  # noqa: E402

SCHEMA: Final = "aura.rlc_wow_pilot_handoff.v2"
STATUS_SCHEMA: Final = "aura.rlc_wow_pilot_handoff_status.v2"
LAUNCH_SCHEMA: Final = "aura.rlc_wow_pilot_handoff_launchd.v2"
CONTROL_ARMS: Final = frozenset({"vanilla", "vanilla_equal_compute"})
PILOT_ARM: Final = "complete_system_closed_book"
SCIENTIFIC_MATCH_FIELDS: Final = (
    "source_commit",
    "python_sha256",
    "difficulty",
    "per_domain",
    "n_slots",
    "max_tokens",
    "memory_fraction",
)
HEAVY_SCRIPTS: Final = frozenset(
    {
        "aura_main.py",
        "launch_recurrent_sft_falsification.py",
        "launch_resident_recurrence_training.py",
        "mlx_worker.py",
        "run_resident_recurrent_grpo_post_training.py",
        "run_resident_recurrent_sft_bootstrap_campaign.py",
        "run_rlc_reconciliation_controller.py",
        "run_test_chunks.py",
        "run_verified_recurrent_grpo_training.py",
        "train_grpo.py",
        "train_intrinsic_recurrence.py",
        "train_recurrent_sft_controls.py",
        "train_resident_recurrent_sft_bootstrap.py",
        "train_and_fuse.py",
    }
)


class HandoffError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffError(f"{role}_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{role}_not_object:{path}")
    return value


def _config_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "config_sha256"}


def _model_identity(config: Mapping[str, Any]) -> str:
    return _sha(config.get("model_manifest"))


def _arms(config: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        part.strip() for part in str(config.get("arms") or "").split(",") if part.strip()
    )


def validate_campaign_pair(
    calibration_config: Mapping[str, Any],
    pilot_config: Mapping[str, Any],
) -> None:
    if _arms(calibration_config) != CONTROL_ARMS:
        raise HandoffError("calibration_arms_invalid")
    if _arms(pilot_config) != {PILOT_ARM}:
        raise HandoffError("pilot_complete_system_arm_invalid")
    if int(calibration_config["seed"]) == int(pilot_config["seed"]):
        raise HandoffError("campaign_seeds_not_disjoint")
    for field in SCIENTIFIC_MATCH_FIELDS:
        if calibration_config.get(field) != pilot_config.get(field):
            raise HandoffError(f"campaign_parameter_mismatch:{field}")
    if calibration_config.get("model") != pilot_config.get("model"):
        raise HandoffError("campaign_model_path_mismatch")
    if _model_identity(calibration_config) != _model_identity(pilot_config):
        raise HandoffError("campaign_model_identity_mismatch")
    if calibration_config.get("source_root") != pilot_config.get("source_root"):
        raise HandoffError("campaign_source_root_mismatch")
    if calibration_config.get("out_dir") == pilot_config.get("out_dir"):
        raise HandoffError("campaign_output_collision")


def build_config(
    *,
    calibration_config_path: Path,
    pilot_config_path: Path,
    out_dir: Path,
    poll_s: float,
    idle_stability_s: float,
    max_wall_s: float,
) -> tuple[dict[str, Any], bytes]:
    calibration_path = calibration_config_path.expanduser().resolve(strict=True)
    pilot_path = pilot_config_path.expanduser().resolve(strict=True)
    calibration = campaign_controller.load_config(calibration_path)
    pilot = campaign_controller.load_config(pilot_path)
    validate_campaign_pair(calibration, pilot)
    if poll_s <= 0 or idle_stability_s < poll_s or max_wall_s <= idle_stability_s:
        raise HandoffError("handoff_time_budget_invalid")
    root = out_dir.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key_path = root / ".handoff.key"
    body = {
        "schema": SCHEMA,
        "calibration_config_path": str(calibration_path),
        "calibration_config_sha256": calibration["config_sha256"],
        "pilot_config_path": str(pilot_path),
        "pilot_config_sha256": pilot["config_sha256"],
        "source_root": calibration["source_root"],
        "source_commit": calibration["source_commit"],
        "model_identity_sha256": _model_identity(calibration),
        "out_dir": str(root),
        "poll_s": float(poll_s),
        "idle_stability_s": float(idle_stability_s),
        "max_wall_s": float(max_wall_s),
        "heartbeat_key_path": str(key_path),
        "launch_label": f"com.aura.rlc-wow-pilot-handoff.{root.name}",
    }
    return {**body, "config_sha256": _sha(body)}, secrets.token_bytes(32)


def write_prepared(config_path: Path, config: Mapping[str, Any], key: bytes) -> None:
    path = config_path.expanduser().absolute()
    key_path = Path(str(config["heartbeat_key_path"]))
    if path.exists() or key_path.exists():
        raise HandoffError("prepared_handoff_artifact_exists")
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(key)
        sink.flush()
        os.fsync(sink.fileno())
    _atomic_json(path, config)


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.expanduser().resolve(strict=True), role="handoff_config")
    if config.get("schema") != SCHEMA or config.get("config_sha256") != _sha(_config_body(config)):
        raise HandoffError("handoff_config_invalid")
    key_path = Path(str(config.get("heartbeat_key_path") or ""))
    try:
        metadata = key_path.stat()
    except OSError as exc:
        raise HandoffError("handoff_key_unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise HandoffError("handoff_key_custody_invalid")
    if len(key_path.read_bytes()) != 32:
        raise HandoffError("handoff_key_invalid")
    calibration = campaign_controller.load_config(Path(config["calibration_config_path"]))
    pilot = campaign_controller.load_config(Path(config["pilot_config_path"]))
    if calibration["config_sha256"] != config.get("calibration_config_sha256"):
        raise HandoffError("calibration_config_identity_drift")
    if pilot["config_sha256"] != config.get("pilot_config_sha256"):
        raise HandoffError("pilot_config_identity_drift")
    validate_campaign_pair(calibration, pilot)
    if _model_identity(calibration) != config.get("model_identity_sha256"):
        raise HandoffError("handoff_model_identity_drift")
    if calibration.get("source_root") != config.get("source_root"):
        raise HandoffError("handoff_source_root_drift")
    if calibration.get("source_commit") != config.get("source_commit"):
        raise HandoffError("handoff_source_commit_drift")
    return config


def _signed_status(config: Mapping[str, Any], *, phase: str, **fields: Any) -> dict[str, Any]:
    body = {
        "schema": STATUS_SCHEMA,
        "config_sha256": config["config_sha256"],
        "phase": phase,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        "claims": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "fusion_authorized": False,
            "wow_signal": False,
        },
        **fields,
    }
    key = Path(str(config["heartbeat_key_path"])).read_bytes()
    return {**body, "hmac_sha256": hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()}


def _write_status(config: Mapping[str, Any], *, phase: str, **fields: Any) -> dict[str, Any]:
    status = _signed_status(config, phase=phase, **fields)
    _atomic_json(Path(str(config["out_dir"])) / "handoff_status.json", status)
    return status


def verify_status(config: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    signature = status.get("hmac_sha256")
    body = {key: value for key, value in status.items() if key != "hmac_sha256"}
    expected = hmac.new(
        Path(str(config["heartbeat_key_path"])).read_bytes(),
        _canonical(body),
        hashlib.sha256,
    ).hexdigest()
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != config["config_sha256"]
        or not isinstance(signature, str)
        or not hmac.compare_digest(signature, expected)
    ):
        raise HandoffError("handoff_status_invalid")


def _process_table() -> list[tuple[int, int, str, str]]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=,ppid=,comm=,args="],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0:
        raise HandoffError("process_table_unavailable")
    rows: list[tuple[int, int, str, str]] = []
    for line in observed.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and fields[0].isdigit() and fields[1].isdigit():
            rows.append((int(fields[0]), int(fields[1]), fields[2], fields[3]))
    return rows


def _process_record(pid: int) -> tuple[int, str]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-o", "ppid=", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0 or not observed.stdout.strip():
        raise HandoffError(f"handoff_lineage_unavailable:{pid}")
    fields = observed.stdout.strip().split(None, 1)
    if len(fields) != 2 or not fields[0].isdigit():
        raise HandoffError(f"handoff_lineage_invalid:{pid}")
    return int(fields[0]), fields[1]


def _verify_launchd_lineage(
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, int]:
    handoff_pid = os.getpid()
    parent, command = _process_record(handoff_pid)
    script = str(Path(str(config["source_root"])) / "tools/run_rlc_wow_pilot_handoff.py")
    exact_config_path = str(config_path.expanduser().resolve(strict=True))
    required = (script, exact_config_path, "--launchd-supervised")
    if parent != 1 or any(value not in command for value in required):
        raise HandoffError("handoff_launchd_caffeinate_lineage_invalid")
    calibration = campaign_controller.load_config(Path(config["calibration_config_path"]))
    caffeinate_required = (
        "/usr/bin/caffeinate",
        "-dims",
        str(calibration["python"]),
        script,
        exact_config_path,
    )
    children = [
        pid
        for pid, child_parent, _executable, child_command in _process_table()
        if child_parent == handoff_pid
        and all(value in child_command for value in caffeinate_required)
    ]
    if len(children) != 1:
        raise HandoffError("handoff_launchd_caffeinate_lineage_invalid")
    return {"launchd_pid": 1, "handoff_pid": handoff_pid, "caffeinate_pid": children[0]}


def _heavy_processes(rows: Sequence[tuple[int, int, str, str]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for pid, parent, executable, command in rows:
        if pid == os.getpid():
            continue
        program = Path(executable).name.lower()
        if program in {"zsh", "bash", "sh", "fish", "rg", "grep"}:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        basenames = {Path(token).name for token in tokens if token and not token.startswith("-")}
        module_pytest = "-m" in tokens and "pytest" in tokens
        matched = sorted(HEAVY_SCRIPTS.intersection(basenames))
        if not matched and not module_pytest:
            continue
        blockers.append(
            {
                "pid": pid,
                "ppid": parent,
                "executable": executable,
                "reason": "pytest" if module_pytest else matched[0],
            }
        )
    return blockers


def calibration_admission(verdict: Mapping[str, Any]) -> tuple[bool, str]:
    if verdict.get("schema") != "aura.rlc_reconciliation_sweep.v1":
        return False, "calibration_verdict_schema_invalid"
    if verdict.get("arms_complete") is not True:
        return False, "calibration_incomplete"
    if (
        verdict.get("coverage_complete") is not True
        or verdict.get("evidence_manifest_valid") is not True
    ):
        return False, "calibration_evidence_invalid"
    if any(
        verdict.get(field)
        for field in (
            "faulted_arms",
            "missing_cells",
            "duplicate_cells",
            "unknown_task_cells",
            "full_stack_runtime_issues",
        )
    ):
        return False, "calibration_contains_faults"
    if verdict.get("battery_informative") is not True:
        return False, "calibration_floor_saturated"
    arms = verdict.get("arms")
    if not isinstance(arms, Mapping) or frozenset(arms) != CONTROL_ARMS:
        return False, "calibration_arm_set_invalid"
    vanilla = arms.get("vanilla")
    equal = arms.get("vanilla_equal_compute")
    if not isinstance(vanilla, Mapping) or not isinstance(equal, Mapping):
        return False, "calibration_control_missing"
    total = int(vanilla.get("total") or 0)
    correct = int(vanilla.get("correct") or 0)
    if total <= 0 or int(equal.get("total") or 0) != total:
        return False, "calibration_control_coverage_invalid"
    if correct >= total:
        return False, "calibration_ceiling_saturated"
    return True, "calibration_admitted"


def pilot_completion(verdict: Mapping[str, Any]) -> tuple[bool, str]:
    if verdict.get("schema") != "aura.rlc_reconciliation_sweep.v1":
        return False, "pilot_verdict_schema_invalid"
    if verdict.get("arms_complete") is not True:
        return False, "pilot_incomplete"
    if (
        verdict.get("coverage_complete") is not True
        or verdict.get("evidence_manifest_valid") is not True
    ):
        return False, "pilot_evidence_invalid"
    if any(
        verdict.get(field)
        for field in (
            "faulted_arms",
            "missing_cells",
            "duplicate_cells",
            "unknown_task_cells",
            "full_stack_runtime_issues",
        )
    ):
        return False, "pilot_contains_faults"
    arms = verdict.get("arms")
    expected_arms = CONTROL_ARMS | {PILOT_ARM}
    if not isinstance(arms, Mapping) or frozenset(arms) != expected_arms:
        return False, "pilot_complete_system_unmeasured"
    return True, "pilot_measured"


def _campaign_phase(config_path: Path) -> tuple[str, dict[str, Any]]:
    state = campaign_controller.status(config_path)
    controller_status = state.get("controller_status") or {}
    return str(controller_status.get("phase") or "starting"), state


def _campaign_started(config_path: Path) -> bool:
    campaign = campaign_controller.load_config(config_path)
    root = Path(str(campaign["out_dir"])).parent
    return any(
        path.is_file()
        for path in (
            root / "launchd_receipt.json",
            root / "controller_status.json",
            root / "controller_heartbeat.json",
            Path(str(campaign["out_dir"])) / "verdict.json",
        )
    )


def _wait_for_idle_host(
    config: Mapping[str, Any],
    *,
    deadline: float,
    lineage: Mapping[str, int],
) -> bool:
    idle_since: float | None = None
    while time.monotonic() < deadline:
        blockers = _heavy_processes(_process_table())
        if blockers:
            idle_since = None
            _write_status(
                config,
                phase="waiting_for_idle_host",
                blockers=blockers,
                lineage=lineage,
            )
        else:
            idle_since = idle_since or time.monotonic()
            remaining = float(config["idle_stability_s"]) - (time.monotonic() - idle_since)
            _write_status(
                config,
                phase="confirming_idle_host",
                idle_remaining_s=max(0.0, round(remaining, 1)),
                lineage=lineage,
            )
            if remaining <= 0:
                return True
        time.sleep(float(config["poll_s"]))
    return False


def _wait_for_campaign(
    handoff: Mapping[str, Any],
    config_path: Path,
    *,
    phase_prefix: str,
    deadline: float,
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        phase, state = _campaign_phase(config_path)
        progress = state.get("progress") or {}
        _write_status(
            handoff,
            phase=f"{phase_prefix}_{phase}",
            campaign_id=state.get("campaign_id"),
            cells=progress.get("cells"),
            sweep_arm=progress.get("sweep_arm"),
            sweep_arm_progress=progress.get("sweep_arm_progress"),
        )
        if phase == "complete":
            campaign = campaign_controller.load_config(config_path)
            return _read_json(
                Path(str(campaign["out_dir"])) / "verdict.json", role="campaign_verdict"
            )
        if phase in {"blocked", "yielded"}:
            raise HandoffError(f"{phase_prefix}_{phase}")
        heartbeat = state.get("heartbeat")
        if heartbeat and time.time() - float(heartbeat.get("observed_unix") or 0.0) > max(
            60.0, float(campaign_controller.load_config(config_path)["poll_s"]) * 4.0
        ):
            raise HandoffError(f"{phase_prefix}_heartbeat_stale")
        time.sleep(float(handoff["poll_s"]))
    raise HandoffError(f"{phase_prefix}_wall_budget_exhausted")


def run(config_path: Path, *, launchd_supervised: bool = False) -> int:
    if not launchd_supervised:
        raise HandoffError("handoff_requires_launchd_supervision")
    config = load_config(config_path)
    lineage = _verify_launchd_lineage(config, config_path)
    calibration_path = Path(str(config["calibration_config_path"]))
    pilot_path = Path(str(config["pilot_config_path"]))
    calibration = campaign_controller.load_config(calibration_path)
    pilot = campaign_controller.load_config(pilot_path)
    deadline = time.monotonic() + float(config["max_wall_s"])
    if not _campaign_started(calibration_path) and not _wait_for_idle_host(
        config,
        deadline=deadline,
        lineage=lineage,
    ):
        _write_status(config, phase="blocked", reason="idle_host_wall_budget_exhausted")
        return 0

    try:
        campaign_controller._source_is_current(calibration)
        campaign_controller._source_is_current(pilot)
        if not _campaign_started(calibration_path):
            campaign_controller.install_launchd(calibration_path)
        calibration_verdict = _wait_for_campaign(
            config,
            calibration_path,
            phase_prefix="calibration",
            deadline=deadline,
        )
        admitted, reason = calibration_admission(calibration_verdict)
        if not admitted:
            _write_status(
                config,
                phase="blocked",
                reason=reason,
                calibration_decision=calibration_verdict.get("decision"),
            )
            return 0
        if not _campaign_started(pilot_path):
            if not _wait_for_idle_host(config, deadline=deadline, lineage=lineage):
                raise HandoffError("pilot_idle_host_wall_budget_exhausted")
            campaign_controller.install_launchd(pilot_path)
        pilot_verdict = _wait_for_campaign(
            config,
            pilot_path,
            phase_prefix="pilot",
            deadline=deadline,
        )
        pilot_measured, pilot_reason = pilot_completion(pilot_verdict)
        if not pilot_measured:
            _write_status(
                config,
                phase="blocked",
                reason=pilot_reason,
                pilot_decision=pilot_verdict.get("decision"),
            )
            return 0
        _write_status(
            config,
            phase="complete",
            reason="pilot_complete_requires_powered_independent_certificate",
            calibration_decision=calibration_verdict.get("decision"),
            pilot_decision=pilot_verdict.get("decision"),
            pilot_reaches_parity=pilot_verdict.get("reaches_parity_with_ordinary_decode"),
            pilot_beats_equal_compute=pilot_verdict.get("beats_equal_compute_control"),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed unattended boundary
        _write_status(config, phase="blocked", reason=f"{type(exc).__name__}:{exc}")
        return 0


def _launch_payload(config_path: Path, config: Mapping[str, Any]) -> bytes:
    python = campaign_controller.load_config(Path(config["calibration_config_path"]))["python"]
    script = Path(str(config["source_root"])) / "tools/run_rlc_wow_pilot_handoff.py"
    payload = {
        "Label": config["launch_label"],
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-dims",
            str(python),
            str(script),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
            "--launchd-supervised",
        ],
        "WorkingDirectory": str(config["source_root"]),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(Path(str(config["out_dir"])) / "handoff.log"),
        "StandardErrorPath": str(Path(str(config["out_dir"])) / "handoff.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(config_path: Path) -> dict[str, Any]:
    path = config_path.expanduser().resolve(strict=True)
    config = load_config(path)
    payload = _launch_payload(path, config)
    launch_agents = Path.home() / "Library/LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True, mode=0o700)
    plist_path = launch_agents / f"{config['launch_label']}.plist"
    temporary = plist_path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, plist_path)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    started = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if started.returncode != 0:
        raise HandoffError(
            f"launchd_bootstrap_failed:{started.returncode}:{started.stderr.strip()}"
        )
    body = {
        "schema": LAUNCH_SCHEMA,
        "config_sha256": config["config_sha256"],
        "label": config["launch_label"],
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(payload).hexdigest(),
        "installed_unix": time.time(),
    }
    receipt = {**body, "launch_sha256": _sha(body)}
    _atomic_json(Path(str(config["out_dir"])) / "launchd_receipt.json", receipt)
    return receipt


def status(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    path = Path(str(config["out_dir"])) / "handoff_status.json"
    if not path.is_file():
        return {"phase": "prepared", "config_sha256": config["config_sha256"]}
    result = _read_json(path, role="handoff_status")
    verify_status(config, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--calibration-config", type=Path, required=True)
    prepare.add_argument("--pilot-config", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--poll-s", type=float, default=30.0)
    prepare.add_argument("--idle-stability-s", type=float, default=120.0)
    prepare.add_argument("--max-wall-s", type=float, default=259_200.0)
    prepare.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--launchd-supervised", action="store_true")
    install = commands.add_parser("install-launchd")
    install.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("status")
    inspect.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            config, key = build_config(
                calibration_config_path=args.calibration_config,
                pilot_config_path=args.pilot_config,
                out_dir=args.out_dir,
                poll_s=args.poll_s,
                idle_stability_s=args.idle_stability_s,
                max_wall_s=args.max_wall_s,
            )
            write_prepared(args.output, config, key)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.action == "run":
            return run(args.config, launchd_supervised=args.launchd_supervised)
        if args.action == "install-launchd":
            print(json.dumps(install_launchd(args.config), indent=2, sort_keys=True))
            return 0
        print(json.dumps(status(args.config), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"run_rlc_wow_pilot_handoff: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
