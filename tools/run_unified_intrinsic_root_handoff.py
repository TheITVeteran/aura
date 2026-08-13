#!/usr/bin/env python3
"""Advance a frozen true-root canary to its powered replication."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import plistlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from tools import adjudicate_unified_intrinsic_resident_replication as replication  # noqa: E402
from tools import adjudicate_unified_intrinsic_resident_transfer as adjudicator  # noqa: E402
from tools import launch_unified_intrinsic_resident_evaluation as evaluator  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)

CONFIG_SCHEMA: Final = "aura.unified_intrinsic.root_handoff_config.v1"
STATUS_SCHEMA: Final = "aura.unified_intrinsic.root_handoff_status.v1"
INTENT_SCHEMA: Final = "aura.unified_intrinsic.root_handoff_launch_intent.v1"
LAUNCH_SCHEMA: Final = "aura.unified_intrinsic.root_handoff_launchd.v1"
LAUNCH_AGENTS_ROOT: Final = Path.home() / "Library/LaunchAgents"


class RootHandoffError(RuntimeError):
    """The frozen canary-to-replication handoff is invalid or unsafe."""


def _fail(message: str) -> Never:
    raise RootHandoffError(message)


def _read_canonical(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RootHandoffError(f"root handoff document is unreadable: {path}") from exc
    if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
        _fail(f"root handoff document is not canonical: {path}")
    return value


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        _fail(f"root handoff source is not a file: {resolved}")
    body = {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def _evaluation_arguments(campaign: Path, root: Path) -> argparse.Namespace:
    return argparse.Namespace(campaign=campaign, output=root)


def _replication_arguments(campaign: Path, root: Path) -> argparse.Namespace:
    return argparse.Namespace(campaign=campaign, output=root, verdict_output=None)


def _config_path(arguments: argparse.Namespace) -> Path:
    return arguments.config.expanduser().absolute()


def _status_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["handoff_root"])) / "controller-status.json"


def _campaign_config(campaign: Path) -> dict[str, Any]:
    config = replication.resident._load_config(campaign / "campaign.json")  # noqa: SLF001
    if Path(str(config["paths"]["campaign_root"])).resolve(strict=True) != campaign:
        _fail("root handoff campaign identity differs")
    return config


def _key(config: Mapping[str, Any]) -> bytes:
    campaign = Path(str(config["campaign"]))
    campaign_config = _campaign_config(campaign)
    if campaign_config.get("config_sha256") != config.get("campaign_config_sha256"):
        _fail("root handoff campaign commitment differs")
    return replication.resident._key(  # noqa: SLF001
        Path(str(campaign_config["paths"]["heartbeat_key"])),
        expected_sha256=str(campaign_config["heartbeat_key_sha256"]),
    )


def _signature(body: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_bytes(dict(body)), hashlib.sha256).hexdigest()


def _source_identity(powered_controller: Path) -> dict[str, Any]:
    body = {
        "handoff": _file_binding(Path(__file__)),
        "evaluator_launcher": _file_binding(Path(evaluator.__file__)),
        "single_adjudicator": _file_binding(Path(adjudicator.__file__)),
        "replication_controller": _file_binding(powered_controller),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    campaign = arguments.campaign.expanduser().resolve(strict=True)
    campaign_config = _campaign_config(campaign)
    canary_root = arguments.canary_root.expanduser().resolve(strict=True)
    powered_root = arguments.powered_root.expanduser().resolve(strict=True)
    if any(
        root == campaign or not root.is_relative_to(campaign)
        for root in (canary_root, powered_root)
    ) or canary_root == powered_root:
        _fail("root handoff evidence roots must be strict campaign children")
    canary_plan = evaluator._existing_plan(  # noqa: SLF001
        _evaluation_arguments(campaign, canary_root)
    )
    if canary_plan is None:
        _fail("root handoff canary plan is unavailable")
    _, _, powered_plan = replication._load_plan(  # noqa: SLF001
        _replication_arguments(campaign, powered_root)
    )
    canary_scientific = canary_plan.get("scientific")
    canary_control = (
        canary_scientific.get("matched_control")
        if isinstance(canary_scientific, Mapping)
        else None
    )
    if canary_control != powered_plan.get("matched_control"):
        _fail("root handoff canary and powered controls differ")
    handoff_root = (
        arguments.output.expanduser().absolute()
        if arguments.output is not None
        else campaign / "resident-root-handoff"
    )
    if handoff_root == campaign or not handoff_root.is_relative_to(campaign):
        _fail("root handoff state must be a strict campaign child")
    if handoff_root in {canary_root, powered_root}:
        _fail("root handoff state must not overlap evidence roots")
    ensure_private_directory(handoff_root)
    powered_controller = arguments.powered_controller.expanduser().resolve(strict=True)
    source = _source_identity(powered_controller)
    body = {
        "schema": CONFIG_SCHEMA,
        "campaign": str(campaign),
        "campaign_id": campaign_config["campaign_id"],
        "campaign_config_sha256": campaign_config["config_sha256"],
        "handoff_root": str(handoff_root),
        "canary_root": str(canary_root),
        "canary_plan_sha256": canary_plan["plan_sha256"],
        "powered_root": str(powered_root),
        "powered_plan_sha256": powered_plan["plan_sha256"],
        "matched_control": powered_plan["matched_control"],
        "powered_controller": str(powered_controller),
        "runtime_python": str(
            Path(str(campaign_config["runtime"]["interpreter"]["executable"]))
        ),
        "source": source,
    }
    config = {**body, "config_sha256": canonical_sha256(body)}
    path = handoff_root / "handoff-config.json"
    payload = canonical_bytes(config) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != config:
            _fail("root handoff config already differs")
    return config


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_canonical(path)
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("schema") != CONFIG_SCHEMA or config.get("config_sha256") != canonical_sha256(
        body
    ):
        _fail("root handoff config identity differs")
    campaign = Path(str(config.get("campaign") or "")).resolve(strict=True)
    _campaign_config(campaign)
    source = config.get("source")
    powered_controller = Path(str(config.get("powered_controller") or ""))
    if not isinstance(source, dict) or source != _source_identity(powered_controller):
        _fail("root handoff source identity differs")
    canary_root = Path(str(config["canary_root"])).resolve(strict=True)
    powered_root = Path(str(config["powered_root"])).resolve(strict=True)
    canary_plan = evaluator._existing_plan(  # noqa: SLF001
        _evaluation_arguments(campaign, canary_root)
    )
    _, _, powered_plan = replication._load_plan(  # noqa: SLF001
        _replication_arguments(campaign, powered_root)
    )
    if (
        canary_plan is None
        or canary_plan.get("plan_sha256") != config.get("canary_plan_sha256")
        or powered_plan.get("plan_sha256") != config.get("powered_plan_sha256")
        or powered_plan.get("matched_control") != config.get("matched_control")
    ):
        _fail("root handoff frozen plan identity differs")
    _key(config)
    return config


def _read_status(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _status_path(config)
    if not path.exists():
        return None
    status = _read_canonical(path)
    body = {key: value for key, value in status.items() if key != "hmac_sha256"}
    signature = status.get("hmac_sha256")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != config.get("config_sha256")
        or type(status.get("sequence")) is not int
        or int(status["sequence"]) < 1
        or not isinstance(signature, str)
        or not hmac.compare_digest(signature, _signature(body, _key(config)))
    ):
        _fail("root handoff status authentication failed")
    return status


def _publish_status(
    config: Mapping[str, Any],
    state: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    previous = _read_status(config)
    body = {
        "schema": STATUS_SCHEMA,
        "config_sha256": config["config_sha256"],
        "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
        "state": state,
        "controller_pid": os.getpid(),
        "controller_start_token": evaluator.detached._process_start_token(  # noqa: SLF001
            os.getpid()
        ),
        "heartbeat_at": time.time(),
        "details": dict(details),
    }
    status = {**body, "hmac_sha256": _signature(body, _key(config))}
    atomic_write_bytes(_status_path(config), canonical_bytes(status) + b"\n", mode=0o600)
    return status


def _write_canary_verdict(config: Mapping[str, Any], verdict: Mapping[str, Any]) -> Path:
    path = Path(str(config["handoff_root"])) / "canary-verdict.json"
    payload = canonical_bytes(dict(verdict)) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != verdict:
            _fail("root handoff canary verdict already differs")
    return path


def _powered_state(config: Mapping[str, Any]) -> dict[str, Any] | None:
    arguments = _replication_arguments(
        Path(str(config["campaign"])),
        Path(str(config["powered_root"])),
    )
    observed = replication.status(arguments)
    controller = observed.get("controller")
    if not isinstance(controller, dict):
        return None
    state = str(controller.get("state") or "unknown")
    liveness = observed.get("controller_liveness")
    if state in {"completed", "refuted", "not_admitted"} or (
        liveness == "alive" and state not in {"failed", "stopped", "unknown"}
    ):
        return {"state": state, "liveness": liveness, "controller": controller}
    return None


def _start_sleep_inhibitor() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(  # noqa: S603 - fixed macOS system utility
        ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _fail("root handoff sleep inhibitor exited during startup")
        token = evaluator.detached._process_start_token(process.pid)  # noqa: SLF001
        if token:
            return process
        time.sleep(0.02)
    process.terminate()
    _fail("root handoff sleep inhibitor identity is unavailable")


def _stop_sleep_inhibitor(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _launch_powered(config: Mapping[str, Any]) -> dict[str, Any]:
    existing = _powered_state(config)
    if existing is not None:
        return {"reopened": True, **existing}
    command = [
        str(config["runtime_python"]),
        str(config["powered_controller"]),
        "install-launchd",
        str(config["campaign"]),
        "--output",
        str(config["powered_root"]),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-600:]
        _fail(f"root handoff powered launch failed: {detail}")
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RootHandoffError("root handoff powered launch receipt is malformed") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("plan_sha256") != config.get("powered_plan_sha256")
    ):
        _fail("root handoff powered launch identity differs")
    return {"reopened": False, "receipt": receipt}


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(_config_path(arguments))
    campaign = Path(str(config["campaign"]))
    canary_root = Path(str(config["canary_root"]))
    deadline = time.monotonic() + float(arguments.timeout)
    inhibitor = _start_sleep_inhibitor()
    try:
        while time.monotonic() < deadline:
            observed = evaluator.status(_evaluation_arguments(campaign, canary_root))
            state = str(observed.get("state") or "unknown")
            if state == "running":
                _publish_status(
                    config,
                    "waiting_for_canary",
                    {
                        "canary_state": state,
                        "attempt": observed.get("attempt"),
                        "sleep_inhibitor_pid": inhibitor.pid,
                    },
                )
                time.sleep(float(arguments.poll_interval))
                continue
            if state != "completed" or not isinstance(observed.get("report"), dict):
                _fail(f"root handoff canary terminated without evidence: {state}")
            verdict = adjudicator.adjudicate_report(observed["report"])
            verdict_path = _write_canary_verdict(config, verdict)
            if verdict.get("supported") is not True:
                status = _publish_status(
                    config,
                    "canary_refuted",
                    {
                        "verdict": verdict.get("verdict"),
                        "verdict_sha256": verdict.get("verdict_sha256"),
                        "verdict_path": str(verdict_path),
                    },
                )
                return {"state": "canary_refuted", "supported": False, "status": status}
            launched = _launch_powered(config)
            status = _publish_status(
                config,
                "powered_launched",
                {
                    "verdict": verdict["verdict"],
                    "verdict_sha256": verdict["verdict_sha256"],
                    "verdict_path": str(verdict_path),
                    "powered": launched,
                },
            )
            return {"state": "powered_launched", "supported": True, "status": status}
        _fail("root handoff timed out waiting for the canary")
    except Exception as exc:
        _publish_status(
            config,
            "failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        _stop_sleep_inhibitor(inhibitor)


def status(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(_config_path(arguments))
    return {
        "schema": "aura.unified_intrinsic.root_handoff_inspection.v1",
        "config_sha256": config["config_sha256"],
        "status": _read_status(config),
        "powered": _powered_state(config),
    }


def _label(config: Mapping[str, Any]) -> str:
    campaign_id = str(config.get("campaign_id") or "")
    if not campaign_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in campaign_id):
        _fail("root handoff launchd label is invalid")
    return f"com.aura.unified-intrinsic-root-handoff.{campaign_id}"


def _launch_contract(config_path: Path, config: Mapping[str, Any], arguments: argparse.Namespace) -> tuple[Path, bytes, dict[str, Any]]:
    script = Path(__file__).resolve(strict=True)
    command = [
        str(config["runtime_python"]),
        str(script),
        "run",
        str(config_path),
        "--poll-interval",
        str(float(arguments.poll_interval)),
        "--timeout",
        str(float(arguments.timeout)),
    ]
    label = _label(config)
    root = Path(str(config["handoff_root"]))
    payload = {
        "Label": label,
        "ProgramArguments": command,
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PYTHONDONTWRITEBYTECODE": "1"},
        "StandardOutPath": str(root / "handoff-launchd.log"),
        "StandardErrorPath": str(root / "handoff-launchd.log"),
    }
    plist = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    path = LAUNCH_AGENTS_ROOT / f"{label}.plist"
    body = {
        "schema": INTENT_SCHEMA,
        "config_sha256": config["config_sha256"],
        "label": label,
        "plist_path": str(path),
        "plist_sha256": hashlib.sha256(plist).hexdigest(),
        "program_arguments": command,
        "controller_source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    return path, plist, {**body, "intent_sha256": canonical_sha256(body)}


def _launchd_pid(label: str) -> int:
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        return 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                return int(stripped.removeprefix("pid = "))
            except ValueError:
                return 0
    return 0


def install_launchd(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = _config_path(arguments)
    config = _load_config(config_path)
    root = Path(str(config["handoff_root"]))
    plist_path, plist, intent = _launch_contract(config_path, config, arguments)
    intent_path = root / "launch-intent.json"
    payload = canonical_bytes(intent) + b"\n"
    if not atomic_write_bytes_if_absent(intent_path, payload, mode=0o400):
        if _read_canonical(intent_path) != intent:
            _fail("root handoff launch intent already differs")
    ensure_private_directory(LAUNCH_AGENTS_ROOT)
    atomic_write_bytes(plist_path, plist, mode=0o600)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", f"{domain}/{_label(config)}"],
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
        _fail(f"root handoff launchd bootstrap failed: {started.stderr.strip()[:300]}")
    deadline = time.monotonic() + 20.0
    controller: dict[str, Any] | None = None
    pid = 0
    while time.monotonic() < deadline:
        pid = _launchd_pid(_label(config))
        controller = _read_status(config)
        if (
            pid > 1
            and isinstance(controller, dict)
            and controller.get("controller_pid") == pid
            and evaluator.detached._identity_state(  # noqa: SLF001
                pid, str(controller.get("controller_start_token") or "")
            )
            == "alive"
        ):
            break
        time.sleep(0.25)
    else:
        _fail("root handoff launchd authenticated start timed out")
    body = {
        "schema": LAUNCH_SCHEMA,
        "config_sha256": config["config_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "plan_sha256": config["powered_plan_sha256"],
        "pid": pid,
        "start_token": controller["controller_start_token"],
        "installed_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "launch_sha256": canonical_sha256(body)}
    atomic_write_bytes(
        root / "launchd-receipt.json",
        canonical_bytes(receipt) + b"\n",
        mode=0o600,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "install-launchd"))
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--canary-root", type=Path)
    parser.add_argument("--powered-root", type=Path)
    parser.add_argument("--powered-controller", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=4 * 60 * 60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.poll_interval <= 0.0 or arguments.timeout <= arguments.poll_interval:
        parser.error("root handoff timing contract is invalid")
    if arguments.action == "prepare":
        if any(
            value is None
            for value in (
                arguments.campaign,
                arguments.canary_root,
                arguments.powered_root,
                arguments.powered_controller,
            )
        ):
            parser.error("prepare requires campaign, canary, powered root and controller")
    elif arguments.config is None:
        parser.error("run, status and install-launchd require a config path")
    try:
        result = {
            "prepare": prepare,
            "run": run,
            "status": status,
            "install-launchd": install_launchd,
        }[arguments.action](arguments)
    except (OSError, ValueError, RootHandoffError) as exc:
        print(f"root handoff failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
