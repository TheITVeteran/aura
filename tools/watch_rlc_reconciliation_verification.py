#!/usr/bin/env python
"""OS-supervised independent verification for a composed RLC campaign.

This watcher deliberately has no training, fusion, attachment, or activation
surface.  It waits for the source-bound controller to produce either a full
terminal matrix or a signed terminal-futility receipt, then invokes the
independent campaign verifier from that same immutable source capsule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import plistlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA = "aura.rlc.reconciliation_verification_watcher.v1"
LAUNCH_SCHEMA = f"{SCHEMA}.launchd"


class WatcherError(RuntimeError):
    """The watcher could not prove its custody or terminal evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WatcherError(f"{role}_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise WatcherError(f"{role}_not_object:{path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_controller(config_path: Path) -> tuple[dict[str, Any], ModuleType]:
    raw = _read_json(config_path, role="controller_config")
    body = {key: value for key, value in raw.items() if key != "config_sha256"}
    if raw.get("config_sha256") != _sha(body):
        raise WatcherError("controller_config_digest_invalid")
    source_root = Path(str(raw.get("source_root") or "")).resolve(strict=True)
    controller_path = Path(str(raw.get("controller_program") or "")).resolve(strict=True)
    if controller_path.parent.parent != source_root:
        raise WatcherError("controller_program_outside_source")
    spec = importlib.util.spec_from_file_location(
        f"aura_frozen_rlc_controller_{raw['config_sha256'][:12]}", controller_path
    )
    if spec is None or spec.loader is None:
        raise WatcherError("controller_import_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = module.load_config(config_path)
    module._source_is_current(loaded)
    return loaded, module


def _result(
    *,
    config: Mapping[str, Any],
    watcher_path: Path,
    decision: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "source_commit": config["source_commit"],
        "watcher_sha256": _sha_file(watcher_path),
        "decision": decision,
        "details": dict(details),
        "fusion_authorized": False,
        "wow_signal_authorized": False,
        "ordinary_serving_authorized": False,
        "observed_unix": time.time(),
    }
    return {**body, "receipt_sha256": _sha(body)}


def watch(
    *,
    config_path: Path,
    output: Path,
    timeout_s: float,
    poll_s: float,
) -> int:
    config_path = config_path.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    config, controller = _load_controller(config_path)
    campaign_dir = Path(str(config["out_dir"])).resolve(strict=True)
    watcher_path = Path(__file__).resolve(strict=True)
    started = time.monotonic()
    status_path = output.with_name(output.stem + "_status.json")

    while time.monotonic() - started < timeout_s:
        controller_status = controller.status(config_path)
        terminal = controller._terminal_verdict(campaign_dir / "verdict.json")
        heartbeat = controller_status.get("heartbeat") or {}
        status = {
            "schema": SCHEMA,
            "campaign_id": config["campaign_id"],
            "config_sha256": config["config_sha256"],
            "phase": "waiting_for_terminal_campaign",
            "committed_cells": (controller_status.get("progress") or {}).get("cells", 0),
            "controller_phase": (controller_status.get("controller_status") or {}).get("phase"),
            "sweep_phase": (controller_status.get("progress") or {}).get("sweep_phase"),
            "signed_heartbeat_observed_unix": heartbeat.get("observed_unix"),
            "heartbeat_unix": time.time(),
            "pid": os.getpid(),
        }
        _atomic_json(status_path, status)
        controller_phase = status.get("controller_phase")
        if controller_phase == "blocked":
            receipt = _result(
                config=config,
                watcher_path=watcher_path,
                decision="campaign_controller_blocked",
                details={"controller_status": controller_status.get("controller_status")},
            )
            _atomic_json(output, receipt)
            return 2
        if terminal is None:
            time.sleep(poll_s)
            continue
        if terminal.get("schema") == "aura.rlc.composed_terminal_futility.v1":
            receipt = _result(
                config=config,
                watcher_path=watcher_path,
                decision="terminal_futility_no_positive_claim",
                details={
                    "terminal_futility_sha256": terminal.get("receipt_sha256"),
                    "fatal_regression_contrasts": terminal.get("fatal_regression_contrasts"),
                },
            )
            _atomic_json(output, receipt)
            return 0

        verifier = Path(str(config["source_root"])) / "tools/verify_rlc_reconciliation_campaign.py"
        verification_output = output.with_name("independent_component_verification.json")
        completed = subprocess.run(
            [
                str(config["python"]),
                str(verifier),
                "--config",
                str(config_path),
                "--campaign-dir",
                str(campaign_dir),
                "--output",
                str(verification_output),
            ],
            cwd=str(config["source_root"]),
            capture_output=True,
            text=True,
            timeout=900.0,
            check=False,
        )
        if completed.returncode != 0 or not verification_output.is_file():
            receipt = _result(
                config=config,
                watcher_path=watcher_path,
                decision="independent_verification_failed",
                details={
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            )
            _atomic_json(output, receipt)
            return 2
        verification = _read_json(
            verification_output, role="independent_component_verification"
        )
        receipt = _result(
            config=config,
            watcher_path=watcher_path,
            decision="independent_component_verification_complete",
            details={
                "verification_path": str(verification_output),
                "verification_sha256": _sha_file(verification_output),
                "verification_decision": (
                    verification.get("composed_recurrent_adjudication") or {}
                ).get("decision"),
            },
        )
        _atomic_json(output, receipt)
        return 0

    receipt = _result(
        config=config,
        watcher_path=watcher_path,
        decision="watch_budget_exhausted_without_terminal_campaign",
        details={"timeout_s": timeout_s},
    )
    _atomic_json(output, receipt)
    return 2


def _launch_payload(
    *, config_path: Path, output: Path, timeout_s: float, poll_s: float
) -> tuple[str, bytes]:
    config, _controller = _load_controller(config_path)
    campaign_id = str(config["campaign_id"])
    label = f"com.aura.rlc-independent-verifier.{campaign_id}"
    root = Path(str(config["out_dir"])).parent
    payload = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-dims",
            str(config["python"]),
            str(Path(__file__).resolve(strict=True)),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
            "--output",
            str(output.expanduser().absolute()),
            "--timeout-s",
            str(timeout_s),
            "--poll-s",
            str(poll_s),
        ],
        "WorkingDirectory": str(config["source_root"]),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "independent_verification_watcher.log"),
        "StandardErrorPath": str(root / "independent_verification_watcher.log"),
    }
    return label, plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(
    *, config_path: Path, output: Path, timeout_s: float, poll_s: float
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve(strict=True)
    label, payload = _launch_payload(
        config_path=config_path,
        output=output,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    launch_agents = Path.home() / "Library/LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True, mode=0o700)
    plist_path = launch_agents / f"{label}.plist"
    temporary = plist_path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, plist_path)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
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
        raise WatcherError(
            f"launchd_bootstrap_failed:{started.returncode}:{started.stderr.strip()}"
        )
    config = _read_json(config_path, role="controller_config")
    body = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "label": label,
        "domain": domain,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(payload).hexdigest(),
        "watcher_sha256": _sha_file(Path(__file__).resolve(strict=True)),
        "installed_unix": time.time(),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "install-launchd"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--timeout-s", type=float, default=259_200.0)
        command.add_argument("--poll-s", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.timeout_s <= 0.0 or not 1.0 <= args.poll_s <= 300.0:
            raise WatcherError("watch_budget_invalid")
        if args.action == "run":
            return watch(
                config_path=args.config,
                output=args.output,
                timeout_s=args.timeout_s,
                poll_s=args.poll_s,
            )
        receipt = install_launchd(
            config_path=args.config,
            output=args.output,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"watch_rlc_reconciliation_verification: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
