from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import run_rlc_wow_pilot_handoff as handoff


def _campaign(*, arms: str, seed: int) -> dict[str, object]:
    model_manifest = {
        "root": "/model",
        "files": [{"path": "weights", "sha256": "a" * 64, "size": 1}],
    }
    return {
        "arms": arms,
        "seed": seed,
        "source_commit": "b" * 40,
        "source_root": "/source",
        "python_sha256": "c" * 64,
        "model": "/model",
        "model_manifest": model_manifest,
        "difficulty": 2,
        "per_domain": 1,
        "n_slots": 16,
        "max_tokens": 1024,
        "memory_fraction": 0.4,
        "out_dir": f"/campaign-{seed}/sweep",
    }


def _verdict(*, correct: int = 3, total: int = 7) -> dict[str, object]:
    return {
        "schema": "aura.rlc_reconciliation_sweep.v1",
        "arms_complete": True,
        "coverage_complete": True,
        "evidence_manifest_valid": True,
        "faulted_arms": {},
        "missing_cells": {},
        "duplicate_cells": [],
        "unknown_task_cells": [],
        "full_stack_runtime_issues": {},
        "battery_informative": correct > 0,
        "arms": {
            "vanilla": {"correct": correct, "total": total},
            "vanilla_equal_compute": {"correct": correct, "total": total},
        },
    }


def test_pair_requires_disjoint_seeds_and_complete_engine() -> None:
    calibration = _campaign(arms="vanilla,vanilla_equal_compute", seed=1)
    pilot = _campaign(arms="complete_system_closed_book", seed=2)
    handoff.validate_campaign_pair(calibration, pilot)

    same_seed = copy.deepcopy(pilot)
    same_seed["seed"] = 1
    with pytest.raises(handoff.HandoffError, match="seeds_not_disjoint"):
        handoff.validate_campaign_pair(calibration, same_seed)

    controls_only = copy.deepcopy(pilot)
    controls_only["arms"] = "vanilla"
    with pytest.raises(handoff.HandoffError, match="complete_system_arm_invalid"):
        handoff.validate_campaign_pair(calibration, controls_only)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update(battery_informative=False), "floor_saturated"),
        (lambda row: row["arms"]["vanilla"].update(correct=7), "ceiling_saturated"),
        (lambda row: row.update(faulted_arms={"vanilla": 1}), "contains_faults"),
        (lambda row: row.update(coverage_complete=False), "evidence_invalid"),
    ],
)
def test_calibration_rejects_uninformative_or_invalid_evidence(mutation, reason) -> None:
    verdict = _verdict()
    mutation(verdict)
    assert handoff.calibration_admission(verdict) == (False, f"calibration_{reason}")


def test_calibration_admits_only_a_nonsaturated_complete_control() -> None:
    assert handoff.calibration_admission(_verdict(correct=3, total=7)) == (
        True,
        "calibration_admitted",
    )


def test_pilot_requires_complete_engine_runtime_evidence() -> None:
    verdict = _verdict()
    verdict["arms"]["complete_system_closed_book"] = {"correct": 4, "total": 7}
    assert handoff.pilot_completion(verdict) == (True, "pilot_measured")
    verdict["full_stack_runtime_issues"] = {
        "complete_system_closed_book": {"task": ["missing"]}
    }
    assert handoff.pilot_completion(verdict) == (False, "pilot_contains_faults")


def test_heavy_process_detection_ignores_shell_watchers_but_catches_real_work() -> None:
    rows = [
        (10, 1, "/bin/zsh", "/bin/zsh -c until pgrep -f run_test_chunks.py; do sleep 10; done"),
        (11, 1, "/usr/bin/python", "/usr/bin/python tools/run_test_chunks.py --chunks 6"),
        (12, 11, "/usr/bin/python", "/usr/bin/python -m pytest tests/test_one.py"),
    ]
    blockers = handoff._heavy_processes(rows)
    assert [row["pid"] for row in blockers] == [11, 12]


def test_status_is_authenticated_and_tamper_evident(tmp_path: Path, monkeypatch) -> None:
    key_path = tmp_path / ".key"
    key_path.write_bytes(b"k" * 32)
    os.chmod(key_path, 0o600)
    config = {"config_sha256": "d" * 64, "heartbeat_key_path": str(key_path)}
    monkeypatch.setattr(os, "getpid", lambda: 42)
    status = handoff._signed_status(config, phase="waiting")
    handoff.verify_status(config, status)
    changed = copy.deepcopy(status)
    changed["phase"] = "complete"
    with pytest.raises(handoff.HandoffError, match="status_invalid"):
        handoff.verify_status(config, changed)
    body = {key: value for key, value in status.items() if key != "hmac_sha256"}
    assert (
        status["hmac_sha256"]
        == __import__("hmac")
        .new(
            b"k" * 32,
            handoff._canonical(body),
            hashlib.sha256,
        )
        .hexdigest()
    )


def test_lineage_requires_launchd_and_exact_caffeinate_child(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "handoff.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {
        "source_root": "/source",
        "config_sha256": "f" * 64,
        "calibration_config_path": "/calibration.json",
    }
    command = (
        "python /source/tools/run_rlc_wow_pilot_handoff.py run "
        f"--config {config_path} --launchd-supervised"
    )
    child = (
        "/usr/bin/caffeinate -dims /venv/python "
        "/source/tools/run_rlc_wow_pilot_handoff.py run "
        f"--config {config_path} --launchd-supervised"
    )
    monkeypatch.setattr(os, "getpid", lambda: 41)
    monkeypatch.setattr(handoff, "_process_record", lambda _pid: (1, command))
    monkeypatch.setattr(handoff, "_process_table", lambda: [(42, 41, "caffeinate", child)])
    monkeypatch.setattr(
        handoff.campaign_controller,
        "load_config",
        lambda _path: {"python": "/venv/python"},
    )
    assert handoff._verify_launchd_lineage(config, config_path) == {
        "launchd_pid": 1,
        "handoff_pid": 41,
        "caffeinate_pid": 42,
    }


def test_completed_handoff_never_claims_wow(tmp_path: Path) -> None:
    key_path = tmp_path / ".key"
    key_path.write_bytes(b"k" * 32)
    os.chmod(key_path, 0o600)
    config = {
        "config_sha256": "e" * 64,
        "heartbeat_key_path": str(key_path),
        "out_dir": str(tmp_path),
    }
    status = handoff._write_status(config, phase="complete", pilot_decision="proceed")
    persisted = json.loads((tmp_path / "handoff_status.json").read_text(encoding="utf-8"))
    assert persisted == status
    assert set(status["claims"].values()) == {False}


def test_launchd_restarts_only_crashed_handoff(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "handoff.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {
        "source_root": "/source",
        "out_dir": str(tmp_path),
        "launch_label": "com.aura.test",
        "calibration_config_path": "/calibration.json",
    }
    monkeypatch.setattr(
        handoff.campaign_controller,
        "load_config",
        lambda _path: {"python": "/venv/python"},
    )
    payload = __import__("plistlib").loads(handoff._launch_payload(config_path, config))
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ThrottleInterval"] == 30


def test_started_campaign_is_attached_without_reinstallation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "controller_status.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        handoff.campaign_controller,
        "load_config",
        lambda _path: {"out_dir": str(root / "sweep")},
    )
    assert handoff._campaign_started(tmp_path / "config.json") is True
