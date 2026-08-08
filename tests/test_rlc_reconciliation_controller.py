from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import pytest

from tools import run_rlc_reconciliation_controller as controller


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative, text in {
        "core/module.py": "VALUE = 1\n",
        "config/runtime.json": "{}\n",
        "tools/run_rlc_reconciliation_sweep.py": "print('sweep')\n",
        "tools/run_rlc_reconciliation_controller.py": "print('controller')\n",
        "pyproject.toml": "[project]\nname='test'\n",
        "requirements_lock.txt": "locked\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _prepared(tmp_path: Path):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    out = tmp_path / "campaign"
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit="a" * 40,
        model=model,
        out_dir=out,
        python=python,
        arms="full_stack",
        seed=7,
        per_domain=1,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
    )
    config_path = out / "controller_config.json"
    controller.write_prepared_campaign(config_path, config, manifest, key)
    return source, out, config_path, controller.load_config(config_path)


def test_source_manifest_detects_execution_source_drift(tmp_path: Path):
    source, _out, _config_path, config = _prepared(tmp_path)
    manifest = controller._read_json(
        Path(config["source_manifest_path"]), role="source_manifest"
    )
    controller.verify_source_manifest(source, manifest)
    (source / "core/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="source_file_drift"):
        controller.verify_source_manifest(source, manifest)


def test_source_manifest_detects_an_injected_python_module(tmp_path: Path):
    source, _out, _config_path, config = _prepared(tmp_path)
    manifest = controller._read_json(
        Path(config["source_manifest_path"]), role="source_manifest"
    )
    (source / "sitecustomize.py").write_text("print('injected')\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="source_file_set_drift"):
        controller.verify_source_manifest(source, manifest)


def test_model_manifest_rejects_weight_drift(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    controller.verify_model_manifest(config["model_manifest"])
    (Path(config["model"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(controller.ControllerError, match="model_file_drift"):
        controller.verify_model_manifest(config["model_manifest"])


def test_config_digest_rejects_a_changed_scientific_parameter(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["arms"] = "vanilla"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="controller_config_invalid"):
        controller.load_config(config_path)


def test_heartbeat_is_authenticated_and_tamper_evident(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    heartbeat = controller._signed_heartbeat(
        config,
        {"controller_pid": 1, "sweep_pid": 2, "observed_unix": 3.0},
    )
    controller.verify_heartbeat(config, heartbeat)
    changed = copy.deepcopy(heartbeat)
    changed["sweep_pid"] = 999
    with pytest.raises(controller.ControllerError, match="heartbeat_invalid"):
        controller.verify_heartbeat(config, changed)


def test_heartbeat_matches_an_independent_hmac_recomputation(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    heartbeat = controller._signed_heartbeat(
        config,
        {"controller_pid": 1, "sweep_pid": 2, "observed_unix": 3.0},
    )
    signature = heartbeat.pop("hmac_sha256")
    key = Path(config["heartbeat_key_path"]).read_bytes()
    expected = hmac.new(key, controller._canonical(heartbeat), hashlib.sha256).hexdigest()
    assert signature == expected


def test_launchd_owns_caffeinate_and_the_exact_controller(tmp_path: Path):
    source, out, config_path, config = _prepared(tmp_path)
    payload = plistlib_loads(controller._launch_payload(config_path, config))
    arguments = payload["ProgramArguments"]
    assert arguments[:2] == ["/usr/bin/caffeinate", "-dims"]
    assert arguments[3] == str(source / "tools/run_rlc_reconciliation_controller.py")
    assert arguments[-2:] == [str(config_path), "--launchd-supervised"]
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["StandardOutPath"] == str(out / "controller.log")


def plistlib_loads(payload: bytes):
    import plistlib

    return plistlib.loads(payload)


def test_sweep_command_preserves_complete_engine_parameters(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    command = controller._sweep_command(config)
    assert command[command.index("--arms") + 1] == "full_stack"
    assert command[command.index("--episode-wall-s") + 1] == "20.0"
    assert command[command.index("--max-wall-s") + 1] == "60.0"
    assert command[command.index("--model") + 1] == config["model"]


def test_exact_group_termination_refuses_a_foreign_process_group(monkeypatch):
    class Process:
        pid = 17

    monkeypatch.setattr(os, "getpgid", lambda _pid: 99)
    with pytest.raises(controller.ControllerError, match="process_group_identity"):
        controller._terminate_exact_group(Process())


def test_controller_requires_launchd_supervision(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    with pytest.raises(controller.ControllerError, match="requires_launchd"):
        controller.run(config_path)


def test_controller_resumes_a_clean_wall_boundary_and_keeps_durable_progress(
    tmp_path: Path,
    monkeypatch,
):
    source = _source(tmp_path)
    sweep = source / "tools/run_rlc_reconciliation_sweep.py"
    sweep.write_text(
        """
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[sys.argv.index('--out-dir') + 1])
out.mkdir(parents=True, exist_ok=True)
marker = out / 'first-attempt-complete'
journal = out / 'journal.jsonl'
if not marker.exists():
    journal.write_text(json.dumps({'event': 'CELL', 'arm': 'vanilla', 'task_id': 'one'}) + '\\n')
    marker.write_text('durable')
    raise SystemExit(3)
(out / 'verdict.json').write_text(json.dumps({'decision': 'synthetic-complete'}))
""".lstrip(),
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    out = tmp_path / "campaign"
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit="b" * 40,
        model=model,
        out_dir=out,
        python=Path(sys.executable),
        arms="full_stack",
        seed=7,
        per_domain=1,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=0.01,
        stale_after_s=40.0,
        retry_backoff_s=0.01,
    )
    config_path = out / "controller_config.json"
    controller.write_prepared_campaign(config_path, config, manifest, key)

    monkeypatch.setattr(
        controller,
        "_verify_launchd_lineage",
        lambda _config: {"launchd_pid": 1, "caffeinate_pid": 2, "controller_pid": 3},
    )
    monkeypatch.setattr(controller, "GLOBAL_MODEL_LOCK", tmp_path / "model.lock")
    assert controller.run(config_path, launchd_supervised=True) == 0
    events = [
        json.loads(line)
        for line in (out / "controller_attempts.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "ATTEMPT_STARTED",
        "ATTEMPT_FINISHED",
        "ATTEMPT_STARTED",
        "ATTEMPT_FINISHED",
    ]
    assert events[1]["returncode"] == 3
    assert events[1]["cells"] == 1
    assert events[3]["returncode"] == 0
    status = json.loads((out / "controller_status.json").read_text())
    assert status["phase"] == "complete"
    assert (out / "sweep/first-attempt-complete").read_text() == "durable"
