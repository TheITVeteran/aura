from __future__ import annotations

import hashlib
import importlib.util
import json
import plistlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "tools/watch_rlc_reconciliation_verification.py"
SPEC = importlib.util.spec_from_file_location("verification_watcher", PATH)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def test_result_never_authorizes_serving_fusion_or_wow(tmp_path: Path) -> None:
    program = tmp_path / "watcher.py"
    program.write_text("bounded watcher", encoding="utf-8")
    result = watcher._result(
        config={
            "campaign_id": "campaign",
            "config_sha256": "a" * 64,
            "source_commit": "b" * 40,
        },
        watcher_path=program,
        decision="independent_component_verification_complete",
        details={"verification_decision": "bounded_causal_canary_positive_replication_required"},
    )
    assert result["fusion_authorized"] is False
    assert result["wow_signal_authorized"] is False
    assert result["ordinary_serving_authorized"] is False
    body = {key: value for key, value in result.items() if key != "receipt_sha256"}
    assert result["receipt_sha256"] == watcher._sha(body)


def test_config_digest_tampering_is_rejected_before_controller_import(tmp_path: Path) -> None:
    config = tmp_path / "controller.json"
    config.write_text(
        json.dumps(
            {
                "config_sha256": "0" * 64,
                "source_root": str(tmp_path),
                "controller_program": str(tmp_path / "controller.py"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(watcher.WatcherError, match="controller_config_digest_invalid"):
        watcher._load_controller(config)


def test_launch_payload_is_sleep_protected_and_non_restarting(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    program = source / "tools" / "run_rlc_reconciliation_controller.py"
    program.parent.mkdir()
    program.write_text("", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    config = {
        "campaign_id": "campaign",
        "config_sha256": "a" * 64,
        "source_root": str(source),
        "controller_program": str(program),
        "python": str(python),
        "out_dir": str(tmp_path / "campaign" / "sweep"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(watcher, "_load_controller", lambda _path: (config, object()))
    label, payload = watcher._launch_payload(
        config_path=config_path,
        output=tmp_path / "campaign" / "watcher.json",
        timeout_s=100.0,
        poll_s=5.0,
    )
    decoded = plistlib.loads(payload)
    assert label == "com.aura.rlc-independent-verifier.campaign"
    assert decoded["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-dims"]
    assert decoded["KeepAlive"] == {"SuccessfulExit": False}
    assert decoded["ProgramArguments"][-4:] == [
        "--timeout-s",
        "100.0",
        "--poll-s",
        "5.0",
    ]


def test_file_hash_streams_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"a" * (2 * 1024 * 1024 + 17))
    assert watcher._sha_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
