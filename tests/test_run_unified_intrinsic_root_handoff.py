from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from tools import run_unified_intrinsic_root_handoff as handoff
from tools.unified_intrinsic_resident_identity import canonical_sha256


def _arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    campaign = tmp_path / "campaign"
    values: dict[str, object] = {
        "action": "prepare",
        "config": None,
        "campaign": campaign,
        "canary_root": campaign / "canary",
        "powered_root": campaign / "powered",
        "powered_controller": tmp_path / "capsule" / "controller.py",
        "output": campaign / "handoff",
        "poll_interval": 0.01,
        "timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _binding() -> dict:
    return {
        "schema": "aura.unified_intrinsic.root_control_binding.v1",
        "mode": "deterministic_pretraining_root",
        "campaign_root": "/tmp/root",
        "stem": "checkpoint_latest",
        "controller_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
    }


def _config(tmp_path: Path) -> dict:
    campaign = tmp_path / "campaign"
    handoff_root = campaign / "handoff"
    handoff_root.mkdir(parents=True, exist_ok=True)
    return {
        "schema": handoff.CONFIG_SCHEMA,
        "config_sha256": "c" * 64,
        "campaign": str(campaign),
        "campaign_id": "campaign",
        "campaign_config_sha256": "d" * 64,
        "handoff_root": str(handoff_root),
        "canary_root": str(campaign / "canary"),
        "canary_plan_sha256": "e" * 64,
        "powered_root": str(campaign / "powered"),
        "powered_plan_sha256": "f" * 64,
        "matched_control": _binding(),
        "powered_controller": str(tmp_path / "controller.py"),
        "runtime_python": str(Path(sys.executable).absolute()),
        "source": {},
    }


def test_prepare_freezes_matching_canary_and_powered_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    campaign = arguments.campaign
    campaign.mkdir()
    arguments.canary_root.mkdir()
    arguments.powered_root.mkdir()
    arguments.powered_controller.parent.mkdir()
    arguments.powered_controller.write_text("# frozen\n", encoding="ascii")
    binding = _binding()
    monkeypatch.setattr(
        handoff,
        "_campaign_config",
        lambda _campaign: {
            "campaign_id": "campaign",
            "config_sha256": "d" * 64,
            "paths": {"campaign_root": str(campaign)},
            "runtime": {"interpreter": {"executable": str(Path(sys.executable).absolute())}},
        },
    )
    monkeypatch.setattr(
        handoff.evaluator,
        "_existing_plan",
        lambda _args: {
            "plan_sha256": "e" * 64,
            "scientific": {"matched_control": binding},
        },
    )
    monkeypatch.setattr(
        handoff.replication,
        "_load_plan",
        lambda _args: (
            campaign,
            {},
            {"plan_sha256": "f" * 64, "matched_control": binding},
        ),
    )
    monkeypatch.setattr(
        handoff,
        "_source_identity",
        lambda _path: {"identity_sha256": "1" * 64},
    )

    config = handoff.prepare(arguments)
    reopened = json.loads((arguments.output / "handoff-config.json").read_bytes())

    assert config == reopened
    assert config["canary_plan_sha256"] == "e" * 64
    assert config["powered_plan_sha256"] == "f" * 64
    assert config["matched_control"] == binding
    assert config["config_sha256"] == canonical_sha256(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )


def test_prepare_rejects_different_root_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    campaign = arguments.campaign
    campaign.mkdir()
    arguments.canary_root.mkdir()
    arguments.powered_root.mkdir()
    arguments.powered_controller.parent.mkdir()
    arguments.powered_controller.write_text("# frozen\n", encoding="ascii")
    monkeypatch.setattr(
        handoff,
        "_campaign_config",
        lambda _campaign: {
            "campaign_id": "campaign",
            "config_sha256": "d" * 64,
            "paths": {"campaign_root": str(campaign)},
            "runtime": {"interpreter": {"executable": str(Path(sys.executable).absolute())}},
        },
    )
    monkeypatch.setattr(
        handoff.evaluator,
        "_existing_plan",
        lambda _args: {
            "plan_sha256": "e" * 64,
            "scientific": {"matched_control": _binding()},
        },
    )
    monkeypatch.setattr(
        handoff.replication,
        "_load_plan",
        lambda _args: (
            campaign,
            {},
            {
                "plan_sha256": "f" * 64,
                "matched_control": {**_binding(), "controller_sha256": "9" * 64},
            },
        ),
    )

    with pytest.raises(handoff.RootHandoffError, match="controls differ"):
        handoff.prepare(arguments)


@pytest.mark.parametrize("supported", [True, False])
def test_run_launches_powered_only_after_supported_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
) -> None:
    config = _config(tmp_path)
    config_path = Path(config["handoff_root"]) / "handoff-config.json"
    config_path.write_text("{}\n", encoding="ascii")
    arguments = _arguments(tmp_path, action="run", config=config_path)
    verdict = {
        "verdict": "supported" if supported else "refuted",
        "supported": supported,
        "verdict_sha256": "a" * 64,
    }
    launched: list[bool] = []
    inhibitor = argparse.Namespace(pid=1234)
    monkeypatch.setattr(handoff, "_load_config", lambda _path: config)
    monkeypatch.setattr(handoff, "_start_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(
        handoff,
        "_stop_sleep_inhibitor",
        lambda observed: observed is inhibitor or pytest.fail("wrong sleep inhibitor"),
    )
    monkeypatch.setattr(
        handoff.evaluator,
        "status",
        lambda _args: {"state": "completed", "report": {"report_sha256": "b" * 64}},
    )
    monkeypatch.setattr(handoff.adjudicator, "adjudicate_report", lambda _report: verdict)
    monkeypatch.setattr(
        handoff,
        "_launch_powered",
        lambda _config: launched.append(True) or {"reopened": False},
    )
    monkeypatch.setattr(
        handoff,
        "_publish_status",
        lambda _config, state, details: {"state": state, "details": details},
    )

    result = handoff.run(arguments)

    assert result["supported"] is supported
    assert result["state"] == ("powered_launched" if supported else "canary_refuted")
    assert launched == ([True] if supported else [])
    assert (Path(config["handoff_root"]) / "canary-verdict.json").exists()


def test_failed_powered_controller_is_not_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        handoff.replication,
        "status",
        lambda _args: {
            "controller": {"state": "failed"},
            "controller_liveness": "alive",
        },
    )

    assert handoff._powered_state(config) is None  # noqa: SLF001


def test_powered_launch_uses_the_frozen_controller_and_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(handoff, "_powered_state", lambda _config: None)

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"plan_sha256": config["powered_plan_sha256"]})

    def run(command: list[str], **_kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(handoff.subprocess, "run", run)

    result = handoff._launch_powered(config)  # noqa: SLF001

    assert result["reopened"] is False
    assert commands == [
        [
            config["runtime_python"],
            config["powered_controller"],
            "install-launchd",
            config["campaign"],
            "--output",
            config["powered_root"],
        ]
    ]


def test_launch_contract_is_restartable_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = Path(config["handoff_root"]) / "handoff-config.json"
    config_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(handoff, "LAUNCH_AGENTS_ROOT", tmp_path / "agents")

    _, plist_bytes, intent = handoff._launch_contract(  # noqa: SLF001
        config_path,
        config,
        _arguments(tmp_path, config=config_path),
    )

    import plistlib

    plist = plistlib.loads(plist_bytes)
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ProgramArguments"][:4] == [
        config["runtime_python"],
        str(Path(handoff.__file__).resolve()),
        "run",
        str(config_path),
    ]
    assert intent["config_sha256"] == config["config_sha256"]
