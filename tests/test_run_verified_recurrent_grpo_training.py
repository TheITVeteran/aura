"""Contract tests for the production recurrent-GRPO launch executable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.learning.verified_recurrent_transition_repository import (
    CAMPAIGN_FINALIZER_ID,
    DURABLE_REPLAY_LOADER_ID,
    INDEPENDENT_SCORER_ID,
    PRODUCTION_EVIDENCE_PRODUCER_ID,
    TOKEN_CODEC_ID,
)
from tools import run_verified_recurrent_grpo_training as runner


def test_runner_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--help"],
        cwd=Path(runner.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-launch-bundle-sha256" in completed.stdout


def test_runtime_components_are_the_fixed_production_set() -> None:
    components = runner.verified_recurrent_runtime_components()

    assert components.evidence_producer_identity == PRODUCTION_EVIDENCE_PRODUCER_ID
    assert components.durable_artifact_loader_identity == DURABLE_REPLAY_LOADER_ID
    assert components.campaign_finalizer_identity == CAMPAIGN_FINALIZER_ID
    assert components.scorer_identity == INDEPENDENT_SCORER_ID
    assert components.token_codec_identity == TOKEN_CODEC_ID
    assert components.evidence_producer.__module__ != "__main__"
    assert components.durable_artifact_loader.__closure__ is None
    assert components.campaign_finalizer.__closure__ is None


def test_main_loads_pinned_bundle_and_forwards_only_training_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "launch.json"
    bundle.write_text("{}\n", encoding="ascii")
    factory = SimpleNamespace(
        training_argv=(
            "tools/train_grpo.py",
            "--model",
            "/models/resident",
            "--execution-mode",
            "recurrent",
        )
    )
    observed: dict[str, Any] = {}

    def fake_loader(
        path: str,
        *,
        expected_bundle_sha256: str,
        expected_preregistration_sha256: str,
        components: Any,
        now_unix: int,
    ) -> object:
        observed["loader"] = (
            path,
            expected_bundle_sha256,
            expected_preregistration_sha256,
            components,
            now_unix,
        )
        return factory

    def fake_training_main(*, verified_group_provider_factory: object) -> int:
        observed["factory"] = verified_group_provider_factory
        observed["argv"] = list(sys.argv)
        return 17

    monkeypatch.setattr(runner, "load_verified_transition_provider_factory", fake_loader)
    monkeypatch.setattr("tools.train_grpo.main", fake_training_main)
    monkeypatch.setattr(runner.time, "time", lambda: 1_900_000_000.9)
    original_argv = list(sys.argv)

    result = runner.main(
        [
            "--verified-launch-bundle",
            str(bundle),
            "--expected-launch-bundle-sha256",
            "a" * 64,
            "--expected-preregistration-sha256",
            "b" * 64,
            "--model",
            "/models/resident",
            "--execution-mode",
            "recurrent",
        ]
    )

    assert result == 17
    assert observed["loader"][0:2] == (str(bundle), "a" * 64)
    assert observed["loader"][1:3] == ("a" * 64, "b" * 64)
    assert observed["loader"][4] == 1_900_000_000
    assert observed["factory"] is factory
    assert observed["argv"] == [
        "tools/train_grpo.py",
        "--model",
        "/models/resident",
        "--execution-mode",
        "recurrent",
    ]
    assert sys.argv == original_argv


def test_main_requires_training_arguments() -> None:
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--verified-launch-bundle",
                "/tmp/launch.json",
                "--expected-launch-bundle-sha256",
                "a" * 64,
                "--expected-preregistration-sha256",
                "b" * 64,
            ]
        )


def test_main_rejects_training_argument_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SimpleNamespace(
        training_argv=("tools/train_grpo.py", "--model", "/models/frozen")
    )
    monkeypatch.setattr(
        runner,
        "load_verified_transition_provider_factory",
        lambda *_args, **_kwargs: factory,
    )
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--verified-launch-bundle",
                "/tmp/launch.json",
                "--expected-launch-bundle-sha256",
                "a" * 64,
                "--expected-preregistration-sha256",
                "b" * 64,
                "--model",
                "/models/substituted",
            ]
        )
