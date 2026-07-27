from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools import launch_recurrent_sft_falsification as containment


def test_forbidden_roots_cover_production_and_model_siblings(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    small = model_root / "small"
    resident = model_root / "resident"
    small.mkdir(parents=True)
    resident.mkdir()

    roots = containment._forbidden_roots(model_dir=small)

    assert resident.resolve() in roots
    assert Path.home().resolve() / ".aura/fusion" in roots
    assert Path.home().resolve() / ".aura/model_registry" in roots


def test_artifact_files_require_exact_existing_names(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    (root / "one.json").write_text("{}")
    assert containment._artifact_files(
        root,
        ("one.json",),
        role="test",
    ) == ((root / "one.json").resolve(),)
    with pytest.raises(FileNotFoundError):
        containment._artifact_files(
            root,
            ("missing.json",),
            role="test",
        )


def test_launch_contract_is_precontained_and_nonresumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self, argv: list[str]) -> argparse.Namespace:
            captured["argv"] = argv
            return argparse.Namespace()

    monkeypatch.setattr(
        containment.run_detached_step,
        "build_parser",
        lambda: _Parser(),
    )
    monkeypatch.setattr(
        containment.run_detached_step,
        "_launch",
        lambda _args, _parser: {"status": "launched"},
    )
    contract = {
        "contract_sha256": "1" * 64,
        "authority_sha256": "2" * 64,
        "output_dir": str(tmp_path / "output"),
        "detached_dir": str(tmp_path / "detached"),
        "timeout_s": 120.0,
        "command": ["/usr/bin/sandbox-exec", "-f", "/private/profile", "python"],
        "environment": {"PATH": "/usr/bin"},
        "claims_not_supported": ["frontier_performance"],
    }

    result = containment.launch_contract(contract)

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--resume-contract") + 1] == "none"
    assert argv[argv.index("--containment-mode") + 1] == "precontained-sandbox"
    assert "--resume-verifier-json" not in argv
    assert result["status"] == "detached_falsification_evaluator_launched"


def test_parser_requires_control_report_hash() -> None:
    parser = containment._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare",
                "--python",
                "/usr/bin/python3",
                "--reference-authority",
                "/a",
                "--expected-authority-sha256",
                "1" * 64,
                "--reference-checkpoint",
                "/b",
                "--expected-checkpoint-sha256",
                "2" * 64,
                "--control-report",
                "/c",
                "--candidate-dir",
                "/d",
                "--evaluator-dir",
                "/e",
                "--model-dir",
                "/f",
                "--execution-spec",
                "/g",
                "--contract-dir",
                "/h",
                "--output-dir",
                "/i",
                "--detached-dir",
                "/j",
            ]
        )


def test_evaluator_environment_is_sanitized(tmp_path: Path) -> None:
    environment = containment._sanitized_environment(
        python=Path("/usr/bin/python3"),
        runtime_root=tmp_path / "runtime",
        model_lane_state=tmp_path / "lane.json",
    )
    assert "AURA_MODEL_PATH" not in environment
    assert not any("EVALUATOR" in key or "REPLAY" in key for key in environment)
    assert environment["PYTHONHASHSEED"] == "0"
