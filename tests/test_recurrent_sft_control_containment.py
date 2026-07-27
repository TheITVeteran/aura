from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import launch_recurrent_sft_controls as containment


def test_sanitized_environment_excludes_inherited_model_and_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_MODEL_PATH", "/forbidden/model")
    monkeypatch.setenv("AURA_EVALUATOR_PATH", "/forbidden/evaluator")
    environment = containment._sanitized_environment(
        python=Path("/usr/bin/python3"),
        runtime_root=tmp_path / "runtime",
        model_lane_state=tmp_path / "lane.json",
    )
    assert "AURA_MODEL_PATH" not in environment
    assert "AURA_EVALUATOR_PATH" not in environment
    assert not any("EVALUATOR" in key or "REPLAY" in key for key in environment)
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"


def test_forbidden_roots_include_evaluator_production_and_model_siblings(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    small = model_root / "small"
    resident = model_root / "resident"
    small.mkdir(parents=True)
    resident.mkdir()
    projected = tmp_path / "campaign" / "run" / "projected.json"
    projected.parent.mkdir(parents=True)
    projected.write_text("{}")
    roots = containment._forbidden_roots(
        projected_dataset=projected,
        model_dir=small,
    )
    assert resident.resolve() in roots
    assert tmp_path / "campaign" / "evaluator" in roots
    assert Path.home().resolve() / ".aura/fusion" in roots


def test_disjoint_contract_rejects_evaluator_read_or_write(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator"
    with pytest.raises(
        containment.RecurrentSFTControlContainmentError,
        match="forbidden_overlap",
    ):
        containment._assert_disjoint(
            reads=[evaluator / "holdout.json"],
            writes=[],
            forbidden=[evaluator],
        )
    with pytest.raises(
        containment.RecurrentSFTControlContainmentError,
        match="forbidden_overlap",
    ):
        containment._assert_disjoint(
            reads=[],
            writes=[evaluator / "output"],
            forbidden=[evaluator],
        )


def test_launch_contract_declares_no_replay_and_precontained_sandbox(
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
        "claims_not_supported": ["heldout_transfer"],
    }
    result = containment.launch_contract(contract)
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--resume-contract") + 1] == "none"
    assert argv[argv.index("--containment-mode") + 1] == "precontained-sandbox"
    assert "--resume-verifier-json" not in argv
    assert result["status"] == "detached_control_training_launched"


def test_parser_requires_external_checkpoint_hash() -> None:
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
                "--projected-dataset",
                "/c",
                "--model-dir",
                "/d",
                "--execution-spec",
                "/e",
                "--contract-dir",
                "/f",
                "--output-dir",
                "/g",
                "--detached-dir",
                "/h",
            ]
        )


def test_contract_hash_is_canonical() -> None:
    left = containment._sha256_json({"b": 2, "a": 1})
    right = containment._sha256_json(json.loads('{"a":1,"b":2}'))
    assert left == right
