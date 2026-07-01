from __future__ import annotations

import json
import sys
from pathlib import Path

import core.learning.preference_trainer as preference_trainer
from core.learning.preference_trainer import (
    PreferenceTrainingRequest,
    build_preference_training_command,
    check_preference_trainer_available,
    export_preference_splits,
    run_verifiable_preference_training,
)
from core.learning.verifiable_preference_harness import Attempt, VerifiablePreferenceHarness
from core.tasks.managed_command import ManagedCommandResult


def _request(tmp_path: Path, *, store: Path | None = None, min_rows: int = 2) -> PreferenceTrainingRequest:
    return PreferenceTrainingRequest(
        model_path=tmp_path / "model",
        store_path=store or (tmp_path / "prefs.jsonl"),
        adapter_path=tmp_path / "adapter",
        data_dir=tmp_path / "data",
        min_rows=min_rows,
        iters=3,
        batch_size=1,
        num_layers=4,
        timeout_seconds=90,
    )


def test_preference_trainer_availability_reports_required_and_optional_modules(monkeypatch):
    origins = {
        "mlx_lm_lora": "/pkg/__init__.py",
        "mlx_lm_lora.train": "/pkg/train.py",
        "mlx_lm_lora.trainer.dpo_trainer": "/pkg/dpo.py",
        "mlx_lm_lora.trainer.orpo_trainer": "/pkg/orpo.py",
        "mlx_lm_lora.trainer.online_dpo_trainer": "/pkg/online.py",
        "mlx_lm_lora.trainer.grpo_trainer": "/pkg/grpo.py",
    }

    class Spec:
        def __init__(self, origin: str) -> None:
            self.origin = origin

    monkeypatch.setattr(preference_trainer.importlib.util, "find_spec", lambda name: Spec(origins[name]))
    monkeypatch.setattr(preference_trainer.importlib.metadata, "version", lambda name: "2.1.0")

    report = check_preference_trainer_available()

    assert report["ok"] is True
    assert report["version"] == "2.1.0"
    assert "dpo" in report["pair_train_modes"]
    assert report["modules"]["mlx_lm_lora.trainer.grpo_trainer"] == "/pkg/grpo.py"


def test_preference_split_export_cleans_dedups_and_preserves_pair_format(tmp_path):
    rows = [
        {"prompt": "p1", "chosen": "good", "rejected": "bad"},
        {"prompt": "p1", "chosen": "good", "rejected": "bad"},
        {"prompt": "", "chosen": "skip", "rejected": "skip2"},
        {"prompt": "p2", "chosen": "right", "rejected": "wrong"},
    ]

    counts = export_preference_splits(rows, tmp_path / "data")

    assert counts == {"train": 2}
    exported = [
        json.loads(line)
        for line in (tmp_path / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exported == [
        {"prompt": "p1", "chosen": "good", "rejected": "bad"},
        {"prompt": "p2", "chosen": "right", "rejected": "wrong"},
    ]


def test_preference_training_command_uses_mlx_lm_lora_pair_trainer(tmp_path):
    request = _request(tmp_path)

    cmd = build_preference_training_command(request)

    assert cmd[:3] == (sys.executable, "-m", "mlx_lm_lora.train")
    assert "--train-mode" in cmd
    assert "dpo" in cmd
    assert "--train-type" in cmd
    assert "lora" in cmd
    assert "--efficient-long-context" in cmd
    assert "--grad-checkpoint" in cmd


def test_preference_training_refuses_without_real_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preference_trainer,
        "check_preference_trainer_available",
        lambda: {"ok": True, "version": "2.1.0"},
    )

    result = run_verifiable_preference_training(_request(tmp_path, min_rows=1), dry_run=True)

    assert result["ok"] is False
    assert result["reason"] == "insufficient_verifiable_preference_rows"
    assert result["rows"] == 0


def test_preference_training_dry_run_exports_real_pairs_and_command(tmp_path, monkeypatch):
    store = tmp_path / "prefs.jsonl"
    harness = VerifiablePreferenceHarness(store_path=store)
    for i in range(3):
        harness.ingest(
            f"problem {i}",
            [
                Attempt(f"correct {i}", verified=True, checked=True),
                Attempt(f"wrong {i}", verified=False, checked=True),
            ],
            domain="logic",
        )
    monkeypatch.setattr(
        preference_trainer,
        "check_preference_trainer_available",
        lambda: {"ok": True, "version": "2.1.0"},
    )

    result = run_verifiable_preference_training(_request(tmp_path, store=store, min_rows=3), dry_run=True)

    assert result["ok"] is True
    assert result["status"] == "dry_run"
    assert result["rows"] == 3
    assert result["split_counts"] == {"train": 3}
    assert (tmp_path / "data" / "train.jsonl").exists()
    assert "mlx_lm_lora.train" in result["command"]


def test_preference_training_runs_managed_command_after_export(tmp_path, monkeypatch):
    store = tmp_path / "prefs.jsonl"
    harness = VerifiablePreferenceHarness(store_path=store)
    for i in range(2):
        harness.ingest(
            f"problem {i}",
            [
                Attempt(f"correct {i}", verified=True, checked=True),
                Attempt(f"wrong {i}", verified=False, checked=True),
            ],
        )
    captured = {}
    monkeypatch.setattr(
        preference_trainer,
        "check_preference_trainer_available",
        lambda: {"ok": True, "version": "2.1.0"},
    )

    def fake_run(command, *, timeout_s):
        captured["command"] = command
        captured["timeout_s"] = timeout_s
        return ManagedCommandResult(tuple(command), 0, "trained", "", 0.5)

    monkeypatch.setattr(preference_trainer, "run_project_command", fake_run)

    result = run_verifiable_preference_training(_request(tmp_path, store=store, min_rows=2))

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["returncode"] == 0
    assert captured["timeout_s"] == 90.0
    assert captured["command"][:3] == (sys.executable, "-m", "mlx_lm_lora.train")
