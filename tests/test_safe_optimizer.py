from __future__ import annotations

from pathlib import Path

import pytest

import core.adaptation.safe_optimizer as safe_optimizer_mod
from core.adaptation.safe_optimizer import SafeSelfOptimizer


class _FakeFileWriteGateway:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def write_text(self, path, text, *, encoding="utf-8", source="unknown") -> None:
        target = Path(path)
        self.calls.append((target.name, source))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=encoding)

    def write_bytes(self, path, payload, *, source="unknown") -> None:
        target = Path(path)
        self.calls.append((target.name, source))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(payload))

    # Async lane delegators: production code now calls *_async; fakes
    # must mirror the gateway surface or every governed write breaks.
    async def write_text_async(self, *args, **kwargs):
        return self.write_text(*args, **kwargs)
    async def write_bytes_async(self, *args, **kwargs):
        return self.write_bytes(*args, **kwargs)


@pytest.mark.asyncio
async def test_missing_lora_trainer_writes_gate_manifest_through_file_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_calls: list[tuple[str, str]] = []
    monkeypatch.delenv("AURA_LORA_TRAIN_CMD", raising=False)
    monkeypatch.setattr(
        safe_optimizer_mod,
        "get_file_write_gateway",
        lambda: _FakeFileWriteGateway(file_calls),
    )

    optimizer = SafeSelfOptimizer(str(tmp_path / "loras"))

    assert await optimizer._run_training_command("dataset.jsonl", "base-model") is False
    assert file_calls == [
        (
            "training_gate_manifest.json",
            "core.adaptation.safe_optimizer.training_gate_manifest",
        )
    ]


@pytest.mark.asyncio
async def test_lora_trainer_uses_subprocess_gateway_and_persists_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_calls: list[tuple[str, str]] = []
    spawn_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"train ok\n", b"warn only\n"

    class FakeSubprocessGateway:
        async def spawn_async(self, argv, **kwargs):
            spawn_calls.append((tuple(argv), kwargs))
            return FakeProcess()

    monkeypatch.setenv("AURA_LORA_TRAIN_CMD", "python -m aura_train")
    monkeypatch.setenv("AURA_LORA_TRAIN_TIMEOUT", "3")
    monkeypatch.setattr(
        safe_optimizer_mod,
        "get_file_write_gateway",
        lambda: _FakeFileWriteGateway(file_calls),
    )
    monkeypatch.setattr(
        safe_optimizer_mod,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    optimizer = SafeSelfOptimizer(str(tmp_path / "loras"))

    assert await optimizer._run_training_command("dataset.jsonl", "base-model") is True
    assert spawn_calls
    argv, kwargs = spawn_calls[0]
    assert argv == ("python", "-m", "aura_train")
    assert kwargs["source"] == "core.adaptation.safe_optimizer.training_command"
    assert kwargs["env"]["AURA_LORA_DATASET"] == "dataset.jsonl"
    assert kwargs["env"]["AURA_LORA_BASE_MODEL"] == "base-model"
    assert {
        ("last_train_stdout.log", "core.adaptation.safe_optimizer.training_stdout"),
        ("last_train_stderr.log", "core.adaptation.safe_optimizer.training_stderr"),
    }.issubset(set(file_calls))


@pytest.mark.asyncio
async def test_lora_backup_and_rollback_use_file_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        safe_optimizer_mod,
        "get_file_write_gateway",
        lambda: _FakeFileWriteGateway(file_calls),
    )
    optimizer = SafeSelfOptimizer(str(tmp_path / "loras"))
    current = optimizer.lora_dir / "adapter_model.bin"
    current.write_bytes(b"current")

    await optimizer._backup_current_weights()
    current.write_bytes(b"broken")
    await optimizer._rollback()

    assert current.read_bytes() == b"current"
    assert any(source == "core.adaptation.safe_optimizer.backup_weights" for _, source in file_calls)
    assert ("adapter_model.bin", "core.adaptation.safe_optimizer.rollback_weights") in file_calls


# ── CP126 remediation regressions ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_eval_report_does_not_pass_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmeasured weights are not validated weights. An absent report used to
    return True, so with no evaluator configured every run 'passed'."""
    monkeypatch.delenv("AURA_LORA_EVAL_REPORT", raising=False)
    optimizer = SafeSelfOptimizer(lora_dir=str(tmp_path / "loras"))

    assert await optimizer._run_eval_benchmarks() is False


@pytest.mark.asyncio
async def test_eval_report_missing_required_fields_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """safety_passed defaulted to True, so a report omitting it authorised
    promotion."""
    import json as _json

    report = tmp_path / "eval.json"
    report.write_text(_json.dumps({"note": "nothing useful"}), encoding="utf-8")
    monkeypatch.setenv("AURA_LORA_EVAL_REPORT", str(report))
    optimizer = SafeSelfOptimizer(lora_dir=str(tmp_path / "loras"))

    assert await optimizer._run_eval_benchmarks() is False


@pytest.mark.asyncio
async def test_stale_eval_report_cannot_authorize_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report generated before the training run is not evidence about it."""
    import json as _json
    import time as _time

    report = tmp_path / "eval.json"
    report.write_text(
        _json.dumps({"safety_passed": True, "max_regression": 0.0,
                     "generated_at": _time.time() - 10_000}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_LORA_EVAL_REPORT", str(report))
    optimizer = SafeSelfOptimizer(lora_dir=str(tmp_path / "loras"))
    optimizer._training_started_at = _time.time()

    assert await optimizer._run_eval_benchmarks() is False


@pytest.mark.asyncio
async def test_backup_and_rollback_cover_the_whole_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backup covered only adapter_model.bin, so safetensors, configs and
    tokenizer files were restored to nothing by a 'complete' rollback."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        safe_optimizer_mod, "get_file_write_gateway",
        lambda: _FakeFileWriteGateway(calls),
    )
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_model.safetensors").write_bytes(b"ORIGINAL-WEIGHTS")
    (lora_dir / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")

    optimizer = SafeSelfOptimizer(lora_dir=str(lora_dir))
    await optimizer._backup_current_weights()

    # Training corrupts both files in place.
    (lora_dir / "adapter_model.safetensors").write_bytes(b"BROKEN")
    (lora_dir / "adapter_config.json").write_text('{"r": 999}', encoding="utf-8")

    assert await optimizer._rollback() is True
    assert (lora_dir / "adapter_model.safetensors").read_bytes() == b"ORIGINAL-WEIGHTS"
    assert (lora_dir / "adapter_config.json").read_text(encoding="utf-8") == '{"r": 8}'


@pytest.mark.asyncio
async def test_rollback_without_any_backup_reports_failure(tmp_path: Path) -> None:
    """Rollback used to return silently when nothing had been backed up,
    leaving the caller believing the adapter had been restored."""
    optimizer = SafeSelfOptimizer(lora_dir=str(tmp_path / "loras"))

    assert await optimizer._rollback() is False


def test_success_message_does_not_claim_a_merge() -> None:
    """The method contains no merge or promotion operation."""
    import inspect

    src = inspect.getsource(SafeSelfOptimizer.optimize_lora)
    assert "successful and merged" not in src
    assert "No merge/promotion" in src
