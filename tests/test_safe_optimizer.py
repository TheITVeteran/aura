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
