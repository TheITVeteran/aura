import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from core.learning.genuine_learning_pipeline import LearningScheduler, LoRATrainer
from core.tasks.managed_command import ManagedCommandResult


def _training_record() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are Aura."},
            {"role": "user", "content": "learn this"},
            {"role": "assistant", "content": "I learned it carefully."},
        ],
        "_meta": {"quality": 0.9},
    }


def test_lora_trainer_uses_managed_command_runner(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], float]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append((command, timeout_s))
        return ManagedCommandResult(command, 0, "trained", "", 0.1)

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        num_epochs=2,
        batch_size=1,
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    saved = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    command = calls[0][0]
    assert success is True
    assert output == "trained"
    assert calls[0][1] == 1800.0
    assert saved == [{"messages": _training_record()["messages"]}]
    assert "--iters" in command
    assert command[command.index("--iters") + 1] == "2"


def test_lora_trainer_reports_managed_timeout(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        return ManagedCommandResult(command, None, "", "late", timeout_s, timed_out=True)

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    assert success is False
    assert output == "timeout"


def test_lora_trainer_reports_runner_error(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        raise OSError("runner unavailable")

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    assert calls
    assert success is False
    assert output == "runner unavailable"


def test_lora_trainer_restores_adapter_snapshot(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    weight_file = adapter_dir / "weights.safetensors"
    weight_file.write_text("old", encoding="utf-8")
    trainer = LoRATrainer(model_path="/models/aura", adapter_dir=str(adapter_dir))

    snapshot = trainer.create_rollback_snapshot()
    weight_file.write_text("new", encoding="utf-8")

    assert trainer.restore_rollback_snapshot(snapshot) is True
    assert weight_file.read_text(encoding="utf-8") == "old"


def test_learning_scheduler_rolls_back_when_benchmark_fails(monkeypatch, tmp_path: Path) -> None:
    import core.will as will_module

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    weight_file = adapter_dir / "weights.safetensors"
    weight_file.write_text("old", encoding="utf-8")

    def runner(command: tuple[str, ...], _timeout_s: float) -> ManagedCommandResult:
        weight_file.write_text("new", encoding="utf-8")
        return ManagedCommandResult(command, 0, "trained", "", 0.1)

    class Buffer:
        def __len__(self) -> int:
            return 1

        def get_training_batch(self, n: int = 50, min_quality: float = 0.65) -> list[dict]:
            return [_training_record()]

    class Benchmark:
        async def run(self, _inference_fn):
            return False, ["identity regression"]

    class Decision:
        reason = "approved"
        receipt_id = "receipt-learning-test"

        def is_approved(self) -> bool:
            return True

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(adapter_dir),
        command_runner=runner,
    )
    trainer.MIN_TRAIN_INTERVAL_S = 0.0
    scheduler = LearningScheduler(Buffer(), trainer, Benchmark(), batch_size=1)
    scheduler._last_activity = time.time() - 120.0

    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: Decision()),
    )

    accepted = asyncio.run(scheduler.run_if_ready(lambda _prompt: "regressed"))

    assert accepted is False
    assert weight_file.read_text(encoding="utf-8") == "old"


def test_learning_scheduler_refuses_training_without_candidate_inference(monkeypatch, tmp_path: Path) -> None:
    import core.will as will_module

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    weight_file = adapter_dir / "weights.safetensors"
    weight_file.write_text("old", encoding="utf-8")

    def runner(command: tuple[str, ...], _timeout_s: float) -> ManagedCommandResult:
        weight_file.write_text("new", encoding="utf-8")
        return ManagedCommandResult(command, 0, "trained", "", 0.1)

    class Buffer:
        def __len__(self) -> int:
            return 1

        def get_training_batch(self, n: int = 50, min_quality: float = 0.65) -> list[dict]:
            return [_training_record()]

    class Benchmark:
        called = False

        async def run(self, _inference_fn):
            self.called = True
            return True, []

    class Decision:
        reason = "approved"
        receipt_id = "receipt-learning-test"

        def is_approved(self) -> bool:
            return True

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(adapter_dir),
        command_runner=runner,
    )
    trainer.MIN_TRAIN_INTERVAL_S = 0.0
    benchmark = Benchmark()
    scheduler = LearningScheduler(Buffer(), trainer, benchmark, batch_size=1)
    scheduler._last_activity = time.time() - 120.0

    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: Decision()),
    )

    accepted = asyncio.run(scheduler.run_if_ready(None))

    assert accepted is False
    assert weight_file.read_text(encoding="utf-8") == "old"
    assert benchmark.called is False
