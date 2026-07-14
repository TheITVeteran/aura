from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning import lora_trainer as trainer_module


@pytest.mark.asyncio
async def test_adapter_only_training_does_not_commit_crsm_consumed_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    dataset_path = tmp_path / "synthetic_training"
    dataset_path.mkdir()
    marker_called = False

    class _Gateway:
        async def run_async(self, *_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout="trained", stderr="")

    def _unexpected_monitor():
        nonlocal marker_called
        marker_called = True
        raise AssertionError("adapter-only training must not publish CRSM closure")

    import core.consciousness.crsm_loop_monitor as monitor_module

    monkeypatch.setattr(trainer_module, "get_subprocess_gateway", lambda: _Gateway())
    monkeypatch.setattr(monitor_module, "get_crsm_loop_monitor", _unexpected_monitor)
    trainer = trainer_module.LoraTrainer.__new__(trainer_module.LoraTrainer)
    trainer.config = SimpleNamespace(llm=SimpleNamespace(local_cortex_path=str(model_path)))

    result = await trainer.train_adapter(
        str(dataset_path),
        str(tmp_path / "adapter"),
        iters=1,
    )

    assert result["status"] == "success"
    assert marker_called is False
