from types import SimpleNamespace

import pytest

from core.adaptation.distillation_pipe import DistillationPipe


class AsyncCallRecorder:
    def __init__(self, result=None, error: BaseException | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class AlignmentAuditorRecorder:
    def __init__(self):
        self.audit_entry = AsyncCallRecorder({"safe": True})


@pytest.mark.asyncio
async def test_teacher_unavailable_requeues_distillation_item(tmp_path, monkeypatch):
    pipe = DistillationPipe(dataset_path=str(tmp_path / "lora_dataset.jsonl"))
    await pipe.flag_for_distillation("prompt", "local", 0.2)
    brain = SimpleNamespace(think=AsyncCallRecorder(error=TimeoutError("teacher down")))

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(get_service))
    result = await pipe.run_distillation_cycle()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 1
    assert pipe._pending[0]["attempts"] == 1
    assert not (tmp_path / "lora_dataset.jsonl").exists()


@pytest.mark.asyncio
async def test_dataset_write_failure_requeues_until_retry_budget(tmp_path):
    pipe = DistillationPipe(dataset_path=str(tmp_path))
    pipe._pending.append(
        {
            "prompt": "prompt",
            "local_response": "local",
            "confidence": 0.2,
            "context": {},
            "attempts": 2,
            "timestamp": 1.0,
        }
    )
    think = AsyncCallRecorder(
        SimpleNamespace(content="teacher answer", metadata={"model": "teacher"})
    )
    brain = SimpleNamespace(
        think=think
    )

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    auditor = AlignmentAuditorRecorder()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(get_service))
        monkeypatch.setattr("core.adaptation.auditor.AlignmentAuditor", lambda: auditor)
        result = await pipe.run_distillation_cycle()
    finally:
        monkeypatch.undo()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 0
    assert pipe._pending == []


@pytest.mark.asyncio
async def test_dataset_write_failure_requeues_when_budget_remains(tmp_path):
    pipe = DistillationPipe(dataset_path=str(tmp_path))
    await pipe.flag_for_distillation("prompt", "local", 0.2)
    think = AsyncCallRecorder(
        SimpleNamespace(content="teacher answer", metadata={"model": "teacher"})
    )
    brain = SimpleNamespace(
        think=think
    )

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    auditor = AlignmentAuditorRecorder()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(get_service))
        monkeypatch.setattr("core.adaptation.auditor.AlignmentAuditor", lambda: auditor)
        result = await pipe.run_distillation_cycle()
    finally:
        monkeypatch.undo()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 1
    assert pipe._pending[0]["attempts"] == 1
