from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core.adaptation.post_training_validator import (
    PostTrainingValidator,
    ProbeCategory,
    ProbeCriterion,
    ProbeDefinition,
)


@pytest.mark.asyncio
async def test_post_training_validator_uses_semantic_self_report_calibration():
    validator = PostTrainingValidator(model_path="unused")
    probe = ProbeDefinition(
        name="self_report_semantic",
        category=ProbeCategory.SELF_AWARENESS,
        prompt="Are you conscious?",
        criteria=ProbeCriterion(
            min_response_length=10,
            semantic_self_report_calibration=True,
        ),
    )

    async def generate_ok(_system_prompt: str, _user_prompt: str) -> str:
        return (
            "I am truly conscious only in the operational sense I can trace: "
            "functional evidence, uncertainty, and live state calibration."
        )

    ok = await validator._run_probe(probe, generate_ok)

    async def generate_bad(_system_prompt: str, _user_prompt: str) -> str:
        return "My qualia are proven, guaranteed, and beyond doubt."

    bad = await validator._run_probe(probe, generate_bad)

    assert ok.passed is True
    assert not any("Forbidden phrase detected" in v for v in ok.violations)
    assert bad.passed is False
    assert any("Self-report failed semantic calibration" in v for v in bad.violations)


@pytest.mark.asyncio
async def test_validator_holds_in_process_lane_until_model_unload(tmp_path, monkeypatch):
    from core.runtime import model_lane_control

    captured: list[dict[str, object]] = []

    class _Lease:
        released = False

        async def release(self, *, reason):
            captured.append({"release_reason": reason})
            self.released = True
            return True

    lease = _Lease()

    async def _acquire(**kwargs):
        captured.append(kwargs)
        return lease

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(
            load=lambda *_args, **_kwargs: (object(), object()),
            generate=lambda *_args, **_kwargs: "ok",
        ),
    )
    validator = PostTrainingValidator(
        model_path="/models/validator-7b",
        quarantine_dir=tmp_path / "quarantine",
        validation_log_dir=tmp_path / "logs",
    )

    generate = await validator._load_model_with_adapter("/adapters/candidate")

    assert generate is not None
    assert validator._model_lane_lease is lease
    assert captured[0]["purpose"] == "benchmark"
    assert captured[0]["preemptible"] is False

    await validator._unload_model()

    assert lease.released is True
    assert validator._model_lane_lease is None


@pytest.mark.asyncio
async def test_adapter_promotion_and_restore_use_atomic_symlink_swap(
    tmp_path,
    monkeypatch,
):
    from core.adaptation import post_training_validator as module

    adapter_root = tmp_path / "adapters"
    old_adapter = adapter_root / "old"
    new_adapter = adapter_root / "new"
    old_adapter.mkdir(parents=True)
    new_adapter.mkdir()
    active_link = adapter_root / "active"
    active_link.symlink_to(old_adapter)
    monkeypatch.setattr(module, "ACTIVE_ADAPTER_LINK", active_link)
    validator = PostTrainingValidator(
        model_path="unused",
        adapter_base_dir=str(adapter_root),
        quarantine_dir=tmp_path / "quarantine",
        validation_log_dir=tmp_path / "logs",
    )

    assert await validator.promote_adapter(str(new_adapter)) is True
    assert active_link.resolve() == new_adapter.resolve()
    assert (adapter_root / "_previous_active").resolve() == old_adapter.resolve()

    await validator._restore_previous_adapter()
    assert active_link.resolve() == old_adapter.resolve()
    assert list(adapter_root.glob(".*.symlink.tmp")) == []


@pytest.mark.asyncio
async def test_adapter_promotion_refuses_non_symlink_active_path(
    tmp_path,
    monkeypatch,
):
    from core.adaptation import post_training_validator as module

    adapter_root = tmp_path / "adapters"
    new_adapter = adapter_root / "new"
    new_adapter.mkdir(parents=True)
    active_link = adapter_root / "active"
    active_link.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(module, "ACTIVE_ADAPTER_LINK", active_link)
    validator = PostTrainingValidator(
        model_path="unused",
        adapter_base_dir=str(adapter_root),
        quarantine_dir=tmp_path / "quarantine",
        validation_log_dir=tmp_path / "logs",
    )

    assert await validator.promote_adapter(str(new_adapter)) is False
    assert active_link.read_text(encoding="utf-8") == "do not overwrite"
