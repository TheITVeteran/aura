from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_unified_recurrent_shadow_lifecycle as lifecycle


def _canary_result(index: int) -> dict:
    return {
        "supported": True,
        "manifest_sha256": "a" * 64,
        "result_sha256": str(index) * 64,
        "plan": {"plan_sha256": "c" * 64},
        "worker_shadow_status": {"controller_sha256": "b" * 64},
    }


@pytest.mark.asyncio
async def test_lifecycle_proves_two_cold_loads_and_post_rollback_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = tmp_path / "activation" / "releases"
    package = releases / "package"
    package.mkdir(mode=0o700, parents=True)
    pointer_path = releases.parent / "active.json"
    model = tmp_path / "model"
    model.mkdir()
    calls: list[str] = []

    monkeypatch.setattr(
        lifecycle,
        "inspect_shadow_package",
        lambda _path: {
            "manifest": {
                "package_id": "package",
                "manifest_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "default_shadow_activation_paths",
        lambda: (pointer_path, releases),
    )

    def publish(_package, **_kwargs):
        pointer_path.write_text("active", encoding="ascii")
        return {"pointer_sha256": "d" * 64}

    def deactivate(**_kwargs):
        pointer_path.unlink()
        return {"pointer_sha256": "d" * 64}

    monkeypatch.setattr(lifecycle, "publish_shadow_pointer", publish)
    monkeypatch.setattr(
        lifecycle,
        "resolve_shadow_pointer",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(lifecycle, "deactivate_shadow_pointer", deactivate)

    async def canary(_package, *, output, discovery_mode, **_kwargs):
        calls.append(discovery_mode)
        result = _canary_result(len(calls))
        output.write_text("canary", encoding="ascii")
        return result

    monkeypatch.setattr(lifecycle, "run_live_canary", canary)
    monkeypatch.setattr(
        lifecycle,
        "_inactive_worker_receipt",
        lambda _model: _async_value({"receipt_sha256": "e" * 64}),
    )

    output = tmp_path / "evidence"
    result = await lifecycle.run_lifecycle(
        package,
        model_path=model,
        output_directory=output,
        minimum_wrong_to_right=1,
        maximum_shadow_latency_ms=120_000,
        maximum_latency_ratio_numerator=8,
        maximum_latency_ratio_denominator=1,
    )

    assert calls == ["durable_pointer", "durable_pointer"]
    assert result["supported"] is True
    assert result["serving_authority"] is False
    assert not pointer_path.exists()
    payload = (output / "lifecycle-result.json").read_bytes()
    assert payload == json.dumps(
        json.loads(payload), sort_keys=True, separators=(",", ":")
    ).encode("ascii")


@pytest.mark.asyncio
async def test_refuted_restart_triggers_exact_emergency_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = tmp_path / "activation" / "releases"
    package = releases / "package"
    package.mkdir(mode=0o700, parents=True)
    pointer_path = releases.parent / "active.json"
    model = tmp_path / "model"
    model.mkdir()
    retired: list[str] = []
    calls = 0
    monkeypatch.setattr(
        lifecycle,
        "inspect_shadow_package",
        lambda _path: {
            "manifest": {
                "package_id": "package",
                "manifest_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "default_shadow_activation_paths",
        lambda: (pointer_path, releases),
    )

    def publish(_package, **_kwargs):
        pointer_path.write_text("active", encoding="ascii")
        return {"pointer_sha256": "d" * 64}

    def deactivate(**kwargs):
        retired.append(kwargs["expected_current_sha256"])
        pointer_path.unlink()
        return {"pointer_sha256": "d" * 64}

    monkeypatch.setattr(lifecycle, "publish_shadow_pointer", publish)
    monkeypatch.setattr(
        lifecycle,
        "resolve_shadow_pointer",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(lifecycle, "deactivate_shadow_pointer", deactivate)

    async def refuted(_package, **_kwargs):
        nonlocal calls
        calls += 1
        return {**_canary_result(1), "supported": False}

    monkeypatch.setattr(lifecycle, "run_live_canary", refuted)

    with pytest.raises(
        lifecycle.UnifiedRecurrentShadowLifecycleError,
        match="first cold-load evidence is refuted",
    ):
        await lifecycle.run_lifecycle(
            package,
            model_path=model,
            output_directory=tmp_path / "evidence",
            minimum_wrong_to_right=1,
            maximum_shadow_latency_ms=120_000,
            maximum_latency_ratio_numerator=8,
            maximum_latency_ratio_denominator=1,
        )

    assert retired == ["d" * 64]
    assert calls == 1
    assert not pointer_path.exists()


@pytest.mark.asyncio
async def test_existing_activation_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer = tmp_path / "active.json"
    pointer.write_text("owned-by-another-run", encoding="ascii")
    monkeypatch.setattr(
        lifecycle,
        "inspect_shadow_package",
        lambda _path: {
            "manifest": {
                "package_id": "package",
                "manifest_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "default_shadow_activation_paths",
        lambda: (pointer, tmp_path / "releases"),
    )

    with pytest.raises(
        lifecycle.UnifiedRecurrentShadowLifecycleError,
        match="inactive initial pointer",
    ):
        await lifecycle.run_lifecycle(
            tmp_path / "package",
            model_path=tmp_path / "model",
            output_directory=tmp_path / "evidence",
            minimum_wrong_to_right=1,
            maximum_shadow_latency_ms=120_000,
            maximum_latency_ratio_numerator=8,
            maximum_latency_ratio_denominator=1,
        )


async def _async_value(value):
    return value
