from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_unified_recurrent_shadow_live_canary as runner


class _Client:
    def __init__(self, *, supported: bool = True) -> None:
        self.closed = False
        self.supported = supported
        self._unified_recurrent_shadow_status = {
            "loaded": True,
            "serving_authority": False,
            "package_id": "cp268-package",
            "manifest_sha256": "a" * 64,
            "controller_sha256": "b" * 64,
        }

    async def warmup(self, **_kwargs) -> bool:
        return True

    async def unified_recurrent_shadow_package_canary_async(
        self,
        _package,
        **_kwargs,
    ):
        plan = {
            "package_id": "cp268-package",
            "controller_sha256": "b" * 64,
            "plan_sha256": "c" * 64,
        }
        verdict = {
            "plan_sha256": "c" * 64,
            "supported": self.supported,
            "serving_authority": False,
            "output_exposed": False,
        }
        return {
            "plan": plan,
            "verdict": verdict,
            "supported": self.supported,
            "reason": "supported" if self.supported else "refuted",
        }

    async def aclose(self) -> None:
        self.closed = True


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    supported: bool = True,
) -> tuple[Path, Path, _Client]:
    package = tmp_path / "package"
    model = tmp_path / "model"
    package.mkdir()
    model.mkdir()
    client = _Client(supported=supported)
    monkeypatch.setattr(
        runner,
        "inspect_shadow_package",
        lambda _path: {
            "manifest": {
                "package_id": "cp268-package",
                "manifest_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_mlx_client",
        lambda _path: client,
    )
    return package, model, client


@pytest.mark.asyncio
async def test_live_canary_persists_canonical_no_output_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, client = _fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    output = output_dir / "canary.json"

    result = await runner.run_live_canary(
        package,
        model_path=model,
        output=output,
        minimum_wrong_to_right=1,
        maximum_shadow_latency_ms=120_000,
        maximum_latency_ratio_numerator=8,
        maximum_latency_ratio_denominator=1,
    )

    assert result["supported"] is True
    assert result["serving_authority"] is False
    assert result["output_exposed"] is False
    assert client.closed is True
    assert output.stat().st_mode & 0o777 == 0o400
    payload = output.read_bytes()
    assert payload == json.dumps(
        json.loads(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


@pytest.mark.asyncio
async def test_negative_canary_is_preserved_without_serving_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, _client = _fixture(tmp_path, monkeypatch, supported=False)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(mode=0o700)

    result = await runner.run_live_canary(
        package,
        model_path=model,
        output=output_dir / "canary.json",
        minimum_wrong_to_right=1,
        maximum_shadow_latency_ms=120_000,
        maximum_latency_ratio_numerator=8,
        maximum_latency_ratio_denominator=1,
    )

    assert result["supported"] is False
    assert result["verdict"]["supported"] is False


@pytest.mark.asyncio
async def test_worker_identity_mismatch_fails_before_evidence_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, client = _fixture(tmp_path, monkeypatch)
    client._unified_recurrent_shadow_status["manifest_sha256"] = "f" * 64
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(mode=0o700)
    output = output_dir / "canary.json"

    with pytest.raises(
        runner.UnifiedRecurrentLiveCanaryError,
        match="identity or authority differs",
    ):
        await runner.run_live_canary(
            package,
            model_path=model,
            output=output,
            minimum_wrong_to_right=1,
            maximum_shadow_latency_ms=120_000,
            maximum_latency_ratio_numerator=8,
            maximum_latency_ratio_denominator=1,
        )

    assert client.closed is True
    assert not output.exists()


@pytest.mark.asyncio
async def test_invalid_threshold_fails_before_worker_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    model = tmp_path / "model"
    constructed = False

    def construct(_path):
        nonlocal constructed
        constructed = True
        return _Client()

    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_mlx_client",
        construct,
    )

    with pytest.raises(
        runner.UnifiedRecurrentLiveCanaryError,
        match="threshold is invalid",
    ):
        await runner.run_live_canary(
            package,
            model_path=model,
            output=tmp_path / "evidence.json",
            minimum_wrong_to_right=1,
            maximum_shadow_latency_ms=0,
            maximum_latency_ratio_numerator=8,
            maximum_latency_ratio_denominator=1,
        )

    assert constructed is False
