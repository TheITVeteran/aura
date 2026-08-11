from __future__ import annotations

import pytest

from tools import evaluate_unified_intrinsic_checkpoint as evaluator
from tools.evaluate_unified_intrinsic_checkpoint import (
    _evaluation_layout,
    _evaluation_preload_evidence,
    _sign_test_p_value,
)


def test_sign_test_is_exact_and_refuses_ties() -> None:
    assert _sign_test_p_value([0.0, 0.0]) is None
    assert _sign_test_p_value([1.0] * 8) == 0.0078125
    assert _sign_test_p_value([1.0] * 4 + [-1.0] * 4) == 1.0


def test_evaluation_layout_supports_legacy_colocation(tmp_path) -> None:
    root = tmp_path.resolve()
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == root
    assert layout.dataset_path == root / "dataset.json"
    assert layout.tokenized_dataset_path == root / "tokenized_dataset.json"


def test_evaluation_layout_uses_resident_frozen_paths(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    inputs = root / "inputs"
    output = root / "training-output"
    inputs.mkdir()
    output.mkdir()
    dataset = inputs / "dataset.json"
    tokenized = inputs / "tokenized_dataset.json"
    dataset.write_text("{}", encoding="ascii")
    tokenized.write_text("{}", encoding="ascii")
    (root / "campaign.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        evaluator,
        "_load_resident_campaign_config",
        lambda _path: {
            "paths": {
                "campaign_root": str(root),
                "training_output": str(output),
                "dataset": str(dataset),
                "tokenized_dataset": str(tokenized),
            }
        },
    )
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == output
    assert layout.dataset_path == dataset
    assert layout.tokenized_dataset_path == tokenized


def test_evaluation_without_external_guard_uses_live_pressure(monkeypatch) -> None:
    observed = {"available": True, "under_pressure": False, "source": "live"}
    monkeypatch.setattr(evaluator, "host_pressure", lambda: observed)
    monkeypatch.setattr(
        evaluator,
        "verify_release",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )

    pressure, release = _evaluation_preload_evidence(
        resource_enabled=False,
        preload_ready_path=None,
        preload_release_path=None,
        preload_key_path=None,
        preload_config_sha256=None,
    )

    assert pressure == observed
    assert release is None


def test_external_evaluation_guard_requires_complete_signed_preload(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires signed preload"):
        _evaluation_preload_evidence(
            resource_enabled=True,
            preload_ready_path=None,
            preload_release_path=None,
            preload_key_path=None,
            preload_config_sha256=None,
        )
    with pytest.raises(ValueError, match="supplied together"):
        _evaluation_preload_evidence(
            resource_enabled=True,
            preload_ready_path=tmp_path / "ready.json",
            preload_release_path=None,
            preload_key_path=None,
            preload_config_sha256=None,
        )


def test_signed_preload_is_the_detached_evaluation_pressure_authority(
    tmp_path,
    monkeypatch,
) -> None:
    release = {
        "host_pressure": {
            "available": True,
            "under_pressure": False,
            "source": "signed-external-sentinel",
        },
        "hmac_sha256": "a" * 64,
    }
    calls = []

    def fake_verify(path, **kwargs):
        calls.append((path, kwargs))
        return release

    monkeypatch.setattr(evaluator, "verify_release", fake_verify)
    monkeypatch.setattr(
        evaluator,
        "host_pressure",
        lambda: (_ for _ in ()).throw(AssertionError("live probe must not run")),
    )
    ready = tmp_path / "ready.json"
    signed = tmp_path / "release.json"
    key = tmp_path / "key"

    pressure, observed_release = _evaluation_preload_evidence(
        resource_enabled=True,
        preload_ready_path=ready,
        preload_release_path=signed,
        preload_key_path=key,
        preload_config_sha256="b" * 64,
    )

    assert pressure["source"] == "signed-external-sentinel"
    assert observed_release is release
    assert calls == [
        (
            signed,
            {
                "ready_path": ready,
                "key_path": key,
                "config_sha256": "b" * 64,
                "require_live_evidence": True,
            },
        )
    ]
