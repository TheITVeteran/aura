from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.latent_cortex import live_adapter_activation


def test_worker_reports_honest_inactive_state_without_certified_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_worker, "state_root", lambda: tmp_path)
    monkeypatch.delenv("AURA_RLC_ACTIVATION_POINTER", raising=False)
    monkeypatch.delenv("AURA_RLC_ACTIVATION_TRUST_ROOT", raising=False)

    status, receipt = mlx_worker._attach_certified_recurrent_adapter(
        object(),
        model_path="/model",
        personality_adapter_path=None,
    )

    assert status["configured"] is False
    assert status["active"] is False
    assert status["reason"] == "no_certified_activation"
    assert receipt is None


def test_worker_fails_closed_when_explicit_pointer_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_worker, "state_root", lambda: tmp_path)
    missing = tmp_path / "missing-pointer.json"
    monkeypatch.setenv("AURA_RLC_ACTIVATION_POINTER", str(missing))

    with pytest.raises(
        RuntimeError,
        match="configured_recurrent_adapter_pointer_missing",
    ):
        mlx_worker._attach_certified_recurrent_adapter(
            object(),
            model_path="/model",
            personality_adapter_path=None,
        )


def test_worker_binds_certified_activation_to_public_runtime_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_root = tmp_path / "data/adapters/latent-cortex"
    pointer = activation_root / "active.json"
    trust_root = activation_root / "trust-root.pem"
    releases = activation_root / "releases"
    releases.mkdir(parents=True)
    pointer.write_bytes(b"pointer")
    trust_root.write_bytes(b"trust")
    pointer.chmod(0o600)
    trust_root.chmod(0o600)
    model = object()
    observed: dict[str, Any] = {}
    receipt = {
        "receipt_sha256": "a" * 64,
        "activation_sha256": "b" * 64,
        "adapter_identity": {"composite_identity_sha256": "c" * 64},
        "campaign_name": "resident-32b-role-v6",
        "claim_tier": "PROVEN",
        "verified_verdict": "gain_proven",
        "loaded_projection_count": 24,
    }

    monkeypatch.setattr(mlx_worker, "state_root", lambda: tmp_path)
    monkeypatch.delenv("AURA_RLC_ACTIVATION_POINTER", raising=False)
    monkeypatch.delenv("AURA_RLC_ACTIVATION_TRUST_ROOT", raising=False)
    monkeypatch.setattr(
        live_adapter_activation,
        "read_live_adapter_trust_root",
        lambda path: b"trust" if Path(path) == trust_root else b"",
    )

    def _attach(loaded_model: Any, **kwargs: Any) -> dict[str, Any]:
        assert loaded_model is model
        observed.update(kwargs)
        return dict(receipt)

    monkeypatch.setattr(
        live_adapter_activation,
        "attach_certified_live_adapter",
        _attach,
    )

    status, actual_receipt = mlx_worker._attach_certified_recurrent_adapter(
        model,
        model_path="/model",
        personality_adapter_path=None,
    )

    assert observed["pointer_path"] == pointer
    assert observed["approved_adapter_roots"] == (releases,)
    assert status == {
        "schema": "aura.latent_cortex.worker_recurrent_adapter_activation.v1",
        "configured": True,
        "active": True,
        "reason": "certified_gain_proven",
        "receipt_sha256": "a" * 64,
        "activation_sha256": "b" * 64,
        "adapter_composite_identity_sha256": "c" * 64,
        "campaign_name": "resident-32b-role-v6",
        "claim_tier": "PROVEN",
        "verified_verdict": "gain_proven",
        "loaded_projection_count": 24,
    }
    assert actual_receipt == receipt
