from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.contract_authority import new_contract_key, sign_job
from core.brain.llm.unified_recurrent_shadow_contract import (
    LOAD_SCHEMA,
    seal_shadow_load_receipt,
    shadow_load_receipt_errors,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
)


def test_worker_reports_sealed_inactive_shadow_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", raising=False)
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow_pointer.default_shadow_activation_paths",
        lambda: (tmp_path / "active.json", tmp_path / "releases"),
    )

    loaded, receipt = mlx_worker._load_unified_recurrent_shadow(
        object(),
        object(),
        model_path="/model",
    )

    assert loaded is None
    assert receipt["configured"] is False
    assert receipt["loaded"] is False
    assert receipt["serving_authority"] is False
    assert shadow_load_receipt_errors(receipt) == []


def test_worker_loads_restart_stable_shadow_pointer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "releases" / "package"
    package.mkdir(parents=True)
    pointer = tmp_path / "active.json"
    monkeypatch.delenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", raising=False)
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow_pointer.default_shadow_activation_paths",
        lambda: (pointer, tmp_path / "releases"),
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow_pointer.resolve_shadow_pointer",
        lambda *_args, **_kwargs: package,
    )
    pointer.write_text("present", encoding="ascii")
    receipt = seal_shadow_load_receipt(
        {
            "schema": LOAD_SCHEMA,
            "configured": True,
            "loaded": True,
            "reason": "certified_shadow_package_loaded",
            "package_id": "package",
            "manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "controller_sha256": "c" * 64,
            "families": ["khop"],
            "task_depths": [1],
            "recurrence_depth": 4,
            "model_identity_strength": "config_behavior_hash_and_weight_extent",
            "mode": "shadow_only",
            "serving_authority": False,
        }
    )
    loaded_shadow = SimpleNamespace(receipt=receipt)
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow.load_unified_recurrent_shadow",
        lambda *_args, **_kwargs: loaded_shadow,
    )

    loaded, observed = mlx_worker._load_unified_recurrent_shadow(
        object(),
        object(),
        model_path="/model",
    )

    assert loaded is loaded_shadow
    assert observed == receipt


def test_worker_fails_closed_on_invalid_restart_pointer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer = tmp_path / "active.json"
    pointer.write_text("invalid", encoding="ascii")
    monkeypatch.delenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", raising=False)
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow_pointer.default_shadow_activation_paths",
        lambda: (pointer, tmp_path / "releases"),
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow_pointer.resolve_shadow_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("tampered")),
    )

    with pytest.raises(
        RuntimeError,
        match="configured_unified_recurrent_shadow_pointer_invalid",
    ):
        mlx_worker._load_unified_recurrent_shadow(
            object(),
            object(),
            model_path="/model",
        )


def test_worker_fails_closed_on_a_configured_invalid_package(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "bad-package"
    package.mkdir()
    monkeypatch.setenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", str(package))

    with pytest.raises(
        RuntimeError,
        match="configured_unified_recurrent_shadow_invalid",
    ):
        mlx_worker._load_unified_recurrent_shadow(
            object(),
            object(),
            model_path="/model",
        )


def test_worker_revalidates_loaded_runtime_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", str(package))
    forged = SimpleNamespace(
        receipt={
            "schema": "forged",
            "configured": True,
            "loaded": True,
            "reason": "loaded",
        }
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow.load_unified_recurrent_shadow",
        lambda *_args, **_kwargs: forged,
    )

    with pytest.raises(
        RuntimeError,
        match="configured_unified_recurrent_shadow_receipt_invalid",
    ):
        mlx_worker._load_unified_recurrent_shadow(
            object(),
            object(),
            model_path="/model",
        )


def test_worker_shadow_probe_requires_bound_authority_and_exposes_no_output() -> None:
    key = new_contract_key()
    contract = seal_shadow_probe_request([1, 2], [3], max_tokens=1)
    receipt = {
        "schema": "fixture",
        "status": "completed",
        "output_exposed": False,
        "serving_authority": False,
    }
    loaded = SimpleNamespace(probe=lambda *_args, **_kwargs: receipt)
    job = sign_job(
        {
            "id": "probe-1",
            "action": "unified_recurrent_shadow_probe",
            "unified_recurrent_shadow_contract": contract,
        },
        key,
        principal="mlx_client.unified_recurrent_shadow_probe",
    )

    response = mlx_worker._handle_unified_recurrent_shadow_probe(
        job,
        loaded_shadow=loaded,
        model=object(),
        contract_key=key,
    )

    assert response == {
        "id": "probe-1",
        "action": "unified_recurrent_shadow_probe",
        "status": "ok",
        "receipt": receipt,
    }
    assert "text" not in response
    assert "tokens" not in response

    job["unified_recurrent_shadow_contract"]["public_token_ids"][0] = 99
    with pytest.raises(ValueError, match="invalid_contract_authority"):
        mlx_worker._handle_unified_recurrent_shadow_probe(
            job,
            loaded_shadow=loaded,
            model=object(),
            contract_key=key,
        )
