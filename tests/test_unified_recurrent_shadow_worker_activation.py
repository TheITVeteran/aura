from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.contract_authority import new_contract_key, sign_job
from core.brain.llm.unified_recurrent_shadow_contract import (
    shadow_load_receipt_errors,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
)


def test_worker_reports_sealed_inactive_shadow_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", raising=False)

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
