from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.contract_authority import new_contract_key, sign_job
from core.brain.llm.unified_recurrent_qualified_decode import (
    seal_qualified_decode_request,
)
from core.brain.llm.unified_recurrent_shadow_contract import (
    LOAD_SCHEMA,
    seal_shadow_load_receipt,
    shadow_load_receipt_errors,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _qualified_evidence():
    from core.brain.llm.unified_recurrent_qualified_activation import (
        seal_qualified_activation,
    )

    pointer_body = {
        "schema": "aura.unified_intrinsic.shadow_pointer.v1",
        "package_path": "/owned/releases/package",
        "package_id": "package",
    }
    manifest_body = {
        "package_id": "package",
        "checkpoint_sha256": "b" * 64,
        "domain_contract": {
            "qualification": "generator_and_grammar_bound",
            "families": ["khop"],
            "task_depths": [1],
            "recurrence_depth": 4,
        },
    }
    manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    pointer_body["manifest_sha256"] = manifest["manifest_sha256"]
    pointer = {**pointer_body, "pointer_sha256": _sha(pointer_body)}
    lifecycle_body = {
        "schema": "aura.unified_intrinsic.shadow_lifecycle_run.v1",
        "package_id": "package",
        "manifest_sha256": manifest["manifest_sha256"],
        "activation_pointer": pointer,
        "canary_plan_sha256": "d" * 64,
        "controller_sha256": "c" * 64,
        "checks": {
            "durable_pointer_reopened": True,
            "first_cold_load_supported": True,
            "restart_cold_load_supported": True,
            "restart_identity_stable": True,
            "pointer_rollback_completed": True,
            "post_rollback_worker_inactive": True,
        },
        "supported": True,
        "serving_authority": False,
        "output_exposed": False,
    }
    lifecycle = {**lifecycle_body, "result_sha256": _sha(lifecycle_body)}
    return seal_qualified_activation(manifest, lifecycle, pointer)


def _loaded_shadow_receipt() -> dict:
    return seal_shadow_load_receipt(
        {
            "schema": LOAD_SCHEMA,
            "configured": True,
            "loaded": True,
            "reason": "certified_shadow_package_loaded",
            "package_id": "package",
            "manifest_sha256": _qualified_evidence()["manifest_sha256"],
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


def test_worker_reports_explicit_inactive_qualified_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_activation_store."
        "default_qualified_activation_path",
        lambda: tmp_path / "qualified-active.json",
    )

    activation, receipt = mlx_worker._load_unified_recurrent_qualified_activation(
        None,
        {"loaded": False},
    )

    assert activation is None
    assert receipt["configured"] is False
    assert receipt["loaded"] is False
    assert receipt["serving_authority"] is False


def test_worker_loads_only_activation_matching_the_loaded_shadow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "qualified-active.json"
    path.write_text("present", encoding="ascii")
    activation = _qualified_evidence()
    shadow_receipt = _loaded_shadow_receipt()
    loaded_shadow = SimpleNamespace(receipt=shadow_receipt)
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_activation_store."
        "default_qualified_activation_path",
        lambda: path,
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_activation_store."
        "read_qualified_activation",
        lambda _path: activation,
    )

    observed, receipt = mlx_worker._load_unified_recurrent_qualified_activation(
        loaded_shadow,
        shadow_receipt,
    )

    assert observed == activation
    assert receipt["loaded"] is True
    assert receipt["serving_authority"] is True

    mismatched = dict(shadow_receipt)
    mismatched["controller_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="shadow_identity_differs"):
        mlx_worker._load_unified_recurrent_qualified_activation(
            loaded_shadow,
            mismatched,
        )


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


def test_worker_qualified_decode_requires_signed_matching_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = new_contract_key()
    activation = _qualified_evidence()
    loaded = SimpleNamespace(receipt=_loaded_shadow_receipt())
    request = seal_qualified_decode_request(
        [1],
        package_id="package",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=1,
        max_tokens=4,
    )
    inactive = {"serving_authority": False}
    authorized = {
        "serving_authority": True,
        "qualified_activation_sha256": activation["activation_sha256"],
    }
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_decode.run_qualified_decode",
        lambda *_args, **_kwargs: inactive,
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_decode."
        "authorize_qualified_decode_result",
        lambda result, observed: authorized
        if result is inactive and observed == activation
        else pytest.fail("worker authorized the wrong result or activation"),
    )
    job = sign_job(
        {
            "id": "decode-1",
            "action": "unified_recurrent_qualified_decode",
            "unified_recurrent_qualified_decode_contract": request,
        },
        key,
        principal="mlx_client.unified_recurrent_qualified_decode",
    )

    response = mlx_worker._handle_unified_recurrent_qualified_decode(
        job,
        loaded_shadow=loaded,
        qualified_activation=activation,
        model=object(),
        contract_key=key,
    )

    assert response["receipt"] == authorized
    job["unified_recurrent_qualified_decode_contract"]["task_depth"] = 2
    with pytest.raises(ValueError, match="invalid_contract_authority"):
        mlx_worker._handle_unified_recurrent_qualified_decode(
            job,
            loaded_shadow=loaded,
            qualified_activation=activation,
            model=object(),
            contract_key=key,
        )
