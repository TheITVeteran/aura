from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.contract_authority import new_contract_key, sign_job
from core.brain.llm.unified_recurrent_qualified_decode import (
    seal_qualified_decode_request,
)
from core.brain.llm.unified_recurrent_shadow_battery import seal_shadow_canary_battery
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
        seal_serving_qualified_activation,
        seal_verified_qualified_activation,
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
    candidate = seal_qualified_activation(manifest, lifecycle, pointer)
    battery = seal_shadow_canary_battery(
        [
            {
                "task_id": "fresh-khop-1",
                "family": "khop",
                "task_depth": 1,
                "prompt_sha256": "1" * 64,
                "expected_sha256": "2" * 64,
                "public_token_ids": [10],
                "expected_token_ids": [20],
                "max_tokens": 1,
            }
        ],
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )
    case = battery["cases"][0]
    expected = _sha(case["expected_token_ids"])
    evidence = [
        {
            "index": 0,
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 1,
            "request_sha256": case["request_sha256"],
            "expected_token_ids_sha256": expected,
            "generated_token_ids_sha256": expected,
            "qualified_result_sha256": "9" * 64,
            "latency_ms": 7,
            "exact": True,
        }
    ]
    canary_body = {
        "schema": "aura.unified_intrinsic.qualified_serving_canary.v3",
        "package_id": candidate["package_id"],
        "manifest_sha256": candidate["manifest_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "controller_sha256": candidate["controller_sha256"],
        "activation_sha256": candidate["activation_sha256"],
        "battery_sha256": battery["battery_sha256"],
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "case_count": 1,
        "exact_count": 1,
        "total_latency_ms": 7,
        "maximum_latency_ms": 7,
        "evidence": evidence,
        "supported": True,
        "serving_authority": False,
        "authority_remains_active": False,
        "canary_authority_was_request_scoped": True,
        "output_exposed": False,
    }
    pending = seal_verified_qualified_activation(
        candidate,
        {**canary_body, "result_sha256": _sha(canary_body)},
        expected_battery=battery,
    )
    pending_canary_body = {
        **canary_body,
        "activation_sha256": pending["activation_sha256"],
    }
    return seal_serving_qualified_activation(
        pending,
        {**pending_canary_body, "result_sha256": _sha(pending_canary_body)},
        expected_battery=battery,
    )


def _qualified_candidate() -> dict:
    durable = _qualified_evidence()
    body = {
        key: value
        for key, value in durable.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "candidate_canary_sha256": "",
            "qualified_canary_sha256": "",
            "mode": "qualified_canary_only",
            "serving_authority": False,
        }
    )
    return {**body, "activation_sha256": _sha(body)}


def _qualified_pending() -> dict:
    candidate = _qualified_candidate()
    body = {
        key: value
        for key, value in candidate.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "mode": "qualified_typed_pending",
            "candidate_canary_sha256": "d" * 64,
        }
    )
    return {**body, "activation_sha256": _sha(body)}


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


def test_worker_keeps_qualified_authority_inactive_when_shadow_is_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "qualified-active.json"
    path.write_text("present", encoding="ascii")
    activation = _qualified_evidence()
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
        None,
        {
            "configured": True,
            "loaded": False,
            "reason": "incompatible:resident binding differs",
        },
    )

    assert observed is None
    assert receipt["configured"] is True
    assert receipt["loaded"] is False
    assert receipt["serving_authority"] is False
    assert receipt["reason"] == "qualified_activation_shadow_unavailable"
    assert receipt["activation"]["activation_sha256"] == activation["activation_sha256"]


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


def test_worker_declines_a_configured_invalid_package_without_dying(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SHADOW fails closed; the worker does not.

    This test asserted that an incompatible package raises. Measured live on
    2026-08-13, that is what happened: every boot logged
    `configured_unified_recurrent_shadow_invalid` and the worker went into a
    respawn loop. The integrity verdict was right — a package built against a
    different resident binding must not be used — but the shadow runs in
    shadow_only mode with serving_authority False, so it is by construction
    not load-bearing. An optional component that cannot serve must not be able
    to stop the thing it decorates.

    Failing closed now means: refused, not loaded, and said so on the receipt.
    """
    package = tmp_path / "bad-package"
    package.mkdir()
    monkeypatch.setenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", str(package))

    loaded, receipt = mlx_worker._load_unified_recurrent_shadow(
        object(),
        object(),
        model_path="/model",
    )

    assert loaded is None, "an incompatible shadow was attached anyway"
    assert receipt["configured"] is True
    assert receipt["loaded"] is False
    assert receipt["reason"].startswith("incompatible:")
    assert receipt["serving_authority"] is False
    assert receipt["model_identity_strength"] == "none"


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
        reclaim=lambda: True,
    )

    assert response == {
        "id": "probe-1",
        "action": "unified_recurrent_shadow_probe",
        "status": "ok",
        "receipt": receipt,
        "allocator_reclaimed": True,
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
            reclaim=lambda: True,
        )


def test_worker_shadow_probe_refuses_unreclaimed_allocator_state() -> None:
    key = new_contract_key()
    loaded = SimpleNamespace(
        probe=lambda *_args, **_kwargs: {"status": "completed"}
    )
    job = sign_job(
        {
            "id": "probe-reclaim",
            "action": "unified_recurrent_shadow_probe",
            "unified_recurrent_shadow_contract": seal_shadow_probe_request(
                [1], [2], max_tokens=1
            ),
        },
        key,
        principal="mlx_client.unified_recurrent_shadow_probe",
    )

    with pytest.raises(RuntimeError, match="memory_not_reclaimed"):
        mlx_worker._handle_unified_recurrent_shadow_probe(
            job,
            loaded_shadow=loaded,
            model=object(),
            contract_key=key,
            reclaim=lambda: False,
        )


def test_worker_shadow_probe_reclaim_is_a_synchronous_allocator_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_mx = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        clear_cache=lambda: calls.append("clear_cache"),
    )
    monkeypatch.setattr(mlx_worker.gc, "collect", lambda: calls.append("gc"))

    assert mlx_worker._reclaim_unified_recurrent_probe_memory(fake_mx) is True
    assert calls == ["gc", "synchronize", "clear_cache", "synchronize"]


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
        lambda result, observed, **_kwargs: authorized
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
        reclaim=lambda: True,
    )

    assert response["receipt"] == authorized
    assert response["allocator_reclaimed"] is True
    job["unified_recurrent_qualified_decode_contract"]["task_depth"] = 2
    with pytest.raises(ValueError, match="invalid_contract_authority"):
        mlx_worker._handle_unified_recurrent_qualified_decode(
            job,
            loaded_shadow=loaded,
            qualified_activation=activation,
            model=object(),
            contract_key=key,
            reclaim=lambda: True,
        )


@pytest.mark.parametrize(
    "activation_factory",
    [_qualified_candidate, _qualified_pending],
)
def test_worker_canary_accepts_only_signed_request_scoped_authority(
    monkeypatch: pytest.MonkeyPatch,
    activation_factory,
) -> None:
    key = new_contract_key()
    activation = activation_factory()
    request = seal_qualified_decode_request(
        [1],
        package_id="package",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=1,
        max_tokens=4,
    )
    from core.brain.llm.unified_recurrent_qualified_decode import (
        seal_qualified_canary_request_authority,
    )
    from core.brain.llm.unified_recurrent_shadow_battery import (
        seal_shadow_canary_battery,
    )

    case = {
        "task_id": "canary-khop-1",
        "family": "khop",
        "task_depth": 1,
        "prompt_sha256": "1" * 64,
        "expected_sha256": "2" * 64,
        "public_token_ids": [1],
        "expected_token_ids": [1],
        "max_tokens": 4,
    }
    battery = seal_shadow_canary_battery(
        [case],
        seed=7,
        replication_plan_sha256="3" * 64,
        replication_verdict_sha256="4" * 64,
        excluded_task_ids_sha256="5" * 64,
        excluded_prompt_sha256s_sha256="6" * 64,
        generator_source_sha256s={"generator.py": "7" * 64},
    )
    loaded = SimpleNamespace(
        receipt=_loaded_shadow_receipt(),
        canary_battery=battery,
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
        lambda result, observed, **_kwargs: authorized
        if result is inactive and observed == activation
        else pytest.fail("worker authorized the wrong canary activation"),
    )

    def signed_job() -> dict:
        now = time.time()
        authority = seal_qualified_canary_request_authority(
            activation_sha256=activation["activation_sha256"],
            battery_sha256=battery["battery_sha256"],
            case_index=0,
            request_sha256=request["request_sha256"],
            nonce="8" * 64,
            issued_at_unix=now,
            expires_at_unix=now + 300.0,
        )
        return sign_job(
            {
                "id": "canary-1",
                "action": "unified_recurrent_qualified_decode",
                "unified_recurrent_qualified_decode_contract": request,
                "unified_recurrent_qualified_canary_activation": activation,
                "unified_recurrent_qualified_canary_authority": authority,
            },
            key,
            principal="mlx_client.unified_recurrent_qualified_decode",
        )

    response = mlx_worker._handle_unified_recurrent_qualified_decode(
        signed_job(),
        loaded_shadow=loaded,
        qualified_activation=None,
        model=object(),
        contract_key=key,
        consumed_canary_nonces=set(),
        reclaim=lambda: True,
    )
    assert response["receipt"] == authorized

    with pytest.raises(
        RuntimeError,
        match="qualified_canary_requires_inactive_durable_authority",
    ):
        mlx_worker._handle_unified_recurrent_qualified_decode(
            signed_job(),
            loaded_shadow=loaded,
            qualified_activation=activation,
            model=object(),
            contract_key=key,
            consumed_canary_nonces=set(),
            reclaim=lambda: True,
        )

    tampered = signed_job()
    tampered["unified_recurrent_qualified_canary_activation"]["task_depths"] = [9]
    with pytest.raises(ValueError, match="invalid_contract_authority"):
        mlx_worker._handle_unified_recurrent_qualified_decode(
            tampered,
            loaded_shadow=loaded,
            qualified_activation=None,
            model=object(),
            contract_key=key,
            consumed_canary_nonces=set(),
            reclaim=lambda: True,
        )


def test_worker_qualified_decode_refuses_unreclaimed_allocator_state(
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
        max_tokens=1,
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_decode.run_qualified_decode",
        lambda *_args, **_kwargs: {"serving_authority": False},
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_decode."
        "authorize_qualified_decode_result",
        lambda *_args, **_kwargs: {"serving_authority": True},
    )
    job = sign_job(
        {
            "id": "decode-reclaim",
            "action": "unified_recurrent_qualified_decode",
            "unified_recurrent_qualified_decode_contract": request,
        },
        key,
        principal="mlx_client.unified_recurrent_qualified_decode",
    )

    with pytest.raises(RuntimeError, match="memory_not_reclaimed"):
        mlx_worker._handle_unified_recurrent_qualified_decode(
            job,
            loaded_shadow=loaded,
            qualified_activation=activation,
            model=object(),
            contract_key=key,
            reclaim=lambda: False,
        )
