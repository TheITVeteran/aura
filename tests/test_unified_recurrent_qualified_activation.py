from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.unified_recurrent_qualified_activation import (
    QUALIFIED_CANARY_SCHEMA,
    activation_matches_shadow_receipt,
    qualified_activation_errors,
    qualified_activation_load_receipt_errors,
    qualified_serving_canary_errors,
    seal_qualified_activation,
    seal_qualified_activation_load_receipt,
    seal_serving_qualified_activation,
    seal_verified_qualified_activation,
)
from core.brain.llm.unified_recurrent_qualified_activation_store import (
    UnifiedRecurrentQualifiedActivationStoreError,
    deactivate_qualified_activation,
    publish_qualified_activation,
    read_qualified_activation,
)
from core.brain.llm.unified_recurrent_shadow_battery import (
    seal_shadow_canary_battery,
)
from core.runtime.atomic_writer import atomic_write_bytes, ensure_private_directory


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _evidence():
    pointer_body = {
        "schema": "aura.unified_intrinsic.shadow_pointer.v1",
        "package_path": "/owned/releases/package",
        "package_id": "package",
    }
    manifest_body = {
        "package_id": "package",
        "checkpoint_sha256": "c" * 64,
        "domain_contract": {
            "qualification": "generator_and_grammar_bound",
            "families": ["khop", "modular", "register_trace"],
            "task_depths": [1, 2, 4],
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
        "controller_sha256": "e" * 64,
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
    return manifest, lifecycle, pointer


def _durable_activation() -> dict:
    manifest, lifecycle, pointer = _evidence()
    candidate = seal_qualified_activation(manifest, lifecycle, pointer)
    canary = _canary(candidate)
    pending = seal_verified_qualified_activation(candidate, canary)
    return seal_serving_qualified_activation(pending, _canary(pending))


def _canary(activation: dict, *, serving: bool = False) -> dict:
    evidence = [
        {
            "index": 0,
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 2,
            "request_sha256": "6" * 64,
            "expected_token_ids_sha256": "7" * 64,
            "generated_token_ids_sha256": "7" * 64,
            "qualified_result_sha256": "9" * 64,
            "latency_ms": 7,
            "exact": True,
        }
    ]
    canary_body = {
        "schema": QUALIFIED_CANARY_SCHEMA,
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "checkpoint_sha256": activation["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "battery_sha256": "8" * 64,
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "case_count": 1,
        "exact_count": 1,
        "total_latency_ms": 7,
        "maximum_latency_ms": 7,
        "evidence": evidence,
        "supported": True,
        "serving_authority": serving,
        "authority_remains_active": serving,
        "canary_authority_was_request_scoped": not serving,
        "output_exposed": False,
    }
    return {**canary_body, "result_sha256": _sha(canary_body)}


def _battery_canary(activation: dict) -> tuple[dict, dict]:
    cases = [
        {
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 1,
            "prompt_sha256": "1" * 64,
            "expected_sha256": "2" * 64,
            "public_token_ids": [10],
            "expected_token_ids": [20],
            "max_tokens": 1,
        },
        {
            "task_id": "fresh-modular-1",
            "family": "modular",
            "task_depth": 2,
            "prompt_sha256": "3" * 64,
            "expected_sha256": "4" * 64,
            "public_token_ids": [11],
            "expected_token_ids": [21],
            "max_tokens": 1,
        },
    ]
    battery = seal_shadow_canary_battery(
        cases,
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )
    evidence = []
    for index, case in enumerate(battery["cases"]):
        expected = _sha(case["expected_token_ids"])
        evidence.append(
            {
                "index": index,
                "task_id": case["task_id"],
                "family": case["family"],
                "task_depth": case["task_depth"],
                "request_sha256": case["request_sha256"],
                "expected_token_ids_sha256": expected,
                "generated_token_ids_sha256": expected,
                "qualified_result_sha256": str(index + 1) * 64,
                "latency_ms": index + 1,
                "exact": True,
            }
        )
    body = {
        "schema": QUALIFIED_CANARY_SCHEMA,
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "checkpoint_sha256": activation["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "battery_sha256": battery["battery_sha256"],
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "case_count": 2,
        "exact_count": 2,
        "total_latency_ms": 3,
        "maximum_latency_ms": 2,
        "evidence": evidence,
        "supported": True,
        "serving_authority": False,
        "authority_remains_active": False,
        "canary_authority_was_request_scoped": True,
        "output_exposed": False,
    }
    return battery, {**body, "result_sha256": _sha(body)}


def _reseal_canary(canary: dict) -> dict:
    body = {key: value for key, value in canary.items() if key != "result_sha256"}
    return {**body, "result_sha256": _sha(body)}


def test_supported_lifecycle_issues_only_nonserving_canary_candidate() -> None:
    manifest, lifecycle, pointer = _evidence()

    activation = seal_qualified_activation(manifest, lifecycle, pointer)

    assert qualified_activation_errors(activation) == []
    assert activation["serving_authority"] is False
    assert activation["mode"] == "qualified_canary_only"
    assert activation["qualified_canary_sha256"] == ""
    assert activation["ordinary_chat_authorized"] is False
    assert activation["arbitrary_reasoning_authorized"] is False


def test_durable_authority_requires_the_complete_exact_canary_artifact() -> None:
    manifest, lifecycle, pointer = _evidence()
    candidate = seal_qualified_activation(manifest, lifecycle, pointer)
    canary = _canary(candidate)

    assert qualified_serving_canary_errors(
        canary,
        expected_activation=candidate,
    ) == []
    pending = seal_verified_qualified_activation(candidate, canary)
    assert pending["candidate_canary_sha256"] == canary["result_sha256"]
    assert pending["qualified_canary_sha256"] == ""
    pending_canary = _canary(pending)
    serving = seal_serving_qualified_activation(pending, pending_canary)
    assert serving["qualified_canary_sha256"] == pending_canary["result_sha256"]

    incomplete = {
        "schema": QUALIFIED_CANARY_SCHEMA,
        "supported": True,
        "serving_authority": False,
        "authority_remains_active": False,
        "canary_authority_was_request_scoped": True,
    }
    incomplete["result_sha256"] = _sha(incomplete)
    with pytest.raises(ValueError, match="not_admissible"):
        seal_verified_qualified_activation(candidate, incomplete)


@pytest.mark.parametrize("mutation", ["foreign", "omitted", "reordered", "wrong_answer"])
def test_canary_must_match_exact_ordered_sealed_battery(mutation: str) -> None:
    manifest, lifecycle, pointer = _evidence()
    candidate = seal_qualified_activation(manifest, lifecycle, pointer)
    battery, canary = _battery_canary(candidate)

    assert qualified_serving_canary_errors(
        canary,
        expected_activation=candidate,
        expected_battery=battery,
    ) == []
    attacked = copy.deepcopy(canary)
    if mutation == "foreign":
        attacked["battery_sha256"] = "f" * 64
    elif mutation == "omitted":
        attacked["evidence"] = attacked["evidence"][:1]
        attacked["case_count"] = 1
        attacked["exact_count"] = 1
        attacked["total_latency_ms"] = 1
        attacked["maximum_latency_ms"] = 1
    elif mutation == "reordered":
        attacked["evidence"].reverse()
        for index, row in enumerate(attacked["evidence"]):
            row["index"] = index
    else:
        attacked["evidence"][0]["generated_token_ids_sha256"] = "f" * 64
    attacked = _reseal_canary(attacked)

    assert qualified_serving_canary_errors(
        attacked,
        expected_activation=candidate,
        expected_battery=battery,
    )


@pytest.mark.parametrize(
    ("families", "depths"),
    [
        (["unknown"], [1]),
        (["khop", "khop"], [1]),
        (["modular", "khop"], [1]),
        (["khop"], [2, 1]),
        (["khop"], [1, 1]),
        (None, None),
    ],
)
def test_activation_rejects_noncanonical_or_unknown_domain(
    families,
    depths,
) -> None:
    manifest, lifecycle, pointer = _evidence()
    manifest["domain_contract"]["families"] = families
    manifest["domain_contract"]["task_depths"] = depths
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = _sha(body)
    pointer["manifest_sha256"] = manifest["manifest_sha256"]
    pointer_body = {
        key: value for key, value in pointer.items() if key != "pointer_sha256"
    }
    pointer["pointer_sha256"] = _sha(pointer_body)
    lifecycle["manifest_sha256"] = manifest["manifest_sha256"]
    lifecycle["activation_pointer"] = pointer
    lifecycle_body = {
        key: value for key, value in lifecycle.items() if key != "result_sha256"
    }
    lifecycle["result_sha256"] = _sha(lifecycle_body)

    with pytest.raises(ValueError):
        seal_qualified_activation(manifest, lifecycle, pointer)


def test_refuted_or_identity_mismatched_lifecycle_cannot_issue_authority() -> None:
    manifest, lifecycle, pointer = _evidence()
    lifecycle["checks"]["restart_identity_stable"] = False
    lifecycle_body = {key: value for key, value in lifecycle.items() if key != "result_sha256"}
    lifecycle["result_sha256"] = _sha(lifecycle_body)

    with pytest.raises(ValueError, match="not_admissible"):
        seal_qualified_activation(manifest, lifecycle, pointer)

    lifecycle["checks"]["restart_identity_stable"] = True
    lifecycle_body = {key: value for key, value in lifecycle.items() if key != "result_sha256"}
    lifecycle["result_sha256"] = _sha(lifecycle_body)
    replacement = copy.deepcopy(pointer)
    replacement["pointer_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="not_admissible"):
        seal_qualified_activation(manifest, lifecycle, replacement)


def test_activation_must_match_the_loaded_shadow_exactly() -> None:
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)
    receipt = {
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "checkpoint_sha256": activation["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "families": activation["families"],
        "task_depths": activation["task_depths"],
        "recurrence_depth": activation["recurrence_depth"],
    }

    assert activation_matches_shadow_receipt(activation, receipt) is True
    receipt["controller_sha256"] = "0" * 64
    assert activation_matches_shadow_receipt(activation, receipt) is False


def test_load_receipt_is_explicitly_active_or_inactive() -> None:
    activation = _durable_activation()

    active = seal_qualified_activation_load_receipt(
        configured=True,
        loaded=True,
        reason="qualified_activation_loaded",
        activation=activation,
    )
    inactive = seal_qualified_activation_load_receipt(
        configured=False,
        loaded=False,
        reason="not_configured",
        activation=None,
    )

    assert qualified_activation_load_receipt_errors(active) == []
    assert active["serving_authority"] is True
    assert qualified_activation_load_receipt_errors(inactive) == []
    assert inactive["serving_authority"] is False


def test_inactive_load_receipt_cannot_retain_activation_authority() -> None:
    activation = _durable_activation()

    receipt = seal_qualified_activation_load_receipt(
        configured=True,
        loaded=True,
        reason="qualified_activation_loaded",
        activation=activation,
    )
    receipt["loaded"] = False
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha(body)

    assert "pending_qualified_activation_state_invalid" in (
        qualified_activation_load_receipt_errors(receipt)
    )


def test_activation_publication_is_pointer_bound_cas_and_revocable(tmp_path) -> None:
    _manifest, _lifecycle, pointer = _evidence()
    activation = _durable_activation()
    root = ensure_private_directory(tmp_path / "authority")
    pointer_path = root / "active.json"
    activation_path = root / "qualified-active.json"
    atomic_write_bytes(
        pointer_path,
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("ascii"),
        mode=0o600,
    )

    published = publish_qualified_activation(
        activation,
        activation_path=activation_path,
        shadow_pointer_path=pointer_path,
    )

    assert read_qualified_activation(activation_path) == published == activation
    assert publish_qualified_activation(
        activation,
        activation_path=activation_path,
        shadow_pointer_path=pointer_path,
    ) == activation

    replacement = copy.deepcopy(activation)
    replacement["canary_plan_sha256"] = "f" * 64
    replacement_body = {
        key: value for key, value in replacement.items() if key != "activation_sha256"
    }
    replacement["activation_sha256"] = _sha(replacement_body)
    with pytest.raises(
        UnifiedRecurrentQualifiedActivationStoreError,
        match="compare-and-swap",
    ):
        publish_qualified_activation(
            replacement,
            activation_path=activation_path,
            shadow_pointer_path=pointer_path,
        )

    assert deactivate_qualified_activation(
        activation_path=activation_path,
        expected_current_sha256=activation["activation_sha256"],
    ) == activation
    assert not activation_path.exists()


def test_nonserving_candidate_cannot_enter_durable_authority_store(tmp_path) -> None:
    manifest, lifecycle, pointer = _evidence()
    candidate = seal_qualified_activation(manifest, lifecycle, pointer)
    root = ensure_private_directory(tmp_path / "authority")
    pointer_path = root / "active.json"
    atomic_write_bytes(
        pointer_path,
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("ascii"),
        mode=0o600,
    )

    with pytest.raises(
        UnifiedRecurrentQualifiedActivationStoreError,
        match="requires persisted typed authority",
    ):
        publish_qualified_activation(
            candidate,
            activation_path=root / "qualified-active.json",
            shadow_pointer_path=pointer_path,
        )


def test_activation_publication_rejects_an_unrelated_shadow_pointer(tmp_path) -> None:
    _manifest, _lifecycle, pointer = _evidence()
    activation = _durable_activation()
    root = ensure_private_directory(tmp_path / "authority")
    pointer_path = root / "active.json"
    replacement = copy.deepcopy(pointer)
    replacement["package_id"] = "other-package"
    replacement_body = {
        key: value for key, value in replacement.items() if key != "pointer_sha256"
    }
    replacement["pointer_sha256"] = _sha(replacement_body)
    atomic_write_bytes(
        pointer_path,
        json.dumps(replacement, sort_keys=True, separators=(",", ":")).encode("ascii"),
        mode=0o600,
    )

    with pytest.raises(
        UnifiedRecurrentQualifiedActivationStoreError,
        match="pointer identity differs",
    ):
        publish_qualified_activation(
            activation,
            activation_path=root / "qualified-active.json",
            shadow_pointer_path=pointer_path,
        )


def test_activation_publication_requires_shared_pointer_custody(tmp_path) -> None:
    _manifest, _lifecycle, pointer = _evidence()
    activation = _durable_activation()
    pointer_root = ensure_private_directory(tmp_path / "pointer-authority")
    activation_root = ensure_private_directory(tmp_path / "other-authority")
    pointer_path = pointer_root / "active.json"
    atomic_write_bytes(
        pointer_path,
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("ascii"),
        mode=0o600,
    )

    with pytest.raises(
        UnifiedRecurrentQualifiedActivationStoreError,
        match="custody differ",
    ):
        publish_qualified_activation(
            activation,
            activation_path=activation_root / "qualified-active.json",
            shadow_pointer_path=pointer_path,
        )
