from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.unified_recurrent_qualified_activation import (
    activation_matches_shadow_receipt,
    qualified_activation_errors,
    qualified_activation_load_receipt_errors,
    seal_qualified_activation,
    seal_qualified_activation_load_receipt,
)
from core.brain.llm.unified_recurrent_qualified_activation_store import (
    UnifiedRecurrentQualifiedActivationStoreError,
    deactivate_qualified_activation,
    publish_qualified_activation,
    read_qualified_activation,
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


def test_supported_lifecycle_can_issue_typed_only_authority() -> None:
    manifest, lifecycle, pointer = _evidence()

    activation = seal_qualified_activation(manifest, lifecycle, pointer)

    assert qualified_activation_errors(activation) == []
    assert activation["serving_authority"] is True
    assert activation["mode"] == "qualified_typed_only"
    assert activation["ordinary_chat_authorized"] is False
    assert activation["arbitrary_reasoning_authorized"] is False


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
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)

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
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)

    receipt = seal_qualified_activation_load_receipt(
        configured=True,
        loaded=True,
        reason="qualified_activation_loaded",
        activation=activation,
    )
    receipt["loaded"] = False
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha(body)

    assert "inactive_qualified_activation_claims_state" in (
        qualified_activation_load_receipt_errors(receipt)
    )


def test_activation_publication_is_pointer_bound_cas_and_revocable(tmp_path) -> None:
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)
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


def test_activation_publication_rejects_an_unrelated_shadow_pointer(tmp_path) -> None:
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)
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
    manifest, lifecycle, pointer = _evidence()
    activation = seal_qualified_activation(manifest, lifecycle, pointer)
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
