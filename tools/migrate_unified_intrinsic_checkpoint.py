#!/usr/bin/env python3
"""Rebind an exact unified-recurrence checkpoint across a source-only repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
    interprocess_file_lock,
)
from tools.run_unified_intrinsic_resident_campaign import _load_config  # noqa: E402
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    TRAINING_SOURCE_FILES,
    _publish_latest_checkpoint_generation,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    campaign_checkpoint_binding,
    canonical_bytes,
    canonical_sha256,
)

MIGRATION_SCHEMA = "aura.unified_intrinsic.checkpoint_source_migration.v2"
MIGRATION_FILENAME = "checkpoint-source-migration.json"


class UnifiedIntrinsicMigrationError(RuntimeError):
    """The source repair cannot preserve the exact training state."""


def _fail(code: str) -> Never:
    raise UnifiedIntrinsicMigrationError(code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256s(source_root: Path) -> dict[str, str]:
    return {relative: _sha256_file(source_root / relative) for relative in TRAINING_SOURCE_FILES}


def _controller_sha256(weights_path: Path, identity: dict[str, Any]) -> str:
    tensors = mx.load(str(weights_path))
    values = [
        (name.removeprefix("bundle.controller."), value)
        for name, value in tensors.items()
        if name.startswith("bundle.controller.")
    ]
    if not values:
        _fail("migration_controller_tensors_missing")
    literal = identity.get("literal_observation_contract")
    opcode = identity.get("opcode_observation_contract")
    if not isinstance(literal, dict) or not isinstance(opcode, dict):
        _fail("migration_controller_contract_missing")
    correction = tensors.get("bundle.controller.correction_a")
    if correction is None or len(correction.shape) != 2:
        _fail("migration_controller_shape_invalid")
    numeric = identity.get("numeric_observation_contract")
    family = identity.get("frontier_family_observation_contract")
    family_patterns = (
        tuple(
            (int(row["opcode"]), tuple(int(value) for value in row["token_ids"]))
            for row in family.get("patterns", ())
        )
        if isinstance(family, dict)
        else ()
    )
    config = UnifiedRecurrenceConfig(
        hidden_size=int(correction.shape[0]),
        correction_rank=int(identity["controller_rank"]),
        state_slots=int(identity.get("state_slots", 5)),
        depth_basis_size=int(identity["depth_basis_size"]),
        minimum_iterations=1,
        literal_digit_token_ids=tuple(int(value) for value in literal["digit_token_ids"]),
        numeric_observation_max_value=(
            int(numeric["max_value"])
            if isinstance(numeric, dict) and "max_value" in numeric
            else 32
        ),
        opcode_token_patterns=tuple(
            (int(row["opcode"]), tuple(int(value) for value in row["token_ids"]))
            for row in opcode["patterns"]
        ),
        opcode_context_patterns=tuple(
            (str(row["name"]), tuple(int(value) for value in row["token_ids"]))
            for row in opcode["contexts"]
        ),
        frontier_family_token_patterns=family_patterns,
        initialization_seed=int(identity["init_seed"]),
    )
    controller = UnifiedRecurrentController(config)
    expected = set(dict(controller.parameters()))
    if {name for name, _value in values} != expected:
        _fail("migration_controller_tensor_inventory_drift")
    controller.update(tree_unflatten(values))
    mx.eval(controller.parameters())
    return controller.parameter_sha256()


def _target_identity(
    source_identity: dict[str, Any],
    *,
    target_config: dict[str, Any],
    controller_sha256: str,
    allowed_source_changes: frozenset[str],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    target = copy.deepcopy(source_identity)
    target.pop("identity_sha256", None)
    source_hashes = source_identity.get("source_sha256s")
    if not isinstance(source_hashes, dict):
        _fail("migration_source_hashes_missing")
    target_root = Path(target_config["source"]["git"]["root"])
    target_hashes = _source_sha256s(target_root)
    differences = {
        name: {"source": str(source_hashes.get(name)), "target": target_hashes[name]}
        for name in TRAINING_SOURCE_FILES
        if source_hashes.get(name) != target_hashes[name]
    }
    if frozenset(differences) != allowed_source_changes:
        _fail("migration_source_change_set_differs")

    target_bootstrap = target_config.get("bootstrap")
    if not isinstance(target_bootstrap, dict):
        _fail("migration_target_bootstrap_missing")
    source_binding = source_identity.get("campaign_binding")
    target_binding = campaign_checkpoint_binding(target_config)
    if not isinstance(source_binding, dict):
        _fail("migration_source_campaign_binding_missing")
    if target_binding.get("training_profile_sha256") != source_binding.get(
        "training_profile_sha256"
    ):
        _fail("migration_training_profile_changed")
    for field in (
        "model_manifest_sha256",
        "runtime_identity_sha256",
        "dataset_identity_sha256",
        "tokenizer_identity_sha256",
        "tokenized_dataset_identity_sha256",
    ):
        if target_binding.get(field) != source_binding.get(field):
            _fail(f"migration_scientific_input_changed:{field}")
    original_bootstrap = source_identity.get("bootstrap")
    if original_bootstrap is not None and not isinstance(original_bootstrap, dict):
        _fail("migration_source_bootstrap_missing")
    target["source_sha256s"] = target_hashes
    target["campaign_binding"] = target_binding
    # This is one experiment crossing a source-only repair. The operational
    # bootstrap points at the resume checkpoint, but the scientific bootstrap
    # remains the experiment's original initialization.
    target["bootstrap"] = copy.deepcopy(original_bootstrap)
    # A source-only migration resumes the same experiment.  The controller in
    # the migrated payload is the current resume state, not a new scientific
    # initialization.  Re-labelling it as the initial controller silently
    # collapses trained and matched-control arms after every repair migration.
    initial_controller_sha256 = source_identity.get("initial_controller_sha256")
    if not isinstance(initial_controller_sha256, str) or len(initial_controller_sha256) != 64:
        _fail("migration_initial_controller_identity_missing")
    target["initial_controller_sha256"] = initial_controller_sha256
    target["source_migration_controller_sha256"] = controller_sha256
    target["identity_sha256"] = canonical_sha256(target)
    return target, differences


def migrate(
    *,
    source_output: Path,
    target_config_path: Path,
    allowed_source_changes: frozenset[str],
    stem: str = "checkpoint_latest",
) -> dict[str, Any]:
    if not allowed_source_changes:
        _fail("migration_source_change_set_empty")
    try:
        source = resolve_checkpoint_generation(source_output, stem=stem, required=True)
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise UnifiedIntrinsicMigrationError("migration_source_checkpoint_invalid") from exc
    if source is None:  # pragma: no cover - required=True is authoritative
        _fail("migration_source_checkpoint_missing")
    config = _load_config(target_config_path)
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, dict) or any(
        bootstrap.get(key) != source.receipt.get(receipt_key)
        for key, receipt_key in (
            ("parent_step", "step"),
            ("parent_checkpoint_sha256", "checkpoint_sha256"),
            ("parent_receipt_sha256", "receipt_sha256"),
        )
    ):
        _fail("migration_target_parent_differs")
    source_identity = source.receipt.get("identity")
    if not isinstance(source_identity, dict) or bootstrap.get(
        "parent_identity_sha256"
    ) != source_identity.get("identity_sha256"):
        _fail("migration_source_identity_invalid")
    migration_tool_sha256 = _sha256_file(Path(__file__).resolve(strict=True))

    target_output = Path(config["paths"]["training_output"])
    observed = target_output.stat()
    if (
        target_output.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or any(target_output.iterdir())
    ):
        _fail("migration_target_output_not_empty")

    controller_sha = _controller_sha256(source.weights_path, source_identity)
    target_identity, source_differences = _target_identity(
        source_identity,
        target_config=config,
        controller_sha256=controller_sha,
        allowed_source_changes=allowed_source_changes,
    )
    payload = source.weights_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source.receipt["checkpoint_sha256"]:
        _fail("migration_source_payload_drift")

    intent_body = {
        "schema": MIGRATION_SCHEMA,
        "state": "intent",
        "source_checkpoint_sha256": source.receipt["checkpoint_sha256"],
        "source_receipt_sha256": source.receipt["receipt_sha256"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "target_config_sha256": config["config_sha256"],
        "target_identity_sha256": target_identity["identity_sha256"],
        "step": source.receipt["step"],
        "allowed_source_changes": sorted(allowed_source_changes),
        "migration_tool_sha256": migration_tool_sha256,
    }
    root = Path(config["paths"]["campaign_root"])
    atomic_write_bytes_if_absent(
        root / "checkpoint-source-migration-intent.json",
        canonical_bytes({**intent_body, "intent_sha256": canonical_sha256(intent_body)}) + b"\n",
        mode=0o400,
    )
    with interprocess_file_lock(target_output / ".unified_checkpoint.lock"):
        complete = _publish_latest_checkpoint_generation(
            target_output,
            stem=stem,
            payload=payload,
            step=int(source.receipt["step"]),
            history=copy.deepcopy(source.receipt.get("history", [])),
            identity=target_identity,
            optimization_phase=str(source.receipt["optimization_phase"]),
            training_state=copy.deepcopy(source.receipt.get("training_state", {})),
        )
    migrated = resolve_checkpoint_generation(target_output, stem=stem, required=True)
    if (
        migrated is None
        or migrated.receipt["checkpoint_sha256"] != source.receipt["checkpoint_sha256"]
        or migrated.receipt["step"] != source.receipt["step"]
        or migrated.receipt["history"] != source.receipt.get("history", [])
        or migrated.receipt["training_state"] != source.receipt.get("training_state", {})
        or migrated.receipt["identity"] != target_identity
    ):
        _fail("migration_destination_verification_failed")
    body = {
        "schema": MIGRATION_SCHEMA,
        "state": "complete",
        "source": {
            "output": str(source_output.expanduser().resolve(strict=True)),
            "generation": source.generation_dir.name,
            "step": source.receipt["step"],
            "checkpoint_sha256": source.receipt["checkpoint_sha256"],
            "receipt_sha256": source.receipt["receipt_sha256"],
            "identity_sha256": source_identity["identity_sha256"],
        },
        "destination": {
            "campaign_id": config["campaign_id"],
            "config_sha256": config["config_sha256"],
            "generation": migrated.generation_dir.name,
            "step": migrated.receipt["step"],
            "checkpoint_sha256": migrated.receipt["checkpoint_sha256"],
            "receipt_sha256": migrated.receipt["receipt_sha256"],
            "identity_sha256": target_identity["identity_sha256"],
        },
        "source_differences": source_differences,
        "migration_tool_sha256": migration_tool_sha256,
        "target_source_commit": config["source"]["git"]["commit"],
        "target_source_manifest_sha256": config["source"]["manifest"]["manifest_sha256"],
        "payload_byte_identical": migrated.weights_path.read_bytes() == payload,
        "optimizer_and_bundle_bytes_preserved": True,
        "history_preserved": True,
        "training_state_preserved": True,
        "scientific_initialization_preserved": True,
        "training_profile_preserved": True,
        "complete_receipt_sha256": complete["receipt_sha256"],
    }
    receipt = {**body, "migration_sha256": canonical_sha256(body)}
    atomic_write_bytes_if_absent(
        root / MIGRATION_FILENAME,
        canonical_bytes(receipt) + b"\n",
        mode=0o400,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--stem", default="checkpoint_latest")
    parser.add_argument("--allow-source-change", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = migrate(
            source_output=args.source_output,
            target_config_path=args.target_config,
            allowed_source_changes=frozenset(args.allow_source_change),
            stem=args.stem,
        )
    except Exception as exc:  # noqa: BLE001 - stable migration CLI boundary
        print(
            f"migrate_unified_intrinsic_checkpoint: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
