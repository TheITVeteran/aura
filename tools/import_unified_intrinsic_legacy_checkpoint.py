#!/usr/bin/env python3
"""Import one hash-pinned legacy controller into immutable checkpoint custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_text,
    ensure_private_directory,
    interprocess_file_lock,
)
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    _canonical_sha256,
    _publish_latest_checkpoint_generation,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    resolve_checkpoint_generation,
)

IMPORT_SCHEMA = "aura.unified_intrinsic.legacy_import.v1"
MAX_RECEIPT_BYTES = 256 * 1024 * 1024
MAX_WEIGHTS_BYTES = 64 * 1024 * 1024 * 1024


def _stable_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise RuntimeError("legacy checkpoint artifact is a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or not 0 < before.st_size <= max_bytes
            ):
                raise RuntimeError("legacy checkpoint artifact custody differs")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RuntimeError("legacy checkpoint artifact is unreadable") from exc
    payload = b"".join(chunks)
    if (
        remaining
        or len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise RuntimeError("legacy checkpoint artifact changed while read")
    return payload


def import_legacy_checkpoint(
    *,
    receipt_path: Path,
    weights_path: Path,
    output_dir: Path,
    stem: str,
    expected_checkpoint_sha256: str,
    expected_identity_sha256: str,
) -> dict[str, Any]:
    """Verify a historical flat checkpoint and republish its exact bytes once."""

    if not stem.startswith("checkpoint_") or not stem.replace("_", "").isalnum():
        raise ValueError("legacy checkpoint stem is invalid")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (expected_checkpoint_sha256, expected_identity_sha256)
    ):
        raise ValueError("legacy checkpoint expected commitment is invalid")

    receipt_raw = _stable_bytes(receipt_path.expanduser(), max_bytes=MAX_RECEIPT_BYTES)
    weights = _stable_bytes(weights_path.expanduser(), max_bytes=MAX_WEIGHTS_BYTES)
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RuntimeError("legacy checkpoint receipt is invalid") from exc
    identity = receipt.get("identity") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "aura.unified_intrinsic_training.v1"
        or not isinstance(identity, dict)
        or identity.get("window_tissue_mode") != "controller_only"
        or type(receipt.get("step")) is not int
        or int(receipt["step"]) < 0
        or not isinstance(receipt.get("history"), list)
        or not isinstance(receipt.get("optimization_phase"), str)
    ):
        raise RuntimeError("legacy checkpoint receipt contract differs")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    identity_body = {
        key: value for key, value in identity.items() if key != "identity_sha256"
    }
    checkpoint_sha256 = hashlib.sha256(weights).hexdigest()
    if (
        receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
        or identity.get("identity_sha256") != _canonical_sha256(identity_body)
        or checkpoint_sha256 != receipt.get("checkpoint_sha256")
        or checkpoint_sha256 != expected_checkpoint_sha256
        or identity.get("identity_sha256") != expected_identity_sha256
    ):
        raise RuntimeError("legacy checkpoint commitment differs")

    tensors = mx.load(str(weights_path.expanduser()))
    if _stable_bytes(weights_path.expanduser(), max_bytes=MAX_WEIGHTS_BYTES) != weights:
        raise RuntimeError("legacy checkpoint changed during tensor inspection")
    bundle_names = sorted(name for name in tensors if name.startswith("bundle."))
    if (
        not bundle_names
        or any(not name.startswith("bundle.controller.") for name in bundle_names)
        or any(value.size < 1 for value in tensors.values())
    ):
        raise RuntimeError("legacy checkpoint is not portable controller-only tissue")

    destination = output_dir.expanduser()
    ensure_private_directory(destination)
    if any(destination.iterdir()):
        raise RuntimeError("legacy checkpoint import destination is not empty")
    import_evidence = {
        "schema": IMPORT_SCHEMA,
        "source_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "source_receipt_commitment": receipt["receipt_sha256"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_identity_sha256": identity["identity_sha256"],
        "bundle_tensor_count": len(bundle_names),
        "optimizer_tensor_count": sum(
            name.startswith("optimizer.") for name in tensors
        ),
        "bytes_preserved_exactly": True,
        "optimizer_bootstrap_authority": False,
    }
    training_state = dict(receipt.get("training_state", {}))
    training_state["legacy_import"] = import_evidence
    with interprocess_file_lock(destination / ".unified_checkpoint.lock"):
        complete = _publish_latest_checkpoint_generation(
            destination,
            stem=stem,
            payload=weights,
            step=int(receipt["step"]),
            history=list(receipt["history"]),
            identity=identity,
            optimization_phase=receipt["optimization_phase"],
            training_state=training_state,
        )
    resolved = resolve_checkpoint_generation(destination, stem=stem, required=True)
    if (
        resolved is None
        or resolved.receipt["checkpoint_sha256"] != checkpoint_sha256
        or resolved.receipt["identity"]["identity_sha256"]
        != expected_identity_sha256
    ):
        raise RuntimeError("immutable legacy checkpoint import verification failed")
    result_body = {
        **import_evidence,
        "destination": str(destination.resolve(strict=True)),
        "stem": stem,
        "step": complete["step"],
        "checkpoint_id": complete["checkpoint_id"],
        "complete_receipt_sha256": complete["receipt_sha256"],
        "pointer": resolved.pointer,
    }
    result = {**result_body, "receipt_sha256": _canonical_sha256(result_body)}
    atomic_write_text(
        destination / "legacy_import_receipt.json",
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        mode=0o400,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="checkpoint_best_trained")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-identity-sha256", required=True)
    args = parser.parse_args()
    result = import_legacy_checkpoint(
        receipt_path=args.receipt,
        weights_path=args.weights,
        output_dir=args.out_dir,
        stem=args.stem,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_identity_sha256=args.expected_identity_sha256,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
