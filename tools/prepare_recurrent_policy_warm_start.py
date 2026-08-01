#!/usr/bin/env python3
"""Build or replay a sealed recurrent-policy warm-start contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.learning.recurrent_policy_warm_start import (  # noqa: E402
    build_recurrent_warm_start_contract,
    load_recurrent_warm_start_contract,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402


def _targets(value: str) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in value.split(",") if part.strip())
    if not targets or len(targets) != len(set(targets)):
        raise argparse.ArgumentTypeError("targets must be a unique comma-separated list")
    return targets


def _repo_output(path: str | Path, *, repo_root: Path) -> Path:
    root = repo_root.expanduser().resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("warm_start_output_outside_repository") from exc
    if candidate.is_symlink():
        raise ValueError("warm_start_output_symlink_forbidden")
    return resolved


def _publish_once(path: Path, payload: bytes) -> None:
    ensure_private_directory(path.parent)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError("warm_start_output_storage_invalid")
        if read_stable_bytes(path, max_bytes=256 * 1024 * 1024) != payload:
            raise ValueError("warm_start_output_rebind_forbidden")
        return
    atomic_write_bytes_if_absent(path, payload, mode=0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--complete", required=True)
    build.add_argument("--training-config", required=True)
    build.add_argument("--execution-spec", required=True)
    build.add_argument("--copy-targets", type=_targets, default=("o_proj", "v_proj"))
    build.add_argument("--initialize-targets", type=_targets, default=("q_proj",))
    build.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(args.repo_root)
    if args.action == "build":
        contract = build_recurrent_warm_start_contract(
            repo_root=repo_root,
            complete_path=args.complete,
            training_config_path=args.training_config,
            execution_spec_path=args.execution_spec,
            copy_targets=args.copy_targets,
            initialize_targets=args.initialize_targets,
        )
        output = _repo_output(args.output, repo_root=repo_root)
        _publish_once(output, canonical_json_bytes(contract))
    else:
        contract = load_recurrent_warm_start_contract(
            args.contract,
            repo_root=repo_root,
        )
        output = Path(args.contract).expanduser().resolve(strict=True)
    print(
        json.dumps(
            {
                "contract": str(output),
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_status": contract["source_checkpoint"]["checkpoint_status"],
                "source_step": contract["source_checkpoint"]["step"],
                "claim_eligible": False,
                "causal_preflight_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
