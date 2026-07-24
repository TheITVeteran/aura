#!/usr/bin/env python3
"""Train and evaluate an adversarial RLC mistake locator from paired captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.adversarial_verifier_curriculum import (  # noqa: E402
    AdversarialVerifierCurriculum,
    VerifiedAdversarialPair,
    VerifiedNegativeStore,
)
from core.runtime.atomic_writer import atomic_write_json  # noqa: E402

INPUT_SCHEMA = "aura.rlc.adversarial_curriculum_input.v1"


def _read_input(path: Path) -> list[VerifiedAdversarialPair]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("adversarial curriculum input file is invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "pairs"}
        or payload.get("schema") != INPUT_SCHEMA
        or not isinstance(payload.get("pairs"), list)
    ):
        raise ValueError("adversarial curriculum input schema differs")
    return [VerifiedAdversarialPair.from_dict(row) for row in payload["pairs"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--negative-store", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    store = VerifiedNegativeStore(args.negative_store)
    try:
        result = AdversarialVerifierCurriculum(rounds=args.rounds, seed=args.seed).run(
            _read_input(args.input),
            repo_root=REPO_ROOT,
            negative_store=store,
        )
        head_sha256 = result.head.save(args.head)
        report = {
            **result.report,
            "head_path": str(args.head.resolve()),
            "head_file_sha256": head_sha256,
        }
        atomic_write_json(
            args.report,
            report,
            schema_version=1,
            schema_name="aura.rlc.adversarial_curriculum_training_report",
        )
    finally:
        store.close()
    print(
        json.dumps(
            {
                "head": str(args.head),
                "head_sha256": head_sha256,
                "report": str(args.report),
                "admitted": result.head.admitted,
            },
            sort_keys=True,
        )
    )
    return 0 if result.head.admitted else 3


if __name__ == "__main__":
    raise SystemExit(main())
