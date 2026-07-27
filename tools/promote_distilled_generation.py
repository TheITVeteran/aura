#!/usr/bin/env python3
"""Run the SPARK-064 promotion transaction from the command line.

Three subcommands, matching the three things an operator actually does at the
end of a campaign:

  open      start a lineage at the frozen pre-treatment artifact
  promote   present gate evidence and either extend the lineage or be told,
            by name, which battery blocked it
  rollback  restore an earlier generation, proving the restored bytes match

`promote` exits non-zero on refusal and prints every responsible gate. That is
the intended way to use it in a campaign script: a refusal is a result, not an
error to be retried until it passes.

The gate report is read from JSON rather than constructed here, because this
tool must not be able to invent a battery that did not run. It can only relay
what the batteries reported, and the transaction refuses an incomplete relay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.permanent_distillation import (  # noqa: E402
    REQUIRED_GATES,
    PermanentDistillationError,
    PermanentDistillationRefusalError,
    baseline_generation,
    gate_report,
    observed_artifact_manifest,
    promote_generation,
    rollback_generation,
)
from core.learning.permanent_distillation_registry import (  # noqa: E402
    append_generation,
    load_lineage,
    write_lineage,
)


def _artifact(args: argparse.Namespace) -> dict:
    return observed_artifact_manifest(
        artifact_id=args.artifact_id,
        base_model_identity=args.base_model,
        adapter_identity=args.adapter_identity,
        root=Path(args.artifact_root),
        names=list(args.file),
    )


def _report(path: str) -> dict:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    rows = document["gates"] if isinstance(document, dict) else document
    return gate_report(rows)


def _print_refusal(decision: dict) -> None:
    print("PROMOTION REFUSED", file=sys.stderr)
    for row in decision["refusals"]:
        detail = ", ".join(
            f"{key}={value}" for key, value in sorted(row.items()) if key != "gate"
        )
        print(f"  {row['gate']}: {detail}", file=sys.stderr)
    print(
        f"  required gates: {', '.join(REQUIRED_GATES)}",
        file=sys.stderr,
    )


def _open(args: argparse.Namespace) -> int:
    record = baseline_generation(
        artifact=_artifact(args),
        provenance=json.loads(args.provenance),
        created_at_unix=int(time.time()),
    )
    head = write_lineage(args.registry, [record])
    print(f"opened lineage at generation 0 ({head})")
    return 0


def _promote(args: argparse.Namespace) -> int:
    lineage = load_lineage(args.registry)
    try:
        record = promote_generation(
            lineage=lineage,
            artifact=_artifact(args),
            report=_report(args.gate_report),
            provenance=json.loads(args.provenance),
            created_at_unix=int(time.time()),
        )
    except PermanentDistillationRefusalError as refusal:
        _print_refusal(refusal.decision)
        return 2
    head = append_generation(args.registry, record)
    print(f"promoted to generation {record['generation_index']} ({head})")
    return 0


def _rollback(args: argparse.Namespace) -> int:
    lineage = load_lineage(args.registry)
    record = rollback_generation(
        lineage=lineage,
        restores_generation_sha256=args.restore,
        observed_artifact=_artifact(args),
        provenance=json.loads(args.provenance),
        created_at_unix=int(time.time()),
    )
    head = append_generation(args.registry, record)
    print(
        f"rolled back to the artifact of {args.restore[:12]}… "
        f"as generation {record['generation_index']} ({head})"
    )
    return 0


def _artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-identity", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--file",
        action="append",
        required=True,
        help="artifact file name, relative to --artifact-root; repeatable",
    )
    parser.add_argument("--provenance", default="{}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    opener = sub.add_parser("open", help="open a lineage at the frozen baseline")
    _artifact_arguments(opener)
    opener.set_defaults(handler=_open)

    promoter = sub.add_parser("promote", help="promote a candidate past every gate")
    _artifact_arguments(promoter)
    promoter.add_argument(
        "--gate-report",
        required=True,
        help="JSON file holding the results of all seven regression batteries",
    )
    promoter.set_defaults(handler=_promote)

    reverter = sub.add_parser("rollback", help="restore an earlier generation exactly")
    _artifact_arguments(reverter)
    reverter.add_argument(
        "--restore",
        required=True,
        help="generation_sha256 of the generation whose artifact is now on disk",
    )
    reverter.set_defaults(handler=_rollback)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PermanentDistillationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
