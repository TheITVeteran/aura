#!/usr/bin/env python3
"""Build a release certificate from evidence that was actually produced.

A certificate assembled by hand certifies the assembler's optimism. This
collects what the gates really said, and submits nothing it did not
observe — a requirement with no harness stays MISSING, which blocks the
release, which is the point.

Usage::

    # Run the gates this tool knows how to run, then certify.
    PYTHON=.venv/bin/python python tools/build_release_certificate.py --run-gates

    # Or fold in evidence produced elsewhere (a soak, a chaos run).
    python tools/build_release_certificate.py \\
        --evidence conversation_soak=pass:turns=200 \\
        --evidence memory_ceiling=fail:peak_gb=71.2

Exit code is 0 only when the certificate certifies. Nothing here prints
"certified" without every blocking requirement being satisfied by evidence
tied to this commit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.runtime.release_certificate import (  # noqa: E402
    REQUIREMENTS,
    CertificateBuilder,
    RequirementStatus,
    current_commit,
)

#: Gates this tool can run itself, and the requirement each one evidences.
#: Deliberately short. A gate mapped here that does not actually establish
#: its requirement would be worse than leaving the requirement MISSING,
#: because MISSING is honest.
_RUNNABLE_GATES: dict[str, tuple[str, ...]] = {
    "hermetic_shards": ("make", "smoke"),
}


def _run_gate(key: str, command: tuple[str, ...], python: str) -> tuple[bool, dict]:
    started = time.time()
    try:
        result = subprocess.run(
            list(command),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
            env={**__import__("os").environ, "PYTHON": python},
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    return result.returncode == 0, {
        "command": " ".join(command),
        "returncode": result.returncode,
        "duration_s": round(time.time() - started, 2),
        "tail": result.stdout.strip().splitlines()[-3:],
    }


def _parse_evidence(raw: str) -> tuple[str, bool, dict]:
    """``key=pass:k=v,k=v`` -> (key, passed, detail)."""
    key, _, rest = raw.partition("=")
    verdict, _, detail_text = rest.partition(":")
    verdict = verdict.strip().lower()
    if verdict not in {"pass", "fail"}:
        raise ValueError(f"{raw!r}: verdict must be 'pass' or 'fail', not {verdict!r}")
    detail: dict[str, str] = {}
    for pair in filter(None, (part.strip() for part in detail_text.split(","))):
        name, _, value = pair.partition("=")
        detail[name.strip()] = value.strip()
    return key.strip(), verdict == "pass", detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gates", action="store_true", help="Run the gates this tool knows")
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="KEY=pass|fail[:k=v,...]",
        help="Evidence produced elsewhere",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = current_commit()
    builder = CertificateBuilder(commit=commit)
    print(f"Building release certificate for {commit[:12]}")

    if args.run_gates:
        for key, command in _RUNNABLE_GATES.items():
            passed, detail = _run_gate(key, command, args.python)
            builder.submit(key, passed=passed, detail=detail, produced_by=" ".join(command))
            print(f"  {'PASS' if passed else 'FAIL'}  {key}  ({' '.join(command)})")

    for raw in args.evidence:
        try:
            key, passed, detail = _parse_evidence(raw)
            builder.submit(key, passed=passed, detail=detail, produced_by="operator_supplied")
            print(f"  {'PASS' if passed else 'FAIL'}  {key}  (supplied)")
        except (KeyError, ValueError) as exc:
            print(f"  REJECTED  {raw}: {exc}", file=sys.stderr)
            return 2

    certificate = builder.build()
    print()
    print(certificate.summary())

    blocked = [
        result
        for result in certificate.results
        if result.status in (RequirementStatus.MISSING, RequirementStatus.STALE)
    ]
    if blocked:
        print("\nNo evidence tied to this commit for:")
        for result in blocked:
            print(f"  - {result.requirement.key}: {result.requirement.description}")
        print(
            "\nThese are not failures. They are unmeasured, which is why the "
            "certificate refuses: an unrun check must never read as a passed one."
        )

    out = args.out or (ROOT / "artifacts" / "release" / f"certificate-{commit[:12]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(certificate.to_dict(), indent=2), encoding="utf-8")
    print(f"\nWritten to {out.relative_to(ROOT)}")
    return 0 if certificate.certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
