#!/usr/bin/env python3
"""Train and report the task-disjoint prefix-stability calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.learning.prefix_stability import (
    CALIBRATION_TARGET,
    PrefixStabilityCalibrator,
    PrefixStabilityExample,
)
from core.runtime.file_read_gateway import read_stable_bytes

MAX_DATASET_BYTES = 32 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_examples_with_sha(
    path: Path,
) -> tuple[list[PrefixStabilityExample], str]:
    raw = read_stable_bytes(path, max_bytes=MAX_DATASET_BYTES)
    if not raw.strip():
        raise ValueError(f"{path} is empty")
    examples: list[PrefixStabilityExample] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_strict_object)
            examples.append(PrefixStabilityExample.from_dict(value))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid example: {exc}") from exc
    if not examples:
        raise ValueError(f"{path} has no examples")
    return examples, hashlib.sha256(raw).hexdigest()


def _load_examples(path: Path) -> list[PrefixStabilityExample]:
    examples, _digest = _load_examples_with_sha(path)
    return examples


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def train(
    *,
    fit_path: Path,
    calibration_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    fit, fit_source_sha256 = _load_examples_with_sha(fit_path)
    calibration, calibration_source_sha256 = _load_examples_with_sha(
        calibration_path
    )
    calibrator = PrefixStabilityCalibrator.fit(fit, calibration)
    artifact_sha = calibrator.save(output_path)
    report = {
        "schema": "aura.rlc.prefix_stability_training_report.v1",
        "target": CALIBRATION_TARGET,
        "fit_source_sha256": fit_source_sha256,
        "calibration_source_sha256": calibration_source_sha256,
        "artifact_path": str(output_path),
        "artifact_sha256": artifact_sha,
        "runtime_eligible": calibrator.admitted,
        "manifest": calibrator.manifest(),
        "claim_boundary": (
            "calibrates future conclusion recurrence only; does not estimate "
            "correctness and grants no branch-selection authority"
        ),
    }
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope(
        "prefix_stability.training_report",
        domain="file_write",
    ):
        get_file_write_gateway().write_bytes(
            report_path,
            _canonical(report),
            source="prefix_stability.training_report",
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = train(
        fit_path=arguments.fit_jsonl,
        calibration_path=arguments.calibration_jsonl,
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["runtime_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
