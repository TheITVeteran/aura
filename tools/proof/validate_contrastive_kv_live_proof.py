#!/usr/bin/env python3
"""Validate a live 32B contrastive/KV proof artifact.

This is deliberately separate from the live runner: the runner proves Aura can
boot and converse; this validator proves the requested foreground contrastive
path was actually active instead of silently falling back to the ordinary 32B
lane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_STDOUT_MARKERS = (
    "contrastive decoding active",
    "Reasoning processors ACTIVE",
    "cd=True",
    "amateur KV cache active",
)

FAILURE_MARKERS = (
    "LIVE PROOF FAILED",
    "Circuit OPEN",
    "Desktop CognitiveEngine produced no acceptable reply",
    "CognitiveEngine desktop chat reply failed reliability gate",
    "Request queue timeout",
    "memory_pressure_refused_worker_spawn",
    "worker_not_alive",
    "conversation_lane:cold",
    "conversation_lane:warming",
    "client_returned_no_text",
)


def _latest(path: Path, suffix: str) -> Path | None:
    matches = sorted(path.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(proof_dir: Path, *, max_peak_rss_mb: float) -> tuple[bool, dict[str, Any]]:
    verdict_path = proof_dir / "LATEST_VERDICT.json"
    if not verdict_path.exists():
        verdict_path = _latest(proof_dir, "_verdict.json") or verdict_path
    stdout_path = _latest(proof_dir, "_stdout.log")

    findings: list[str] = []
    if not verdict_path.exists():
        findings.append(f"missing verdict artifact: {verdict_path}")
        verdict: dict[str, Any] = {}
    else:
        verdict = _load_json(verdict_path)

    stdout = ""
    if stdout_path is None:
        findings.append("missing stdout log artifact")
    else:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")

    if verdict and verdict.get("passed") is not True:
        findings.append("live proof verdict did not pass")

    peak_rss = float(verdict.get("peak_rss_mb") or 0.0)
    if peak_rss <= 0.0:
        findings.append("missing peak_rss_mb in verdict")
    elif peak_rss > max_peak_rss_mb:
        findings.append(f"peak RSS {peak_rss:.1f}MB exceeded ceiling {max_peak_rss_mb:.1f}MB")

    missing_markers = [marker for marker in REQUIRED_STDOUT_MARKERS if marker not in stdout]
    if missing_markers:
        findings.append(f"missing required stdout markers: {', '.join(missing_markers)}")

    present_failures = [marker for marker in FAILURE_MARKERS if marker in stdout]
    if present_failures:
        findings.append(f"failure markers present in stdout: {', '.join(present_failures)}")

    steps = verdict.get("steps") or []
    required_steps = {
        "boot_health",
        "chat_identity",
        "chat_continuity",
        "chat_conversation_soak",
        "desktop_action",
        "shutdown",
        "runtime_stream_scan",
    }
    passed_steps = {str(step.get("step")) for step in steps if step.get("ok") is True}
    missing_steps = sorted(required_steps - passed_steps)
    if missing_steps:
        findings.append(f"missing passed proof steps: {', '.join(missing_steps)}")

    report = {
        "schema": "aura.contrastive_kv_live_proof_validation.v1",
        "proof_dir": str(proof_dir),
        "verdict_path": str(verdict_path) if verdict_path.exists() else "",
        "stdout_path": str(stdout_path) if stdout_path else "",
        "passed": not findings,
        "findings": findings,
        "peak_rss_mb": peak_rss,
        "max_peak_rss_mb": max_peak_rss_mb,
        "required_stdout_markers": list(REQUIRED_STDOUT_MARKERS),
        "git_commit": verdict.get("git_commit"),
        "git_dirty": verdict.get("git_dirty"),
    }
    return not findings, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof_dir", type=Path)
    parser.add_argument("--max-peak-rss-mb", type=float, default=32_768.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    ok, report = validate(args.proof_dir, max_peak_rss_mb=args.max_peak_rss_mb)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
