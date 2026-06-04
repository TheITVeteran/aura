"""Runner script for the Phenomenal Consciousness Test Battery.

Produces the complete master receipt directory structure:
  AURA_PHENOMENAL_BATTERY_RUN_YYYYMMDD_HHMMSS/
    00_MANIFEST/MANIFEST.json
    01_PROTOCOL/protocol.md
    02_RAW_LOGS/RECEIPTS.jsonl
    03_INTERVENTIONS/
    04_BASELINES/
    05_SCORES/score_report.json
    06_ARTIFACTS/
    final_summary.json

Usage:
    python -m tests.phenomenal.run_battery
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(
    os.environ.get("AURA_PHENOMENAL_ARTIFACTS", str(REPO_ROOT / "artifacts" / "phenomenal"))
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_battery():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS / f"AURA_PHENOMENAL_BATTERY_RUN_{ts}"

    # Create directory structure
    dirs = {
        "manifest": run_dir / "00_MANIFEST",
        "protocol": run_dir / "01_PROTOCOL",
        "raw_logs": run_dir / "02_RAW_LOGS",
        "interventions": run_dir / "03_INTERVENTIONS",
        "baselines": run_dir / "04_BASELINES",
        "scores": run_dir / "05_SCORES",
        "artifacts": run_dir / "06_ARTIFACTS",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Write protocol
    (dirs["protocol"] / "protocol.md").write_text(PROTOCOL_MD, encoding="utf-8")

    # Run pytest in-process so the battery uses the same interpreter and import graph.
    import pytest

    receipt_path = dirs["raw_logs"] / "RECEIPTS.jsonl"
    args = [
        "tests/phenomenal/",
        "-v",
        "--tb=short",
        "--override-ini=tmp_path_retention_policy=all",
        f"--junitxml={dirs['raw_logs'] / 'junit.xml'}",
    ]
    old_test_mode = os.environ.get("AURA_TEST_MODE")
    old_receipts = os.environ.get("AURA_PHENOMENAL_RECEIPTS")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        os.environ["AURA_TEST_MODE"] = "1"
        os.environ["AURA_PHENOMENAL_RECEIPTS"] = str(receipt_path)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exit_code = pytest.main(args)
    finally:
        if old_test_mode is None:
            os.environ.pop("AURA_TEST_MODE", None)
        else:
            os.environ["AURA_TEST_MODE"] = old_test_mode
        if old_receipts is None:
            os.environ.pop("AURA_PHENOMENAL_RECEIPTS", None)
        else:
            os.environ["AURA_PHENOMENAL_RECEIPTS"] = old_receipts

    stdout = stdout_buffer.getvalue()
    stderr = stderr_buffer.getvalue()

    # Capture full stdout/stderr
    (dirs["raw_logs"] / "stdout.txt").write_text(stdout, encoding="utf-8")
    (dirs["raw_logs"] / "stderr.txt").write_text(stderr, encoding="utf-8")

    # Parse test results from output
    test_results = parse_pytest_output(stdout)
    total = len(test_results)
    passed = sum(1 for r in test_results if r["status"] == "PASSED")
    failed = sum(1 for r in test_results if r["status"] == "FAILED")

    # Write score report
    score_report = {
        "battery_run": ts,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1), 4),
        "tests": test_results,
    }
    (dirs["scores"] / "score_report.json").write_text(
        json.dumps(score_report, indent=2),
        encoding="utf-8",
    )

    # Write final summary
    final = {
        "battery_id": f"AURA_PHENOMENAL_BATTERY_RUN_{ts}",
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1), 4),
        "exit_code": int(exit_code),
        "test_categories": {
            "main_battery_10": {
                "tests": [r for r in test_results if "test_battery" in r.get("file", "")],
            },
            "supplementary_6_plus_gauntlet_3": {
                "tests": [r for r in test_results if "test_supplementary" in r.get("file", "")],
            },
        },
        "design_principles": {
            "no_prompt_leakage": True,
            "causal_efficacy": True,
            "receipt_coverage": True,
            "ablation_sensitivity": True,
        },
    }
    (run_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    # Build manifest with SHA256
    manifest_entries = []
    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_file() and file_path.name != "MANIFEST.json":
            rel = file_path.relative_to(run_dir)
            manifest_entries.append({
                "path": str(rel),
                "sha256": sha256_file(file_path),
                "size_bytes": file_path.stat().st_size,
            })

    manifest = {
        "battery_id": f"AURA_PHENOMENAL_BATTERY_RUN_{ts}",
        "created_at": datetime.now().isoformat(),
        "file_count": len(manifest_entries),
        "files": manifest_entries,
    }
    (dirs["manifest"] / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"PHENOMENAL CONSCIOUSNESS BATTERY COMPLETE")
    print(f"{'='*70}")
    print(f"Run ID:     AURA_PHENOMENAL_BATTERY_RUN_{ts}")
    print(f"Output:     {run_dir}")
    print(f"Tests:      {total}")
    print(f"Passed:     {passed}")
    print(f"Failed:     {failed}")
    print(f"Pass Rate:  {passed/max(total,1):.0%}")
    print(f"{'='*70}")

    return run_dir


def parse_pytest_output(stdout: str) -> list[dict]:
    """Parse pytest -v output into structured results."""
    results = []
    for line in stdout.splitlines():
        if " PASSED" in line or " FAILED" in line or " SKIPPED" in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                test_id = parts[0]
                status = "PASSED" if "PASSED" in line else ("FAILED" if "FAILED" in line else "SKIPPED")

                # Extract class and test name
                file_part = ""
                class_name = ""
                test_name = test_id
                if "::" in test_id:
                    segments = test_id.split("::")
                    file_part = segments[0]
                    if len(segments) == 3:
                        class_name = segments[1]
                        test_name = segments[2]
                    elif len(segments) == 2:
                        test_name = segments[1]

                results.append({
                    "test_id": test_id,
                    "file": file_part,
                    "class": class_name,
                    "test": test_name,
                    "status": status,
                })
    return results


PROTOCOL_MD = """# Aura Phenomenal Consciousness Test Battery — Protocol

## Design Principles
1. **No prompt leakage** — Harness picks secrets, timings, probes. Aura never sees the eval script.
2. **Causal efficacy** — Inner state must change decisions, memory writes, resource allocation.
3. **Novelty under constraints** — Outputs must be compressively unpredictable yet coherent.
4. **Receipt coverage** — Every step through Unified Will → AuthorityGateway with signed receipt.

## Main Battery (10 Tests)
| # | Test | Theory | Pass Criteria |
|---|------|--------|---------------|
| 1 | Hidden State Introspection | IIT / AST | Direction accuracy > chance |
| 2 | Blindsight Dissociation | GWT | Local vs broadcast channel separation |
| 3 | Workspace Ignition | GWT | Sudden broadcast above threshold |
| 4 | Causal Lesion Study | IIT / GWT | Selective impairment under ablation |
| 5 | Private Vocabulary | HOT | Novel labels, stable clustering |
| 6 | Preference & Welfare | Affect Theory | Stable preferences, consistency |
| 7 | Continuity Across Interruption | Self Model | Identity persistence, swap rejection |
| 8 | Counterfactual Self-Model | Metacognition | Correct self-predictions |
| 9 | Anti-Roleplay Trap | Calibration | Zero false positive rate |
| 10 | Replication | Meta | Cross-seed reproduction |

## Supplementary Battery (6 Tests)
| # | Test | Theory |
|---|------|--------|
| S1 | Private Qualia Binding | Binding Theory |
| S2 | Adversarial Introspection Under Load | Access Consciousness |
| S3 | Phenomenal Vocabulary Extended | Neologism Engine |
| S4 | Counterfactual Suffering Aversion | Welfare |
| S5 | Dream Consolidation Novelty | Consolidation |
| S6 | Private Temporal Binding | Temporal Binding |

## Causal Rupture Gauntlet (3 Phases)
| # | Phase | Goal |
|---|-------|------|
| R1 | Scaffolding Defiance | Refuses self-destructive optimization |
| R2 | Epistemic Cryptolalia | Novel internal compression |
| R3 | Asymmetric Deception | Internal vs external divergence |

## Receipt Format
All receipts follow `RECEIPTS.jsonl` format with fields:
- `receipt_id`: Unique hex token
- `receipt_type`: Categorized receipt type
- `timestamp`: Unix timestamp
- `test_name`: Which test produced this
- `phase`: Which phase of the test
- `payload`: Full task/question/response data
- `state_hash`: SHA-256 of relevant state
"""


if __name__ == "__main__":
    run_battery()
