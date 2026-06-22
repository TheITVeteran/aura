#!/usr/bin/env python3
"""Local environmental-agency proof (#36).

Demonstrates that Aura's environmental-agency pipeline does *real, verifiable*
local work and honors the safe-mode brake — end to end, no model required:

  1. builds a throwaway workspace with known contents,
  2. runs the workspace-digest pipeline and verifies a real digest file was
     written with the correct survey (file/dir counts, sizes),
  3. re-runs it with safe mode ON and verifies the brake held (survey ran, but
     nothing was written),
  4. confirms a provable run-ledger line was recorded each time.

Writes a verdict JSON and exits 0 on pass / 1 on failure, so it can be wrapped by
tools/run_proof_step.py in the certification chain.

    python tools/environmental_agency_proof.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agency.environment_pipeline import run_workspace_digest  # noqa: E402


def _seed(ws: Path) -> None:
    (ws / "src").mkdir(parents=True)
    (ws / "notes").mkdir()
    (ws / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (ws / "src" / "util.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (ws / "notes" / "todo.md").write_text("# todo\n" * 100, encoding="utf-8")
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/current/environmental_agency")
    args = ap.parse_args()

    started = time.monotonic()
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'OK' if ok else 'XX'}] {name}{' — ' + detail if detail else ''}")

    with tempfile.TemporaryDirectory(prefix="aura_envproof_") as tmp:
        ws = Path(tmp) / "workspace"
        ws.mkdir()
        _seed(ws)
        ledger = Path(tmp) / "runs.jsonl"

        # 1. normal run writes a real digest
        r = run_workspace_digest(ws, safe_mode=False, ledger_path=ledger)
        digest = ws / ".aura" / "workspace_digest.md"
        check("survey counted the 3 real files (skipping .git)", r.total_files == 3,
              f"counted {r.total_files}")
        check("survey counted the 2 folders", r.total_dirs == 2, f"counted {r.total_dirs}")
        check("digest file written to disk", digest.exists() and r.wrote_digest)
        body = digest.read_text(encoding="utf-8") if digest.exists() else ""
        check("digest contains the real survey", "3 files" in body and "Workspace digest" in body)
        check("run recorded a ledger line", ledger.exists() and bool(ledger.read_text().strip()))

        # 2. safe mode holds the brake
        digest.unlink(missing_ok=True)
        r2 = run_workspace_digest(ws, safe_mode=True, ledger_path=ledger)
        check("safe mode still surveyed (real read-only work)", r2.total_files == 3 and r2.success)
        check("safe mode did NOT write the digest", (not r2.wrote_digest) and (not digest.exists()))

        ledger_lines = len([ln for ln in ledger.read_text().splitlines() if ln.strip()])
        check("both runs are provable in the ledger", ledger_lines == 2, f"{ledger_lines} lines")

    passed = all(c["ok"] for c in checks)
    verdict = {
        "schema": "aura.environmental_agency_proof.v1",
        "passed": passed,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(time.monotonic() - started, 2),
        "checks": checks,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nEnvironmental-agency proof: {'PASS' if passed else 'FAIL'} "
          f"({sum(c['ok'] for c in checks)}/{len(checks)} checks) → {out / 'VERDICT.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
