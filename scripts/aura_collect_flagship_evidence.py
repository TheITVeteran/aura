#!/usr/bin/env python3
"""Collect a flagship-readiness evidence bundle for Aura.

This does not claim metaphysical proof. It creates a concrete artifact with:
- source health gate results
- task ownership findings
- persistence audit findings
- morphogenesis file/integration presence
- recent log evidence when logs are present
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

_COMMAND_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)
_READ_RECOVERABLE_ERRORS = (OSError, UnicodeDecodeError)


def run_cmd(cmd: list[str], cwd: Path, *, source: str) -> dict[str, Any]:
    started = time.time()
    try:
        proc = get_subprocess_gateway().run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            timeout=90,
            offline_tooling=True,
            source=source,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:],
            "duration_s": round(time.time() - started, 3),
        }
    except _COMMAND_RECOVERABLE_ERRORS as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}", "duration_s": round(time.time() - started, 3)}


def read_tail(path: Path, max_chars: int = 5000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[-max_chars:]
    except _READ_RECOVERABLE_ERRORS:
        return ""


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def find_logs(root: Path) -> list[Path]:
    candidates = []
    for base in [root / "logs", Path.home() / ".aura" / "logs"]:
        if base.exists():
            candidates.extend(sorted(base.glob("*.log"), key=_safe_mtime, reverse=True)[:8])
    return candidates


def collect(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "schema": "aura.flagship.evidence.v1",
        "created_at": time.time(),
        "root": str(root),
        "python": sys.version,
        "checks": {},
        "presence": {},
        "logs": {},
    }

    evidence["presence"] = {
        "aura_main": (root / "aura_main.py").exists(),
        "morphogenesis_runtime": (root / "core" / "morphogenesis" / "runtime.py").exists(),
        "morphogenesis_hooks": (root / "core" / "morphogenesis" / "hooks.py").exists(),
        "task_ownership": (root / "core" / "runtime" / "task_ownership.py").exists(),
        "persistence_ownership": (root / "core" / "runtime" / "persistence_ownership.py").exists(),
        "flagship_readiness": (root / "core" / "runtime" / "flagship_readiness.py").exists(),
    }

    commands = {
        "flagship_readiness": [sys.executable, "-m", "core.runtime.flagship_readiness", "--json", "."],
        "task_ownership": [sys.executable, "scripts/aura_task_ownership_codemod.py", ".", "--json"],
        "persistence_audit": [sys.executable, "scripts/aura_persistence_audit.py", ".", "--json"],
    }
    for name, cmd in commands.items():
        if (root / cmd[1]).exists() or cmd[1] == "-m":
            evidence["checks"][name] = run_cmd(
                cmd,
                root,
                source=f"maintenance_tooling:flagship_evidence:{name}",
            )
        else:
            evidence["checks"][name] = {"skipped": True, "reason": f"{cmd[1]} not found"}

    for log in find_logs(root):
        tail = read_tail(log)
        evidence["logs"][str(log)] = {
            "tail": tail,
            "contains_morphogenesis_started": "MorphogeneticRuntime started" in tail,
            "contains_hooks_wired": "Morphogenesis hooks" in tail or "Morphogenesis hooks wired" in tail,
            "contains_consciousness_online": "Consciousness System ONLINE" in tail,
        }

    json_path = out_dir / "flagship_evidence.json"
    atomic_write_text(json_path, json.dumps(evidence, indent=2, sort_keys=True, default=repr), encoding="utf-8")

    md_lines = [
        "# Aura Flagship Evidence Bundle",
        "",
        f"Created: {time.ctime(evidence['created_at'])}",
        f"Root: `{root}`",
        "",
        "## Presence",
    ]
    for k, v in evidence["presence"].items():
        md_lines.append(f"- {k}: {'yes' if v else 'no'}")
    md_lines.append("")
    md_lines.append("## Checks")
    for k, v in evidence["checks"].items():
        rc = v.get("returncode", "skipped" if v.get("skipped") else "error")
        md_lines.append(f"- {k}: {rc}")
    md_lines.append("")
    md_lines.append("See `flagship_evidence.json` for complete stdout/stderr/log tails.")
    atomic_write_text((out_dir / "flagship_evidence.md"), "\n".join(md_lines), encoding="utf-8")

    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--out", default="flagship_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    evidence = collect(root, out)
    print(f"Wrote evidence bundle to {out}")
    print(json.dumps({"presence": evidence["presence"], "checks": {k: v.get("returncode", v.get("skipped")) for k, v in evidence["checks"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
